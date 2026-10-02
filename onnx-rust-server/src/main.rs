// laya-rust-server — drop-in Rust replacement for the Node/ONNX warm server.
// Protocol: POST /decide, POST /mcp (streamable-http), GET /health,
// POST /admin/{load,unload}. Hybrid residency identical to the Node server.
use std::io::Read;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use serde_json::{json, Value};
use tiny_http::Server;

mod model;

const PRESSURE_GRACE_MS: u128 = 120_000;

static READY: AtomicBool = AtomicBool::new(false);
static LOADING: AtomicBool = AtomicBool::new(false);
static INFLIGHT: AtomicUsize = AtomicUsize::new(0);
static UNLOAD_COUNT: AtomicUsize = AtomicUsize::new(0);
static LAST_USED_MS: Mutex<u128> = Mutex::new(0);
static READY_SINCE_MS: Mutex<u128> = Mutex::new(0);
static LOAD_ERROR: Mutex<Option<String>> = Mutex::new(None);

struct Engine {
	session: Mutex<ort::session::Session>,
	tok: tokenizers::Tokenizer,
	cls: u32,
	sep: u32,
	mask: u32,
	mask_tok: String,
	cfg: Value,
	model_dir: String,
}
static ENGINE: RwLock<Option<Arc<Engine>>> = RwLock::new(None);

fn now_ms() -> u128 {
	std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0)
}

fn mono_ms() -> u128 {
	std::time::Instant::now().elapsed().as_millis()
}

fn env_u64(name: &str, d: u64) -> u64 {
	std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn load_model(model_dir: &str, threads: usize) {
	if READY.load(Ordering::SeqCst) || LOADING.swap(true, Ordering::SeqCst) {
		return;
	}
	*LOAD_ERROR.lock().unwrap() = None;
	let model_dir = model_dir.to_string();
	std::thread::spawn(move || {
		// catch_unwind: ort panics (in its own thread) when the dylib is missing —
		// never wedge the LOADING flag on a panic.
		let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<Engine, String> {
			let read_json = |p: &str| -> Result<Value, String> {
				let raw = std::fs::read_to_string(p).map_err(|e| e.to_string())?;
				serde_json::from_str(&raw).map_err(|e| e.to_string())
			};
			let cfg = read_json(&format!("{model_dir}/laya_config.json"))?;
			let tok_cfg = read_json(&format!("{model_dir}/tokenizer/tokenizer_config.json"))?;
			let tok = tokenizers::Tokenizer::from_file(&format!("{model_dir}/tokenizer/tokenizer.json"))
				.map_err(|e| e.to_string())?;
			let id_of = |t: &str| -> Result<u32, String> {
				tok.token_to_id(t)
					.ok_or_else(|| format!("special token {t} missing from tokenizer"))
			};
			let cls = id_of(tok_cfg.get("cls_token").and_then(|v| v.as_str()).unwrap_or("[CLS]"))?;
			let sep = id_of(tok_cfg.get("sep_token").and_then(|v| v.as_str()).unwrap_or("[SEP]"))?;
			let mask = id_of(tok_cfg.get("mask_token").and_then(|v| v.as_str()).unwrap_or("[MASK]"))?;
			let mask_tok = tok_cfg.get("mask_token").and_then(|v| v.as_str()).unwrap_or("[MASK]").to_string();
			let builder = ort::session::Session::builder().map_err(|e| e.to_string())?;
			let mut session = builder
				.with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
				.map_err(|e| e.to_string())?
				.with_intra_threads(threads)
				.map_err(|e| e.to_string())?
				.commit_from_file(&format!("{model_dir}/laya.onnx"))
				.map_err(|e| e.to_string())?;
			Ok(Engine { session: Mutex::new(session), tok, cls, sep, mask, mask_tok, cfg, model_dir })
		}));
		let result = match outcome {
			Ok(r) => r,
			Err(p) => {
				let msg = p
					.downcast_ref::<String>()
					.cloned()
					.or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
					.unwrap_or_else(|| format!("non-string panic payload: {:?}", p.type_id()));
				eprintln!("[laya-rust] LOAD THREAD PANIC: {msg}");
				Err(format!("panicked: {msg}"))
			}
		};
		match result {
			Ok(engine) => {
				*ENGINE.write().unwrap() = Some(Arc::new(engine));
				READY.store(true, Ordering::SeqCst);
				*READY_SINCE_MS.lock().unwrap() = mono_ms();
				*LAST_USED_MS.lock().unwrap() = mono_ms();
				println!("[laya-rust] ready (load took {:.1}s since boot)", mono_ms() as f64 / 1000.0);
			}
			Err(e) => {
				*LOAD_ERROR.lock().unwrap() = Some(e.clone());
				println!("[laya-rust] load failed: {e}");
			}
		}
		LOADING.store(false, Ordering::SeqCst);
	});
}

fn void<T>(_: &T) {}

fn unload_model() -> bool {
	if INFLIGHT.load(Ordering::SeqCst) > 0 {
		return false;
	}
	let mut guard = ENGINE.write().unwrap();
	if guard.is_none() {
		return false;
	}
	*guard = None;
	READY.store(false, Ordering::SeqCst);
	UNLOAD_COUNT.fetch_add(1, Ordering::SeqCst);
	println!("[laya-rust] unloaded weights");
	true
}

fn mem_pressure_level() -> i32 {
	std::process::Command::new("/usr/sbin/sysctl")
		.arg("-n")
		.arg("kern.memorystatus_vm_pressure_level")
		.output()
		.ok()
		.and_then(|o| String::from_utf8(o.stdout).ok()?.trim().parse().ok())
		.unwrap_or(1)
}

fn watchdog(idle_secs: u64) {
	let _ = idle_secs;
	loop {
		std::thread::sleep(std::time::Duration::from_secs(15));
		if LOADING.load(Ordering::SeqCst) {
			continue;
		}
		let pressure = mem_pressure_level();
		let since_ready = mono_ms().saturating_sub(*READY_SINCE_MS.lock().unwrap());
		if READY.load(Ordering::SeqCst) && pressure >= 2 && since_ready >= PRESSURE_GRACE_MS {
			println!("[laya-rust] memory pressure level={pressure} — unloading weights");
			unload_model();
			continue;
		}
		if idle_secs > 0
			&& READY.load(Ordering::SeqCst)
			&& mono_ms().saturating_sub(*LAST_USED_MS.lock().unwrap()) >= idle_secs as u128 * 1000
		{
			unload_model();
		}
	}
}

fn read_body(req: &mut tiny_http::Request) -> Result<Value, String> {
	let mut body = String::new();
	req.as_reader().read_to_string(&mut body).map_err(|e| e.to_string())?;
	serde_json::from_str(&body).map_err(|e| e.to_string())
}

fn main() {
	let port = env_u64("LAYA_PORT", 8755);
	let idle_secs = env_u64("LAYA_IDLE_UNLOAD_SECS", 1800);
	let model_dir = std::env::var("LAYA_MODEL_DIR").unwrap_or_else(|_| "onnx-export/fp32".into());
	// ort's load-dynamic feature panics (in its own thread) when the dylib is
	// missing — default it to the onnxruntime dylib that ships with
	// onnxruntime-node so the server works out of the box.
	if std::env::var("ORT_DYLIB_PATH").is_err() {
		let default_dylib = std::env::current_dir()
			.unwrap_or_default()
			.join("../onnx-server/node_modules/onnxruntime-node/bin/napi-v6/darwin/arm64/libonnxruntime.1.30.0.dylib");
		std::env::set_var("ORT_DYLIB_PATH", default_dylib);
	}
	std::thread::spawn(move || load_model(&model_dir, env_u64("LAYA_THREADS", 4) as usize));
	std::thread::spawn(move || watchdog(idle_secs));

	let server = Arc::new(Server::http(("127.0.0.1", port as u16)).expect("bind failed"));
	println!("[laya-rust] listening on 127.0.0.1:{port} (idle unload {idle_secs}s, runtime rust)");
	let server2 = server.clone();
	for _ in 0..3 {
		let s = server2.clone();
		std::thread::spawn(move || loop {
			match s.recv() {
				Ok(req) => handle(req),
				Err(e) => println!("[laya-rust] recv error: {e}"),
			}
		});
	}
	loop {
		match server.recv() {
			Ok(req) => handle(req),
			Err(e) => println!("[laya-rust] recv error: {e}"),
		}
	}
}

fn handle(mut req: tiny_http::Request) {
	let method = req.method().as_str().to_string();
	let url = req.url().to_string();
	let result = route(&mut req, &method, &url);
	match result {
		Ok((code, body, sse)) => {
			let data = body.into_bytes();
			let header = if sse {
				tiny_http::Header::from_bytes(&b"content-type"[..], &b"text/event-stream"[..]).unwrap()
			} else {
				tiny_http::Header::from_bytes(&b"content-type"[..], &b"application/json"[..]).unwrap()
			};
			let _ = req.respond(
				tiny_http::Response::from_data(data)
					.with_status_code(code)
					.with_header(header),
			);
		}
		Err(e) => {
			let body = format!("{{\"ok\":false,\"error\":\"{e}\"}}");
			let _ = req.respond(tiny_http::Response::from_string(body).with_status_code(500));
		}
	}
}

fn route(req: &mut tiny_http::Request, method: &str, url: &str) -> Result<(u16, String, bool), String> {
	if method == "GET" {
		if url == "/mcp" {
			return Ok((405, String::new(), false));
		}
		if url == "/health" || url == "/" {
			let body = json!({
				"ok": true,
				"runtime": "rust-ort",
				"ready": READY.load(Ordering::SeqCst),
				"loaded": READY.load(Ordering::SeqCst),
				"loading": LOADING.load(Ordering::SeqCst),
				"idleUnloadSecs": env_u64("LAYA_IDLE_UNLOAD_SECS", 1800),
				"idleForSecs": (mono_ms().saturating_sub(*LAST_USED_MS.lock().unwrap())) as f64 / 1000.0,
				"memPressure": mem_pressure_level(),
				"inflight": INFLIGHT.load(Ordering::SeqCst),
				"unloadCount": UNLOAD_COUNT.load(Ordering::SeqCst),
				"model": std::env::var("LAYA_MODEL_DIR").unwrap_or_default(),
				"error": *LOAD_ERROR.lock().unwrap(),
			});
			return Ok((200, body.to_string(), false));
		}
		return Ok((404, "{\"ok\":false,\"error\":\"not found\"}".into(), false));
	}
	if method != "POST" {
		return Ok((404, "{\"ok\":false,\"error\":\"not found\"}".into(), false));
	}
	match url {
		"/admin/load" => {
			let was = READY.load(Ordering::SeqCst) || LOADING.load(Ordering::SeqCst);
			let model_dir = std::env::var("LAYA_MODEL_DIR").unwrap_or_else(|_| "onnx-export/fp32".into());
			if !was {
				std::thread::spawn(move || load_model(&model_dir, env_u64("LAYA_THREADS", 4) as usize));
			}
			Ok((200, json!({"ok": true, "started": !was, "ready": READY.load(Ordering::SeqCst)}).to_string(), false))
		}
		"/admin/unload" => {
			let ok = unload_model();
			Ok((200, json!({"ok": ok}).to_string(), false))
		}
		"/decide" => {
			let body = read_body(req)?;
			let state = body.get("state").cloned().unwrap_or(Value::Null);
			let questions = match body.get("questions").and_then(|q| q.as_object()) {
				Some(q) => q.clone(),
				None => return Ok((400, "{\"ok\":false,\"error\":\"questions object required\"}".into(), false)),
			};
			if !READY.load(Ordering::SeqCst) {
				let model_dir = std::env::var("LAYA_MODEL_DIR").unwrap_or_else(|_| "onnx-export/fp32".into());
				std::thread::spawn(move || load_model(&model_dir, env_u64("LAYA_THREADS", 4) as usize));
				let err = LOAD_ERROR.lock().unwrap().clone().unwrap_or_else(|| "model loading".into());
				return Ok((
					503,
					json!({"ok": false, "wakeAccepted": true, "error": err}).to_string(),
					false,
				));
			}
			let engine = { ENGINE.read().unwrap().clone() };
			let engine = match engine {
				Some(e) => e,
				None => return Ok((503, "{\"ok\":false,\"error\":\"model loading\"}".into(), false)),
			};
			INFLIGHT.fetch_add(1, Ordering::SeqCst);
			*LAST_USED_MS.lock().unwrap() = mono_ms();
			let result = {
				let mut session = engine.session.lock().unwrap();
				let encode = |s: &str| -> Vec<u32> {
					engine.tok.encode(s, false).map(|e| e.get_ids().to_vec()).unwrap_or_default()
				};
				let state_str = model::serialize_state(&state);
				model::predict(
					&mut session, &encode, engine.cls, engine.sep, engine.mask,
					&engine.mask_tok, &state_str, &questions, &engine.cfg,
				)
			};
			INFLIGHT.fetch_sub(1, Ordering::SeqCst);
			*LAST_USED_MS.lock().unwrap() = mono_ms();
			match result {
				Ok((answers, usage)) => {
					Ok((200, json!({"ok": true, "answers": answers, "usage": usage}).to_string(), false))
				}
				Err(e) => Ok((500, json!({"ok": false, "error": e}).to_string(), false)),
			}
		}
		"/mcp" => {
			let body = read_body(req)?;
			let method = body.get("method").and_then(|m| m.as_str()).unwrap_or("").to_string();
			let id = body.get("id").cloned().unwrap_or(Value::Null);
			let send_mcp = |payload: Value| -> (u16, String, bool) {
				(200, format!("event: message\ndata: {}\n\n", payload), true)
			};
			Ok(match method.as_str() {
				"initialize" => send_mcp(json!({
					"jsonrpc": "2.0", "id": id,
					"result": {"protocolVersion": "2024-11-05",
						"capabilities": {"tools": {"listChanged": false}},
						"serverInfo": {"name": "laya-rust", "version": "1.0.0"}}
				})),
				m if m.starts_with("notifications/") => (202, String::new(), false),
				"tools/list" => send_mcp(json!({
					"jsonrpc": "2.0", "id": id,
					"result": {"tools": [{
						"name": "jev_decide",
						"description": "Ask the warm local Laya System-1 decision model (Rust host, hybrid-resident). Calibrated probabilities in one forward pass. Not for nuanced reasoning or >16 options.",
						"inputSchema": {"type": "object",
							"properties": {"state": {"type": "string"}, "questions": {"type": "object"}},
							"required": ["state", "questions"]}}
					]}}
				)),
				"tools/call" => {
					let params = body.get("params").cloned().unwrap_or(Value::Null);
					let name = params.get("name").and_then(|n| n.as_str()).unwrap_or("");
					if name != "jev_decide" {
						return Ok(send_mcp(json!({"jsonrpc": "2.0", "id": id, "error": {"code": -32602, "message": format!("unknown tool {name}")}})));
					}
					let args = params.get("arguments").cloned().unwrap_or(Value::Null);
					let state = args.get("state").cloned().unwrap_or(Value::Null);
					let questions = args.get("questions").cloned().unwrap_or(json!({}));
					let text = if !READY.load(Ordering::SeqCst) {
						load_model_async();
						format!("Laya is loading (wake accepted); retry shortly. {}", LOAD_ERROR.lock().unwrap().clone().unwrap_or_default())
					} else {
						let engine = { ENGINE.read().unwrap().clone() };
						match engine {
							Some(engine) => {
								INFLIGHT.fetch_add(1, Ordering::SeqCst);
								*LAST_USED_MS.lock().unwrap() = mono_ms();
								let out = {
									let mut session = engine.session.lock().unwrap();
									let encode = |s: &str| -> Vec<u32> {
										engine.tok.encode(s, false).map(|e| e.get_ids().to_vec()).unwrap_or_default()
									};
									let state_str = model::serialize_state(&state);
									let qmap = questions.as_object().cloned().unwrap_or_default();
									model::predict(&mut session, &encode, engine.cls, engine.sep, engine.mask, &engine.mask_tok, &state_str, &qmap, &engine.cfg)
								};
								INFLIGHT.fetch_sub(1, Ordering::SeqCst);
								match out {
									Ok((a, u)) => json!({"answers": a, "usage": u}).to_string(),
									Err(e) => format!("Laya error: {e}"),
								}
							}
							None => "Laya not ready".into(),
						}
					};
					send_mcp(json!({"jsonrpc": "2.0", "id": id,
						"result": {"content": [{"type": "text", "text": text}], "isError": false}}))
				}
				other => {
					if id.is_null() {
						(202, String::new(), false)
					} else {
						send_mcp(json!({"jsonrpc": "2.0", "id": id, "error": {"code": -32601, "message": format!("method not supported: {other}")}}))
					}
				}
			})
		}
		_ => Ok((404, "{\"ok\":false,\"error\":\"not found\"}".into(), false)),
	}
}

fn load_model_async() {
	let model_dir = std::env::var("LAYA_MODEL_DIR").unwrap_or_else(|_| "onnx-export/fp32".into());
	std::thread::spawn(move || load_model(&model_dir, env_u64("LAYA_THREADS", 4) as usize));
}
