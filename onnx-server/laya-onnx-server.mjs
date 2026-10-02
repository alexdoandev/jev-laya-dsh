// laya-onnx-server.mjs — drop-in Node/ONNX replacement for the Python warm
// server, same wire protocol (turn-9 A/B candidate):
//
//   POST /decide {state, questions}   -> {ok, answers, usage}
//   POST /mcp    (JSON-RPC/SSE)       -> tool jev_decide (parity with python)
//   GET  /health                      -> {ok, ready, loaded, loading, ...}
//   POST /admin/load | /admin/unload  -> hybrid wake / free
//
// Hybrid residency identical to python v3.2: process resident, weights reload
// in seconds (ONNX mmap), idle unload (default 1800s) + macOS memory-pressure
// valve, single-flight load, inflight guard.
//
//   LAYA_PORT=8756 LAYA_MODEL_DIR=../onnx/fp32 node laya-onnx-server.mjs
import { createRequire } from "node:module";
import { execFile } from "node:child_process";
import http from "node:http";

const require = createRequire(import.meta.url);
const { Laya } = require("@receptron/laya");

const PORT = Number(process.env.LAYA_PORT || 8756);
const IDLE_UNLOAD_SECS = Number(process.env.LAYA_IDLE_UNLOAD_SECS || 1800);
const MODEL_DIR = process.env.LAYA_MODEL_DIR || new URL("./fp32", import.meta.url).pathname;

let LAYA = null;
let LOADING = false;
let LOAD_ERROR = null;
let LOAD_PROMISE = null;
let READY = false;
let READY_SINCE = 0;
let LAST_USED = Date.now();
let INFLIGHT = 0;
let UNLOAD_COUNT = 0;
const BOOT_TS = Date.now();
// Grace after a finished load: under sustained memory pressure, an immediate
// unload + caller wake + reload would thrash (seen live during the A/B eval).
// The valve only fires 120s after the latest load completed.
const PRESSURE_GRACE_MS = 120_000;

const MCP_TOOLS = [
	{
		name: "jev_decide",
		description:
			"Ask the warm local Laya System One decision model (ONNX runtime, hybrid-resident) to answer typed questions about a state in one fast pass. Returns calibrated probabilities. Do NOT use for nuanced reasoning or >16 options.",
		inputSchema: {
			type: "object",
			properties: { state: { type: "string" }, questions: { type: "object" } },
			required: ["state", "questions"],
		},
	},
];

function memPressureLevel() {
	return new Promise((resolve) => {
		execFile("/usr/sbin/sysctl", ["-n", "kern.memorystatus_vm_pressure_level"], { timeout: 2000 }, (err, out) => {
			if (err) return resolve(1);
			const n = Number(String(out).trim());
			resolve(Number.isFinite(n) && n > 0 ? n : 1);
		});
	});
}

function loadAsync() {
	if (READY || LOADING) return false;
	LOADING = true;
	LOAD_ERROR = null;
	LOAD_PROMISE = (async () => {
		try {
			// Cap intra-op threads (default 4): onnxruntime defaults to all cores,
			// which starves the rest of the box when the machine is already busy —
			// under load it is BOTH slower overall and a bad neighbor.
			const threads = Number(process.env.LAYA_THREADS || 4);
			console.log(`[laya-onnx] loading bundle ${MODEL_DIR} (intraOpNumThreads=${threads}) ...`);
			LAYA = await Laya.load({ modelDir: MODEL_DIR, sessionOptions: { intraOpNumThreads: threads } });
			READY = true;
			LAST_USED = Date.now();
			console.log(`[laya-onnx] ready (load took ${((Date.now() - BOOT_TS) / 1000).toFixed(1)}s since boot)`);
		} catch (error) {
			LOAD_ERROR = `${error && error.name ? error.name : "Error"}: ${error && error.message ? error.message : error}`;
			console.log(`[laya-onnx] load failed: ${LOAD_ERROR}`);
		} finally {
			LOADING = false;
			if (READY) READY_SINCE = Date.now();
		}
	})();
	return true;
}

async function unloadModel() {
	if (!READY || !LAYA) return false;
	LAYA = null;
	READY = false;
	UNLOAD_COUNT += 1;
	console.log(`[laya-onnx] unloaded weights (#${UNLOAD_COUNT})`);
	return true;
}

async function watchdog() {
	for (;;) {
		await new Promise((r) => setTimeout(r, 15_000));
		if (LOADING) continue;
		const pressure = await memPressureLevel();
		if (READY && pressure >= 2 && Date.now() - READY_SINCE >= PRESSURE_GRACE_MS) {
			console.log(`[laya-onnx] memory pressure level=${pressure} — unloading weights`);
			if (INFLIGHT === 0) await unloadModel();
			continue;
		}
		if (IDLE_UNLOAD_SECS > 0 && READY && Date.now() - LAST_USED >= IDLE_UNLOAD_SECS * 1000) {
			if (INFLIGHT === 0) await unloadModel();
		}
	}
}

async function predict(state, questions) {
	const out = await LAYA.systemOne(state, questions);
	return { answers: out.answers, usage: out.usage };
}

function send(res, code, body) {
	const buf = Buffer.from(body);
	res.writeHead(code, { "content-type": "application/json", "content-length": buf.length });
	res.end(buf);
}

function sendMcp(res, payload) {
	const frame = Buffer.from(`event: message\ndata: ${JSON.stringify(payload)}\n\n`);
	res.writeHead(200, { "content-type": "text/event-stream", "content-length": String(frame.length) });
	res.end(frame);
}

function readJson(req) {
	return new Promise((resolve, reject) => {
		let size = Number(req.headers["content-length"] || 0);
		const chunks = [];
		req.on("data", (c) => chunks.push(c));
		req.on("end", () => {
			try {
				resolve(JSON.parse(Buffer.concat(chunks).toString("utf8") || "null"));
			} catch (e) {
				reject(e);
			}
		});
		req.on("error", reject);
		void size;
	});
}

const server = http.createServer(async (req, res) => {
	try {
		if (req.method === "GET") {
			if (req.url === "/mcp") {
				res.writeHead(405, { allow: "POST", "content-length": "0" });
				return res.end();
			}
			return send(res, 200, JSON.stringify({
				ok: true, runtime: "onnx-node", ready: READY, loaded: READY, loading: LOADING,
				idleUnloadSecs: IDLE_UNLOAD_SECS, idleForSecs: Math.round((Date.now() - LAST_USED) / 100) / 10,
				memPressure: await memPressureLevel(), inflight: INFLIGHT, unloadCount: UNLOAD_COUNT,
				model: MODEL_DIR, error: LOAD_ERROR,
			}));
		}
		if (req.method !== "POST") return send(res, 404, '{"ok":false,"error":"not found"}');

		if (req.url === "/admin/load") {
			const started = loadAsync();
			return send(res, 200, JSON.stringify({ ok: true, started, ready: READY }));
		}
		if (req.url === "/admin/unload") {
			const ok = INFLIGHT === 0 ? await unloadModel() : false;
			return send(res, 200, JSON.stringify({ ok }));
		}

		const body = await readJson(req);
		if (req.url === "/decide") {
			if (!READY) {
				loadAsync();
				return send(res, 503, JSON.stringify({ ok: false, wakeAccepted: true, error: LOAD_ERROR || "model loading" }));
			}
			INFLIGHT += 1;
			LAST_USED = Date.now();
			try {
				const { answers, usage } = await predict(body.state, body.questions);
				return send(res, 200, JSON.stringify({ ok: true, answers, usage }));
			} catch (error) {
				return send(res, 500, JSON.stringify({ ok: false, error: `${error?.name || "Error"}: ${error?.message || error}` }));
			} finally {
				INFLIGHT -= 1;
				LAST_USED = Date.now();
			}
		}
		if (req.url === "/mcp") {
			const rpc = body;
			const id = rpc && rpc.id;
			if (rpc?.method === "initialize") {
				return sendMcp(res, { jsonrpc: "2.0", id, result: { protocolVersion: "2024-11-05", capabilities: { tools: { listChanged: false } }, serverInfo: { name: "laya-onnx", version: "1.0.0" } } });
			}
			if (typeof rpc?.method === "string" && rpc.method.startsWith("notifications/")) {
				res.writeHead(202);
				return res.end();
			}
			if (rpc?.method === "tools/list") return sendMcp(res, { jsonrpc: "2.0", id, result: { tools: MCP_TOOLS } });
			if (rpc?.method === "tools/call") {
				const p = rpc.params || {};
				if (p.name !== "jev_decide") return sendMcp(res, { jsonrpc: "2.0", id, error: { code: -32602, message: `unknown tool ${p.name}` } });
				const a = p.arguments || {};
				let text;
				if (!READY) {
					loadAsync();
					text = "Laya is loading (wake accepted); retry shortly. " + (LOAD_ERROR || "");
				} else {
					INFLIGHT += 1;
					LAST_USED = Date.now();
					try {
						const { answers, usage } = await predict(a.state, a.questions);
						text = JSON.stringify({ answers, usage });
					} catch (error) {
						text = `Laya error: ${error?.name || "Error"}: ${error?.message || error}`;
					} finally {
						INFLIGHT -= 1;
						LAST_USED = Date.now();
					}
				}
				return sendMcp(res, { jsonrpc: "2.0", id, result: { content: [{ type: "text", text }], isError: false } });
			}
			if (id === undefined || id === null) {
				res.writeHead(202);
				return res.end();
			}
			return sendMcp(res, { jsonrpc: "2.0", id, error: { code: -32601, message: `method not supported: ${rpc?.method}` } });
		}
		return send(res, 404, '{"ok":false,"error":"not found"}');
	} catch (error) {
		try { send(res, 500, JSON.stringify({ ok: false, error: `${error?.name || "Error"}: ${error?.message || error}` })); } catch { /* socket gone */ }
	}
});

server.listen(PORT, "127.0.0.1", () => {
	console.log(`[laya-onnx] listening on 127.0.0.1:${PORT} (idle unload ${IDLE_UNLOAD_SECS}s)`);
	loadAsync();
});
watchdog();
