// Model + sequence-building port of @receptron/laya (dist/{sequence,laya}.js).
// Faithful port: same specials handling, marker positions, temperature buckets,
// softmax, confidence (1 - normalized entropy), 4-decimal rounding, pyJsonDumps.
use ndarray::Array2;
use serde_json::{json, Map, Value};

pub struct Question {
	pub qtype: u8, // 0 choice, 1 score, 2 noul
	pub qname: &'static str,
	pub instructions: String,
	pub crit: Value,
}

pub fn to_internal(q: &Value) -> Question {
	let t = q["type"].as_str().unwrap_or("noul").to_string();
	let instructions = match &q["instructions"] {
		Value::String(s) => s.clone(),
		other => py_json_dumps(other),
	};
	let qtype = match t.as_str() {
		"choice" => 0u8,
		"score" => 1u8,
		_ => 2u8,
	};
	Question {
		qtype,
		qname: match qtype {
			0 => "choice",
			1 => "score",
			_ => "noul",
		},
		instructions,
		crit: q["criteria"].clone(),
	}
}

/// Option texts in label-index order. Noul is always [false, true] so p[1] == noul.
pub fn render_options(q: &Question) -> Vec<String> {
	match q.qtype {
		0 => match &q.crit {
			Value::Object(o) => o
				.iter()
				.map(|(k, v)| match v.as_str() {
					Some(s) => format!("{}: {}", k, s),
					None => k.clone(),
				})
				.collect(),
			Value::Array(a) => a
				.iter()
				.map(|v| v.as_str().map(|s| s.to_string()).unwrap_or_else(|| py_json_dumps(v)))
				.collect(),
			_ => vec![],
		},
		1 => match &q.crit {
			Value::Array(a) => a
				.iter()
				.enumerate()
				.map(|(i, c)| format!("level {}: {}", i, c.as_str().unwrap_or(&py_json_dumps(c))))
				.collect(),
			_ => vec![],
		},
		_ => {
			let (f, tr) = match q.crit.as_object() {
				Some(o) => (
					o.get("false").and_then(|v| v.as_str()).unwrap_or("no, the statement does not hold"),
					o.get("true").and_then(|v| v.as_str()).unwrap_or("yes, the statement holds"),
				),
				None => ("no, the statement does not hold", "yes, the statement holds"),
			};
			vec![format!("false: {f}"), format!("true: {tr}")]
		}
	}
}

/// Python json.dumps(obj, ensure_ascii=False): ", " / ": " separators, insertion key order.
pub fn py_json_dumps(v: &Value) -> String {
	match v {
		Value::Null => "null".into(),
		Value::String(s) => serde_json::to_string(s).unwrap_or_default(),
		Value::Number(n) => {
			if n.is_i64() || n.is_u64() {
				n.to_string()
			} else {
				let f = n.as_f64().unwrap_or(0.0);
				if f == f.trunc() && f.abs() < 1e21 {
					format!("{}", f as i64)
				} else {
					format!("{f}")
				}
			}
		}
		Value::Bool(b) => if *b { "true" } else { "false" }.into(),
		Value::Array(a) => {
			let items: Vec<String> = a.iter().map(py_json_dumps).collect();
			format!("[{}]", items.join(", "))
		}
		Value::Object(o) => {
			let items: Vec<String> = o
				.iter()
				.map(|(k, x)| format!("{}: {}", serde_json::to_string(k).unwrap_or_default(), py_json_dumps(x)))
				.collect();
			format!("{{{}}}", items.join(", "))
		}
	}
}

pub fn temp_bucket(qtype: u8, k: usize) -> String {
	let name = ["choice", "score", "noul"][qtype as usize];
	let bucket = if k <= 2 {
		"2"
	} else if k <= 5 {
		"3-5"
	} else if k <= 10 {
		"6-10"
	} else {
		"11+"
	};
	format!("{name}:{bucket}")
}

/// Jev-style confidence: 1 - normalized entropy of the answer distribution.
pub fn confidence_from_probs(p: &[f64]) -> f64 {
	let k = p.len();
	if k < 2 {
		return 1.0;
	}
	let mut ent = 0.0;
	for &x in p {
		ent -= x * x.max(1e-12).ln();
	}
	1.0 - ent / (k as f64).ln()
}

pub fn softmax(z: &[f64]) -> Vec<f64> {
	let zmax = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
	let e: Vec<f64> = z.iter().map(|v| (v - zmax).exp()).collect();
	let sum: f64 = e.iter().sum();
	e.iter().map(|v| v / sum).collect()
}

pub fn round4(x: f64) -> f64 {
	(x * 1e4).round() / 1e4
}

pub struct Built {
	pub ids: Vec<i64>,
	pub markers: Vec<i64>,
}

/// Port of buildSequence:
/// [CLS] <type> question: instructions [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]
/// Returns the ids and the position of each option's [MASK] marker.
pub fn build_sequence(
	encode: &dyn Fn(&str) -> Vec<u32>,
	cls: u32,
	sep: u32,
	mask: u32,
	mask_tok: &str,
	state: &str,
	q: &Question,
	max_len: usize,
	head_max_len: usize,
) -> Built {
	let scrub = |s: &str| s.split(mask_tok).collect::<Vec<_>>().join(" ");
	let opts = render_options(q);
	let mut head_ids = encode(&format!("{} question: {}", q.qname, scrub(&q.instructions)));
	let mut opt_ids: Vec<Vec<u32>> = opts
		.iter()
		.map(|o| {
			let mut v = vec![mask];
			v.extend(encode(&format!(" {}", scrub(o))).into_iter().take(48));
			v
		})
		.collect();
	let total: usize = opt_ids.iter().map(|o| o.len()).sum();
	let mut opt_budget = head_max_len as isize - total as isize;
	if opt_budget < 16 {
		let per = ((head_max_len as isize - 16) / opt_ids.len().max(1) as isize).max(4);
		for o in opt_ids.iter_mut() {
			o.truncate(per.max(0) as usize);
		}
		opt_budget = head_max_len as isize - opt_ids.iter().map(|o| o.len()).sum::<usize>() as isize;
	}
	let keep = (opt_budget.max(8) as usize).min(head_ids.len());
	head_ids.truncate(keep);
	let mut seq: Vec<i64> = vec![cls as i64];
	seq.extend(head_ids.iter().map(|&x| x as i64));
	seq.push(sep as i64);
	let mut markers: Vec<i64> = Vec::new();
	for o in &opt_ids {
		markers.push(seq.len() as i64);
		seq.extend(o.iter().map(|&x| x as i64));
	}
	seq.push(sep as i64);
	let room = (max_len as isize - seq.len() as isize - 1).max(0) as usize;
	let st: Vec<u32> = encode(&scrub(state)).into_iter().take(room).collect();
	seq.extend(st.iter().map(|&x| x as i64));
	seq.push(sep as i64);
	seq.truncate(max_len);
	Built {
		ids: seq,
		markers: markers.into_iter().filter(|&m| m < max_len as i64).collect(),
	}
}

pub fn serialize_state(state: &Value) -> String {
	match state {
		Value::String(s) => s.clone(),
		other => py_json_dumps(other),
	}
}

/// Batched inference for one /decide request; returns (answers, usage).
#[allow(clippy::too_many_arguments)]
pub fn predict(
	session: &mut ort::session::Session,
	encode: &dyn Fn(&str) -> Vec<u32>,
	cls: u32,
	sep: u32,
	mask: u32,
	mask_tok: &str,
	state: &str,
	questions: &Map<String, Value>,
	cfg: &Value,
) -> Result<(Value, Value), String> {
	let qids: Vec<String> = questions.keys().cloned().collect();
	if qids.is_empty() {
		return Err("systemOne: at least one question is required".into());
	}
	let max_len = cfg["max_len"].as_u64().unwrap_or(512) as usize;
	let head_max_len = cfg["head_max_len"].as_u64().unwrap_or(256) as usize;

	let mut items: Vec<(String, Question, Built)> = Vec::new();
	for qid in &qids {
		let q = to_internal(&questions[qid]);
		let built = build_sequence(encode, cls, sep, mask, mask_tok, state, &q, max_len, head_max_len);
		if built.markers.len() != render_options(&q).len() {
			return Err(format!("question {qid}: options do not fit in head_max_len={head_max_len}"));
		}
		items.push((qid.clone(), q, built));
	}
	let n = items.len();
	let seq_len = items.iter().map(|i| i.2.ids.len()).max().unwrap_or(1);
	let k_max = items.iter().map(|i| i.2.markers.len()).max().unwrap_or(1);

	let mut input_ids = vec![0i64; n * seq_len];
	let mut attention = vec![0i64; n * seq_len];
	let mut marker_pos = vec![0i64; n * k_max];
	let mut marker_mask = vec![false; n * k_max];
	let mut qtype = vec![0i64; n];
	let mut n_tokens = 0usize;
	for (i, item) in items.iter().enumerate() {
		for (j, &v) in item.2.ids.iter().enumerate() {
			input_ids[i * seq_len + j] = v;
			attention[i * seq_len + j] = 1;
		}
		n_tokens += item.2.ids.len();
		for (j, &m) in item.2.markers.iter().enumerate() {
			marker_pos[i * k_max + j] = m;
			marker_mask[i * k_max + j] = true;
		}
		qtype[i] = item.1.qtype as i64;
	}

	let ids_arr = Array2::from_shape_vec((n, seq_len), input_ids).map_err(|e| e.to_string())?;
	let att_arr = Array2::from_shape_vec((n, seq_len), attention).map_err(|e| e.to_string())?;
	let mp_arr = Array2::from_shape_vec((n, k_max), marker_pos).map_err(|e| e.to_string())?;
	let mm_arr = Array2::from_shape_vec((n, k_max), marker_mask).map_err(|e| e.to_string())?;
	let qt_arr = ndarray::Array1::from_shape_vec(n, qtype).map_err(|e| e.to_string())?;

	let ids_t = ort::value::Tensor::from_array(ids_arr).map_err(|e| e.to_string())?;
	let att_t = ort::value::Tensor::from_array(att_arr).map_err(|e| e.to_string())?;
	let mp_t = ort::value::Tensor::from_array(mp_arr).map_err(|e| e.to_string())?;
	let mm_t = ort::value::Tensor::from_array(mm_arr).map_err(|e| e.to_string())?;
	let qt_t = ort::value::Tensor::from_array(qt_arr).map_err(|e| e.to_string())?;

	let outputs = session
		.run(ort::inputs![
			"input_ids" => ids_t,
			"attention_mask" => att_t,
			"marker_pos" => mp_t,
			"marker_mask" => mm_t,
			"qtype" => qt_t,
		])
		.map_err(|e| format!("{e}"))?;

	let (_lshape, logit_data) = outputs["logits"]
		.try_extract_tensor::<f32>()
		.map_err(|e| e.to_string())?;
	let (_ashape, act_data) = outputs["act_probs"]
		.try_extract_tensor::<f32>()
		.map_err(|e| e.to_string())?;
	let act_row = act_data.len() / items.len().max(1);

	let temp_by_options = &cfg["temperature_by_options"];
	let temperature = &cfg["temperature"];

	let mut answers = Map::new();
	for (r, item) in items.iter().enumerate() {
		let kk = item.2.markers.len();
		let bucket = temp_bucket(item.1.qtype, kk);
		let temp = temp_by_options
			.get(&bucket)
			.and_then(|v| v.as_f64())
			.or_else(|| temperature.get(item.1.qtype as usize).and_then(|v| v.as_f64()))
			.unwrap_or(1.0);
		let z: Vec<f64> = (0..kk).map(|i| logit_data[r * k_max + i] as f64 / temp).collect();
		let p = softmax(&z);
		let act_probability = act_data[r * act_row + 0] as f64;
		let q = &item.1;
		let mut ans = Map::new();
		match q.qtype {
			0 => {
				let keys: Vec<String> = match &q.crit {
					Value::Object(o) => o.keys().cloned().collect(),
					Value::Array(a) => a.iter().map(|v| v.as_str().unwrap_or_default().to_string()).collect(),
					_ => vec![],
				};
				let best = p.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i).unwrap_or(0);
				ans.insert("type".into(), json!("choice"));
				if let Some(k0) = keys.get(best) {
					ans.insert("choice".into(), json!(k0));
				}
				let mut probs = Map::new();
				for (i, key) in keys.iter().enumerate() {
					probs.insert(key.clone(), json!(round4(p.get(i).copied().unwrap_or(0.0))));
				}
				ans.insert("probabilities".into(), Value::Object(probs));
				ans.insert("confidence".into(), json!(round4(confidence_from_probs(&p))));
				ans.insert("rl_agent".into(), json!({ "act_probability": act_probability }));
			}
			1 => {
				ans.insert("type".into(), json!("score"));
				ans.insert("score".into(), json!(round4(p.iter().enumerate().map(|(i, v)| i as f64 * v).sum())));
				if let Value::Array(a) = &q.crit {
					let mut legend = Map::new();
					for (i, c) in a.iter().enumerate() {
						legend.insert(i.to_string(), json!(c));
					}
					ans.insert("legend".into(), Value::Object(legend));
				}
				let mut probs = Map::new();
				for (i, v) in p.iter().enumerate() {
					probs.insert(i.to_string(), json!(round4(*v)));
				}
				ans.insert("probabilities".into(), Value::Object(probs));
				ans.insert("confidence".into(), json!(round4(confidence_from_probs(&p))));
				ans.insert("rl_agent".into(), json!({ "act_probability": act_probability }));
			}
			_ => {
				ans.insert("type".into(), json!("noul"));
				ans.insert("noul".into(), json!(round4(p.get(1).copied().unwrap_or(0.0))));
				ans.insert("rl_agent".into(), json!({ "act_probability": act_probability }));
			}
		}
		answers.insert(item.0.clone(), Value::Object(ans));
	}
	Ok((
		Value::Object(answers),
		json!({ "input_tokens": n_tokens, "output_tokens": 0 }),
	))
}
