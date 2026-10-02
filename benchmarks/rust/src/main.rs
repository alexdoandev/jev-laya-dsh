// Runtime benchmark for the exported Laya ONNX graph through ort
// (Rust bindings to onnxruntime — same engine family as the Node/ONNX row,
// Rust host process instead of Node).
//
// Fair-comparison notes:
// - Same ONNX graph and same tensor shapes as a real router call.
// - Tokenization is NOT included (trivial text processing in every runtime);
//   the encoder forward pass dominates latency in all implementations.
//
// Usage: ORT_DYLIB_PATH=<libonnxruntime.dylib> cargo run --release -- \
//          ../onnx-export/fp32/laya.onnx [warm_runs]
use std::time::Instant;

use ndarray::Array2;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;

fn lcg(state: &mut u64) -> u64 {
	// deterministic pseudo-random token ids (no rand dependency)
	*state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
	*state >> 33
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
	let args: Vec<String> = std::env::args().collect();
	let model_path = args.get(1).cloned().unwrap_or_else(|| "../onnx-export/fp32/laya.onnx".into());
	let warm_runs: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(20);

	let t_load0 = Instant::now();
	let mut session = Session::builder()?
		.with_optimization_level(GraphOptimizationLevel::Level3)?
		.with_intra_threads(4)?
		.commit_from_file(&model_path)?;
	let load_s = t_load0.elapsed().as_secs_f64();
	eprintln!("session created in {load_s:.2}s");

	// deterministic inputs shaped like a real router call
	let mut st: u64 = 0x9E37_79B9_7F4A_7C15;
	let input_ids: Vec<i64> = (0..256).map(|_| 5 + (lcg(&mut st) % 30_000) as i64).collect();
	let markers: Vec<i64> = vec![3, 9, 15];
	let qtype: Vec<i64> = vec![0];

	let ids_arr = Array2::from_shape_vec((1, 256), input_ids)?;
	let mask_arr = Array2::from_shape_vec((1, 256), vec![1i64; 256])?;
	let mk_arr = Array2::from_shape_vec((1, 3), markers)?;
	let mk_mask = Array2::from_shape_vec((1, 3), vec![true, true, true])?;
	let qt_arr = ndarray::Array1::from_shape_vec((1,), qtype)?;

	let ids_t = Tensor::from_array(ids_arr.clone())?;
	let mask_t = Tensor::from_array(mask_arr.clone())?;
	let mk_t = Tensor::from_array(mk_arr.clone())?;
	let mask2_t = Tensor::from_array(mk_mask.clone())?;
	let qt_t = Tensor::from_array(qt_arr.clone())?;

	let t_first = Instant::now();
	let outputs = session.run(ort::inputs![
		"input_ids" => ids_t.clone(),
		"attention_mask" => mask_t.clone(),
		"marker_pos" => mk_t.clone(),
		"marker_mask" => mask2_t.clone(),
		"qtype" => qt_t.clone(),
	])?;
	let first_ms = t_first.elapsed().as_secs_f64() * 1000.0;
	let logits_len: i64 = outputs["logits"].shape().iter().product();
	drop(outputs);

	let mut times = Vec::with_capacity(warm_runs);
	for _ in 0..warm_runs {
		let t = Instant::now();
		let outputs = session.run(ort::inputs![
			"input_ids" => ids_t.clone(),
			"attention_mask" => mask_t.clone(),
			"marker_pos" => mk_t.clone(),
			"marker_mask" => mask2_t.clone(),
			"qtype" => qt_t.clone(),
		])?;
		times.push(t.elapsed().as_secs_f64() * 1000.0);
		drop(outputs);
	}
	times.sort_by(|a, b| a.partial_cmp(b).unwrap());
	let avg = times.iter().sum::<f64>() / times.len() as f64;
	let p50 = times[times.len() / 2];
	let min = times[0];
	let max = times[times.len() - 1];

	println!(
		"{{\"engine\":\"ort-rust\",\"load_s\":{load_s:.2},\"first_inference_ms\":{first_ms:.1},\"warm_avg_ms\":{avg:.1},\"warm_p50_ms\":{p50:.1},\"warm_min_ms\":{min:.1},\"warm_max_ms\":{max:.1},\"warm_runs\":{warm_runs},\"logits_elements\":{logits_len}}}"
	);
	Ok(())
}
