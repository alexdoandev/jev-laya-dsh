# RUST_MIGRATION.md — Evaluate: replace the Node/ONNX server with Rust

Question: *the Rust host benchmarked 4–8× faster than Node under CPU contention —
should the production warm server migrate from Node to Rust?*

Answer: **yes, worth doing as a phased A/B** — the largest technical risk
(tokenization parity) was just proven away, and the remaining port is deterministic
logic with a byte-comparable reference. Estimated effort: 300–400 lines of Rust +
verification. Do not delete Node until the Rust server passes the parity gate below.

## 1. What the Rust server must implement

The Node server (`onnx-server/laya-onnx-server.mjs`, ~250 lines) delegates all model
work to `@receptron/laya`. Its responsibilities, and the Rust equivalent:

| Responsibility | Node today | Rust port | Risk |
|---|---|---|---|
| Tokenize `state` | `@huggingface/tokenizers` (JS wrapper **of the Rust `tokenizers` crate**) | `tokenizers` crate (the native engine itself) | **none — proven** (see §2) |
| Build sequences | `dist/sequence.js` `buildSequence`: cls/sep wrap, per-question-type marker rendering, option truncation | port ~150 lines from the readable dist source | low — deterministic, byte-comparable |
| Calibrate + decode | temperature buckets from `laya_config.json`, softmax, 4-decimal rounding | same port | low, same file |
| Specials for mmBERT | `patches/apply-multilingual-patch.js` (postinstall on node_modules) | read `tokenizer_config.json` natively | **simpler in Rust** — the whole patch disappears |
| Inference | `onnxruntime-node` | `ort` crate, `load-dynamic` + the same dylib | **none — benchmarked** (§2) |
| HTTP `/decide` `/health` `/admin/*`, MCP SSE | `node:http` | `axum`/`hyper` | none |
| Hybrid residency (idle 1800 s, pressure valve, 120 s grace, in-flight guard) | timer thread + sysctl | same, `libc`/sysctl | none |
| launchd unit | plist → node | plist → rust binary | none |

## 2. Evidence gathered (spikes, 2026-10-02)

- **Inference through Rust**: `benchmarks/rust/` (ort 2.0.0-rc.13) — session create
  1.2 s, warm 3-question inference 0.3–0.5 s on an idle box; **1.5–2.4 s vs the Node
  service's 6.8–13.6 s in the same heavy-load window** (3 concurrent rustc jobs,
  load ~80). Peak RSS mid-run ~0.57 GB; binary 566 KB reusing the same dylib.
- **Tokenization parity**: same prompt, same `tokenizer.json` —
  JS ids `[2, 4926, 18335, 235292, …]` vs Rust ids `[4926, 18335, 235292, …]`
  (the leading `2` = `<bos>`, added by the sequence builder in both designs).
  Rust encode latency 0.9 ms; tokenizer load 1.7 s one-time.
- **Same engine**: the JS `@huggingface/tokenizers` package wraps the Rust
  `tokenizers` crate; onnxruntime-node wraps the same onnxruntime C++ library the
  `ort` crate binds. Rust is not a different math stack — it is the same one,
  minus one process and one IPC hop.

## 3. Expected production gains

| Metric | Node/ONNX (today) | Rust (projected) |
|---|---|---|
| Process cold start | ~2.5–50 s (node + require chain; 50 s observed under load) | **~3 s** (session 1.2 s + tokenizer 1.7 s), stable |
| Decide under CPU contention | 6.8–13.6 s (fail-open window) | **1.5–2.4 s** — routing survives build storms |
| RAM loaded | ~0.9 GB | ~0.57 GB |
| Dependency surface | node_modules + postinstall patch on `@receptron/laya` | single 566 KB binary + dylib; the multilingual patch becomes native code |

## 4. Phased plan (the repo's own A/B pattern)

1. **Phase A — port & parity**: implement `onnx-rust-server/` (axum + tokenizers +
   ort); run `eval/run-eval.js --url http://127.0.0.1:8757` and require **identical
   86.9% fused accuracy** plus byte-comparable probabilities (4 decimals) against the
   Node server on all 61 prompts. Gate: no parity, no cutover.
2. **Phase B — shadow soak**: run the Rust server on 8756 alongside Node for 24–48 h
   of real usage; compare `p50/p95` decide latency and memory in `health`.
3. **Phase C — cutover**: repoint the `ai.local.laya-warm` plist to the Rust binary
   on 8755 (Node stays on disk for rollback), then deprecate
   `onnx-server/` + the venv once the soak is accepted.

## 5. Non-goals / guards

- The Node server is retained as the protocol reference; the Rust server must match
  it on the wire, not the other way around.
- The Python `laya` package remains the numerical ground truth for parity checks.
- No change to the DSH plugin, the profile wiring, or the MCP contract during the
  migration — the server is a drop-in.
