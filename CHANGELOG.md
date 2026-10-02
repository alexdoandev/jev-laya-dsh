# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning is SemVer.

## [Unreleased]

### Added
- **`onnx-rust-server/`** — Rust host (`ort` bindings): drop-in protocol-compatible
  warm server, single 566 KB binary. Same-graph parity verified on the full VN
  benchmark (max |Δp| = 0.000 across 61 prompts × 3 questions). Under heavy CPU
  contention it served the same inference 4–8× faster than the Node server.
- `eval/parity.js` — Node-vs-Rust answer-parity gate (the cutover check).
- `docs/RUST_MIGRATION.md` — migration evaluation (tokenization parity spike,
  phased plan, go/no-go).

### Changed
- Production deployment cut over to the Rust server (launchd `ai.local.laya-warm`).

### Fixed
- **i18n**: English is now the primary documentation language (`README.md`); added
  `README.zh-CN.md` (简体中文) and `README.vi.md` (Tiếng Việt) with a language switcher.
- Router notice strings (advisory hints injected into turns) are now English by default —
  better cross-model portability; tier names (`tier1a/1b/2`) and `[Routing|jev]` prefix
  are unchanged, so existing log greps keep working.
- `CONTRIBUTING.md` and `CHANGELOG.md` converted to English.

## [1.0.0] - 2026-10-02

### Added
- **`laya-server/`** — reference warm server kept for parity (Python/torch): `/decide`, `/mcp`,
  `/health`, `/admin/{load,unload}`; hybrid residency (30-min idle unload + memory-pressure valve).
- **`onnx-server/`** — recommended Node/ONNX server: ~2.5 s cold start (vs 7–20 min torch),
  parity `max|Δlogits| = 7.4e-06`, identical protocol; postinstall patch enabling multilingual
  (mmBERT) checkpoints.
- **`packages/dsh-jev-router/`** — DeepSeek Harness host plugin implementing
  `User → Jev(Laya) → Agent`:
  - Router via `agent/pre-step` (Laya `choice`/`noul`/`score` × vector/regex evidence fusion).
  - Risk gate via `tools/pre-execute` (3-probe vote, single batched forward pass).
  - Model-swap via `agent/request` by tier; peak/off-peak (Asia/Ho_Chi_Minh) and quota tags.
- **`eval/`** — 61 gold-labeled Vietnamese prompts + threshold sweep; fused accuracy 86.9%
  (laya-win threshold 0.75 chosen from the sweep); 0 cloud tokens to run.
- **`onnx-export/`** — script exporting the multilingual checkpoint to ONNX (parity 7.4e-06).

### Evaluation references
- RouteLLM / FrugalGPT / Hybrid LLM / AutoMix / Tabi (routing & cascades).
- NeMo Guardrails / Prompt Guard / Semantic Router (rails & vector routing).
See README §References.
