# jev-laya-dsh

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![ci](https://github.com/alexdoandev/jev-laya-dsh/actions/workflows/ci.yml/badge.svg)](../../actions/workflows/ci.yml)
![Node](https://img.shields.io/badge/node-%E2%89%A520-brightgreen)
![Runtime](https://img.shields.io/badge/runtime-ONNX%20%7C%20torch-blue)

<p align="center">
  English | <a href="README.zh-CN.md">简体中文</a> | <a href="README.vi.md">Tiếng Việt</a>
</p>

**jev-laya-dsh** adds a local **System-1 decision layer** in front of
[DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) agents, following the
`User → Jev(Laya) → Agent` pattern:

```
User
 │ prompt
 ▼
[1] ROUTER — event agent/pre-step
 │  Laya classifies the prompt (task_type / needs_code_model / urgency) in one
 │  forward pass; the fused verdict is appended as an advisory notice
 ▼
[2] GATE — event tools/pre-execute (optional)
 │  3-probe noul vote on risky tool calls; escalate to the human when hot
 ▼
[3] AGENT — the large model does the actual work
 │  optional per-tier model swap via event agent/request
 ▼
answer
```

Decisions come from **Laya** — the open, Jev-compatible calibrated decision model
(mmBERT multilingual encoder, 100+ languages): no text generation, no hallucinated
confidence, one forward pass, $0/token, 100% local. Large-model calls are reserved for
the work only they can do.

## Features

- **Prompt router** — every user prompt is classified (chat / code / architecture) with
  calibrated probabilities before the main model spends its first token; the verdict is
  injected as a DSH notice, advisory by design.
- **Risk gate** — a 3-probe calibrated vote (destructive / irreversible /
  outside-workspace) on `bash`/`pwsh` calls, batched in a single forward pass;
  escalates to a user confirmation instead of silently running.
- **Per-tier model swap** — hard turns (architecture/planning) can be pinned to the
  strong model for the whole turn via the `agent/request` event
  ([RouteLLM](https://arxiv.org/abs/2406.18665)/[Hybrid LLM](https://arxiv.org/abs/2404.14618)-style
  cost routing).
- **Hybrid-resident serving** — the ONNX runtime stays warm (2.5 s cold start,
  0.3 s warm decisions) while weights are auto-unloaded after idle or under memory
  pressure, with predictive wake on session start.
- **Fail-open by construction** — every layer degrades to "no routing" instead of
  blocking a turn; the decision layer is never a single point of failure.
- **Zero cloud tokens for decisions** — routing, gating and the eval harness run 100%
  locally; the large model is only billed for real work.

## Requirements

- [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) (any deployment with
  the Cordis plugin API) for the router plugin — the servers themselves are standalone.
- Node ≥ 20 (ONNX server, eval, smoke test).
- Python 3.12+ with torch — only for the one-time ONNX export.
- Model weights: [`convaiinnovations/laya-multilingual`](https://huggingface.co/convaiinnovations/laya-multilingual)
  (Apache-2.0, ~1.3 GB). macOS arm64 is the tested platform; anything running
  onnxruntime-node should work.

## Installation

```bash
git clone https://github.com/alexdoandev/jev-laya-dsh.git
cd dsh-jev-meta

# 1) weights (one-time)
HF_HUB_OFFLINE=0 python3 -c "from huggingface_hub import snapshot_download; \
  snapshot_download('convaiinnovations/laya-multilingual')"

# 2) ONNX bundle (one-time; needs torch)
python3 onnx-export/export_onnx_multilingual.py \
  ~/.cache/huggingface/hub/models--convaiinnovations--laya-multilingual/snapshots/<rev> \
  onnx-export/fp32

# 3a) server — Rust/ort (production shape; single 566 KB binary)
cd onnx-rust-server && cargo build --release
LAYA_PORT=8755 LAYA_MODEL_DIR=$PWD/../onnx-export/fp32 \
  ORT_DYLIB_PATH=<libonnxruntime.dylib> ./target/release/laya-rust-server

# 3b) server — Node/ONNX (equivalent, needs the multilingual patch)
cd onnx-server && npm install
LAYA_PORT=8755 LAYA_MODEL_DIR=../onnx-export/fp32 npm start
curl http://127.0.0.1:8755/health        # {"ok":true,"ready":true,...}
```

### Wire the router into DeepSeek Harness

Add a bundle + insert entry to a profile (same pattern for the `desktop` and `web`
profiles):

```yaml
# <profile>/package.json → dsh.profile.bundles += "@local/dsh-jev-router"
# <profile>/package.json → dependencies += { "@local/dsh-jev-router": "workspace:*" }
# <profile>/node_modules/@local/dsh-jev-router → symlink to packages/ copy
```

```yaml
# <profile>/cordis.patch.yml
- insert:
  - id: local-jev-router
    name: "@local/dsh-jev-router"
    config:
      routing: true
      gate: false
      modelSwap: true
      modelByTier:
        tier2: { provider: zai-coding-cn, model: glm-5.3 }
```

Restart the app; every new turn is now routed. `packages/dsh-jev-decide` (the
pull-mode `jev_decide` tool) and the `/mcp` endpoint are optional companions.

## Usage

```bash
node smoke-test.js "<any prompt>"        # end-to-end routing check (fake Cordis ctx)
node eval/run-eval.js [--url http://127.0.0.1:8756]   # 61-prompt benchmark + threshold sweep

curl -X POST http://127.0.0.1:8755/decide \
  -H 'content-type: application/json' \
  -d '{"state":"...","questions":{"urgent":{"type":"noul","instructions":"Is this urgent?"}}}'

curl http://127.0.0.1:8755/health        # ready / loading / memPressure / idleForSecs
curl -X POST http://127.0.0.1:8755/admin/unload   # free the weights now
curl -X POST http://127.0.0.1:8755/admin/load     # warm in background
```

The server also speaks MCP (streamable-http, tool `jev_decide`) at `POST /mcp`, so the
decision layer can be consumed by any MCP client, not only by this plugin.

## Configuration

Plugin (`config` in the profile patch entry):

| Option | Default | Description |
|---|---|---|
| `routing` | `true` | classify every turn on `agent/pre-step` and inject a notice |
| `gate` | `false` | enable the risk gate on `tools/pre-execute` |
| `gateThreshold` | `0.85` | mean-noul threshold for the ask-user escalation |
| `gateTools` | `[bash, pwsh]` | tools subject to the gate |
| `timeoutMs` | `2500` | per-request timeout; keeps turns unblocked |
| `wake` | `true` | predictive wake on `api-session/added` |
| `wakeUrl` | `http://127.0.0.1:8755/admin/load` | wake endpoint |
| `coldRetryMs` | `2000` | retry once after the server wakes, then fail-open |
| `maxPromptChars` | `2000` | prompt prefix sent as the Laya state |
| `modelSwap` | `false` | pin tier-2 turns to `modelByTier.tier2` via `agent/request` |
| `modelByTier` | `{}` | per-tier `{provider, model}` overrides |
| `routeUtterances` | `{}` | extra utterances per tier for the vector router |

Server environment variables: `LAYA_PORT` (8755), `LAYA_MODEL_DIR`,
`LAYA_IDLE_UNLOAD_SECS` (1800), `LAYA_THREADS` (4).

## Benchmarks

61 gold-labeled Vietnamese prompts (`eval/eval-set.jsonl`); threshold sweep reported by
`eval/run-eval.js`. Decisions run locally — no cloud tokens.

| Signal | Accuracy |
|---|---|
| Laya only (argmax) | 62.3% |
| Vector router only (char-ngram centroids) | 80.3% |
| Regex only | 78.7% |
| **Fused (Laya × evidence, laya-win ≥ 0.75)** | **86.9%** |

Per-tier at the operating point: 14/20 chat · 18/20 code · 21/21 architecture.

### Runtime comparison (same benchmark, same questions)

| Metric | Python/torch | ONNX Runtime (Node) | Rust host (`ort`) |
|---|---|---|---|
| Cold start | 7–20 min | **2.5 s** | **1.2 s** (566 KB binary) |
| Warm decide (3 questions) | 0.14–0.36 s | **0.29–0.41 s** (idle box) | ≤ 0.5 s (est.) |
| Warm decide (heavy CPU contention) | — | 6.8–13.6 s | **1.5–2.4 s** |
| Parity | reference | max \|Δlogits\| = 7.4e-06 | same graph, same engine family |
| Serving disk | venv 922 MB + 1.7 GB snapshot | 1.29 GB bundle + node_modules | 1.29 GB bundle + **566 KB** binary |
| RAM (loaded) | ~0.9–1.8 GB | ~0.9 GB | **~0.57 GB** |

The Rust row (`benchmarks/rust/`) runs the identical ONNX graph through `ort`
(onnxruntime Rust bindings) as a lean single binary — no Node, no HTTP hop. Under a
heavy concurrent-load window (three `rustc` jobs, load average ~80) it completed the
same inference **4–8× faster than the Node service in the same window**; on an idle
box both are sub-second and the difference collapses to the HTTP hop. Rust is the
strongest option for embedding the decision layer without a server process at all.

## How it works

- **Fusion policy** — the Laya verdict and a lexical/vector evidence tier are combined:
  agree ⇒ verdict; disagreement ⇒ the model must exceed a tuned confidence (0.75) to
  overrule the evidence; low confidence ⇒ no notice at all. All thresholds in
  `packages/dsh-jev-router/lib/index.js` were selected by the sweep in `eval/`.
- **Vector router** — per-tier utterance centroids over char-trigram + word TF vectors,
  pure JS, microseconds; seeds are configurable via `routeUtterances`.
- **Fail-open** — Laya down, cold, or timing out ⇒ routing is skipped for that turn and
  a wake is triggered; tool gating fails open identically.
- **Hybrid residency** — the runtime process stays resident (HTTP always bound);
  weights are unloaded after `LAYA_IDLE_UNLOAD_SECS` idle or on macOS memory-pressure
  warning, with a 120 s post-load grace and an in-flight guard. See
  [docs/OPERATIONS.md](docs/OPERATIONS.md) for the operational write-up (sizing, AV
  on-access storms, deployment checklist).
- **Advisory by design** — injected notices steer the agent; they never veto a turn.
  The gate layer is the only component that can block, and it only escalates to the
  human.

## Project layout

```
├── laya-server/            Python/torch reference server (protocol-compatible)
├── onnx-server/            Node/ONNX server (recommended) + multilingual patch
├── onnx-export/            checkpoint → ONNX export scripts (dynamo + legacy)
├── packages/
│   └── dsh-jev-router/     DeepSeek Harness host plugin (router/gate/swap)
├── benchmarks/rust/        pure-Rust host benchmark (ort) with the same graph
├── eval/                   benchmark set + threshold-sweep runner
├── docs/OPERATIONS.md      operational write-up (sizing, AV storms, deploy)
└── smoke-test.js           end-to-end routing check
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Routing behavior changes must include
before/after numbers from `eval/run-eval.js`.

## License

MIT — see [LICENSE](LICENSE). Third-party components keep their own licenses
(Laya weights: Apache-2.0, © Convai Innovations — weights are not redistributed here;
`@receptron/laya`: MIT). "Jev" is a TypeSafe product; this project uses only the
open, Jev-compatible Laya model as an independent local component.
