# dsh-jev-router

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

**User → Jev(Laya) → Agent** — a [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness)
host plugin that puts a local **System-1 decision model** ([Laya](https://huggingface.co/convaiinnovations/laya-multilingual),
the open Jev-compatible calibrated decision engine) between the user and the agent:

- **Prompt router** — every turn is classified (chat / code / architecture, plus
  urgency and code-need probes) in one sub-second forward pass; the fused verdict
  is injected as an advisory DSH notice before the main model spends its first token.
- **Risk gate** (optional) — 3-probe calibrated vote on `bash`/`pwsh` calls;
  escalates to a user confirmation instead of running silently.
- **Peak/quota awareness** — notices carry live peak/off-peak (Asia/Ho_Chi_Minh)
  and low-quota tags.
- **Fail-open by construction** — if the decision server is down, cold, or slow,
  turns run untouched. The layer steers; it never blocks.

A companion decision server is required at `http://127.0.0.1:8755` — pick one from
the [jev-laya-dsh](https://github.com/alexdoandev/jev-laya-dsh) repo:

- `onnx-rust-server/` — Rust/ort single binary (recommended; macOS arm64 release
  artifact available)
- `onnx-server/` — Node/ONNX equivalent
- `laya-server/` — Python/torch reference

## Install (DeepSeek Harness profile)

```bash
# copy lib/ into your profile, e.g.
#   <profile>/packages/dsh-jev-router/lib/index.js
# then wire it in <profile>/cordis.patch.yml:
```

```yaml
- insert:
  - id: local-jev-router
    name: "@local/dsh-jev-router"
    config:
      routing: true
      gate: false
      modelSwap: true
      modelByTier:
        tier2: { provider: your-provider, model: your-strong-model }
```

Full configuration table and architecture:
[jev-laya-dsh README](https://github.com/alexdoandev/jev-laya-dsh#readme).

## License

MIT. Uses the open Laya model (Apache-2.0, © Convai Innovations) via a local
decision server — "Jev" (TypeSafe) is not used.
