---
license: apache-2.0
base_model: convaiinnovations/laya-multilingual
library_name: onnx
tags:
- onnx
- onnxruntime
- decision-model
- system-one
- jev
- laya
- multilingual
- modernbert
- mmbert
- routing
language:
- multilingual
- vi
- en
- zh
---

# Laya multilingual — ONNX (fp32, dynamo export)

ONNX export of the [Laya multilingual System-1 decision model](https://huggingface.co/convaiinnovations/laya-multilingual)
(© Convai Innovations, Apache-2.0) — a calibrated decision engine that answers typed
questions (`choice` / `score` / `noul`) about a state in **one forward pass**: no text
generation, sub-second on CPU.

This is the **first publicly available multilingual ONNX bundle** of Laya — the
published [`receptron/laya-onnx`](https://huggingface.co/receptron/laya-onnx) ships
English-only weights.

## Why the community needs this

LLM routing and guardrails need thousands of cheap calibrated decisions, not one
expensive generation. Run this 322M-parameter encoder locally and ask it typed
questions — "which team handles this?", "is this urgent?", "P(destructive)?" — in
~0.15–0.4 s per batch, offline, $0/token. Reference integration:
[jev-laya-dsh](https://github.com/alexdoandev/jev-laya-dsh) (a System-1 decision
layer for DeepSeek Harness agents).

## Files

| File | Purpose |
|---|---|
| `laya.onnx` + `laya.onnx.data` | the graph (opset 18, dynamo export) + fp32 weights |
| `laya_config.json` | `max_len`, `head_max_len`, per-cardinality `temperature` |
| `tokenizer/` | mmBERT tokenizer |

Inputs: `input_ids [B,L] i64`, `attention_mask [B,L] i64`, `marker_pos [B,K] i64`,
`marker_mask [B,K] bool`, `qtype [B] i64` (0=choice, 1=score, 2=noul).
Outputs: `logits [B,K] f32` (uncalibrated; apply the temperature from
`laya_config.json` then softmax) and `act_probs [B,2] f32`.

## Usage

**Node** — [`@receptron/laya`](https://github.com/receptron/laya) ≥ 0.1.2 with the
multilingual specials patch (or any onnxruntime binding that builds the sequences
below):

```js
import { Laya } from "@receptron/laya";
const laya = await Laya.load({ repo: "alexdoandev/laya-multilingual-onnx" });
// note: multilingual (mmBERT) checkpoints need specials read from
// tokenizer_config.json — the upstream JS hardcodes BERT tokens;
// see jev-laya-dsh onnx-server/patches/apply-multilingual-patch.js
```

**Python** — reference `laya` package loads the original torch checkpoint; use the
export script in [jev-laya-dsh/onnx-export/](https://github.com/alexdoandev/jev-laya-dsh)
to reproduce this bundle.

**Rust** — `ort` crate + `Tensor::from_shape` (see jev-laya-dsh `benchmarks/rust/`).

## Provenance & parity

- Exported 2026-10-02 from `convaiinnovations/laya-multilingual`
  (revision `e4e9ddf2`) with torch 2.14, opset 18, dynamo exporter.
- Parity vs PyTorch: **max |Δlogits| = 7.4e-06**.
- Served in production as a prompt router + risk gate for coding agents
  (fused routing accuracy 86.9% on a 61-prompt Vietnamese benchmark — see the
  integration repo).

## License

Weights and this export: **Apache-2.0** (inherited from the Convai checkpoint).
"Jev" is a TypeSafe product and is unrelated to this artifact.
