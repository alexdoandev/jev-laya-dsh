# Contributing

Thanks for your interest! The repo is small and so are the rules.

## Run it

```bash
# 1. Laya server (pick one — see README §4/§6)
cd onnx-server && npm install && npm start           # ONNX/Node (recommended)
cd laya-server && python laya_warm_server.py         # Python/torch (reference)

# 2. Smoke test + eval (0 cloud tokens, runs locally)
node smoke-test.js
node eval/run-eval.js
```

## Rules

- **Never commit**: weights (`*.onnx*`), `node_modules/`, any secret/key — not even "temporarily".
- One logical change per PR; describe the change and the actual result of `smoke-test`/`eval`.
- Any routing behavior change (fusion/thresholds/tiers) must include before/after numbers from
  `eval/run-eval.js`.
- Code and identifiers in English; discussion in Vietnamese, Chinese or English is welcome.

## Reporting issues

Include: OS/arch, node version, the tail of `~/.local/laya/onnx.log` (or `warm.log`),
the output of `curl 127.0.0.1:8755/health`, and the prompt that triggered the issue (if any).
