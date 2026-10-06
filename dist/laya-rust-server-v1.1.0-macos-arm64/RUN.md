# RUN — laya-rust-server (macOS arm64)

Standalone warm server for the Laya System-1 decision model.
Protocol: POST /decide · POST /mcp · GET /health · POST /admin/{load,unload} on 127.0.0.1:8755.

## 1. Prerequisites

- The ONNX bundle (`laya.onnx` + `laya.onnx.data` + `laya_config.json` + `tokenizer/`)
  exported from the `convaiinnovations/laya-multilingual` checkpoint — see the repo
  README §Installation for the one-time export.
- Nothing else: the onnxruntime dylib ships in this archive (MIT, from the
  onnxruntime project).

## 2. Run in foreground

```bash
export LAYA_PORT=8755
export LAYA_MODEL_DIR=/absolute/path/to/onnx-export/fp32
export ORT_DYLIB_PATH=$(pwd)/libonnxruntime.1.30.0.dylib
./laya-rust-server
# → [laya-rust] listening on 127.0.0.1:8755
# → [laya-rust] ready (load took Ns since boot)
```

Sanity check:

```bash
curl -s -X POST http://127.0.0.1:8755/decide \
  -H 'content-type: application/json' \
  -d '{"state":"hello","questions":{"q":{"type":"noul","instructions":"ok?"}}}'
```

## 3. Run as a launchd service (auto-start at login, KeepAlive)

```bash
sed -e "s|__BINARY__|$PWD/laya-rust-server|" \
    -e "s|__MODEL_DIR__|$LAYA_MODEL_DIR|" \
    -e "s|__DYLIB__|$ORT_DYLIB_PATH|" \
    ai.local.laya-warm.plist.example > ~/Library/LaunchAgents/ai.local.laya-warm.plist
launchctl bootstrap "gui/$(id -u)" ~/Library/LaunchAgents/ai.local.laya-warm.plist
```

Defaults (override in the plist env): `LAYA_IDLE_UNLOAD_SECS=1800`,
`LAYA_THREADS=4`. Weights unload on idle or macOS memory pressure — the process
itself stays resident and reloads in seconds.

## 4. Endpoints

| Endpoint | Description |
|---|---|
| `POST /decide` `{state, questions}` | calibrated answers (choice/score/noul) in one forward pass |
| `POST /mcp` | MCP streamable-http (tool `jev_decide`) |
| `GET /health` | ready / loading / memPressure / idleForSecs |
| `POST /admin/unload` · `/admin/load` | free RAM now · warm in background |
