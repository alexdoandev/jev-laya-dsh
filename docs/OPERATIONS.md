# Operations

Deployment and operational notes for the jev-laya-dsh stack, from a long-running
macOS arm64 installation (DeepSeek Harness desktop 0.2.0-rc.2, launchd-managed warm
server). Treat the numbers as reference points for Apple-silicon laptops.

## Serving runtimes

Two protocol-compatible servers are shipped (`/decide`, `/mcp`, `/health`,
`/admin/{load,unload}` on `127.0.0.1:8755`):

| | `onnx-server/` (Node, recommended) | `laya-server/` (Python/torch, reference) |
|---|---|---|
| Cold start | 2.5 s | 7–20 min (torch import + Gatekeeper) |
| Warm decide (3-question batch) | 0.29–0.41 s under heavy CPU contention | 0.14–0.36 s on an idle box; degrades sharply under contention |
| RAM (loaded) | ~0.9 GB (mmap, file-backed, OS-purgeable) | ~1.5–1.8 GB RSS (4 GB spikes at import) |
| Serving disk | 1.29 GB bundle | 922 MB venv + 1.7 GB snapshot |
| Accuracy vs benchmark | identical (export parity max \|Δlogits\| = 7.4e-06) | reference |

The production deployment runs the ONNX server under launchd:

```
Label: ai.local.laya-warm · KeepAlive · ThrottleInterval 10
ProgramArguments: node onnx-server/laya-onnx-server.mjs
Environment: LAYA_PORT=8755 · LAYA_MODEL_DIR=<bundle> · LAYA_IDLE_UNLOAD_SECS=1800 · LAYA_THREADS=4
Logs: ~/.local/laya/onnx.log
```

`LAYA_THREADS=4` caps onnxruntime's intra-op threads: the default (all cores) starves
the rest of the machine when it is already busy — measured live at load average 338,
decides timing out at 20 s; capped, they stay usable and the box keeps breathing.

## Hybrid residency

- The process is kept alive by launchd (`KeepAlive`); HTTP binds immediately, so
  health probes and wakes always work.
- Weights unload when (a) idle ≥ `LAYA_IDLE_UNLOAD_SECS` (1800 s) or (b) macOS memory
  pressure (`kern.memorystatus_vm_pressure_level`) reaches warning level, or manually
  via `POST /admin/unload`. Unloading is refused while requests are in flight or while
  the model is loading.
- **120 s post-load grace**: under sustained memory pressure, an unload immediately
  after a caller-triggered load would thrash (observed live: unload → wake → load →
  unload killed an eval run). The valve only fires 120 s after a finished load.
- **Wake paths**: `POST /admin/load` (also fired by the plugin on `api-session/added`),
  a cold `/decide` (returns `503 {wakeAccepted}` and self-wakes), and the plugin's
  2 s cold-retry.
- Bug class to watch: the idle watchdog must refresh its "last used" timestamp when a
  load completes, otherwise a long first load gets unloaded right after becoming ready
  (v3 had this; fixed in v3.1).

With the ONNX runtime, cold loads are short enough that the idle window could shrink
below 30 minutes; with the Python runtime it could not (a weights-only reload still
costs 7–11 minutes of pure-Python CPU inside `laya.load()`), which is why the idle
window defaults to 1800 s.

## Antivirus on-access storms

Endpoint-security products that hook file operations turn a cold weight read into tens
of thousands of scan events. Measured with Kaspersky Endpoint Software
(`KAV/Data/Report/Database/reports.db`, `DailyProtectionStatistics`, `threats=0` — no
real detections, pure overhead) while installing, exporting, and restarting the stack:

| 15-minute window | files scanned | concurrent activity |
|---|---|---|
| 10:00–10:30 | 330k–449k | pip installs, zcode npx MCP servers |
| 11:45 | 174k | `npm install @receptron/laya`, ONNX export |
| 12:00–12:45 | up to 905k | A/B eval + restarts + 1.7 GB page-in |
| 13:00–13:30 | 407k–701k | cutover: python killed, node cold-mmap 1.29 GB |

≈ 5.6 M scan events in 3.5 h; every event crosses the AV system extension as an
uninterruptible wait, inflating load average (peaked at 338 with ~124% peak CPU) and
stretching decides to 6–20 s. The router fail-opened throughout — no turn was blocked.

Mitigation (keep the AV everywhere else): scope exclusions for the inference paths
(`~/jev-laya-dsh/*`, `~/.cache/huggingface/*`, `~/.local/laya/*`, `~/.npm/_npx/*`) and
register the node binary as a trusted application. The consumer Mac app exposes no
scan-exclusion UI — exclusions come from the management policy (Security Center) or
AV-specific tooling; otherwise pause protection briefly around installs/exports.

## Deployment checklist

1. Weights downloaded; ONNX bundle exported (parity ≤ 1e-05).
2. Server starts (`/health` → `ready:true`); `/decide` returns `ok:true` — assert the
   body, not just latency (a 500 with a green timing number is the classic miss).
3. launchd/KeepAlive unit installed; log path writable.
4. Plugin wired into the DSH profile; one turn produces a `[Routing|jev]` notice.
5. `node eval/run-eval.js` matches the published numbers (±2%).
6. Optional: weekly eval schedule (launchd calendar job) and AV exclusions.

## Known trade-offs

- Laya-only accuracy is 62.3% on the VN benchmark — the evidence layer (vector 80.3%,
  regex 78.7%) carries real weight; fusion is mandatory, never ship laya solo.
- RSS is a noisy metric on macOS (compressed/purgeable pages); judge runtimes by
  accuracy, latency and cold-start.
- The multilingual ONNX bundle is self-exported; the published `receptron/laya-onnx`
  repo currently ships English-only weights. Re-export after upstream publishes a
  multilingual bundle to pick up their improvements.

## Rust host (`benchmarks/rust/`)

A third serving shape: the identical ONNX graph consumed by `ort` (onnxruntime Rust
bindings) inside a single 566 KB binary — no Node, no HTTP hop. Same engine family as
the Node/ONNX row (results within noise of each other on an idle box); the win shows
under contention: during a heavy-load window (three concurrent `rustc` jobs, load
average ~80) the Rust host completed the same 3-question inference in 1.5–2.4 s while
the Node service needed 6.8–13.6 s in the same window. Peak RSS mid-run ~0.57 GB,
cold session create 1.2 s.

Notes for reproducing:
- The dynamo-exported graph fails tract's fact analysis (`Range` node, symbolic/val
  unification) — `export_onnx_legacy.py` re-exports via the legacy exporter
  (`dynamo=False`, opset 18, `dynamic_axes`) with parity 1.2e-04, which tract and ort
  both load.
- `ort` 2.0.0-rc.13 pairs with ndarray 0.17 — keep the versions aligned or the
  tensor-data traits will not resolve.
- `kstring 2.0.5` requires rustc ≥ 1.96; `cargo update kstring@2.0.5 --precise 2.0.2`
  on rustc 1.93 toolchains. `Cargo.lock` is committed with the working pins.
- Run: `ORT_DYLIB_PATH=<libonnxruntime.dylib> cargo run --release -- <model> [runs]`
  (the dylib from `onnx-server/node_modules/onnxruntime-node/` is reused — nothing
  extra to download).
