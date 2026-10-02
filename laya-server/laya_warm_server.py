#!/usr/bin/env python3
"""Warm Laya decision server v3 — HYBRID residency, dual protocol on 127.0.0.1:8755.

- POST /decide {state, questions}          -> plain JSON (direct use)
- POST /mcp   (JSON-RPC, streamable-http)  -> MCP server exposing tool jev_decide
- GET  /health                             -> {"ok", "ready", "loaded", "loading", ...}
- POST /admin/load                         -> wake: load weights in background (idempotent)
- POST /admin/unload                       -> free weights now (idempotent, skipped if inflight)

Hybrid residency model (v3, thay cho "luôn ấm" của v2):
- PROCESS stays resident under launchd (torch imported once — the expensive part),
  HTTP always bound, so probes and wakes are instant.
- WEIGHTS unload after LAYA_IDLE_UNLOAD_SECS (default 300s) without requests:
  MODEL_OBJ = None + gc.collect() frees ~1.2-1.4 GB RSS; torch import cost is NOT
  paid again. Reload on next request/wake is weights-only (fast).
- /decide while cold: kicks a background load and answers 503 {wakeAccepted:true};
  callers fail-open. Wake-early endpoints make this rare.

Model stays loaded while active; designed to run under launchd (KeepAlive).
"""
import gc
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(os.environ.get("LAYA_PORT", "8755"))
# With the Python laya package a load (cold OR weights-only) costs ~7-11 min of
# CPU, so idle-unload must be coarse: only long absences (default 30 min) or
# real memory pressure justify freeing the weights. The ONNX client (roadmap)
# is what will make reload seconds-fast.
IDLE_UNLOAD_SECS = int(os.environ.get("LAYA_IDLE_UNLOAD_SECS", "1800"))

# Pinned snapshot downloaded on 2026-10-02 (convaiinnovations/laya-multilingual).
PINNED_MODEL = os.path.expanduser(
    "~/.cache/huggingface/hub/models--convaiinnovations--laya-multilingual/"
    "snapshots/e4e9ddf21a7b1903b7acffd8814ad4307bf63a67"
)
MODEL = os.environ.get("LAYA_MODEL_PATH", PINNED_MODEL)

MODEL_OBJ = None
LOAD_ERROR = None
LOADING = False
READY = threading.Event()
LOAD_LOCK = threading.Lock()
STATE_LOCK = threading.Lock()

LAST_USED = time.time()
INFLIGHT = 0
UNLOAD_COUNT = 0
BOOT_TS = time.time()


def _load_model_async():
    """Wake path: start one background load; no-op if loaded or already loading."""
    global LOADING, LOAD_ERROR
    with LOAD_LOCK:
        if READY.is_set() or LOADING:
            return False
        LOADING = True
        LOAD_ERROR = None

    def _run():
        global MODEL_OBJ, LOADING, LOAD_ERROR, LAST_USED
        try:
            if not os.path.isdir(MODEL):
                raise FileNotFoundError(
                    f"pinned model dir not found: {MODEL} "
                    f"(re-download: HF_HUB_OFFLINE=0 {sys.executable} -c "
                    f"'from huggingface_hub import snapshot_download; snapshot_download(\"convaiinnovations/laya-multilingual\")')"
                )
            print(f"[laya-warm] loading model from {MODEL} ...", flush=True)
            from laya import load

            MODEL_OBJ = load(MODEL)
            READY.set()
            # A finished load counts as fresh use: the idle watchdog must never
            # unload a model that just became ready (slow first load > idle window).
            LAST_USED = time.time()
            print(f"[laya-warm] ready (load took {round(time.time() - BOOT_TS, 1)}s since boot)", flush=True)
        except Exception as exc:  # noqa: BLE001
            LOAD_ERROR = f"{type(exc).__name__}: {exc}"
            print(f"[laya-warm] load failed: {LOAD_ERROR}", flush=True)
        finally:
            LOADING = False

    threading.Thread(target=_run, name="laya-load", daemon=True).start()
    return True


def _mem_pressure_level():
    """macOS kern.memorystatus_vm_pressure_level: 1 normal, 2 warning, 3 critical."""
    try:
        import subprocess

        out = subprocess.run(
            ["/usr/sbin/sysctl", "-n", "kern.memorystatus_vm_pressure_level"],
            capture_output=True, text=True, timeout=2,
        ).stdout.strip()
        return int(out)
    except Exception:
        return 1


def _unload_model(force=False):
    """Free weights; refuses while a request is in flight unless force."""
    global MODEL_OBJ, UNLOAD_COUNT
    with STATE_LOCK:
        if INFLIGHT > 0 and not force:
            return False
        if MODEL_OBJ is None:
            return False
        MODEL_OBJ = None
        READY.clear()
        gc.collect()
        UNLOAD_COUNT += 1
        print(f"[laya-warm] idle-unloaded weights (#{UNLOAD_COUNT}), "
              f"idle after {IDLE_UNLOAD_SECS}s", flush=True)
        return True


def _idle_watchdog():
    while True:
        time.sleep(15)
        if LOADING:
            continue
        pressure = _mem_pressure_level()
        if READY.is_set() and pressure >= 2:
            print(f"[laya-warm] memory pressure level={pressure} — unloading weights", flush=True)
            _unload_model()
            continue
        if IDLE_UNLOAD_SECS <= 0:
            continue
        if READY.is_set() and time.time() - LAST_USED >= IDLE_UNLOAD_SECS:
            _unload_model()


threading.Thread(target=_idle_watchdog, name="laya-idle", daemon=True).start()
_load_model_async()  # boot: warm immediately (hybrid keeps process + first-load eager)


def mark_used():
    global LAST_USED, INFLIGHT
    with STATE_LOCK:
        INFLIGHT += 1
        LAST_USED = time.time()


def release_used():
    global INFLIGHT, LAST_USED
    with STATE_LOCK:
        INFLIGHT = max(0, INFLIGHT - 1)
        LAST_USED = time.time()


def predict(state, questions):
    if not READY.wait(timeout=0):
        _load_model_async()  # wake accepted; caller sees 503 this round
        raise RuntimeError("model loading (wake accepted)")
    out = MODEL_OBJ.predict(state, questions)
    answers = {
        k: {kk: vv for kk, vv in v.items() if kk != "action"}
        for k, v in out.get("answers", {}).items()
    }
    return answers, out.get("usage")


MCP_TOOLS = [
    {
        "name": "jev_decide",
        "description": (
            "Ask the warm local Laya System One decision model (mmBERT multilingual "
            "encoder, 100+ languages including Vietnamese, hybrid-resident via launchd: "
            "instant when active, weights unload when idle) to answer typed questions "
            "about a state in one fast pass (sub-second warm, no token generation). "
            "Returns calibrated probabilities. Use for cheap high-volume gating "
            "decisions (keep/drop, route, classify, urgency) where a big model is "
            "wasteful. Do NOT use for nuanced reasoning or >16 options."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"state": {"type": "string"}, "questions": {"type": "object"}},
            "required": ["state", "questions"],
        },
    }
]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        try:
            print(
                f"[laya-warm] {time.strftime('%H:%M:%S')} {self.address_string()} {fmt % args}",
                flush=True,
            )
        except Exception:
            pass

    # ---- helpers ----
    def _send_mcp(self, payload):
        body = json.dumps(payload).encode()
        frame = b"event: message\ndata: " + body + b"\n\n"
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("content-length", str(len(frame)))
        self.end_headers()
        self.wfile.write(frame)

    def _send(self, code, body):
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # ---- GET ----
    def do_GET(self):
        if self.path == "/mcp":
            # MCP streamable-http spec: server without SSE stream support answers 405.
            self.send_response(405)
            self.send_header("allow", "POST")
            self.send_header("content-length", "0")
            self.end_headers()
            return
        body = json.dumps(
            {
                "ok": True,
                "ready": READY.is_set(),
                "loaded": READY.is_set(),
                "loading": LOADING,
                "idleUnloadSecs": IDLE_UNLOAD_SECS,
                "idleForSecs": round(time.time() - LAST_USED, 1),
                "memPressure": _mem_pressure_level(),
                "inflight": INFLIGHT,
                "unloadCount": UNLOAD_COUNT,
                "model": MODEL,
                "error": LOAD_ERROR,
            }
        ).encode()
        self._send(200, body)

    # ---- POST ----
    def do_POST(self):
        if self.path == "/decide":
            return self._do_decide()
        if self.path == "/mcp":
            return self._do_mcp()
        if self.path == "/admin/load":
            started = _load_model_async()
            return self._send(200, json.dumps({"ok": True, "started": started, "ready": READY.is_set()}).encode())
        if self.path == "/admin/unload":
            return self._send(200, json.dumps({"ok": _unload_model()}).encode())
        self._send(404, b'{"ok":false,"error":"not found"}')

    def _read_json(self):
        length = int(self.headers.get("content-length", 0))
        return json.loads(self.rfile.read(length) or b"null")

    def _do_decide(self):
        global LAST_USED
        try:
            req = self._read_json() or {}
            if not READY.is_set():
                _load_model_async()
                return self._send(
                    503,
                    json.dumps(
                        {"ok": False, "wakeAccepted": True, "error": LOAD_ERROR or "model loading"}
                    ).encode(),
                )
            mark_used()
            try:
                answers, usage = predict(req["state"], req["questions"])
            finally:
                release_used()
            body = json.dumps({"ok": True, "answers": answers, "usage": usage}, default=str).encode()
            self._send(200, body)
        except Exception as exc:  # noqa: BLE001
            self._send(500, json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}).encode())

    def _do_mcp(self):
        try:
            req = self._read_json()
            print(f"[laya-warm] {time.strftime('%H:%M:%S')} mcp <- {req.get('method')} id={req.get('id')}", flush=True)
        except Exception as exc:  # noqa: BLE001
            return self._send(400, json.dumps({"error": str(exc)}).encode())
        if not isinstance(req, dict) or "method" not in req:
            return self._send(400, b'{"error":"bad rpc"}')
        method = req["method"]
        req_id = req.get("id")

        if method == "initialize":
            return self._send_mcp(
                {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {"listChanged": False}},
                        "serverInfo": {"name": "laya-warm", "version": "0.3.0"},
                    },
                }
            )
        if method.startswith("notifications/"):
            self.send_response(202)
            self.end_headers()
            return
        if method == "tools/list":
            return self._send_mcp({"jsonrpc": "2.0", "id": req_id, "result": {"tools": MCP_TOOLS}})
        if method == "tools/call":
            params = req.get("params") or {}
            if params.get("name") != "jev_decide":
                return self._send_mcp(
                    {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {"code": -32602, "message": f"unknown tool {params.get('name')}"},
                    }
                )
            args = params.get("arguments") or {}
            if not READY.is_set():
                _load_model_async()
                text = "Laya is loading (wake accepted); retry shortly. " + (LOAD_ERROR or "")
            else:
                mark_used()
                try:
                    answers, usage = predict(args["state"], args["questions"])
                    text = json.dumps({"answers": answers, "usage": usage}, default=str)
                except Exception as exc:  # noqa: BLE001
                    text = f"Laya error: {type(exc).__name__}: {exc}"
                finally:
                    release_used()
            return self._send_mcp(
                {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"content": [{"type": "text", "text": text}], "isError": False},
                }
            )
        if req_id is None:
            self.send_response(202)
            self.end_headers()
            return
        self._send_mcp(
            {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"method not supported: {method}"}}
        )


if __name__ == "__main__":
    ThreadingHTTPServer.allow_reuse_address = True
    try:
        ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
    except Exception as exc:
        print(f"[laya-warm] fatal: {exc}", file=sys.stderr, flush=True)
        raise
