"""franzen_overlays/watchdog.py against a stub HTTP server: a test kill and a crash are each relaunched
once and come back healthy, the relaunch cap holds, and a set stop flag ends the watch.

    python3 solution/sim/test_watchdog.py      # no GPU, stdlib only
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OVERLAY = REPO / "solution/kaggle/franzen_overlays/watchdog.py"
# /health answers 200; with a flag file present the server deletes it and exits after 1 s (one crash).
STUB = """
import http.server, os, sys, threading
port, flag, always = int(sys.argv[1]), sys.argv[2], sys.argv[3] == "1"
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200 if self.path == "/health" else 404); self.end_headers()
    def log_message(self, *a):
        pass
if always or os.path.exists(flag):
    if os.path.exists(flag):
        os.remove(flag)
    threading.Timer(1.0, lambda: os._exit(3)).start()
http.server.HTTPServer(("127.0.0.1", port), H).serve_forever()
"""
FAST = {"poll_s": 0.2, "grace_s": 0.2, "ready_timeout": 20.0}


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait(predicate, timeout: float = 20.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.1)
    return False


def scenario(tmp: Path, *, crash_once: bool = False, always_crash: bool = False, **kwargs):
    ns: dict = {}
    exec(compile(OVERLAY.read_text(), str(OVERLAY), "exec"), ns)
    port, flag = free_port(), tmp / f"crash-{time.monotonic_ns()}"
    if crash_once:
        flag.touch()
    args = [sys.executable, "-c", STUB, str(port), str(flag), "1" if always_crash else "0"]
    log = tmp / "serve.log"
    state = {"proc": subprocess.Popen(args, start_new_session=True)}
    first_pid = state["proc"].pid
    url = f"http://127.0.0.1:{port}/health"
    assert wait(lambda: ns["_healthy"](url)), "stub never came up"
    record = ns["start"](state, args, dict(os.environ), log, url, **{**FAST, **kwargs})
    return ns, state, record, first_pid, url


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="watchdog-") as tmp:
        tmp = Path(tmp)

        cached = []
        ns, state, record, first, url = scenario(tmp, kill_at=0.5, precache=lambda *p, threads: cached.append((p, threads)),
                                                  precache_paths=("model", "draft"))
        assert os.environ["ARC3_HTTP_RETRIES"] == ns["HTTP_RETRIES"]
        assert wait(lambda: any("READY again" in e for e in record["events"])), record["events"]
        assert record["restarts"] == 1 and state["proc"].pid != first and ns["_healthy"](url)
        assert any(e.startswith("TEST kill") for e in record["events"])
        assert cached == [(("model", "draft"), 3)], cached  # one precache per relaunch, as the first boot
        record["stop"].set()
        state["proc"].kill()
        print("ok   test kill relaunched once and came back healthy")

        ns, state, record, first, url = scenario(tmp, crash_once=True)
        assert wait(lambda: any("READY again" in e for e in record["events"])), record["events"]
        assert record["restarts"] == 1 and state["proc"].pid != first and ns["_healthy"](url)
        time.sleep(1.5)  # the relaunched server has no flag and stays up
        assert record["restarts"] == 1 and state["proc"].poll() is None
        record["stop"].set()
        state["proc"].kill()
        print("ok   a crash was relaunched once; the relaunch stays up")

        ns, state, record, first, url = scenario(tmp, always_crash=True, max_restarts=2)
        assert wait(lambda: any("relaunches used, stopping" in e for e in record["events"]), 30), record["events"]
        assert record["restarts"] == 2
        print("ok   the relaunch cap holds")

        ns, state, record, first, url = scenario(tmp)
        record["stop"].set()
        time.sleep(0.5)
        state["proc"].kill()
        time.sleep(1.0)
        assert record["restarts"] == 0 and state["proc"].pid == first
        print("ok   a set stop flag ends the watch before any relaunch")


if __name__ == "__main__":
    main()
