"""Relaunch the model server if it dies mid-run, so a crash costs minutes instead of the rest of the run.

Why. On 2026-10-11 two benches lost their SGLang server after READY to a CUDA OOM (free device memory
in this family's runs dips to 0.15 to 0.5 GiB): l1a1y8sv3 at once, l1a1y0sv2 after 1 h 54 min. Nothing
in the notebook restarts it, so every game then fails 11 analyzer calls and gives up. In a 9 h scored
run that banks whatever was reached before the death; the fail-closed check only catches runs that end
under 20 levels.

What. start() runs a daemon thread that polls the server process. When it has exited, the thread kills
what is left of its process group, waits for the GPU memory to come back, restarts the notebook's own
model precache (cell 9's precache(), the reason a first boot loads weights in 260 s; a relaunch without it
took 1,467 s in the 10-11 rehearsal), relaunches the same command with the same environment (appending to
the same log) and waits for /health. Games keep their state:
requests to a dead endpoint retry in place (ARC3_HTTP_RETRIES, set here to 30 min at 5 s), which keeps a
turn's tool work instead of rolling it back. Healthy runs see no retryable HTTP error at all (0 in five
logged runs), so the longer budget only acts during an outage. At most MAX_RESTARTS relaunches.

kill_at (bench test builds only, never a scored run) kills the server once, that many seconds after
start(), to rehearse the path on the card. solution/sim/test_watchdog.py drives the loop against a stub
HTTP server.
"""
from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
import urllib.request

MAX_RESTARTS = 3
HTTP_RETRIES = "360"  # in-place retries at 5 s: 30 min, longer than a relaunch (about 9 to 17 min)


def _healthy(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return response.status == 200
    except Exception:
        return False


def _gpu_used_mib() -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout
        return int(out.split()[0])
    except Exception:
        return None


def _kill_group(proc, grace_s: float) -> None:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except (ProcessLookupError, PermissionError):
            return
        time.sleep(grace_s)


def start(state: dict, args: list, env: dict, log, health_url: str, *, kill_at: float | None = None,
          precache=None, precache_paths: tuple = (), max_restarts: int = MAX_RESTARTS,
          ready_timeout: float = 1800.0, gpu_free_mib: int = 4096, wait_gpu_s: float = 300.0,
          poll_s: float = 5.0, grace_s: float = 5.0) -> dict:
    """Watch state["proc"] (the notebook's globals) and relaunch it with args/env when it exits; before each
    relaunch run precache(*precache_paths, threads=3) in a thread, as cell 15 does before the first boot.
    Returns a record with 'restarts', 'events' and a 'stop' Event to set once the run is over."""
    os.environ["ARC3_HTTP_RETRIES"] = HTTP_RETRIES
    t0 = time.time()
    record = {"restarts": 0, "events": [], "stop": threading.Event()}

    def note(message: str) -> None:
        record["events"].append(message)
        print(f"watchdog: {message}", flush=True)

    def relaunch() -> None:
        waited = time.time()
        while time.time() - waited < wait_gpu_s:
            used = _gpu_used_mib()
            if used is None or used < gpu_free_mib:
                break
            time.sleep(poll_s)
        if precache is not None and precache_paths:
            threading.Thread(target=precache, args=tuple(precache_paths), kwargs={"threads": 3},
                             daemon=True, name="arc3-watchdog-precache").start()
        with open(log, "ab", buffering=0) as logf:
            state["proc"] = subprocess.Popen(args, env=env, stdout=logf, stderr=subprocess.STDOUT,
                                             start_new_session=True)
        up = time.time()
        while time.time() - up < ready_timeout and state["proc"].poll() is None:
            if _healthy(health_url):
                note(f"READY again after {int(time.time() - up)}s (pid {state['proc'].pid})")
                return
            time.sleep(poll_s)
        note(f"relaunch {record['restarts']} did not come up within {int(time.time() - up)}s")

    def loop() -> None:
        test_kill_pending = kill_at is not None
        while not record["stop"].wait(poll_s):
            proc = state["proc"]
            if test_kill_pending and kill_at is not None and time.time() - t0 >= kill_at:
                test_kill_pending = False
                note(f"TEST kill of server pid {proc.pid} {int(time.time() - t0)}s after start")
                _kill_group(proc, grace_s)
            if proc.poll() is None:
                continue
            if record["restarts"] >= max_restarts:
                note(f"server exited ({proc.returncode}); {record['restarts']} relaunches used, stopping")
                return
            record["restarts"] += 1
            note(f"server exited ({proc.returncode}) {int(time.time() - t0)}s after start; "
                 f"relaunch {record['restarts']} of {max_restarts}")
            _kill_group(proc, grace_s)
            relaunch()

    threading.Thread(target=loop, daemon=True, name="arc3-watchdog").start()
    note(f"watching pid {state['proc'].pid}; in-place HTTP retries {HTTP_RETRIES}"
         + (f"; TEST kill at {kill_at:.0f}s" if kill_at is not None else ""))
    return record
