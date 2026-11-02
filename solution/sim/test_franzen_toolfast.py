"""Offline check for toolfast (franzen_overlays/toolfast.py): the sandbox sees byte-identical state.

    python3 solution/sim/test_franzen_toolfast.py

Two bundles: the shipped harness overlays (wastefix, replycache, agentfix) and the same plus toolfast. Each runs
one scenario in its own process: a history grown one entry at a time as the solver does (the same list, the same
entry objects), with a result holding a tuple (not JSON-native), an empty-grid frame, a history list rebuilt from
scratch, and a file rewritten behind the cache. After every write: load_runtime_state, then the payload piped to
the sandbox (_ascii_frame_view_payload and _ascii_history_view_payload). The toolfast bundle must produce
byte-identical payloads at every step, and its load must equal a fresh parse of the file. Also: the edit lands once
in either order with the other overlays, only runtime_state.py and tool_agent.py change, a rerun refuses, the ascii
memo empties when its frames are freed, and the toolfast path is much faster at 300 entries.
"""
from __future__ import annotations

import gc
import hashlib
import json
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SIM = Path(__file__).resolve().parent
STEPS = 160


def scenario(bundle: Path) -> dict:
    """Runs inside a child process for one bundle; returns per-step digests and checks."""
    sys.path.insert(0, str(SIM))
    from test_franzen_sandbox import import_tool_agent  # noqa: PLC0415
    tool_agent = import_tool_agent(bundle)
    rs = sys.modules["inference.agent.runtime_state"]
    rng = random.Random(7)
    base = [[rng.choice([0, 0, 0, 3, 5, 9, 12]) for _ in range(64)] for _ in range(64)]

    def frame(i: int):
        g = [row[:] for row in base]
        g[i % 64][(i * 7) % 64] = 11
        return rs.Frame(grid=tuple(tuple(r) for r in g), step=i, level=1 + i // 50)

    def parse(path: Path):
        payload = json.loads(path.read_text())
        cur = rs.frame_from_payload(payload.get("current_frame"))
        hist = [e for e in (rs.history_entry_from_payload(x) for x in payload.get("history", [])) if e is not None]
        return cur, hist

    tmp = Path(tempfile.mkdtemp())
    path = tmp / "tool_runtime_state.json"
    history: list = []
    digests, load_ok, times = [], True, []

    def step(current) -> None:
        nonlocal load_ok
        t = time.perf_counter()
        rs.write_runtime_state(path, current_frame=current, history=history)
        cur, hist = rs.load_runtime_state(path)
        pipe = json.dumps({"current_frame": tool_agent._ascii_frame_view_payload(cur),
                           "history": tool_agent._ascii_history_view_payload(hist)})
        times.append((len(history), time.perf_counter() - t))
        digests.append(hashlib.sha256(pipe.encode()).hexdigest())
        if (cur, hist) != parse(path):
            load_ok = False

    step(None)
    for i in range(STEPS):
        result = {"executed": True, "i": i}
        if i == 5:
            result["pair"] = (1, 2)  # becomes a list in the file
        f = rs.Frame(grid=(), step=i, level=1) if i == 9 else frame(i)
        history.append(rs.HistoryEntry(action=f"ACTION{1 + i % 5}", frame=f, result=result))
        step(history[-1].frame)
    history[:] = [rs.HistoryEntry(action=e.action, frame=e.frame, result=dict(e.result)) for e in history[:40]]
    step(history[-1].frame)  # rebuilt list: no identity prefix past the start
    external = {"current_frame": None, "history": [rs.history_entry_to_payload(e) for e in history[:3]]}
    path.write_text(json.dumps(external, indent=2))
    cur, hist = rs.load_runtime_state(path)  # rewritten behind the cache: must parse
    behind_ok = (cur, hist) == parse(path) and len(hist) == 3
    memo = getattr(tool_agent, "_TOOLFAST_ASCII", None)
    memo_before = len(memo) if memo is not None else None
    history.clear()
    cache = getattr(rs, "_TOOLFAST_CACHE", None)
    if cache is not None:
        cache.clear()
    del cur, hist, f, result  # the loop's last frame is still alive through f
    gc.collect()
    memo_after = len(memo) if memo is not None else None
    per_entry = {n: dt for n, dt in times if n in (100, 150)}
    return {"digests": digests, "load_ok": load_ok, "behind_ok": behind_ok,
            "memo_before": memo_before, "memo_after": memo_after, "t": per_entry}


def main() -> None:
    if len(sys.argv) == 3 and sys.argv[1] == "--emit":
        print(json.dumps(scenario(Path(sys.argv[2]))))
        return
    from test_franzen_sandbox import INF, TREE, apply  # noqa: PLC0415
    shipped = ("wastefix", "replycache", "agentfix")
    with tempfile.TemporaryDirectory(prefix="toolfast-") as tmp:
        tmp = Path(tmp)
        bundles = {}
        for name, order in (("shipped", shipped), ("toolfast", shipped + ("toolfast",)),
                            ("toolfast-first", ("toolfast",) + shipped)):
            bundles[name] = tmp / name
            shutil.copytree(TREE, bundles[name])
            for overlay in order:
                apply(overlay, bundles[name])
        changed = sorted(str(p.relative_to(bundles["toolfast"])) for p in bundles["toolfast"].rglob("*.py")
                         if p.read_bytes() != (bundles["shipped"] / p.relative_to(bundles["toolfast"])).read_bytes())
        assert changed == [str(INF / "agent/runtime_state.py"), str(INF / "agent/tool_agent.py")], changed
        for name in ("toolfast", "toolfast-first"):
            try:
                apply("toolfast", bundles[name])
            except RuntimeError as exc:
                assert "already applied" in str(exc), exc
            else:
                raise AssertionError("a second application did not refuse")
        print("ok   lands once in either order; only runtime_state.py and tool_agent.py change; a rerun refuses")

        out = {}
        for name in ("shipped", "toolfast"):
            r = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--emit", str(bundles[name])],
                               capture_output=True, text=True, cwd=SIM)
            assert r.returncode == 0, r.stderr[-3000:]
            out[name] = json.loads(r.stdout.strip().splitlines()[-1])
        a, b = out["shipped"], out["toolfast"]
        assert len(a["digests"]) == len(b["digests"]) == STEPS + 2
        diff = [i for i, (x, y) in enumerate(zip(a["digests"], b["digests"])) if x != y]
        assert not diff, f"payload differs at steps {diff[:10]}"
        print(f"ok   the piped payload is byte-identical at all {STEPS + 2} steps")
        assert a["load_ok"] and b["load_ok"], (a["load_ok"], b["load_ok"])
        assert b["behind_ok"], "a file rewritten behind the cache was not re-parsed"
        print("ok   load equals a fresh parse at every step, and a file rewritten behind the cache is re-parsed")
        assert b["memo_before"] and b["memo_after"] == 0, (b["memo_before"], b["memo_after"])
        print(f"ok   the ascii memo held {b['memo_before']} frames and emptied when they were freed")
        speed = {n: a["t"][n] / b["t"][n] for n in a["t"]}
        assert all(v > 3 for v in speed.values()), speed
        print("ok   faster per action: " + ", ".join(f"{n} entries {a['t'][n]*1000:.0f} -> {b['t'][n]*1000:.0f} ms"
                                             for n in sorted(a["t"])))


if __name__ == "__main__":
    main()
