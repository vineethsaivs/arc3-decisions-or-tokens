"""Offline check for the sandbox memory cap (franzen_overlays/memcap.py).

    python3 solution/sim/test_franzen_memcap.py

Applies the overlay to a temp copy of Franzen's patched bundle after the shipped harness overlays (wastefix,
replycache, agentfix), and before them: only python_tool_sandbox.py changes, it and the child bootstrap compile,
the child sets RLIMIT_AS to 8 GiB and oom_score_adj to 1000, and a second application refuses. On Linux it also
runs the real sandbox: a 16 GB list dies with MemoryError, the child's oom_score_adj reads 1000, and the next
snippet still runs. macOS ignores RLIMIT_AS, so there the functional half is skipped and says so.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

from test_franzen_sandbox import INF, TREE, apply
from test_franzen_wastefix import load, run

SANDBOX = INF / "agent/python_tool_sandbox.py"
SHIPPED = ("wastefix", "replycache", "agentfix")


def check_static(bundle: Path) -> None:
    text = (bundle / SANDBOX).read_text()
    assert text.count("overlay memcap") == 2, "both memcap lines must land once"
    assert '(getattr(resource, "RLIMIT_AS", None), 8 << 30)' in text
    assert 'open("/proc/self/oom_score_adj", "w")' in text
    try:
        apply("memcap", bundle)
    except RuntimeError as exc:
        assert "already applied" in str(exc), exc
    else:
        raise AssertionError("a second application did not refuse")


def child_oom_score_adj(sandbox) -> tuple[str, list]:
    """Run a busy snippet in a thread and read the sandbox child's /proc/<pid>/oom_score_adj from the host.
    Returns (value of the first child that answered, every (pid, oom_score_adj, argv head) seen)."""
    import os
    import threading
    import time
    done = threading.Event()
    threading.Thread(target=lambda: (run(sandbox, "n = 0\nwhile True:\n    n += 1", timeout=4), done.set()),
                     daemon=True).start()
    me, seen, found = str(os.getpid()), "", []
    while not done.is_set() and seen != "1000":  # the child spawns before it applies its limits
        for proc in Path("/proc").iterdir():
            try:
                if proc.name.isdigit() and (proc / "stat").read_text().rsplit(")", 1)[1].split()[1] == me:
                    value = (proc / "oom_score_adj").read_text().strip()
                    argv = (proc / "cmdline").read_bytes().split(b"\0")[:3]
                    if not found or found[-1][:2] != (proc.name, value):
                        found.append((proc.name, value, [a[:20] for a in argv]))
                    if b"-I" in argv:
                        seen = value
            except OSError:
                continue
        time.sleep(0.02)
    done.wait(10)
    return seen, found[:6]


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="memcap-") as tmp:
        for order in (SHIPPED + ("memcap",), ("memcap",) + SHIPPED):
            bundle = Path(tmp) / "-".join(order)
            shutil.copytree(TREE, bundle)
            for name in order:
                apply(name, bundle)
            check_static(bundle)
            if order[0] == "memcap":
                continue
            after_shipped = Path(tmp) / "shipped-only"
            shutil.copytree(TREE, after_shipped)
            for name in SHIPPED:
                apply(name, after_shipped)
            diff = [p.relative_to(bundle) for p in bundle.rglob("*.py")
                    if p.read_bytes() != (after_shipped / p.relative_to(bundle)).read_bytes()]
            assert diff == [SANDBOX], diff
        print("ok   the edit lands once in either order, only the sandbox changes, a rerun refuses")

        if not sys.platform.startswith("linux"):
            print("skip functional checks: RLIMIT_AS is not enforced on", sys.platform)
            return
        _, sandbox = load(Path(tmp) / "-".join(SHIPPED + ("memcap",)))
        hog = run(sandbox, "x = [0] * (2 * 10**9)\nprint('allocated')", timeout=20)
        assert "MemoryError" in str(hog.get("error")) or "exited unexpectedly" in str(hog.get("error")), hog
        value, found = child_oom_score_adj(sandbox)  # snippets have no open(); read it from the host
        assert value == "1000", (value, found)
        assert run(sandbox, "print(1)").get("stdout", "").strip() == "1"
        big = run(sandbox, "seen = {(r, c, k): k for r in range(64) for c in range(64) for k in range(256)}\n"
                           "print(len(seen))", timeout=60)
        assert big.get("stdout", "").strip() == str(64 * 64 * 256), big  # about 1M tuples, well under the cap
        print("ok   a 16 GB list raises MemoryError, oom_score_adj is 1000, the next snippet and a 1M-tuple search run")


if __name__ == "__main__":
    main()
