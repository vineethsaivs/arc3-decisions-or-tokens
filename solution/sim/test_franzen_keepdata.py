"""Offline check for keepdata (franzen_overlays/keepdata.py) on a real game through the real host and sandbox.

    python3 solution/sim/test_franzen_keepdata.py

Re-runs itself under the engine interpreter the agentfix test uses (reference/re-arc-3/.venv/bin/python or
$AGENTFIX_PYTHON) and borrows its Play harness: taaf's GameAPI on the OFFLINE arcade, the solver's step_env,
ToolAgent._run_python_tool and the real sandbox child, under cell 7's settings. The bundle carries the overlays in
the order cell 7 runs them. Checks: plain data (int, list, set of tuples, dict) is restored in the next call with
its type; a frozenset, a FrameView, a value over 20K chars and a private name are not kept, and a provided name is
never overwritten; an augmented assignment updates the kept value; an error keeps what ran before it; a
SyntaxError and a timeout keep the old set; a function that reads kept data is still not retained; time and
hashlib import; an executed action empties the set and the name is gone next call. Then: only the sandbox, the
host and the prompt module list change, and a second application refuses.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
ENGINE_PY = Path(os.environ.get("AGENTFIX_PYTHON", REPO / "reference/re-arc-3/.venv/bin/python"))


def _reexec() -> None:
    try:
        for name in ("numpy", "pydantic", "flask", "matplotlib", "imageio", "dotenv", "requests", "PIL"):
            __import__(name)
        return
    except ImportError as exc:
        missing = exc.name
    if os.environ.get("_KEEPDATA_CHILD") or not ENGINE_PY.exists():
        raise SystemExit(f"{sys.executable} lacks {missing}; set AGENTFIX_PYTHON ({ENGINE_PY} not usable)")
    env = dict(os.environ, _KEEPDATA_CHILD="1", _AGENTFIX_CHILD="1", PYTHONDONTWRITEBYTECODE="1")
    raise SystemExit(subprocess.run([str(ENGINE_PY), str(Path(__file__).resolve())], env=env).returncode)


_reexec()
import test_franzen_agentfix as AF  # noqa: E402  (engine wheels, cell 7's env, the Play harness)
from test_franzen_sandbox import INF, apply  # noqa: E402

CELL7_ORDER = ("keepdata", "toolfast", "memcap", "agentfix", "replycache", "wastefix")
SETUP = ("w = 5\nplan = ['UP', 'LEFT']\nwalls = {(1, 2), (3, 4)}\ninfo = {'a': [1, 2]}\n"
         "fs = frozenset({1})\ncf = current_frame\nbig = 'x' * 25000\n_hidden = 1\nprint('set')")


def stdout(out: dict) -> str:
    return str(out.get("stdout", "")).strip()


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="keepdata-") as tmp:
        tmp = Path(tmp)
        bundle = AF.copy_tree(tmp / "bundle")
        for name in CELL7_ORDER:
            apply(name, bundle)
        g = AF.Play(AF.load(bundle), "cd82", tmp)
        kept = lambda: g.agent._kept_data  # noqa: E731

        out = g.snippet(SETUP)
        assert "error" not in out and stdout(out) == "set", out
        assert set(kept()) == {"w", "plan", "walls", "info"}, kept()
        out = g.snippet("print(w, plan, sorted(walls), info, type(walls).__name__)")
        assert stdout(out) == "5 ['UP', 'LEFT'] [(1, 2), (3, 4)] {'a': [1, 2]} set", out
        print("ok   plain data comes back with its type; frozenset, FrameView, 25K str and _private are not kept")

        out = g.snippet("w += 2\nprint(w)")
        assert stdout(out) == "7" and kept()["w"] == "7", (out, kept())
        g.agent._kept_data["current_frame"] = "1"  # a provided name must never be overwritten
        out = g.snippet("print(type(current_frame).__name__)")
        assert stdout(out) != "int" and "current_frame" not in kept(), (out, kept())
        print("ok   an augmented assignment updates the value; a provided name is never overwritten")

        out = g.snippet("z = 9\nraise ValueError('boom')")
        assert "ValueError" in str(out.get("error")) and kept().get("z") == "9" and "w" in kept(), (out, kept())
        before = dict(kept())
        out = g.snippet("a = (")
        assert "error" in out and kept() == before, (out, kept())
        timeout = g.agent._python_timeout
        g.agent._python_timeout = 2
        out = g.snippet("t = 1\nwhile True:\n    pass")
        g.agent._python_timeout = timeout
        assert "timed out" in str(out.get("error")).lower() and kept() == before, (out, kept())
        print("ok   an error keeps what ran before it; a SyntaxError and a timeout keep the old set")

        out = g.snippet("k = 3\ndef f():\n    return k\nprint(f())")
        assert stdout(out) == "3" and "f" not in g.agent._kept_functions and kept().get("k") == "3", (out, kept())
        out = g.snippet("import time, hashlib\nprint(hashlib.md5(b'a').hexdigest()[:6], type(time.time()).__name__)")
        assert stdout(out) == "0cc175 float", out
        print("ok   a function reading kept data is still not retained; time and hashlib import")

        actions = g.actions
        out = g.snippet("q = 1\naction(['LEFT'])")
        assert g.actions == actions + 1 and kept() == {}, (out, kept())
        out = g.snippet("print(q)")
        assert "NameError" in str(out.get("error")), out
        print("ok   an executed action empties the set; the name is gone in the next call")

        shipped = AF.copy_tree(tmp / "shipped")
        for name in CELL7_ORDER[1:]:
            apply(name, shipped)
        changed = sorted(str(p.relative_to(bundle)) for p in bundle.rglob("*.py")
                         if p.read_bytes() != (shipped / p.relative_to(bundle)).read_bytes())
        assert changed == sorted(str(INF / f) for f in ("agent/prompts.py", "agent/python_tool_sandbox.py",
                                                         "agent/tool_agent.py")), changed
        try:
            apply("keepdata", bundle)
        except RuntimeError as exc:
            assert "already applied" in str(exc), exc
        else:
            raise AssertionError("a second application did not refuse")
        print("ok   only the sandbox, the host and the prompt module list change; a rerun refuses")


if __name__ == "__main__":
    main()
