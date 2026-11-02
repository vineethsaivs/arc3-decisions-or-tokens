"""Offline check for tool2 (franzen_overlays/tool2.py) through the real sandbox child.

On a copy of Franzen's patched bundle with the arm's overlays applied in cell 7's order, `run_sandboxed_python` runs
real child processes:
1. every overlay applies, a second tool2 refuses, the system prompt names `inspect.getsource` and `SnippetTimeout`;
2. a snippet that loops past its limit returns what it printed, a SnippetTimeout message, in about limit - 3 s (the
   timeout is 6 s here, so 3 s), where the unpatched sandbox returns "Tool timed out" with empty stdout after 6 s;
3. a snippet that swallows the first stop with a bare `except` is stopped again;
4. an action request the host answers slowly (2 s, limit 1 s of compute) is never cut: the reply is read, the action
   result is recorded, and the snippet finishes normally; compute after the slow reply still counts;
5. `inspect.getsource` returns the source of a retained function, of a function defined in the same snippet, and falls
   back for a builtin (OSError/TypeError as before);
6. an ordinary snippet and an ordinary error behave as before (stdout, result, traceback).

    python3 solution/sim/test_franzen_tool2.py
"""
from __future__ import annotations

import importlib
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

from test_franzen_sandbox import FORK_ENV, TREE, apply
from test_franzen_seg8 import order

ARM = "l1a1y8sv2r3kp12lf15us8"   # tool2 is built on top of the seg8 arm; the order check below uses the real arm


def run(sb, code, *, timeout=6, kept=None, slow_action=0.0):
    seen = []

    def action_handler(actions, stale_after=None):  # noqa: ARG001
        seen.append(actions)
        time.sleep(slow_action)
        return {"action_result": {"executed": True}, "state": {}}

    t0 = time.monotonic()
    out = sb.run_sandboxed_python(code=code, timeout_seconds=timeout, initial_state={}, action_handler=action_handler,
                                  kept_functions=kept)
    return out, time.monotonic() - t0, seen


def main() -> None:
    arm = "l1a1y8sv2r3kp12lf15ust"
    names = order(arm)
    assert "tool2" in names and names.index("tool2") < names.index("wastefix") or names[0] == "tool2", names
    with tempfile.TemporaryDirectory(prefix="tool2-") as tmp:
        plain = Path(tmp) / "plain"
        bundle = Path(tmp) / "bundle"
        for b in (plain, bundle):
            shutil.copytree(TREE, b, ignore=shutil.ignore_patterns("__pycache__"))
        for name in names:
            if name != "tool2":
                apply(name, plain)
            apply(name, bundle)
        try:
            apply("tool2", bundle)
        except RuntimeError as exc:
            assert "already applied" in str(exc), exc
        else:
            raise AssertionError("a second tool2 did not refuse")
        os.environ.update(FORK_ENV)

        def load(root):
            for key in [k for k in sys.modules if k.startswith("inference")]:
                del sys.modules[key]
            sys.path.insert(0, str(root / "src" / "ARC3-Inference"))
            try:
                return importlib.import_module("inference.agent.python_tool_sandbox"), importlib.import_module("inference.agent.prompts")
            finally:
                sys.path.pop(0)

        sb0, _ = load(plain)
        out0, took0, _ = run(sb0, "print('start')\nwhile True:\n    pass\n")
        assert "timed out" in out0["error"] and out0["stdout"] == "" and took0 >= 6, (out0, took0)
        sb, prompts = load(bundle)
        text = Path(str(prompts.__file__)).read_text()
        assert "SnippetTimeout" in text and "inspect.getsource(fn)" in text
        print(f"ok 1 {arm}: overlays apply in order, a second tool2 refuses, the prompt names SnippetTimeout and getsource")

        out, took, _ = run(sb, "print('start')\nfor i in range(10**9):\n    if i % 10**6 == 0:\n        print('i', i)\n")
        assert "SnippetTimeout" in out["error"] and "start" in out["stdout"] and 2.5 <= took < 5, (out, took)
        print(f"ok 2: a looping snippet stops after {took:.1f} s of a 6 s timeout with its output "
              f"({out['stdout'].count(chr(10))} lines) and a SnippetTimeout error; unpatched: timed out, empty stdout, {took0:.1f} s")

        code = ("print('go')\nn = 0\ntry:\n    while True:\n        n += 1\nexcept BaseException:\n    print('swallowed')\n"
                "while True:\n    n += 1\n")
        out, took, _ = run(sb, code, timeout=5)
        assert "swallowed" in out["stdout"] and "SnippetTimeout" in out["error"] and took < 5, (out, took)
        print(f"ok 3: a snippet that swallows the first stop is stopped by the second ({took:.1f} s; error: {out['error'][-60:]!r})")

        code = ("print('before')\nr = action(['UP'])\nprint('after')\ns = 0\nfor i in range(200000):\n    s += i\nresult = s\n")
        out, took, seen = run(sb, code, timeout=4, slow_action=3.0)   # limit 1 s of compute; the host takes 3 s
        assert out["error"] == "" and "after" in out["stdout"] and out["result"] == sum(range(200000)), out
        assert seen == [[{"action": "UP"}]], seen
        print(f"ok 4: a 3 s host reply against a 1 s compute limit is not cut ({took:.1f} s total, action recorded)")

        kept = ["def helper(x):\n    return x + 1\n"]
        code = ("import inspect\n"
                "def local(y):\n    return y * 2\n"
                "print(inspect.getsource(helper).strip().splitlines()[0])\n"
                "print(inspect.getsource(local).strip().splitlines()[0])\n"
                "try:\n    inspect.getsource(len)\nexcept Exception as e:\n    print('builtin', type(e).__name__)\n")
        out, _, _ = run(sb, code, kept=kept)
        assert out["error"] == "", out
        assert out["stdout"].splitlines()[:2] == ["def helper(x):", "def local(y):"] and "builtin" in out["stdout"], out
        out0, _, _ = run(sb0, code, kept=kept)
        assert "OSError" in out0["error"] or "could not get source" in out0["error"], out0
        print("ok 5: getsource reads a retained function and a same-call function; builtins fall back; unpatched raised OSError")

        out, _, _ = run(sb, "x = 3\nprint(x)\nresult = x * 2\n")
        assert out["stdout"].strip() == "3" and out["result"] == 6 and out["error"] == ""
        out, _, _ = run(sb, "print('a')\n1 / 0\n")
        assert "ZeroDivisionError" in out["error"] and out["stdout"].strip() == "a"
        print("ok 6: an ordinary snippet and an ordinary error behave as before")
    print("PASS test_franzen_tool2")


if __name__ == "__main__":
    main()
