"""Offline check for keepvars (franzen_overlays/keepvars.py).

On a copy of Franzen's patched bundle with l1a1y8sv2r3kp12lf15kv's overlays applied in the order its cell 7 runs them:
1. every overlay applies, the tree compiles, a second keepvars refuses; the prompt line says variables are kept;
2. plain data assigned in one call (ints, strings, nested lists, tuple-keyed dicts, sets) is there in the next, on the
   success path and on the error path, and `globals()['x'] = ...` works;
3. functions, classes, objects, NaN, private names, `result` and provided globals are never kept, a provided global is
   never overwritten by a kept one, and oversized values are dropped;
4. the agent clears the kept text at a level transition and at a new session (source check of the edited lines);
5. the pristine sandbox still raises NameError on the second call (what the overlay fixes).

    python3 solution/sim/test_franzen_keepvars.py
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

from test_franzen_sandbox import FORK_ENV, TREE, apply, import_tool_agent

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kaggle"))
import build_franzen_arms as B  # noqa: E402

ARM = "l1a1y8sv2r3kp12lf15kv"
STATE = {"current_frame": None, "history": [], "valid_actions": [], "last_action_result": {}}


def order() -> list[str]:
    nb = json.loads((B.BASE / "submission.ipynb").read_text())
    B.apply_edits(nb, B.REPLICA + B.ARMS[ARM][1]())
    return re.findall(r"^# arm overlay: (\w+)$", B.cell_text(nb["cells"][B.ENV]), re.M)


def call(mod, code, kept=None, **kw):
    return mod.run_sandboxed_python(code=code, timeout_seconds=10, initial_state=STATE,
                                    action_handler=lambda *_, **__: {}, kept_functions=[],
                                    **({"kept_vars": kept} if kept is not None else {}), **kw)


def main() -> None:
    names = order()
    assert names[0] == "keepvars" and {"imports", "levelfirst", "wastefix"} <= set(names), names
    with tempfile.TemporaryDirectory(prefix="keepvars-") as tmp:
        bundle, plain = Path(tmp) / "bundle", Path(tmp) / "plain"
        for path, run in ((bundle, names), (plain, [n for n in names if n != "keepvars"])):
            shutil.copytree(TREE, path, ignore=shutil.ignore_patterns("__pycache__"))
            for name in run:
                apply(name, path)
        try:
            apply("keepvars", bundle)
        except RuntimeError as exc:
            assert "already applied" in str(exc), exc
        else:
            raise AssertionError("a second keepvars did not refuse")
        agent_dir = bundle / "src/ARC3-Inference/inference/agent"
        prompts = (agent_dir / "prompts.py").read_text()
        assert "Plain-data variables you assign at top level" in prompts and "Python variables reset" not in prompts
        print(f"ok 1: {len(names)} overlays in cell 7 order (keepvars first); compiles; a second keepvars refuses; "
              "prompt updated")

        os.environ.update(FORK_ENV)
        sys.path.insert(0, str(bundle / "src" / "ARC3-Inference"))
        import inference.agent.python_tool_sandbox as sb  # noqa: E402
        ta = import_tool_agent(bundle)

        # 2: kept on success and on error, globals() works
        first = call(sb, "G = {(1, 2): [3, 4]}\nplan = ['UP', 'LEFT']\nseen = {1, 2}\nn = 5\nx = 2.5\n"
                         "s = 'ab'\nflag = True\nnothing = None\nempty = set()\nglobals()['y'] = 7\n"
                         "print('first ok')", kept={})
        assert not first.get("error"), first
        kept = first["kept_vars"]
        assert set(kept) == {"G", "plan", "seen", "n", "x", "s", "flag", "nothing", "empty", "y"}, kept
        second = call(sb, "print(G[(1, 2)], plan[-1], sorted(seen), n + y, x, s, flag, nothing, empty)", kept=kept)
        assert not second.get("error") and "[3, 4] LEFT [1, 2] 12 2.5 ab True None set()" in second["stdout"], second
        err = call(sb, "z = [1, 2]\nraise ValueError('boom')", kept=kept)
        assert "ValueError" in err.get("error", "") and err["kept_vars"].get("z") == "[1, 2]" and "G" in err["kept_vars"]
        print("ok 2: plain data survives the next call (tuple-keyed dict, set, empty set, None, float), on the error "
              "path too; globals()['y'] works")

        # 3: never kept
        third = call(sb, "def f():\n    return 1\nclass C:\n    pass\nobj = C()\nnan = float('nan')\n_private = 1\n"
                         "result = 3\nbig = list(range(30000))\nhuge = 'x' * 200000\nmixed = [1, obj]\n"
                         "current_frame = 'shadow'\n", kept=kept)
        got = third["kept_vars"]
        for name in ("f", "C", "obj", "nan", "_private", "result", "big", "huge", "mixed", "current_frame"):
            assert name not in got, (name, sorted(got))
        clash = call(sb, "print('frame', current_frame)", kept={**kept, "current_frame": "'stale'", "action": "1"})
        assert "frame None" in clash["stdout"] and not clash.get("error"), clash
        print("ok 3: functions, classes, objects, NaN, private names, result, provided globals, over-long and "
              "over-large values are not kept; a kept name never shadows a provided global")

        # 4: the agent side
        src = Path(ta.__file__).read_text()
        assert src.count("overlay keepvars") == 4
        assert "            self._kept_vars = {}  # overlay keepvars: variables never cross a level\n" in src
        assert "            kept_vars=dict(getattr(self, \"_kept_vars\", None) or {}),  # overlay keepvars\n" in src
        print("ok 4: the agent passes the kept text, records it after each call, clears it at a level transition "
              "and at a new session")

        # 5: the pristine sandbox
        for key in [k for k in sys.modules if k.startswith("inference")]:
            del sys.modules[key]
        sys.path.remove(str(bundle / "src" / "ARC3-Inference"))
        sys.path.insert(0, str(plain / "src" / "ARC3-Inference"))
        import inference.agent.python_tool_sandbox as psb  # noqa: E402
        a = call(psb, "G = 1")
        b = call(psb, "print(G)")
        assert "NameError" in (b.get("error") or "") and "kept_vars" not in a, (a, b)
        print("ok 5: the pin's sandbox raises NameError on the second call")
    print("PASS test_franzen_keepvars")


if __name__ == "__main__":
    main()
