"""Offline check for keepvars2 (franzen_overlays/keepvars2.py).

On a copy of Franzen's patched bundle with l1a1y8sv2r3kp12lf15kv2's overlays applied in the order its cell 7 runs them:
1. every overlay applies, the tree compiles, a second keepvars refuses; both prompt edits are in;
2. plain data (frozensets too) assigned in one call is there in the next, on the success and the error path;
3. objects, NaN, private names, `result`, provided globals and oversized values are never kept, and a kept name never
   shadows a provided global;
4. a function that reads data defined in the same call is retained and works next call, and so does one that reads data
   kept from an earlier call; after a level change clears the data the function is refused with the harness's own
   "unavailable dependencies" note and nothing crashes (with and without repair hints);
5. on the same tree with v1 (keepvars) that function is refused: the defect v2 fixes;
6. the agent clears the kept text at a level transition and at a new session (source check).

    python3 solution/sim/test_franzen_keepvars2.py
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

ARM = "l1a1y8sv2r3kp12lf15kv2"
STATE = {"current_frame": None, "history": [], "valid_actions": [], "last_action_result": {}}


def order() -> list[str]:
    nb = json.loads((B.BASE / "submission.ipynb").read_text())
    B.apply_edits(nb, B.REPLICA + B.ARMS[ARM][1]())
    return re.findall(r"^# arm overlay: (\w+)$", B.cell_text(nb["cells"][B.ENV]), re.M)


def call(mod, code, kept=None, funcs=(), hints=True):
    return mod.run_sandboxed_python(code=code, timeout_seconds=10, initial_state=STATE,
                                    action_handler=lambda *_, **__: {}, kept_functions=list(funcs),
                                    retain_imports=hints, repair_hints=hints,
                                    **({"kept_vars": kept} if kept is not None else {}))


def sources(result):
    return [f["source"] for f in result.get("keepable_functions") or []]


def load(tree: Path):
    for key in [k for k in sys.modules if k.startswith("inference")]:
        del sys.modules[key]
    sys.path[:] = [p for p in sys.path if not p.endswith("ARC3-Inference")]
    sys.path.insert(0, str(tree / "src" / "ARC3-Inference"))
    import inference.agent.python_tool_sandbox as sb  # noqa: E402
    return sb


def main() -> None:
    names = order()
    assert names[0] == "keepvars2" and {"imports", "levelfirst", "wastefix"} <= set(names), names
    with tempfile.TemporaryDirectory(prefix="keepvars2-") as tmp:
        bundle, v1 = Path(tmp) / "bundle", Path(tmp) / "v1"
        for path, run in ((bundle, names), (v1, ["keepvars" if n == "keepvars2" else n for n in names])):
            shutil.copytree(TREE, path, ignore=shutil.ignore_patterns("__pycache__"))
            for name in run:
                apply(name, path)
        for again in ("keepvars", "keepvars2"):
            try:
                apply(again, bundle)
            except RuntimeError as exc:
                assert "already applied" in str(exc), exc
            else:
                raise AssertionError(f"a second {again} did not refuse")
        agent_dir = bundle / "src/ARC3-Inference/inference/agent"
        prompts = (agent_dir / "prompts.py").read_text()
        assert "Plain-data variables you assign at top level" in prompts and "Python variables reset" not in prompts
        assert "and your kept plain-data variables (a function that reads them is dropped" in prompts
        print(f"ok 1: {len(names)} overlays in cell 7 order (keepvars2 first); compiles; a second keepvars refuses; "
              "both prompt edits in")

        os.environ.update(FORK_ENV)
        sb = load(bundle)

        # 2: kept on success and on error
        first = call(sb, "G = {(1, 2): [3, 4]}\nplan = ['UP', 'LEFT']\nseen = {1, 2}\nfz = frozenset({(1, 2), (3, 4)})\n"
                         "n = 5\nx = 2.5\ns = 'ab'\nflag = True\nnothing = None\nempty = set()\nglobals()['y'] = 7\n"
                         "print('first ok')", kept={})
        assert not first.get("error"), first
        kept = first["kept_vars"]
        assert set(kept) == {"G", "plan", "seen", "fz", "n", "x", "s", "flag", "nothing", "empty", "y"}, kept
        second = call(sb, "print(G[(1, 2)], plan[-1], sorted(seen), sorted(fz), type(fz).__name__, n + y, x, s, flag, "
                          "nothing, empty)", kept=kept)
        assert not second.get("error") and ("[3, 4] LEFT [1, 2] [(1, 2), (3, 4)] frozenset 12 2.5 ab True None set()"
                                            in second["stdout"]), second
        err = call(sb, "z = [1, 2]\nraise ValueError('boom')", kept=kept)
        assert "ValueError" in err.get("error", "") and err["kept_vars"].get("z") == "[1, 2]" and "G" in err["kept_vars"]
        print("ok 2: plain data and frozensets survive the next call, on the error path too; globals()['y'] works")

        # 3: never kept
        third = call(sb, "class C:\n    pass\nobj = C()\nnan = float('nan')\n_private = 1\nresult = 3\n"
                         "big = list(range(30000))\nhuge = 'x' * 200000\nmixed = [1, obj]\ncurrent_frame = 'shadow'\n",
                     kept=kept)
        got = third["kept_vars"]
        for name in ("C", "obj", "nan", "_private", "result", "big", "huge", "mixed", "current_frame"):
            assert name not in got, (name, sorted(got))
        clash = call(sb, "print('frame', current_frame)", kept={**kept, "current_frame": "'stale'", "action": "1"})
        assert "frame None" in clash["stdout"] and not clash.get("error"), clash
        print("ok 3: objects, NaN, private names, result, provided globals, over-large values are not kept; a kept name "
              "never shadows a provided global")

        # 4: functions that read kept data
        for hints in (True, False):
            a = call(sb, "DIRS = {'UP': (-1, 0), 'DOWN': (1, 0)}\ndef step(p, d):\n    dr, dc = DIRS[d]\n"
                         "    return (p[0] + dr, p[1] + dc)\nprint(step((5, 5), 'UP'))", kept={}, hints=hints)
            assert not a.get("error") and "(4, 5)" in a["stdout"], a
            assert [f["name"] for f in a["keepable_functions"]] == ["step"], a.get("retention_rejected")
            b = call(sb, "W = 3\ndef wide(p):\n    return step(p, 'DOWN')[0] * W\nprint(wide((1, 1)))",
                     kept=a["kept_vars"], funcs=sources(a), hints=hints)
            assert not b.get("error") and "6" in b["stdout"], b
            assert sorted(f["name"] for f in b["keepable_functions"]) == ["step", "wide"], b.get("retention_rejected")
            c = call(sb, "print(step((0, 0), 'DOWN'), wide((2, 2)))", kept=b["kept_vars"], funcs=sources(b), hints=hints)
            assert not c.get("error") and "(1, 0) 9" in c["stdout"], c
            # level change: the agent sends no kept text; the functions are refused, not crashed on
            d = call(sb, "print('new level')", kept={}, funcs=sources(c), hints=hints)
            assert not d.get("error") and "new level" in d["stdout"], d
            assert not d["keepable_functions"], d
            why = json.dumps(d.get("retention_rejected"))
            assert "DIRS" in why and "W" in why, why
        print("ok 4: a function reading same-call or kept data is retained and works next call; after a level change "
              "it is refused with an unavailable-dependencies note, no crash (with and without repair hints)")

        # 5: v1 refuses the same function
        s1 = load(v1)
        a1 = call(s1, "DIRS = {'UP': (-1, 0)}\ndef step(p, d):\n    dr, dc = DIRS[d]\n    return (p[0] + dr, p[1] + dc)\n"
                      "print(step((5, 5), 'UP'))", kept={})
        assert not a1["keepable_functions"] and "DIRS" in json.dumps(a1["retention_rejected"]), a1
        print("ok 5: v1 (keepvars) refuses that function: unavailable dependencies: DIRS")

        # 6: the agent side
        sys.path[:] = [p for p in sys.path if not p.endswith("ARC3-Inference")]
        ta = import_tool_agent(bundle)
        src = Path(ta.__file__).read_text()
        assert src.count("overlay keepvars") == 4
        assert "            self._kept_vars = {}  # overlay keepvars: variables never cross a level\n" in src
        print("ok 6: the agent passes the kept text, records it after each call, clears it at a level transition and "
              "at a new session")
    print("PASS test_franzen_keepvars2")


if __name__ == "__main__":
    main()
