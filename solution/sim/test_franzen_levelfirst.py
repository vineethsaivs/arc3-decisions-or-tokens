"""Offline check for levelfirst and imports (franzen_overlays/levelfirst.py, imports.py).

    python3 solution/sim/test_franzen_levelfirst.py

On a copy of Franzen's patched bundle with l1a1y8sv2r3kp12lf's overlays applied in the order its cell 7 runs them:
1. every overlay applies (levelfirst and imports run first), the tree compiles, a second application refuses;
2. the real trimmer on a three-level history over the budget: the current level keeps every message the pristine
   trimmer keeps, earlier levels drain to the 12K tail, the request starts with a user message, the input is not
   mutated, the eviction flag is set, and a control-tagged resume opener repeating the level-up text is not taken as
   the level start; with a long current level the result equals the pristine trimmer's;
3. under the budget, on level 1 (no level-up opener) and with ARC3_LEVEL_DRAIN_TAIL=-1, it equals the pristine trimmer;
4. the sandbox imports the seven modules, time.sleep(30) returns at once, and sys and pickle stay blocked.
No GPU, no network.
"""
from __future__ import annotations

import copy
import json
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

from test_franzen_sandbox import FORK_ENV, TREE, apply, import_tool_agent

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kaggle"))
import build_franzen_arms as B  # noqa: E402

ARM = "l1a1y8sv2r3kp12lf"
BUNDLE_ONLY = {"keepalive"}  # solver.py only; applied too, it just does not touch what this checks
IMG = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
LEVEL_UP = "You have completed the previous level. `current_frame` now contains the starting board"


def order() -> list[str]:
    nb = json.loads((B.BASE / "submission.ipynb").read_text())
    B.apply_edits(nb, B.REPLICA + B.ARMS[ARM][1]())
    return re.findall(r"^# arm overlay: (\w+)$", B.cell_text(nb["cells"][B.ENV]), re.M)


def turns(level: int, n: int, *, opener: str, size: int = 6000) -> list[dict]:
    """n tool rounds; a fresh opener every 2 rounds (a turn ends when an action runs, about 8 turns per level)."""
    out = [{"role": "user", "content": [{"type": "text", "text": opener + " " + "o" * 2500}, IMG]}]
    for t in range(n):
        if t and t % 2 == 0:
            out.append({"role": "user", "content": [{"type": "text", "text": f"The code executed {t} actions. "
                                                     "You are still on the same level. " + "o" * 2500}, IMG]})
        out.append({"role": "assistant", "reasoning_content": f"think L{level}.{t} " + "r" * size,
                    "tool_calls": [{"id": f"c{level}{t}", "type": "function", "function": {
                        "name": "python", "arguments": json.dumps({"code": f"x={t}\n" + "c" * 800})}}]})
        out.append({"role": "tool", "tool_call_id": f"c{level}{t}", "content": f"result L{level}.{t} " + "t" * 1500})
    return out


def history(l1: int, l2: int, l3: int) -> tuple[list[dict], int]:
    msgs = [{"role": "system", "content": "s" * 15000}]
    msgs += turns(1, l1, opener="The code executed 3 actions. You are still on the same level.")
    msgs += turns(2, l2, opener=LEVEL_UP)
    start = len(msgs)
    level3 = turns(3, l3, opener=LEVEL_UP)
    # a yield inside level 3: a control-tagged resume opener that repeats the level-up text, then more turns
    mid = 1 + 2 * (l3 // 2)
    level3[mid:mid] = [{**level3[0], "_arc3_control": "resume"},
                       {"role": "assistant", "reasoning_content": "think resume " + "r" * 3000, "content": None}]
    return msgs + level3, start


def main() -> None:
    names = order()
    assert names[:2] == ["imports", "levelfirst"], names
    with tempfile.TemporaryDirectory(prefix="levelfirst-") as tmp:
        bundle, plain_bundle = Path(tmp) / "bundle", Path(tmp) / "plain"
        for path, run in ((bundle, names), (plain_bundle, [n for n in names if n not in ("imports", "levelfirst")])):
            shutil.copytree(TREE, path, ignore=shutil.ignore_patterns("__pycache__"))
            for name in run:
                apply(name, path)
        for name in ("levelfirst", "imports"):
            try:
                apply(name, bundle)
            except RuntimeError as exc:
                assert "already applied" in str(exc), exc
            else:
                raise AssertionError(f"a second {name} did not refuse")
        print(f"ok 1: {len(names)} overlays in cell 7 order {names}; compiles; a second application refuses")

        os.environ.update(FORK_ENV)
        os.environ.update({"ARC3_IMAGE_TOKENS_FLAT": "400"})
        os.environ.pop("ARC3_LEVEL_DRAIN_TAIL", None)
        ta = import_tool_agent(bundle)
        for key in [k for k in sys.modules if k.startswith("inference")]:
            del sys.modules[key]
        sys.path.remove(str(bundle / "src" / "ARC3-Inference"))
        pa = import_tool_agent(plain_bundle)
        assert "overlay levelfirst" in Path(ta.__file__).read_text() and "_drain_older_levels" not in Path(pa.__file__).read_text()

        def agent(mod):
            a = object.__new__(mod.ToolAgent)
            a._context_budget_tokens = max(1024, 128 * 1024 - 12 * 1024 - 512)
            a._text_chars_per_token = 3.05
            a._has_evicted = False
            a._context_was_trimmed = False
            return a

        def trim(mod, msgs):
            a = agent(mod)
            frozen = copy.deepcopy(msgs)
            out = a._trim_messages_for_context(list(msgs))
            assert msgs == frozen, "input mutated"
            return out, a

        def toks(a, msgs):
            return a._estimate_request_input_tokens(msgs) - a._estimate_request_input_tokens([])

        # 2a: earlier levels are most of the history
        msgs, start = history(22, 14, 6)
        base, _ = trim(pa, msgs)
        new, a = trim(ta, msgs)
        cur = msgs[start:]
        ids = {id(m) for m in cur}  # identity: the level-2 and level-3 openers are equal dicts
        assert [id(m) for m in base if id(m) in ids] == [id(m) for m in new if id(m) in ids] == [id(m) for m in cur], \
            "current level not kept whole"
        older = [m for m in new[1:] if id(m) not in ids]
        assert 0 < toks(a, older) <= 12288, toks(a, older)
        assert new[1]["role"] == "user" and a._context_was_trimmed and a._has_evicted
        assert toks(a, new) < toks(a, base), (toks(a, new), toks(a, base))
        print(f"ok 2a: before {toks(a, msgs)} tokens; pristine keeps {toks(a, base)}, levelfirst {toks(a, new)} "
              f"(earlier levels {toks(a, older)}); the current level ({len(cur)} messages, a resume opener inside) whole")

        # 2b: a long current level: the drain runs into it, exactly as the pristine trimmer does
        msgs, start = history(6, 6, 32)
        base, _ = trim(pa, msgs)
        new, a = trim(ta, msgs)
        assert new == base, "long current level differs from the pristine trimmer"
        print(f"ok 2b: long current level: same {len(new) - 1} messages as the pristine trimmer")

        # 3: under the budget, level 1, the knob off
        msgs, _ = history(3, 3, 3)
        assert trim(ta, msgs)[0] == msgs and not trim(ta, msgs)[1]._context_was_trimmed
        level1 = [{"role": "system", "content": "s" * 15000}] + turns(1, 46, opener="The code executed 3 actions.")
        assert trim(ta, level1)[0] == trim(pa, level1)[0] != level1
        os.environ["ARC3_LEVEL_DRAIN_TAIL"] = "-1"
        msgs, _ = history(22, 14, 6)
        assert trim(ta, msgs)[0] == trim(pa, msgs)[0]
        os.environ.pop("ARC3_LEVEL_DRAIN_TAIL")
        print("ok 3: under the budget unchanged; level 1 and ARC3_LEVEL_DRAIN_TAIL=-1 equal the pristine trimmer")

        # 4: the sandbox
        state = {"current_frame": None, "history": [], "valid_actions": [], "last_action_result": {}}
        code = ("import time, inspect, traceback, types, hashlib, pprint, timeit, typing, enum\n"
                "from dataclasses import dataclass\n"
                "t = time.time(); time.sleep(30); from time import sleep; sleep(30)\n"
                "print('slept', round(time.time() - t, 2), hashlib.md5(b'x').hexdigest()[:6], types.SimpleNamespace(a=1))\n"
                "class Node:\n"
                "    def __init__(self, pos):\n"
                "        super().__init__()\n"
                "        self.pos = pos\n"
                "@dataclass(frozen=True)\n"
                "class State:\n"
                "    pos: tuple\n"
                "    keys: int = 0\n"
                "n = Node((1, 2)); setattr(n, 'g', 3); o = object()\n"
                "print('classes', n.pos, vars(n)['g'], State((1, 2)) == State((1, 2)), id(o) > 0, __name__)\n"
                "for bad in ('sys', 'pickle', 'builtins'):\n"
                "    try:\n"
                "        __import__(bad)\n"
                "        print('LEAK', bad)\n"
                "    except ImportError:\n"
                "        print('blocked', bad)\n")
        t0 = time.time()
        out = ta.run_sandboxed_python(code=code, timeout_seconds=10, initial_state=state,
                                      action_handler=lambda *_, **__: {})
        took = time.time() - t0
        text = out.get("stdout", "")
        assert not out.get("error"), out
        assert "slept 0.0" in text and "classes (1, 2) 3 True True __main__" in text, out
        assert "LEAK" not in text and text.count("blocked") == 3 and took < 8, (took, out)
        plain = pa.run_sandboxed_python(code="class A:\n    pass\n", timeout_seconds=10, initial_state=state,
                                        action_handler=lambda *_, **__: {})
        assert "__build_class__" in (plain.get("error") or ""), plain
        print(f"ok 4: 10 modules import, two sleep(30) calls took {took:.1f} s in all; a class, a dataclass, super, "
              "setattr, vars, object and except ImportError work (the pin's sandbox raises on `class A`); "
              "sys/pickle/builtins blocked")
    print("PASS test_franzen_levelfirst")


if __name__ == "__main__":
    main()
