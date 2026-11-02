"""Offline check for verify (franzen_overlays/verify.py) through the real sandbox child, on both bases.

For l1a1y8sv2r3kp12lf15ustv (with tool2) and l1a1y8sv2r3kp12lf15usv (without), on a copy of Franzen's patched bundle with
the arm's overlays applied in cell 7's order, `run_sandboxed_python` runs real child processes on a synthetic history
(a one-cell mover; level 1, a level change, then level 2 with a model RESET, a death and its automatic RESET):
1. every overlay applies, a second verify refuses, the system prompt documents both helpers and the opener names them;
2. a correct observe/step pair matches every level-2 transition, with RESET, death and level-change skips counted and no
   level-1 transition checked;
3. a wrong step names the newest mismatching history index;
4. dict mode (cells only) matches; None abstains; a raising step is reported and the snippet goes on;
5. a step that calls action() is refused inside the check and no real action runs;
6. check_goal returns exactly the boards where the test is true; a raising goal test is reported;
7. a (board, action) pair seen twice with different results is reported as hidden state;
8. a slow step stops at the time limit with a note (and, with tool2, before the snippet's own SnippetTimeout);
9. an ordinary snippet is unchanged.

    python3 solution/sim/test_franzen_verify.py
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

W, H = 6, 4


def board(r, c, level):
    rows = [["." for _ in range(W)] for _ in range(H)]
    rows[r][c] = "P"
    asc = "\n".join("".join(x) for x in rows)
    grid = [[1 if ch == "P" else 0 for ch in line] for line in asc.splitlines()]
    return {"ascii": asc, "step": 0, "level": level, "shape": [H, W], "grid": grid}


def history():
    h = []

    def add(action, r, c, level, **result):
        h.append({"action": action, "frame": board(r, c, level), "result": result})

    add("", 0, 0, 1)                                   # seeded frame, level 1
    add("RIGHT", 0, 1, 1)                              # 1
    add("RIGHT", 0, 0, 2, level_completed=True)        # 2: level change, next level's first board
    add("RIGHT", 0, 1, 2)                              # 3
    add("DOWN", 1, 1, 2)                               # 4
    add("RESET", 0, 0, 2)                              # 5: model RESET
    add("RIGHT", 0, 1, 2)                              # 6
    add("RIGHT", 0, 2, 2, game_over=True)              # 7: death
    add("RESET", 0, 0, 2, automatic=True)              # 8: automatic RESET
    add("RIGHT", 0, 1, 2)                              # 9
    add("RIGHT", 0, 2, 2)                              # 10
    return h


def state(h):
    return {"current_frame": h[-1]["frame"], "history": h, "valid_actions": ["RIGHT", "DOWN", "RESET"],
            "last_action_call_result": {}}


PRELUDE = ("MOVES = {'RIGHT': (0, 1), 'DOWN': (1, 0), 'LEFT': (0, -1), 'UP': (-1, 0)}\n"
           "def parse(f):\n    rows = f.ascii.splitlines()\n"
           "    return next((r, c) for r, row in enumerate(rows) for c, ch in enumerate(row) if ch == 'P')\n"
           "def sim(s, a):\n    dr, dc = MOVES[a]\n    return (s[0] + dr, s[1] + dc)\n")


def run(sb, code, h, timeout=30):
    calls = []

    def action_handler(actions, stale_after=None):  # noqa: ARG001
        calls.append(actions)
        return {"action_result": {"executed": True}, "state": {}}

    out = sb.run_sandboxed_python(code=PRELUDE + code, timeout_seconds=timeout, initial_state=state(h),
                                  action_handler=action_handler)
    return out, calls


def check_arm(arm: str, with_tool2: bool) -> None:
    names = order(arm)
    assert names[-1] == "verify" and ("tool2" in names) == with_tool2, names
    with tempfile.TemporaryDirectory(prefix="verify-") as tmp:
        bundle = Path(tmp) / "bundle"
        shutil.copytree(TREE, bundle, ignore=shutil.ignore_patterns("__pycache__"))
        for name in names:
            apply(name, bundle)
        try:
            apply("verify", bundle)
        except RuntimeError as exc:
            assert "already applied" in str(exc), exc
        else:
            raise AssertionError("a second verify did not refuse")
        os.environ.update(FORK_ENV)
        for key in [k for k in sys.modules if k.startswith("inference")]:
            del sys.modules[key]
        sys.path.insert(0, str(bundle / "src" / "ARC3-Inference"))
        try:
            sb = importlib.import_module("inference.agent.python_tool_sandbox")
            prompts_text = (bundle / "src/ARC3-Inference/inference/agent/prompts.py").read_text()
            agent_text = (bundle / "src/ARC3-Inference/inference/agent/tool_agent.py").read_text()
        finally:
            sys.path.pop(0)
        assert prompts_text.count("`check_model(step, observe=None)` replays") == 1
        assert prompts_text.count("`check_goal(is_goal)` runs your goal test") == 1
        assert "run `check_model` on it" in prompts_text
        assert "`check_model(step, observe)`, `check_goal(is_goal)`, and `action(actions)`" in agent_text
        assert "run check_model on it" in agent_text
        print(f"ok 1 {arm}: {len(names)} overlays, verify last; a second verify refuses; prompt, opener and stall text name both")

        h = history()
        out, calls = run(sb, "r = check_model(sim, observe=parse)\nprint('R', r['checked'], r['matched'], r['skipped'])\n", h)
        assert out["error"] == "", out
        assert "R 5 5 {'RESET': 2, 'death': 1, 'level change': 0}" in out["stdout"], out["stdout"]
        assert "5 of 5 checked transitions" in out["stdout"] and not calls
        print(f"ok 2 {arm}: a correct rule matches all 5 level-2 transitions; 2 RESETs and 1 death skipped; level 1 not checked")

        out, _ = run(sb, "def bad(s, a):\n    dr, dc = MOVES[a]\n    return (s[0] + 2 * dr, s[1] + 2 * dc)\n"
                         "r = check_model(bad, observe=parse)\nprint('M', r['mismatches'][:2])\n", h)
        assert "M [10, 9]" in out["stdout"] and "history[10] 'RIGHT': predicted (0, 3) got (0, 2)" in out["stdout"], out["stdout"]
        print(f"ok 3 {arm}: a wrong rule names history[10] first (newest first) with predicted and actual state")

        code = ("def cells(f, a):\n    r, c = sim(parse(f), a)\n    return {(r, c): 'P'}\n"
                "r1 = check_model(cells)\n"
                "def picky(f, a):\n    return None if a == 'DOWN' else cells(f, a)\n"
                "r2 = check_model(picky)\n"
                "def boom(f, a):\n    raise ValueError('nope')\n"
                "r3 = check_model(boom)\nprint('D', r1['matched'], r2['abstained'], len(r3['mismatches']))\n")
        out, _ = run(sb, code, h)
        assert out["error"] == "" and "D 5 1 5" in out["stdout"] and "ValueError: nope" in out["stdout"], out
        print(f"ok 4 {arm}: dict mode matches, None abstains, a raising rule is reported and the snippet continues")

        code = ("def actor(s, a):\n    action([a])\n    return sim(s, a)\n"
                "r = check_model(actor, observe=parse)\nprint('A', r['matched'], len(r['mismatches']))\n"
                "print('after', callable(action))\n")
        out, calls = run(sb, code, h)
        assert "A 0 5" in out["stdout"] and "disabled inside check_model" in out["stdout"] and not calls, (out, calls)
        assert "after True" in out["stdout"]
        print(f"ok 5 {arm}: action() inside a rule is refused during the check, no real action runs, action() is restored")

        code = ("hits = check_goal(lambda f: parse(f)[1] >= 2)\nprint('G', hits)\n"
                "hits2 = check_goal(lambda f: 1 / 0)\n")
        out, _ = run(sb, code, h)
        assert "G [7, 10]" in out["stdout"] and "true on 2 of" in out["stdout"], out["stdout"]
        assert "ZeroDivisionError" in out["stdout"], out["stdout"]
        print(f"ok 6 {arm}: check_goal flags exactly the recorded boards where the goal test holds; a raising test is reported")

        h2 = history()
        h2.append({"action": "LEFT", "frame": board(0, 1, 2), "result": {}})     # 11: (0,2) LEFT -> (0,1)
        h2.append({"action": "RIGHT", "frame": board(0, 2, 2), "result": {}})    # 12
        h2.append({"action": "LEFT", "frame": board(0, 2, 2), "result": {}})     # 13: same board and action, no move
        out, _ = run(sb, "r = check_model(sim, observe=parse)\nprint('C', r['contradictions'])\n", h2)
        assert "C 1" in out["stdout"] and "hidden counter" in out["stdout"], out["stdout"]
        print(f"ok 7 {arm}: a (board, action) pair with two results is reported as hidden state")

        long_h = history()
        for k in range(40):
            long_h.append({"action": "RIGHT" if k % 2 == 0 else "LEFT", "frame": board(0, 3 if k % 2 == 0 else 2, 2),
                           "result": {}})
        code = ("import time as T\ndef slow(s, a):\n    t = T.monotonic()\n    while T.monotonic() - t < 0.3:\n        pass\n"
                "    return sim(s, a)\nr = check_model(slow, observe=parse)\nprint('S', r['checked'])\n")
        t0 = time.monotonic()
        out, _ = run(sb, code, long_h, timeout=8 if with_tool2 else 30)
        took = time.monotonic() - t0
        assert "time limit" in out["stdout"] and out["error"] == "", out
        assert took < (6 if with_tool2 else 12), took
        print(f"ok 8 {arm}: a slow rule stops at the time limit ({took:.1f} s) with a note and no snippet timeout")

        out, _ = run(sb, "x = 3\nprint(x)\nresult = x * 2\n", h)
        assert out["stdout"].strip() == "3" and out["result"] == 6 and out["error"] == ""
        print(f"ok 9 {arm}: an ordinary snippet is unchanged")


def main() -> None:
    check_arm("l1a1y8sv2r3kp12lf15ustv", with_tool2=True)
    check_arm("l1a1y8sv2r3kp12lf15usv", with_tool2=False)
    print("PASS test_franzen_verify")


if __name__ == "__main__":
    main()
