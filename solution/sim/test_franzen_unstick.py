"""Offline check for unstick (franzen_overlays/unstick.py).

On a copy of Franzen's patched bundle with l1a1y8sv2r3kp12lf15unst's overlays applied in the order its cell 7 runs them:
1. every overlay applies, both files compile, a second unstick refuses; the system prompt of a real ToolAgent carries
   the step-verification rules with the new "suspect the mechanics first" and replay lines;
2. the real opener builder (`_build_user_prompt`) shows nothing on the first turns of a level, the stall check at the
   3rd, 5th and 7th turn in a row without an action and not at the 4th or 6th, and nothing once an action executes;
3. with the stall check, a click game lists the object types no click on this level touched: not the clicked type
   (even where its color changed after the click), not objects within one cell of a click, not the background, not a
   thin border strip, not a type clicked only on an earlier level's board; at most six types;
4. a game over or a model RESET on the level shows the untested list once, without the stall check; a keyboard game
   lists the valid actions not yet used on this level (not RESET);
5. a level change resets the count; broken input returns no lines instead of raising.

    python3 solution/sim/test_franzen_unstick.py
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

ARM = "l1a1y8sv2r3kp12lf15unst"
STALL = "Stall check:"
UNTESTED = "Untested on this level:"


def order() -> list[str]:
    nb = json.loads((B.BASE / "submission.ipynb").read_text())
    B.apply_edits(nb, B.REPLICA + B.ARMS[ARM][1]())
    return re.findall(r"^# arm overlay: (\w+)$", B.cell_text(nb["cells"][B.ENV]), re.M)


def board(level: int, *, red_color: int = 8):
    """64x64, white background (0), a thin blue timer strip on the last row, three red 4x4 squares, four light-gray
    1x3 vertical bars, two yellow 2x2 blocks, one orange 3x5 block, one magenta 1x1, one pink 2x1, one gray 3x3."""
    g = [[0] * 64 for _ in range(64)]
    for c in range(64):
        g[63][c] = 9
    for i, (r, c) in enumerate(((10, 10), (10, 20), (10, 30))):
        color = red_color if i == 0 else 8
        for y in range(r, r + 4):
            for x in range(c, c + 4):
                g[y][x] = color
    for c in (10, 14, 18, 22):
        for y in range(30, 33):
            g[y][c] = 1
    for r, c in ((40, 10), (40, 20)):
        for y in range(r, r + 2):
            for x in range(c, c + 2):
                g[y][x] = 11
    for y in range(50, 53):
        for x in range(30, 35):
            g[y][x] = 12
    g[45][45] = 6
    g[45][50] = g[46][50] = 7
    for y in range(20, 23):
        for x in range(50, 53):
            g[y][x] = 2
    return g


def main() -> None:
    names = order()
    assert names[0] == "unstick" and {"imports", "levelfirst", "wastefix", "agentfix"} <= set(names), names
    with tempfile.TemporaryDirectory(prefix="unstick-") as tmp:
        bundle = Path(tmp) / "bundle"
        shutil.copytree(TREE, bundle, ignore=shutil.ignore_patterns("__pycache__"))
        for name in names:
            apply(name, bundle)
        try:
            apply("unstick", bundle)
        except RuntimeError as exc:
            assert "already applied" in str(exc), exc
        else:
            raise AssertionError("a second unstick did not refuse")

        os.environ.update(FORK_ENV)
        os.environ.pop("ARC3_STEP_VERIFICATION_HINT", None)
        ta = import_tool_agent(bundle)
        from inference.agent.runtime_state import Frame, HistoryEntry
        agent = ta.ToolAgent(model="stub")
        system = agent._system_prompt
        assert "During a multi-action plan, verify important effects" in system
        assert "Suspect the mechanics first: an action or object type you have not tried" in system
        assert "change the goal last" in system and "Reconsider your mechanics, goal" not in system
        assert "replay this level's recorded `transitions` through it" in system
        print(f"ok 1: {len(names)} overlays in cell 7 order (unstick first); compiles; a second unstick refuses; "
              "system prompt carries step verification with both new lines")

        def frame(level, step, grid=None):
            return Frame(grid=tuple(tuple(r) for r in (grid or board(level))), step=step, level=level)

        def opener(action_num, current, history, valid, summary=None):
            return agent._build_user_prompt(action_num, valid_actions=valid, current_frame=current,
                                            history_entries=history, previous_step_summary=summary)

        # 2: stall cadence. Level 2 board; one click on the first red square (it turns orange-red after the click)
        start = frame(2, 5)
        clicked = frame(2, 6, board(2, red_color=13))
        level1 = frame(1, 4)
        history = [HistoryEntry(action="", frame=frame(1, 0)),
                   HistoryEntry(action="MOUSE(row=50, col=31)", frame=level1),     # level 1: orange block clicked
                   HistoryEntry(action="UP", frame=start),                         # the level-1 winning move
                   HistoryEntry(action="MOUSE(row=11, col=11)", frame=clicked),    # level 2: first red square
                   HistoryEntry(action="MOUSE(row=46, col=46)", frame=frame(2, 7, board(2, red_color=13)))]
        current = history[-1].frame
        valid = ["RESET", "MOUSE", "UP", "SPACE"]
        seen = []
        for turn in range(9):
            text = opener(7, current, history, valid)
            seen.append((STALL in text, UNTESTED in text))
        assert seen == [(False, False), (False, False), (False, False), (True, True), (False, False),
                        (True, True), (False, False), (True, True), (False, False)], seen
        text = opener(8, current, history, valid)
        assert STALL not in text and UNTESTED not in text, text
        print("ok 2: no lines on the first turns; stall check at 3, 5 and 7 idle turns, not at 4 or 6; "
              "cleared by an executed action")

        # 3: what the untested list holds
        for _ in range(3):
            text = opener(8, current, history, valid)
        assert STALL in text, text
        line = next(x for x in text.splitlines() if x.startswith(UNTESTED))
        types = re.findall(r"([A-Za-z]) (\d+)x(\d+)(?: x(\d+))?", line.split(":", 1)[1])
        got = {(c, int(h), int(w)) for c, h, w, _ in types}
        assert ("w", 3, 1) in got and ("Y", 2, 2) in got and ("g", 3, 3) in got and ("O", 3, 5) in got, line
        assert ("R", 4, 4) not in got, line              # clicked on this level (the clicked one changed color)
        assert ("M", 1, 1) not in got, line              # within one cell of the (46, 46) click
        assert all(c != "W" for c, _, _ in got), line    # background
        assert all(c != "b" for c, _, _ in got), line    # thin strip on the border
        assert "w 3x1 x4 (e.g. row 31, col " in line, line
        assert len(types) <= 6, line
        assert "Valid actions not yet used on this level: UP, SPACE." in text, text   # UP ran on level 1's board
        print("ok 3: untested list: unclicked types only (light-gray bars x4 first); the clicked red type, the "
              "object next to a click, background and the border strip are left out; level-1 actions do not "
              "count; unused actions listed")

        # 4: a failed attempt shows the list once, without the stall check; keyboard game
        text = opener(9, current, history, valid, {"game_over": True, "executed_actions": ["SPACE"]})
        assert STALL not in text and UNTESTED in text, text
        text = opener(9, current, history, valid, {"game_over": True, "executed_actions": ["SPACE"]})
        assert UNTESTED not in text, text
        text = opener(10, current, history, valid, {"executed_actions": ["RESET", "UP"]})
        assert UNTESTED in text, text
        kb = [HistoryEntry(action="", frame=frame(3, 0)), HistoryEntry(action="UP", frame=frame(3, 1)),
              HistoryEntry(action="UP", frame=frame(3, 2))]
        kvalid = ["RESET", "UP", "DOWN", "SPACE"]
        opener(2, kb[-1].frame, kb, kvalid)
        for _ in range(3):
            text = opener(2, kb[-1].frame, kb, kvalid)
        assert STALL in text and UNTESTED not in text, text
        assert "Valid actions not yet used on this level: DOWN, SPACE." in text, text
        print("ok 4: a game over or a model RESET shows the list once without the stall check; a keyboard game lists "
              "its unused actions, RESET excluded")

        # 5: level change and broken input
        opener(2, frame(4, 3), kb, kvalid)
        assert agent._unstick_idle == 0 and agent._unstick_level == 4
        odd = [HistoryEntry(action="MOUSE(row=99, col=99)", frame=frame(4, 4)),
               HistoryEntry(action="MOUSE(row=x)", frame=Frame(grid=(), step=5, level=4))]
        real = ta._unstick_coverage

        def boom(*_args):
            raise ValueError("boom")
        ta._unstick_coverage = boom
        for _ in range(3):
            text = opener(2, frame(4, 4), odd, ["MOUSE"])
        assert STALL in text and UNTESTED not in text, text
        ta._unstick_coverage = real
        assert isinstance(real(Frame(grid=(), step=5, level=4), odd, ["MOUSE"]), list)
        assert ta._unstick_lines(agent, 1, "not a frame", None, None, None) == []
        print("ok 5: a level change resets the count; a failing untested list keeps the stall check; an off-board click and an "
              "empty board raise nothing; broken input returns no lines")
    print("PASS test_franzen_unstick")


if __name__ == "__main__":
    main()
