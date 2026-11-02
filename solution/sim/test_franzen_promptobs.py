"""Offline check for promptobs lite (franzen_overlays/promptobs.py).

    python3 solution/sim/test_franzen_promptobs.py

Applies the overlays in the order cell 7 runs them for the next stacked arm (promptobs, keepdata, toolfast, memcap,
agentfix, replycache, wastefix) to a copy of Franzen's patched bundle, then on the real code: the system prompt
types the animation fields and has none of the killed edits (row_runs, a stated image size, teal); U, D, L, R (any
case) map to ACTION1 to ACTION4 and nothing else changes in the action names; the real sandbox reads step, level,
ascii and action_display through history entries and transitions, keeps deepcopy working and names .frame in the
error for any other attribute. A second application refuses. No GPU, no network.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

from test_franzen_sandbox import FORK_ENV, TREE, apply, import_tool_agent

ORDER = ("promptobs", "keepdata", "toolfast", "memcap", "agentfix", "replycache", "wastefix")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="promptobs-") as tmp:
        bundle = Path(tmp) / "bundle"
        shutil.copytree(TREE, bundle, ignore=shutil.ignore_patterns("__pycache__"))
        for name in ORDER:
            apply(name, bundle)
        try:
            apply("promptobs", bundle)
        except RuntimeError as exc:
            assert "already applied" in str(exc), exc
        else:
            raise AssertionError("a second application did not refuse")
        os.environ.update(FORK_ENV)
        tool_agent = import_tool_agent(bundle)
        system = tool_agent.ToolAgent(model="stub")._system_prompt
        for needle in ("`frames` (an int: how many frames", "`transient_pixels` (an int count)",
                       "`changed` (an int: how many cells changed)", "`bbox` (a text string"):
            assert needle in system, needle
        for absent in ("row_runs", "640x640", "teal"):
            assert absent not in system, absent
        print("ok   the animation fields are typed in the prompt; no killed edit is present")

        names = sys.modules["inference.agent.action_names"]
        assert [names.to_engine_action(x) for x in ("U", "D", "L", "R", "r")] == [
            "ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION4"]
        assert names.to_engine_action("RESET") == "RESET" and names.to_engine_action("Q") is None
        assert names.to_model_actions(["ACTION1", "ACTION4", "RESET"]) == ["UP", "RIGHT", "RESET"]
        print("ok   U/D/L/R map to ACTION1-4; other names unchanged")

        sandbox = sys.modules["inference.agent.python_tool_sandbox"]
        frame = {"ascii": "WWbb\nWWWW", "step": 3, "level": 2, "shape": [2, 4], "grid": [[0, 0, 9, 9], [0, 0, 0, 0]]}
        state = {"current_frame": frame, "history": [{"action": "", "frame": frame, "result": {}},
                                                     {"action": "UP", "frame": frame, "result": {}}],
                 "valid_actions": ["UP"], "last_action_call_result": {}}

        def run(code: str) -> dict:
            return sandbox.run_sandboxed_python(code=code, timeout_seconds=10, initial_state=state,
                                                action_handler=lambda a, **k: {"action_result": {}, "state": state},
                                                animation_handler=None, kept_functions=[])

        out = run("h = history[-1]; t = transitions[-1]\n"
                  "print(h.step, h.level, t.level, t.action_display, h.ascii == current_frame.ascii)")
        assert not out.get("error") and out["stdout"] == "3 2 2 UP True\n", out
        out = run("import copy\nc = copy.deepcopy(history)\nprint(c[-1].step, hasattr(history[-1], 'grid'))")
        assert not out.get("error") and out["stdout"] == "3 False\n", out
        out = run("history[-1].grid")
        assert "has no attribute 'grid'; its board is .frame" in out["error"], out
        print("ok   history entries and transitions forward frame fields; deepcopy works; other names say .frame")


if __name__ == "__main__":
    main()
