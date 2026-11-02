"""Offline check for the B4 Wang/Ludvig port overlay (franzen_overlays/sandbox.py).

    python3 solution/sim/test_franzen_sandbox.py

Applies the overlay to a temp copy of Franzen's patched bundle and checks, on the
real code: the system prompt lost the two bar lines and kept Franzen's budget-bar
wording; the turn opener lost every static line, kept every dynamic one, and kept
the act/batch cue ("When ready, call action(actions) ... prefer batching"). Then
applies every overlay in franzen_overlays/ in every order. No GPU, no network.
"""
from __future__ import annotations

import itertools
import overlay_orders
import os
import py_compile
import shutil
import sys
import tempfile
from pathlib import Path
from unittest import mock

OVERLAYS = Path(__file__).resolve().parents[1] / "kaggle" / "franzen_overlays"
TREE = Path(os.environ.get(
    "FRANZEN_TREE",
    str(__import__("franzen_tree").pristine_tree()),  # vendored bundle plus the notebook patch
))
INF = Path("src/ARC3-Inference/inference")
TOUCHED = {INF / "agent/prompts.py", INF / "agent/tool_agent.py"}
BATCH_CUE = ("When ready, call `action(actions)` from inside the `python` tool with the best valid action or "
             "ordered batch selected by your code. If your code has found a reliable short sequence, prefer "
             "batching it in one call.")

# cell 7 of the fork, the settings the opener and system prompt are built under
FORK_ENV = {
    "LOCAL_ANALYZER_MODEL_ID": "stub", "ARC3_REASONING_HISTORY_KEY": "reasoning_content",
    "LOCAL_ANALYZER_ENABLE_THINKING": "true", "LOCAL_ANALYZER_TOOL_STEPS": "0",
    "LOCAL_ANALYZER_TOOL_TIMEOUT": "30", "LOCAL_ANALYZER_YIELD_SECONDS": "0",
    "LOCAL_ANALYZER_YIELD_TOKENS": "2048", "LOCAL_ANALYZER_CONTEXT_WINDOW": str(128 * 1024),
    "LOCAL_ANALYZER_MAX_OUTPUT": str(12 * 1024), "ARC3_CONTEXT_DRAIN_TOKENS": str(58 * 1024),
    "ARC3_HISTORY_ASSISTANT_TURNS": "150", "ARC3_HISTORY_TURN_DRAIN": "30",
    "ARC3_HISTORY_DRAIN_COALESCE": "1", "LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS": "3072",
    "ARC3_MIDDLE_TRUNCATION": "1", "ARC3_ACTION_ECHO": "1", "ARC3_EXPLAIN_GAMEPLAY_CHANGED": "1",
    "ARC3_NEW_CHANGED_PROMPTS": "1", "ARC3_LEVEL_TRANSFER_GUIDANCE": "1", "ARC3_ACTION_INFO": "1",
    "ARC3_DEDUPE_MULTICALL_LINE": "1", "ARC3_FRAME_DIFF_HINT": "1",
    "MULTIMODAL_CONTEXT": "current_grid", "MULTIMODAL_UPSCALE": "10", "ARC3_DIFF_IMAGE": "1",
    "ARC3_GAMEOVER_DIFF_IMAGE": "1", "ARC3_ANIMATION": "1", "ARC3_GUARDS_FROM_LEVEL": "2",
    "ARC3_BATCH_NOOP_BLOCK": "1", "ARC3_STALE_STATE_BLOCK": "1", "ARC3_MEMORY_SECTIONS": "off",
    "ARC3_PERSISTENT_FUNCTIONS": "1", "ARC3_PERSISTENT_FUNCTIONS_SCOPE": "game",
    "ARC3_PERSISTENT_FUNCTIONS_IMPORTS": "1", "EXPOSE_UNDO": "on", "ARC3_MAX_ACTIVE_STREAMS": "10",
}

def apply(name: str, bundle: Path) -> None:
    path = OVERLAYS / f"{name}.py"
    exec(compile(path.read_text(), str(path), "exec"), {"BUNDLE_DIR": bundle})


def import_tool_agent(bundle: Path):
    """Import the patched package, stubbing requests/PIL if this python lacks them."""
    sys.path.insert(0, str(bundle / "src" / "ARC3-Inference"))
    stubbed: list[str] = []
    while True:
        try:
            import inference.agent.tool_agent as tool_agent
            return tool_agent
        except ImportError as exc:
            name = (exc.name or "").split(".")[0]
            assert name and name not in stubbed and not name.startswith("inference"), exc
            stubbed.append(name)
            sys.modules[name] = mock.MagicMock()
            for key in [k for k in sys.modules if k.startswith("inference")]:
                del sys.modules[key]


def check_agent(bundle: Path) -> None:
    os.environ.update(FORK_ENV)
    tool_agent = import_tool_agent(bundle)
    assert Path(tool_agent.__file__).resolve() == (bundle / INF / "agent/tool_agent.py").resolve()

    agent = tool_agent.ToolAgent(model="stub")
    system = agent._system_prompt
    assert "do not get distracted" not in system and "countdown bars" not in system
    # Franzen's own budget-bar wording stays
    assert "count remaining budget into your plans" in system and "DON'T DO THIS" in system

    static = sorted(tool_agent._WANG_STATIC_OPENER_LINES)
    assert len(static) == 10 and tool_agent.TOOL_CALL_FORMAT_GUIDANCE in static and BATCH_CUE not in static
    valid = ["ACTION1", "ACTION2", "ACTION6", "ACTION7"]

    def opener(num, summary):
        text = agent._build_user_prompt(num, valid_actions=valid, current_frame=None,
                                        history_entries=[], previous_step_summary=summary)
        for line in static:
            assert line not in text, (line, text)
        assert "Current state: step" in text and "Valid actions right now: UP, DOWN, MOUSE, UNDO." in text
        # the opener's only act/batch cue stays, exactly once
        assert text.count(BATCH_CUE) == 1, text
        return text

    first = opener(0, None)
    assert first.startswith("No previous sequence has been executed yet.") and len(first) < 450, first
    moved = opener(5, {"executed_count": 3, "executed_actions": ["UP", "UP", "LEFT"], "level": 1,
                       "gameplay_changed": True, "board_changed": True})
    for line in ("The code executed 3 actions in the previous sequence.", "Executed actions: UP, UP, LEFT.",
                 "You are still on the same level."):
        assert line in moved, moved
    assert BATCH_CUE in moved
    dead = opener(9, {"executed_count": 1, "executed_actions": ["DOWN"], "level": 1, "game_over": True,
                      "fatal_action": "DOWN", "board_changed": True, "gameplay_changed": True})
    assert "GAME OVER occurred during the previous sequence." in dead and "BAR RULE" in dead
    assert "Do NOT re-submit this same series of moves" in dead
    up = opener(12, {"executed_count": 2, "executed_actions": ["UP", "UP"], "level": 2,
                     "level_transition": True, "board_changed": True, "gameplay_changed": True})
    assert "You have completed the previous level." in up
    saved = sum(len(line) + 1 for line in static if not line.startswith("Ground yourself"))
    print(f"opener: {len(moved)} chars, about {saved} static chars removed per turn")


def main() -> None:
    assert TREE.is_dir(), f"pristine Franzen tree missing: {TREE} (set FRANZEN_TREE)"
    text = (OVERLAYS / "sandbox.py").read_text()
    assert "\u2014" not in text and "\u2013" not in text, "dash in overlay"

    with tempfile.TemporaryDirectory(prefix="franzen-sandbox-") as tmp:
        bundle = Path(tmp) / "bundle"
        shutil.copytree(TREE, bundle)
        apply("sandbox", bundle)
        touched = {p.relative_to(bundle) for p in bundle.rglob("*.py")
                   if p.read_bytes() != (TREE / p.relative_to(bundle)).read_bytes()}
        assert touched == TOUCHED, touched
        for rel in touched:
            py_compile.compile(str(bundle / rel), doraise=True)
        try:
            apply("sandbox", bundle)
            raise AssertionError("second application should refuse")
        except RuntimeError as exc:
            assert "already applied" in str(exc), exc
        check_agent(bundle)

        # serving patches never touch the bundle; keepvars2 supersedes keepvars (same marker, never stacked)
        names = sorted(p.stem for p in OVERLAYS.glob("*.py") if not p.stem.endswith("_serve") and p.stem != "keepvars")
        for order in overlay_orders.orders(names, "sandbox"):
            combo = Path(tmp) / ("combo-%08x" % __import__("zlib").crc32("-".join(order).encode()))
            shutil.copytree(TREE, combo)
            for name in order:
                apply(name, combo)
            for path in combo.rglob("*.py"):
                py_compile.compile(str(path), doraise=True)
            agent_text = (combo / INF / "agent/tool_agent.py").read_text()
            assert agent_text.count("_WANG_STATIC_OPENER_LINES = frozenset((") == 1
            assert agent_text.count("if line not in _WANG_STATIC_OPENER_LINES)") == 1
            shutil.rmtree(combo)
            print("composed:", " -> ".join(order))
    print("PASS")


if __name__ == "__main__":
    main()
