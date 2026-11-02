"""Offline check for the B2 strategy-audit overlay (franzen_overlays/audit.py).

    python3 solution/sim/test_franzen_audit.py
    uv --directory ARC-AGI-3-Agents run python ../solution/sim/test_franzen_audit.py

Applies the overlay to a temp copy of Franzen's patched bundle, drives the real
ToolAgent's turn-opener builder with synthetic per-level counters, and applies
every overlay in franzen_overlays/ in every order. No GPU, no network.
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
REL = Path("src/ARC3-Inference/inference/agent/tool_agent.py")
MARKER = "Strategy check: this level has gone a long time without a level-up."


def copy_tree() -> Path:
    dst = Path(tempfile.mkdtemp(prefix="franzen-audit-")) / "bundle"
    shutil.copytree(TREE, dst)
    return dst


def apply(name: str, bundle: Path) -> None:
    path = OVERLAYS / f"{name}.py"
    exec(compile(path.read_text(), str(path), "exec"), {"BUNDLE_DIR": bundle})


def changed_files(bundle: Path) -> list[Path]:
    # a file another overlay created has no pristine counterpart: changed too
    return [p for p in bundle.rglob("*.py")
            if not (TREE / p.relative_to(bundle)).exists()
            or p.read_bytes() != (TREE / p.relative_to(bundle)).read_bytes()]


def import_tool_agent(bundle: Path):
    """Import the patched module, stubbing requests/PIL if this python lacks them."""
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


def check_trigger(bundle: Path) -> None:
    for key in ("ARC3_AUDIT_TOKENS", "ARC3_AUDIT_MIN_ACTIONS", "ARC3_PERSISTENT_FUNCTIONS",
                "ARC3_PRIORITY_PACE"):
        os.environ.pop(key, None)
    os.environ["LOCAL_ANALYZER_MODEL_ID"] = "stub"
    tool_agent = import_tool_agent(bundle)
    assert Path(tool_agent.__file__).resolve() == (bundle / REL).resolve(), tool_agent.__file__
    agent = tool_agent.ToolAgent(model="stub")

    def opener() -> str:
        # current_frame=None keeps the message plain text; the audit path is the same
        return agent._build_user_message("TURN", None)["content"]

    def step(level: int, end_action: int, tokens: int = 0, level_up: bool = False) -> int:
        agent._accumulate_usage_tokens({"completion_tokens": tokens})
        agent._last_step_summary = {"level": level, "end_action_num": end_action,
                                    "level_transition": level_up}
        # the harness's own level-up bookkeeping, which restarts both counts
        agent._update_summarized_knowledge_from_step_summary()
        return opener().count(MARKER)

    # level 1: neither threshold alone is enough
    assert opener().count(MARKER) == 0  # first turn, nothing executed
    assert step(1, 10, 40_000) == 0  # 40K tokens but only 10 actions
    agent._session_generated_tokens = 0
    assert step(1, 25, 29_999) == 0  # 25 actions but 1 token short
    assert step(1, 26, 1) == 1  # both crossed: fires
    assert step(1, 80, 60_000) == 0  # never again on the same level
    assert step(1, 90, 60_000) == 0
    assert agent._audited_levels == {1}

    # level-up resets the per-level counts, so level 2 starts from zero
    assert step(2, 91, 5_000, level_up=True) == 0
    assert agent._actions_at_level_start == 91
    assert step(2, 110, 30_000) == 0  # 19 actions on this level
    assert step(2, 111) == 1  # 20 actions, 30K tokens since the level-up
    assert step(2, 200, 99_000) == 0
    assert agent._audited_levels == {1, 2}

    # thresholds come from the env at call time
    os.environ["ARC3_AUDIT_TOKENS"] = "1000"
    os.environ["ARC3_AUDIT_MIN_ACTIONS"] = "3"
    assert step(3, 201, 0, level_up=True) == 0
    assert step(3, 203, 999) == 0
    assert step(3, 204, 1) == 1
    del os.environ["ARC3_AUDIT_TOKENS"], os.environ["ARC3_AUDIT_MIN_ACTIONS"]

    # the question is the last text of the opener, after the turn prompt
    agent._session_generated_tokens += 50_000
    agent._last_step_summary = {"level": 4, "end_action_num": 300}
    text = opener()
    assert text.startswith("TURN") and text.endswith(tool_agent._STRATEGY_AUDIT_PROMPT), text
    assert "\u2014" not in text and "\u2013" not in text

    # a bad summary is logged and skipped, never raised into the game
    agent._last_step_summary = {"level": "not-a-number", "end_action_num": 999}
    assert opener() == "TURN"

    # an agent unpickled from before the overlay has no _audited_levels
    del agent._audited_levels
    agent._last_step_summary = {"level": 5, "end_action_num": 999}
    assert opener().count(MARKER) == 1


def main() -> None:
    assert TREE.is_dir(), f"pristine Franzen tree missing: {TREE} (set FRANZEN_TREE)"
    overlay_text = (OVERLAYS / "audit.py").read_text()
    assert "\u2014" not in overlay_text and "\u2013" not in overlay_text, "dash in overlay"

    bundle = copy_tree()
    apply("audit", bundle)
    touched = changed_files(bundle)
    assert touched == [bundle / REL], touched
    for path in touched:
        py_compile.compile(str(path), doraise=True)
    try:
        apply("audit", bundle)
        raise AssertionError("second application should refuse")
    except RuntimeError as exc:
        assert "already applied" in str(exc), exc
    check_trigger(bundle)

    names = sorted(p.stem for p in OVERLAYS.glob("*.py") if not p.stem.endswith("_serve"))  # serving patches never touch the bundle
    for order in overlay_orders.orders(names, "audit"):
        combo = copy_tree()
        for name in order:
            apply(name, combo)
        for path in changed_files(combo):
            py_compile.compile(str(path), doraise=True)
        assert (combo / REL).read_text().count("user_prompt += self._strategy_audit_context()") == 1
        print("composed:", " -> ".join(order))
    print("PASS")


if __name__ == "__main__":
    main()
