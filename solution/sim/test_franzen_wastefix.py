"""Offline check for the transcript-waste overlay (franzen_overlays/wastefix.py).

    python3 solution/sim/test_franzen_wastefix.py

Applies the overlay to a temp copy of Franzen's patched bundle: exactly the four files
change, each compiles, and a second application refuses. Then, on the real code with no
GPU and no network (check_pristine first shows the functional checks failing unpatched):
  - the real sandbox runs json, Counter, deque, defaultdict, math and re without
    imports, every pre-bound module is on its allowlist, blocked modules stay blocked;
  - a snippet whose host handlers sleep 4 s against a 2 s deadline finishes, while the
    snippet's own time still times out;
  - a timed-out snippet keeps the retained functions, a finished one still drops what
    it no longer defines;
  - last_animation_timeline answers items() and values() with the documented keys on a
    real, a non-animating and a disabled timeline;
  - bbox on a non-rectangular node, host and sandbox, in frame_diff's bbox convention;
  - the fork's assembled system prompt and tool schema carry the corrected text.
Last, the overlay composes with every other overlay applied before or after it (the
other overlay tests apply every order).
"""
from __future__ import annotations

import ast
import json
import os
import py_compile
import shutil
import sys
import tempfile
import time
from pathlib import Path

from test_franzen_sandbox import FORK_ENV, INF, OVERLAYS, TREE, apply, import_tool_agent

TOUCHED = {INF / "agent/python_tool_sandbox.py", INF / "agent/tool_agent.py",
           INF / "agent/prompts.py", INF / "utils/segmentation.py"}
STOP_RULE = ("- If an action result reports `game_over`, `run_complete`, `level_completed`, or `done`, "
             "stop acting immediately and re-ground on the next turn.\n")
GRID = [[0] * 5 for _ in range(5)]  # an L of color 3 on background 0
for _r, _c in ((1, 1), (2, 1), (3, 1), (3, 2), (3, 3)):
    GRID[_r][_c] = 3
STATE = {"current_frame": {"ascii": "", "step": 0, "level": 1, "shape": [5, 5], "grid": GRID},
         "history": [], "valid_actions": ["UP"], "last_action_call_result": {}}
# 5 actions and 5 timeline fetches: 4 s of host time against a 2 s deadline
SLOW = "for _ in range(5):\n    action(['UP'])\n    len(last_animation_timeline)\nprint('finished')"
PREBOUND = ("c = Counter('abca')\n"
            "print(json.dumps({'a': c['a']}), deque([7]).pop(), defaultdict(int)['k'], "
            "math.isqrt(10), re.sub('b', 'x', 'abc'))")
TIMELINE = ("tl = last_animation_timeline\n"
            "print(json.dumps([dict(tl.items()), list(tl.values()), tl['steps'], tl['frames'], tl['action']]))")
NODE_BBOXES = "print([n['bbox'] for n in current_frame.segmentation['nodes']])"
KEPT = "def f(x):\n    return x + 1"


def changed(bundle: Path) -> list[Path]:
    """Files an overlay edited or created, relative to the bundle."""
    return [p.relative_to(bundle) for p in bundle.rglob("*.py")
            if not (TREE / p.relative_to(bundle)).exists()
            or p.read_bytes() != (TREE / p.relative_to(bundle)).read_bytes()]


def load(bundle: Path):
    """Import this bundle's harness, dropping any other copy already imported."""
    for key in [k for k in sys.modules if k == "inference" or k.startswith("inference.")]:
        del sys.modules[key]
    tool_agent = import_tool_agent(bundle)
    assert Path(tool_agent.__file__).resolve().is_relative_to(bundle.resolve()), tool_agent.__file__
    return tool_agent, sys.modules["inference.agent.python_tool_sandbox"]


def run(sandbox, code: str, *, timeout: int = 10, host_sleep: float = 0.0, timeline=None, kept=None) -> dict:
    """The real sandbox, with host handlers that take host_sleep seconds per request."""
    def on_action(actions, *, stale_after=None):
        time.sleep(host_sleep)
        return {"action_result": {"executed": True, "gameplay_changed": True,
                                  "executed_actions": [a["action"] for a in actions]}, "state": STATE}

    def on_animation(request):
        time.sleep(host_sleep)
        return [] if request.get("kind") == "frames" else timeline

    return sandbox.run_sandboxed_python(code=code, timeout_seconds=timeout, initial_state=STATE,
                                        action_handler=on_action, animation_handler=on_animation,
                                        kept_functions=kept)


def check_pristine(bundle: Path) -> None:
    """Every functional check below fails here first."""
    tool_agent, sandbox = load(bundle)
    slow = run(sandbox, SLOW, timeout=2, host_sleep=0.4)
    assert slow["error"] == "Tool timed out after 2s" and "keepable_functions" not in slow, slow
    assert "NameError: name 'Counter' is not defined" in run(sandbox, PREBOUND)["error"]
    assert "has no attribute 'items'" in run(sandbox, "last_animation_timeline.items()")["error"]
    assert "KeyError: 'bbox'" in run(sandbox, NODE_BBOXES)["error"]
    agent = tool_agent.ToolAgent(model="stub")
    agent._kept_functions = {"f": KEPT}
    agent._record_retained_functions(slow, {})
    assert agent._kept_functions == {}, "the pristine timeout should wipe the retained functions"
    assert STOP_RULE not in agent._system_prompt


def check_sandbox(sandbox) -> None:
    out = run(sandbox, PREBOUND)
    assert out["error"] == "" and out["stdout"] == '{"a": 2} 7 0 3 axc\n', out
    child = ast.parse(sandbox._SANDBOX_BOOTSTRAP)
    allowed = ast.literal_eval(next(node.value for node in child.body if isinstance(node, ast.Assign)
                                    and getattr(node.targets[0], "id", "") == "SAFE_MODULES"))
    names = set(ast.literal_eval(run(sandbox, "print(sorted(dir()))")["stdout"]))
    assert allowed | {"deque", "Counter", "defaultdict"} <= names, names
    blocked = {"builtins", "inspect", "io", "os", "pickle", "resource", "subprocess", "sys", "time", "traceback"}
    assert not blocked & allowed and not blocked & names, blocked & names
    for module in sorted(blocked):
        assert f"Module '{module}' is not allowed" in run(sandbox, f"import {module}")["error"], module

    start = time.monotonic()
    slow = run(sandbox, SLOW, timeout=2, host_sleep=0.4)
    took = time.monotonic() - start
    assert slow["error"] == "" and slow["stdout"] == "finished\n" and len(slow["action_results"]) == 5, slow
    assert took > 4, "the handlers did not sleep"
    spin = run(sandbox, "while True:\n    pass", timeout=1, kept=[KEPT])
    assert spin["error"] == "Tool timed out after 1s" and "keepable_functions" not in spin, spin
    print(f"sandbox: 4 s of host time fit a 2 s deadline ({took:.1f} s wall); own time still times out")

    def timeline(reply) -> dict:
        out = run(sandbox, TIMELINE, timeline=reply)
        assert out["error"] == "", out
        items, values, steps, frames, action = json.loads(out["stdout"])
        assert values == list(items.values()) and [steps, frames, action] == [items[k] for k in ("steps", "frames", "action")]
        return items

    animation = sys.modules["inference.utils.animation"]
    moved = [[[5 if (r, c) in cells else v for c, v in enumerate(row)] for r, row in enumerate(GRID)]
             for cells in ({(0, 0)}, {(0, 0), (0, 1)})]
    real = json.loads(json.dumps(animation.build_animation_view({"chain": [GRID, *moved], "action_display": "UP"})))
    assert real["frames"] == 2 and len(real["steps"]) == 2, real
    assert timeline(real) == real
    none = animation.build_animation_view(None)  # the host's reply after a single-frame action
    assert set(none) == {"error"}, none
    assert timeline(none) == {"action": None, "frames": 0, "steps": [], **none}
    assert timeline({}) == {"action": None, "frames": 0, "steps": []}  # ARC3_ANIMATION_TIMELINE=0
    assert run(sandbox, "print(len(last_animation_frames))")["stdout"] == "0\n"

    out = run(sandbox, NODE_BBOXES)
    assert out["stdout"] == "[[0, 0, 4, 4], [1, 1, 3, 3]]\n", out


def check_bbox() -> None:
    segmentation = sys.modules["inference.utils.segmentation"]
    frame_diff = sys.modules["inference.utils.frame_diff"]
    colors = sys.modules["inference.utils.grid_utils"].ARC_COLOR_CHARS
    nodes = segmentation.segment_layer(GRID, colors)["nodes"]
    assert [n["bbox"] for n in nodes] == [[0, 0, 4, 4], [1, 1, 3, 3]], nodes
    ell = nodes[1]
    assert ell["pixels"] == 5 and ell["boundary"][2] != ell["bbox"][2:], ell  # a corner list, not a bbox
    assert all(tuple(n["bbox"]) == frame_diff._node_bbox(n) for n in nodes)  # frame_diff's appeared/disappeared bbox


def check_agent(tool_agent, tmp: Path) -> None:
    agent = tool_agent.ToolAgent(model="stub")
    system = agent._system_prompt
    assert STOP_RULE in system and "`run_complete`, or `done`, stop acting" not in system
    assert "usually rendered" in system and "Sometimes they" in system and "1x1 tokens.\n- Some games" in system
    assert "`pixels` (cell count), `bbox` (`[min_row, min_col, max_row, max_col]`, inclusive), `boundary` (" in system
    for field in ("`executed` (a bool)", "`executed_actions` (the list of actions that ran)", "`stop_reason` (",
                  "`gameplay_changed_per_action` (one bool per executed action"):
        assert field in system, field
    assert "When the board changed since your previous analyzer turn, a diff image follows" in system
    assert "Alongside the current grid image you receive" not in system

    def schema() -> str:
        return agent._tools(tmp / "state.json")[0]["function"]["parameters"]["properties"]["code"]["description"]

    assert schema() == "Python code to run. Variables do not persist between calls; eligible functions do."
    os.environ["ARC3_PERSISTENT_FUNCTIONS"] = "0"
    assert schema() == "Python code to run. The snippet is ephemeral and is not saved across tool calls."
    os.environ["ARC3_PERSISTENT_FUNCTIONS"] = "1"

    # a killed snippet leaves the retained set alone; a finished one still drops what it lost
    sandbox = sys.modules["inference.agent.python_tool_sandbox"]
    spin = run(sandbox, "while True:\n    pass", timeout=1, kept=[KEPT])
    agent._kept_functions = {"f": KEPT}
    payload: dict = {}
    agent._record_retained_functions(spin, payload)
    assert agent._kept_functions == {"f": KEPT} and payload == {}, payload
    agent._record_retained_functions(run(sandbox, "print(f(1))", kept=[KEPT]), payload)
    assert agent._kept_functions == {"f": KEPT}, "a finished call keeps a restored function"
    agent._record_retained_functions({"stdout": "", "error": "", "keepable_functions": []}, payload)
    assert agent._kept_functions == {} and "no longer retained" in payload["function_retention"], payload


def main() -> None:
    assert TREE.is_dir(), f"pristine Franzen tree missing: {TREE} (set FRANZEN_TREE)"
    text = (OVERLAYS / "wastefix.py").read_text()
    assert "\u2014" not in text and "\u2013" not in text, "dash in overlay"
    os.environ.update(FORK_ENV)  # read at import: set before either harness loads

    with tempfile.TemporaryDirectory(prefix="franzen-wastefix-") as name:
        tmp = Path(name)
        pristine = tmp / "pristine"
        shutil.copytree(TREE, pristine)
        check_pristine(pristine)
        print("pristine: every check below fails there")

        bundle = tmp / "bundle"
        shutil.copytree(TREE, bundle)
        apply("wastefix", bundle)
        touched = changed(bundle)
        assert set(touched) == TOUCHED, touched
        for rel in touched:
            py_compile.compile(str(bundle / rel), doraise=True)
        try:
            apply("wastefix", bundle)
            raise AssertionError("second application should refuse")
        except RuntimeError as exc:
            assert "already applied" in str(exc), exc

        tool_agent, sandbox = load(bundle)
        check_sandbox(sandbox)
        check_bbox()
        check_agent(tool_agent, tmp)
        print("patched: pre-bound stdlib, host time excluded, timeouts keep functions, API text corrected")

        others = sorted(p.stem for p in OVERLAYS.glob("*.py") if p.stem != "wastefix")
        for order in (["wastefix", *others], [*others, "wastefix"]):
            combo = tmp / ("combo-%08x" % __import__("zlib").crc32("-".join(order).encode()))
            shutil.copytree(TREE, combo)
            for overlay in order:
                apply(overlay, combo)
            for rel in changed(combo):
                py_compile.compile(str(combo / rel), doraise=True)
            assert (combo / INF / "agent/python_tool_sandbox.py").read_text().count("# overlay wastefix") == 5
            print("composed:", " -> ".join(order))
    print("PASS")


if __name__ == "__main__":
    main()
