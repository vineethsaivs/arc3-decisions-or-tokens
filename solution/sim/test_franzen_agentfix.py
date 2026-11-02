"""Offline check for the A1 agent-fix overlay (franzen_overlays/agentfix.py).

    python3 solution/sim/test_franzen_agentfix.py

Re-runs itself under reference/re-arc-3/.venv/bin/python (or $AGENTFIX_PYTHON) when this
interpreter lacks the engine's dependencies. The engine is the pair Kaggle installs, arc_agi 0.9.8
and arcengine 0.9.3, imported straight from the wheels in
solution/dev/cloud_public25_2026_09_13/engine-wheels, and any other version stops the test (that
venv carries arc_agi 0.9.4). No GPU, no network.

The games are real: taaf's GameAPI on an OFFLINE arcade over the 25 public game files, the
harness's own _HarnessGameSession.step_env, ToolAgent._run_python_tool and the real sandbox
child, under cell 7's settings (FORK_ENV, ONLY_RESET_LEVELS=true) plus the arm's
EXPOSE_RESET=on. Level 1 is cleared with RE-ARC-3's winning moves for cd82 and sp80.

1. Pristine tree, where every functional check below fails: a RESET at a level start runs and
   costs an action; the engine facts the overlay's docstring states (with ONLY_RESET_LEVELS it
   changes nothing, a mid-level RESET keeps levels_completed either way, and without the variable
   a RESET at a level start is a FULL reset to level 1); after a caught stale refusal every later
   call in the snippet is refused; level_starts is a NameError.
2. The overlay alone, and with wastefix in both orders (the builder's first):
   - RESET at a level start is skipped at the game start, after the warmup RESET, after a
     level-up, right after a manual RESET and after a game over's automatic RESET: it does not
     run, no action is spent, its echo line shows, nothing raises and the snippet goes on;
   - at the head of a batch at a level start the RESET is skipped and the rest of the batch runs;
   - a mid-level RESET runs, keeps levels_completed and brings back the level's first board;
   - a caught stale refusal lets the next call through, 6 no-ops are waived, the 7th re-arms the
     latch and the next refusal is not waived;
   - level_starts and attempt_starts after the warmup RESET, a level-up (refreshed inside the
     snippet), a manual RESET and a game over; the prompt, skip and refusal texts.
3. agentfix_counters.py counts the tool results those games printed.
4. The builder snippet's cell 7 edits land once and compile.
5. The overlay touches exactly three files, refuses a second application, and composes with
   every other bundle overlay (overlay_orders).
"""
from __future__ import annotations

import json
import os
import py_compile
import shutil
import subprocess
import sys
import tempfile
import threading
from importlib.metadata import version
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
ENGINE_PY = Path(os.environ.get("AGENTFIX_PYTHON", REPO / "reference/re-arc-3/.venv/bin/python"))
WHEELS = REPO / "solution/dev/cloud_public25_2026_09_13/engine-wheels"
ENV_DIR = REPO / "ARC-AGI-3-Kaggle-Starter/environment_files"


def _reexec() -> None:
    """Re-run under an interpreter that carries arc_agi's and taaf's dependencies."""
    missing = []
    for name in ("numpy", "pydantic", "flask", "matplotlib", "imageio", "dotenv", "requests", "PIL"):
        try:
            __import__(name)
        except ImportError:
            missing.append(name)
    if not missing:
        return
    if os.environ.get("_AGENTFIX_CHILD") or not ENGINE_PY.exists():
        raise SystemExit(f"{sys.executable} lacks {', '.join(missing)}; set AGENTFIX_PYTHON to a python "
                         f"3.12 with arc_agi's dependencies ({ENGINE_PY} not usable)")
    env = dict(os.environ, _AGENTFIX_CHILD="1", PYTHONDONTWRITEBYTECODE="1")
    raise SystemExit(subprocess.run([str(ENGINE_PY), str(Path(__file__).resolve())], env=env).returncode)


_reexec()
sys.path[:0] = [str(p) for p in sorted(WHEELS.glob("*.whl"))]  # exact Kaggle engine, imported from the zips
sys.path.insert(0, str(REPO / "solution/analysis"))
sys.path.insert(0, str(REPO / "solution/kaggle"))

import logging  # noqa: E402

logging.disable(logging.ERROR)
import agentfix_counters  # noqa: E402
import build_franzen_arms as B  # noqa: E402
import overlay_orders  # noqa: E402
from test_franzen_sandbox import FORK_ENV, INF, OVERLAYS, TREE, apply  # noqa: E402

sys.path.insert(0, str(TREE / "src/tufa-arc-agi-framework/src"))  # taaf: the overlay never touches it
os.environ.update(FORK_ENV)  # read at import: set before any harness loads
os.environ.update({"ONLY_RESET_LEVELS": "true", "EXPOSE_RESET": "on"})  # cell 7, plus the arm's edit
import arc_agi  # noqa: E402
from taaf.game import RunSession  # noqa: E402
from taaf.game_api import ArcadeSpec, GameAPI  # noqa: E402

TOUCHED = {INF / "framework/solver.py", INF / "agent/python_tool_sandbox.py", INF / "agent/prompts.py"}
CD82_LEVEL1 = ("LEFT", "LEFT", "LEFT", "LEFT", "DOWN", "DOWN", "RIGHT", "SPACE")  # RE-ARC-3 precomputed_actions
SP80_LEVEL1 = ("RIGHT", "RIGHT", "RIGHT", "SPACE")
RESET_ECHO = "[action] RESET -> not executed | stop_reason: reset_at_level_start"
RESET_DETAIL = "RESET was skipped and no action was spent"
RESET_SENTENCE = ("RESET restarts only the current level, so when that level is already at its start (right after a "
                  "level-up, a game over or another RESET) it is skipped, spends no action, and the rest of the "
                  "snippet runs.")
CONTRACT = ("The first time a snippet catches this, the check is waived for its next 6 actions that change "
            "nothing; the 7th re-arms it, and later refusals in that snippet are not waived.")
REFRESHED = ("`current_frame`, `previous_frame`, `history`, `transitions`, `level_starts`, `attempt_starts`, "
             "`valid_actions`, and `last_action_call_result` are refreshed.")
# on level 2 of cd82, UP and DOWN change nothing and LEFT moves something (guards start at level 2)
STALE = """
log, messages = [], []
def press(name):
    try:
        action([name])
        log.append(name + ' ran')
    except StaleStateActionError as exc:
        log.append(name + ' refused')
        messages.append(str(exc))
press('UP')
press('DOWN')
for _ in range(6):
    press('UP')
press('DOWN')
press('UP')
press('LEFT')
print(log)
print(messages[0])
"""
STALE_PATCHED = ["UP ran", "DOWN refused", *["UP ran"] * 6, "DOWN ran", "UP refused", "LEFT refused"]
STALE_PRISTINE = ["UP ran", "DOWN refused", *["UP refused"] * 6, "DOWN refused", "UP refused", "LEFT refused"]


def copy_tree(dst: Path) -> Path:
    shutil.copytree(TREE, dst, ignore=shutil.ignore_patterns("__pycache__"))
    return dst


def load(bundle: Path):
    """Import this bundle's harness, dropping any other copy already imported."""
    for key in [k for k in sys.modules if k == "inference" or k.startswith("inference.")]:
        del sys.modules[key]
    sys.path[:] = [p for p in sys.path if not p.endswith("ARC3-Inference")]
    sys.path.insert(0, str(bundle / "src" / "ARC3-Inference"))
    import inference.agent.tool_agent as tool_agent
    import inference.framework.solver as solver
    assert Path(solver.__file__).resolve().is_relative_to(bundle.resolve()), solver.__file__
    return tool_agent, solver


class Play:
    """One real game through the harness: offline arcade, the solver's session, a stub ToolAgent."""

    def __init__(self, harness, name: str, work: Path):
        self.tool_agent, self.solver = harness
        self.game = GameAPI(env_name=name, arcade_spec=ArcadeSpec(
            operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=str(ENV_DIR)))
        self.game.start_game(RunSession())
        self.agent = self.tool_agent.ToolAgent(model="stub")
        self.session = self.solver._HarnessGameSession(
            solver=self.solver.HarnessSolver(label="agentfix"), game=self.game, analyzer=self.agent,
            game_index=0, pass_index=0, state_path=work / f"{name}.json", transcript_path=work / f"{name}.txt",
            analysis_html_relpath=f"{name}.html", stop_event=threading.Event(),
            viewer_data_path=work / f"{name}_viewer.json")
        self.session.seed_initial_history()
        self.session.write_runtime_state()
        self.agent._step_env_callback = self.session.step_env

    def press(self, *names: str) -> dict:
        """Setup moves, one step_env call each (no snippet, so no stale latch)."""
        for name in names:
            out = self.session.step_env({"actions": [{"action": name}]})
            assert out.get("executed"), out
        return out

    def snippet(self, code: str) -> dict:
        """The model's path: sandbox child -> ToolAgent._handle_action -> step_env."""
        self.agent._current_valid_actions = self.tool_agent._normalize_valid_actions(
            self.solver._engine_action_names(self.game))
        return json.loads(self.agent._run_python_tool(self.session.state_path, {"code": code}).content)

    @property
    def actions(self) -> int:
        return len(self.game.game_run.history)

    @property
    def levels(self) -> int:
        return self.game.current_state.levels_completed

    @property
    def level(self) -> int:
        return self.session.current_frame().level

    def board(self):
        return self.session.current_frame().grid


def skipped(g: Play, code: str = "action(['RESET'])") -> dict:
    """A lone RESET at a level start: it does not run or cost an action, nothing raises and the
    snippet goes on."""
    before = (g.actions, g.levels, g.board())
    out = g.snippet(code + "\nprint('went on')")
    assert "error" not in out and RESET_ECHO + "\n" in out["stdout"] and out["stdout"].endswith("went on\n"), out
    assert (g.actions, g.levels, g.board()) == before, "a skipped RESET must not run or cost an action"
    last = g.agent._last_action_call_result
    assert last["stop_reason"] == "reset_at_level_start" and RESET_DETAIL in last["stop_detail"], last
    return out


def check_pristine(harness, work: Path) -> None:
    """Every functional check below fails here, and the engine does what the docstring says."""
    g = Play(harness, "cd82", work)
    out = g.snippet("action(['RESET'])")
    assert out["stdout"].startswith("[action] RESET -> NO-OP") and g.actions == 1, out  # spent at the start
    g.press(*CD82_LEVEL1)
    assert (g.levels, g.level) == (1, 2)
    first = g.board()
    out = g.snippet("action(['RESET'])")  # ONLY_RESET_LEVELS=true: runs, changes nothing, costs an action
    assert out["stdout"].startswith("[action] RESET -> NO-OP") and (g.levels, g.level, g.board()) == (1, 2, first)
    os.environ.pop("ONLY_RESET_LEVELS")
    try:
        g.press("LEFT")
        g.snippet("action(['RESET'])")  # mid-level: a level reset even without the variable
        assert (g.levels, g.level, g.board()) == (1, 2, first)
        g.snippet("action(['RESET'])")  # at the level start without it: a FULL reset
        assert (g.levels, g.level) == (0, 1), "arcengine without ONLY_RESET_LEVELS should restart the game"
    finally:
        os.environ["ONLY_RESET_LEVELS"] = "true"
    g.press(*CD82_LEVEL1)
    before = g.actions
    out = g.snippet(STALE)
    assert out["stdout"].splitlines()[-2] == repr(STALE_PRISTINE) and g.actions == before + 1, out
    assert "NameError: name 'level_starts' is not defined" in g.snippet("print(level_starts)")["error"]
    system = harness[0].ToolAgent(model="stub")._system_prompt
    assert "RESET must be the first game action" in system and RESET_SENTENCE not in system
    assert "level_starts" not in system


def check_cd82(harness, work: Path, captured: list) -> None:
    g = Play(harness, "cd82", work)
    out = skipped(g, "print(level_starts, attempt_starts)\nr = action(['RESET'])\n"  # the seeded start
                     "print(r['executed'], r['level'], r['score'], r['action_num'])")
    assert out["stdout"].startswith(f"[0] [0]\n{RESET_ECHO}\nFalse 1 0 0\n"), out
    g.session._execute_auto_reset()  # the warmup RESET: the same path as the one after a game over
    out = skipped(g, "print(level_starts, attempt_starts)\naction(['RESET'])")
    assert out["stdout"].startswith("[0] [0, 1]\n"), out
    captured.append(out)
    g.press(*CD82_LEVEL1)
    assert (g.levels, g.level) == (1, 2)
    first = g.board()
    before = g.actions  # heading a batch: the RESET is skipped and the LEFT runs
    out = g.snippet("r = action(['RESET', 'LEFT'])\nprint(r['executed_actions'])")
    assert "error" not in out and out["stdout"] == f"{RESET_ECHO}\n[action] LEFT -> effect\n['LEFT']\n", out
    assert (g.actions, g.levels) == (before + 1, 1) and g.board() != first
    captured.append(out)
    before = g.actions
    out = g.snippet("r = action(['RESET'])\nprint(r['executed'], r['level'], r['score'])")
    assert out["stdout"].endswith("True 2 1\n") and "error" not in out, out
    assert (g.actions, g.levels, g.board()) == (before + 1, 1, first), "a mid-level RESET restarts the level only"
    captured.append(out)
    captured.append(skipped(g))  # right after that manual RESET
    before = g.actions
    out = g.snippet(STALE)
    log, message = out["stdout"].splitlines()[-2:]
    assert log == repr(STALE_PATCHED) and g.actions == before + 8, out
    assert CONTRACT in message, message
    for name in ("DOWN", "UP", "LEFT"):
        assert out["stdout"].count(f"[action] {name} -> not executed | stop_reason: stale_state") == 1, out
    captured.append(out)


def check_sp80(harness, work: Path, captured: list) -> None:
    g = Play(harness, "sp80", work)
    g.press(*SP80_LEVEL1[:-1])
    out = g.snippet("before = (list(level_starts), list(attempt_starts))\n"
                    "r = action(['SPACE'])\nprint(before, level_starts, attempt_starts, r['level_completed'])")
    assert out["stdout"].endswith("([0], [0]) [0, 4] [4] True\n"), out  # refreshed by action() itself
    captured.append(skipped(g))  # after the level-up
    g.press("RIGHT")
    out = g.snippet("action(['RESET'])")  # mid-level: runs
    assert "error" not in out and g.levels == 1, out
    captured.append(out)
    captured.append(skipped(g))  # after a manual RESET
    for _ in range(5):  # on level 2 SPACE changes nothing and the fifth ends the attempt
        last = g.press("SPACE")
    assert last["game_over"], last
    g.session._execute_auto_reset()  # what the play loop does next
    captured.append(skipped(g))  # after a game over
    out = g.snippet("print(level_starts, attempt_starts, len(history))\n"
                    "print(history[level_starts[1]].frame.level, history[level_starts[1] - 1].frame.level, "
                    "history[attempt_starts[-1]].frame.ascii == history[level_starts[1]].frame.ascii)")
    assert out["stdout"] == "[0, 4] [4, 6, 12] 13\n2 1 True\n", out


def check_patched(harness, work: Path, captured: list) -> None:
    system = harness[0].ToolAgent(model="stub")._system_prompt
    assert RESET_SENTENCE in system and REFRESHED in system, "prompt text missing"
    assert "- `level_starts` and `attempt_starts` are lists of `history` indices." in system
    assert "`history[attempt_starts[-1]].frame` is the board the current attempt started from." in system
    check_cd82(harness, work, captured)
    check_sp80(harness, work, captured)


def check_counters(captured: list, work: Path) -> None:
    """The counter reads what the patched harness printed: one transcript turn per captured call,
    then a burn-to-reset episode and a level drop."""
    run = work / "counters-run"
    run.mkdir()
    turns = []
    for out in captured:
        result = out.get("stdout", "") + (f"\nerror:\n{out['error']}" if out.get("error") else "")
        turns.append(("level 2", "probe", result))
    turns += [("level 2", "To trigger a reset, I'd have to burn ~26 actions.", "[action] UP -> effect\n"),
              ("level 2", "keep going", "[action] UP -> effect\n"),
              ("level 1. GAME OVER occurred during the previous sequence.", "start again", "")]
    text = "".join(f"--- analysis_step={i} | action={i} | 00:00:{i:02d} | tool-agent ---\n[USER PROMPT]\n"
                   f"Current state: step {i}, {where}\n\n[MODEL RESPONSE META]\nfinish_reason: tool_calls\n"
                   f"tool_call_count: 1\n\n[THINKING]\n{think}\n\n[TOOL CALL: python]\ncode\n\n"
                   f"[TOOL RESULT: python]\n{result}\n\n" for i, (where, think, result) in enumerate(turns, 1))
    (run / "cd82-fb555c5d_p0.txt").write_text(text)
    (run / "benchmark.json").write_text(json.dumps({"game_runs": [{
        "game_id": "cd82-fb555c5d", "state": "gave_up", "levels_completed": 1,
        "history": [{"action": {"id": "RESET", "data": {}}, "generated_tokens": 0},
                    {"action": {"id": "RESET", "data": {}}, "generated_tokens": 812}]}]}))
    c = agentfix_counters.counters(run)
    expect = {"manual_resets": 1, "automatic_resets": 1, "manual_resets_echoed": 2, "reset_skips": 6,
              "reset_not_first": 0, "stale_refusals": 3, "snippets_with_stale_refusal": 1,
              "snippets_with_2plus_stale_refusals": 1, "burn_episodes": 1, "burn_actions": 2,
              "burn_episodes_final_level": 1, "level_drops": 1}
    assert {k: c[k] for k in expect} == expect, {k: c[k] for k in expect}


def check_builder() -> tuple[str, ...]:
    """The snippet handed to the integrator, applied to the base notebook. Returns the order in which
    cell 7 then runs agentfix and wastefix, since overlay() inserts each one right after the marker."""
    edits = [(B.ENV, "    'EXPOSE_UNDO': 'on',",
              "    'EXPOSE_UNDO': 'on',\n    'EXPOSE_RESET': 'on',  # arm agentfix: RESET behind the level-start guard")]
    edits += B.overlay("agentfix")
    if hasattr(B, "agentfix"):
        assert B.agentfix() == edits, "build_franzen_arms.agentfix() differs from the snippet this test checks"
    nb = json.loads((B.BASE / "submission.ipynb").read_text())
    B.apply_edits(nb, B.REPLICA + B.overlay("wastefix") + edits)
    cell = B.cell_text(nb["cells"][B.ENV])
    assert cell.count("'EXPOSE_RESET': 'on'") == 1 and cell.count("franzen_overlays/agentfix.py") == 1
    compile("\n".join("pass" if line.lstrip().startswith(("!", "%")) else line for line in cell.split("\n")),
            "cell7", "exec")
    return tuple(sorted(("agentfix", "wastefix"), key=lambda name: cell.index(f"# arm overlay: {name}\n")))


def main() -> None:
    assert TREE.is_dir(), f"pristine Franzen tree missing: {TREE} (set FRANZEN_TREE)"
    assert ENV_DIR.is_dir(), f"public game files missing: {ENV_DIR}"
    for path in (OVERLAYS / "agentfix.py", Path(__file__), REPO / "solution/analysis/agentfix_counters.py"):
        assert not {chr(0x2014), chr(0x2013)} & set(path.read_text()), f"dash in {path.name}"
    engine = (version("arc-agi"), version("arcengine"))
    if engine != ("0.9.8", "0.9.3"):  # the engine facts checked below are the Kaggle pair's
        raise SystemExit(f"need arc_agi 0.9.8 and arcengine 0.9.3, the wheels in {WHEELS}; found {engine}")
    print(f"engine: arc_agi {engine[0]}, arcengine {engine[1]} ({Path(arc_agi.__file__).parents[1]})")
    ship_order = check_builder()
    print("builder snippet: EXPOSE_RESET and the overlay land once in cell 7; it runs", " -> ".join(ship_order))

    with tempfile.TemporaryDirectory(prefix="franzen-agentfix-") as name:
        tmp = Path(name)
        check_pristine(load(copy_tree(tmp / "pristine")), tmp)
        print("pristine: level-start RESETs run, stale refusals stick, no level_starts (the checks below fail here)")

        alone = copy_tree(tmp / "alone")
        apply("agentfix", alone)
        touched = {p.relative_to(alone) for p in alone.rglob("*.py")
                   if p.read_bytes() != (TREE / p.relative_to(alone)).read_bytes()}
        assert touched == TOUCHED, touched
        for rel in touched:
            py_compile.compile(str(alone / rel), cfile=str(tmp / "check.pyc"), doraise=True)
        try:
            apply("agentfix", alone)
            raise AssertionError("a second application must refuse")
        except RuntimeError as exc:
            assert "already applied" in str(exc), exc
        captured: list = []
        check_patched(load(alone), tmp, captured)
        print("patched: level-start RESETs skipped, the snippet goes on, mid-level RESET keeps levels, catch contract kept, "
              "globals right")

        for order in (ship_order, ship_order[::-1]):  # the builder's order first
            ship = copy_tree(tmp / ("ship-" + "-".join(order)))
            for overlay in order:
                apply(overlay, ship)
            check_patched(load(ship), tmp, [])
            print("patched with wastefix,", " -> ".join(order) + ": same")

        check_counters(captured, tmp)
        print("counters: reset skips, manual RESETs, stale refusals per snippet, burn episodes, level drops")

        names = sorted(p.stem for p in OVERLAYS.glob("*.py") if not p.stem.endswith("_serve"))
        for order in overlay_orders.orders(names, "agentfix"):
            combo = copy_tree(tmp / ("combo-%08x" % __import__("zlib").crc32("-".join(order).encode())))
            for overlay in order:
                apply(overlay, combo)
            for path in combo.rglob("*.py"):
                py_compile.compile(str(path), cfile=str(tmp / "check.pyc"), doraise=True)
            text = (combo / INF / "agent/python_tool_sandbox.py").read_text()
            assert text.count("# overlay agentfix") == 5 and text.count("_stale_waiver: list[int] = []") == 1
            shutil.rmtree(combo)
            print("composed:", " -> ".join(order))
    print("PASS")


if __name__ == "__main__":
    main()
