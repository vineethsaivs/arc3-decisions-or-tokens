"""Fail-closed check (franzen_overlays/failclosed.py) on the transport the scored run uses.

The check only helps if it can tell a dead run from a real one in a SCORED run, where nothing
can be watched. So this plays through taaf's localhost COMPETITION-mode Arcade (the gateway's
shape, real arc_agi client, real taaf.Benchmark) with no GPU:

  1. unit: summarize() and check() on plain objects (raises below the minimum, passes at it);
  2. a live run: a stub solver replays the fastest public human's level-1 actions on bp35 and
     does nothing on ls20; levels_completed must read 1 and 0, and check() must agree;
  3. a dead run: the solver raises at once (a dead server or gateway); Benchmark.run swallows it
     and returns normally with every game "crashed" (the 0.00 shape), and check() must raise.

    python3 solution/sim/test_failclosed.py
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
OVERLAY = REPO / "solution/kaggle/franzen_overlays/failclosed.py"
ENV_FILES = REPO / "ARC-AGI-3-Kaggle-Starter/environment_files"
REPLAY = REPO / "reference/re-arc-3/re_arc/dsl/official_human_replays/bp35.json"


def _reexec_under_uv() -> None:
    """Re-run inside the ARC-AGI-3-Agents venv (arc_agi with its server module, and arcengine)."""
    if os.environ.get("_FAILCLOSED_CHILD"):
        return
    sys.path.insert(0, str(REPO / "solution/sim"))
    from franzen_tree import pristine_tree  # noqa: PLC0415

    tree = pristine_tree()
    env = dict(os.environ, _FAILCLOSED_CHILD="1")
    env["PYTHONPATH"] = os.pathsep.join(
        [str(tree / "src/tufa-arc-agi-framework/src"), env.get("PYTHONPATH", "")]).strip(os.pathsep)
    cmd = ["uv", "--directory", str(REPO / "ARC-AGI-3-Agents"), "run", "--with", "imageio", "--with", "matplotlib",
           "--with", "scipy", "--with", "pandas", "--with", "arc-agi>=0.9.9", "python", str(Path(__file__).resolve())]
    raise SystemExit(subprocess.run(cmd, env=env).returncode)


def load_overlay() -> dict:
    namespace: dict = {}
    exec(compile(OVERLAY.read_text(), str(OVERLAY), "exec"), namespace)
    return namespace


def unit_checks(fc: dict) -> int:
    runs = [SimpleNamespace(levels_completed=3, history=[1, 2], state="gave_up"),
            SimpleNamespace(levels_completed=0, history=[], state="crashed"),
            SimpleNamespace(state="crashed")]  # a run without the attributes still counts as 0
    summary = fc["summarize"](runs)
    assert summary == {"games": 3, "levels": 3, "actions": 2,
                       "states": {"crashed": 2, "gave_up": 1}}, summary
    assert fc["check"](runs, min_levels=3)["levels"] == 3
    try:
        fc["check"](runs, min_levels=4)
    except RuntimeError as error:
        assert "dead run, 3 levels across 3 games" in str(error), error
    else:
        raise AssertionError("check() passed a run below the minimum")
    assert fc["MIN_LEVELS"] == 20
    print("ok  unit: summarize counts levels, actions and states; check raises below the minimum")
    return 1


def transport_checks(fc: dict) -> int:
    import arc_agi
    import arcengine
    import taaf.benchmark
    import taaf.competition_arcade as ca
    import taaf.game_api as ga
    import taaf.solver

    if not hasattr(arc_agi.EnvironmentInfo, "private_tags"):  # older local arc_agi pin, label only
        arc_agi.EnvironmentInfo.private_tags = None
    level1 = json.loads(REPLAY.read_text())["levels"][0]

    class ReplaySolver(taaf.solver.Solver):
        async def _run_games(self, games):
            for game in games:
                if game.game_run.game_id.startswith("bp35"):
                    for action_id, data in level1:
                        game.execute_action(arcengine.ActionInput(id=arcengine.GameAction.from_id(action_id), data=data))
                game.finish_game()

    class DeadSolver(taaf.solver.Solver):
        async def _run_games(self, games):
            raise ConnectionError("analyzer server is gone")  # what a dead SGLang server looks like

    def play(solver) -> list:
        server = ca.CompetitionArcadeServer(game_ids=("bp35", "ls20"), environments_dir=str(ENV_FILES))
        server.start()
        try:
            games = [ga.GameAPI(env_name=gid, arcade_spec=server.arcade_spec) for gid in server.exposed_game_ids]
            with tempfile.TemporaryDirectory(prefix="failclosed-") as job:
                bm = taaf.benchmark.Benchmark(label="failclosed", games=games, solver=solver, job_dir=Path(job))
                asyncio.run(bm.run(minimal_diagnostics=True))  # the scored run's flag
                return bm.game_runs
        finally:
            server.stop()

    runs = play(ReplaySolver())
    by_game = {run.game_id.split("-")[0]: run for run in runs}
    assert by_game["bp35"].levels_completed == 1, by_game["bp35"]
    assert by_game["ls20"].levels_completed == 0, by_game["ls20"]
    assert fc["check"](runs, min_levels=1)["levels"] == 1
    print(f"ok  live run on the COMPETITION transport: bp35 levels_completed=1 after the human's "
          f"{len(level1)} level-1 actions, ls20 0; check() passes at 1")

    dead = play(DeadSolver())
    assert {run.state for run in dead} == {"crashed"}, [run.state for run in dead]
    try:
        fc["check"](dead)
    except RuntimeError as error:
        assert "dead run, 0 levels across 2 games" in str(error), error
    else:
        raise AssertionError("check() passed a dead run")
    print("ok  dead run: Benchmark.run swallowed the solver error and returned with every game crashed; "
          "check() raised")
    return 2


def main() -> None:
    _reexec_under_uv()
    fc = load_overlay()
    checks = unit_checks(fc) + transport_checks(fc)
    print(f"\n{checks} checks passed")


if __name__ == "__main__":
    main()
