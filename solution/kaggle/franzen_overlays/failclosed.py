"""Fail closed on a dead scored run: raise at the end of the run so Kaggle records an error.

Why. taaf's Benchmark.run swallows any exception from the solver task (R2.15), marks every game
still playing as crashed, and returns normally. So a run whose model server or gateway dies
early ends the notebook cleanly, and Kaggle banks it as "complete" with a near-zero score:
2026-10-03 (Kaggle outage) and 2026-10-10 (unexplained) both completed within 2.5 h at 0.00,
each spending the day's only slot. A run that errors does not spend the slot: on 2026-09-30
submission 56690848 errored and Kaggle accepted resubmission 56718936 the same UTC day. So in a
scored run, finishing with fewer than MIN_LEVELS levels across all games raises instead, and
daily_submit.sh retries the same day.

MIN_LEVELS is far below any healthy run: the lowest public draws of Franzen's notebook score
about 20, which is well over a hundred levels across 110 games. A run under 20 levels is worth
nothing on a board that keeps our maximum (31.90), so raising cannot cost a useful draw.

The cell embeds this file and calls check() only when TRUE_SUBMISSION; bench and probe runs
call summarize() and print it. solution/sim/test_failclosed.py checks both functions and that
levels_completed is counted on the COMPETITION transport the scored run uses.
"""
from __future__ import annotations

from collections import Counter

MIN_LEVELS = 20


def summarize(game_runs) -> dict:
    """Games, levels, actions and terminal states across a benchmark's game runs."""
    runs = list(game_runs)
    return {
        "games": len(runs),
        "levels": sum(int(getattr(run, "levels_completed", 0) or 0) for run in runs),
        "actions": sum(len(getattr(run, "history", None) or ()) for run in runs),
        "states": dict(sorted(Counter(str(getattr(run, "state", "?")) for run in runs).items())),
    }


def check(game_runs, min_levels: int = MIN_LEVELS) -> dict:
    """Print the summary; raise when the run completed fewer than min_levels levels."""
    summary = summarize(game_runs)
    print(f"failclosed: {summary}", flush=True)
    if summary["levels"] < min_levels:
        raise RuntimeError(
            f"failclosed: dead run, {summary['levels']} levels across {summary['games']} games "
            f"(minimum {min_levels}). Raising so this submission errors instead of banking a "
            "near-zero score; the daily job retries while the slot is unspent.")
    return summary
