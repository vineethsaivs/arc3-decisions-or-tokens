"""Apply the REAL leaderboard formula to a real benchmark.json. What do we actually bench at?

WHY. The campaign's largest unexplained number is a board-to-bench ratio of about 0.44: our
Flash-Next family is quoted at 9.3-10.5 by the shipped scorer on the public 25, and the board reads
4.47. Two explanations are usually offered, the hidden games are harder or we are overfit, and they
point opposite ways. There is a third that nobody has checked: THE 9.3-10.5 FIGURE MIGHT BE WRONG.

score_model.py is PARAMETRIC. It takes a level count, a completed count and one efficiency number
and models a hypothetical board. It has never been pointed at an actual run. But benchmark.json
carries `actions_per_level` and `base_actions_per_level` per game, which is everything the real
formula needs, so the true bench score is computable exactly and has simply never been computed.

The formula, transcribed in score_model.py from arc_agi/scorecard.py:
    level_score_i = min(115, (baseline_i / actions_i)^2 * 100)  if completed else 0
    weight_i      = i, 1-indexed, over ALL levels including unreached ones
    raw           = sum(level_score_i * i) / sum_over_ALL(i)
    cap           = sum_over_COMPLETED(i) / sum_over_ALL(i) * 100
    game_score    = min(raw, cap)
    board         = mean of game_score over games

    python3 solution/analysis/score_benchmark.py
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "solution/kaggle/generated/kernel-outputs"


def game_score(actions_per_level: list, baselines: list, levels_done: int, n_levels: int) -> float | None:
    if not n_levels:
        return None
    total_w = sum(range(1, n_levels + 1))
    done_w = sum(range(1, levels_done + 1))
    raw = 0.0
    for i in range(1, levels_done + 1):
        if i - 1 >= len(actions_per_level) or i - 1 >= len(baselines or []):
            return None
        a = actions_per_level[i - 1]
        b = baselines[i - 1]
        if not a or not b:
            return None
        raw += min(115.0, (b / a) ** 2 * 100.0) * i
    raw /= total_w
    cap = done_w / total_w * 100.0
    return min(raw, cap)


def main() -> None:
    print(f"{'run':34s} {'games':>5} {'levels':>7} {'BOARD-STYLE SCORE':>19} {'scorable':>9}")
    rows = []
    for d in sorted(OUT.iterdir()):
        bm = d / "benchmark.json"
        if not bm.is_file():
            continue
        runs = json.loads(bm.read_text()).get("game_runs", [])
        scores, lv = [], 0
        for r in runs:
            n = int(r.get("number_of_levels", 0) or 0)
            done = int(r.get("levels_completed", 0) or 0)
            lv += done
            s = game_score(r.get("actions_per_level") or [], r.get("base_actions_per_level") or [],
                           done, n)
            # A game with no baselines cannot be scored locally. It is NOT a zero; excluding it and
            # saying so is the honest handling, because the engine hides baselines in submission
            # mode (taaf.game: "base_actions_per_level may legitimately be None").
            if s is not None:
                scores.append(s)
            elif done == 0 and n:
                scores.append(0.0)
        if not scores:
            continue
        mean = statistics.mean(scores)
        rows.append((d.name, len(runs), lv, mean, len(scores)))
        mark = "  <- Flash-Next" if "flashnext" in d.name else ""
        print(f"{d.name:34s} {len(runs):>5} {lv:>7} {mean:>19.2f} {len(scores):>9}{mark}")

    fx = [r for r in rows if "flashnext" in r[0]]
    if fx:
        print(f"\nFLASH-NEXT FAMILY, the config we actually ship:")
        for r in sorted(fx, key=lambda x: -x[3]):
            print(f"  {r[0]:34s} levels={r[2]:>3}  board-style score={r[3]:.2f}")
        m = statistics.mean(r[3] for r in fx)
        print(f"  mean {m:.2f} over {len(fx)} runs")
        print(f"\nAGAINST THE ACTUAL BOARD: our per-draw mean for this config is 3.96 "
              f"(draws 4.47 and 3.45).")
        print(f"  ratio board / bench = {3.96 / m:.2f}")
        print("\n  If this bench number is far BELOW the 9.3-10.5 the campaign has been quoting,")
        print("  then the '0.44 anomaly' is partly an artefact of that quote and there is less to")
        print("  explain than assumed.")


if __name__ == "__main__":
    main()
