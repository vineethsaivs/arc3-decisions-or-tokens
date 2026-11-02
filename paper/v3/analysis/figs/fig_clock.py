#!/usr/bin/env python3
"""F11 "Where the nine hours go": per game, clock on cleared levels vs on the level open at the end (a); share of the clock
by actions executed in a turn (b).

Run from the repo root:
  uv run --no-project --with matplotlib --with numpy python -I -B paper/v3/analysis/figs/fig_clock.py

Reads the analyzer transcripts and benchmark.json of six replica runs in solution/kaggle/generated/kernel-outputs/
(kaggle-lf15uns2-1030-1..3: stall check; kaggle-lf15us8-1031-1..3: stall check plus object groups) through clock_tail.py.
Writes paper/v3/analysis/figs/data/clock_by_game_run.csv (one row per run and game, seconds).
"""
import csv
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt  # noqa: E402

from clock_tail import run_split  # noqa: E402
from figstyle import C, HERE, REPO, W, apply, check_no_overlap, panel, save  # noqa: E402

KO = REPO / "solution/kaggle/generated/kernel-outputs"
RUNS = [f"kaggle-lf15uns2-1030-{i}" for i in (1, 2, 3)] + [f"kaggle-lf15us8-1031-{i}" for i in (1, 2, 3)]
data = {r: run_split(KO / r) for r in RUNS}
games = sorted(data[RUNS[0]])
assert all(sorted(data[r]) == games for r in RUNS) and len(games) == 25

with open(HERE / "data/clock_by_game_run.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["run", "game", "levels_completed", "levels_in_game", "clock_cleared_s", "clock_tail_s",
                "s_0_actions", "s_1_action", "s_2_to_3_actions", "s_4_plus_actions", "s_first_idle_of_streak"])
    for r in RUNS:
        for g in games:
            c, t, by, fi, lc, nl = data[r][g]
            w.writerow([r, g, lc, nl, c, t, *by, fi])

# ---- numbers ----------------------------------------------------------------------------------------------
per_game = {g: (st.mean(data[r][g][0] for r in RUNS) / 60, st.mean(data[r][g][1] for r in RUNS) / 60,
                st.mean(data[r][g][4] for r in RUNS), data[RUNS[0]][g][5]) for g in games}   # cleared min, tail min, levels, n
tot_c = sum(data[r][g][0] for r in RUNS for g in games)
tot_t = sum(data[r][g][1] for r in RUNS for g in games)
tail_share = tot_t / (tot_c + tot_t)

CATS = ["0 actions, first of a streak", "0 actions, later in a streak", "1 action", "2 to 3 actions", "4 or more actions"]


def shares(run_list):
    by = [sum(data[r][g][2][k] for r in run_list for g in games) for k in range(4)]
    fi = sum(data[r][g][3] for r in run_list for g in games)
    tot = sum(by)
    return [fi / tot, (by[0] - fi) / tot, by[1] / tot, by[2] / tot, by[3] / tot], tot


pooled, clock6 = shares(RUNS)
per_run = [shares([r])[0] for r in RUNS]
uns2 = shares(RUNS[:3])[0]
assert abs(sum(pooled) - 1) < 1e-9
assert 0.27 <= uns2[0] + uns2[1] <= 0.29, uns2          # idle turns, stall-check runs: about 28 percent of the clock
assert 0.17 <= uns2[0] <= 0.21, uns2                     # first idle turn of a streak: 17 to 20 percent
assert round(pooled[0] + pooled[1], 3) == 0.269 and round(pooled[0], 3) == 0.198, pooled
print("tail share", round(tail_share, 3), "pooled shares", [round(x, 3) for x in pooled], "clock h", round(clock6 / 3600, 1))
print("stall-check runs idle", round(uns2[0] + uns2[1], 3), "first idle", round(uns2[0], 3))
# the game tail minutes printed in the figure must equal the CSV
rows = list(csv.DictReader(open(HERE / "data/clock_by_game_run.csv")))
for g in games:
    m = st.mean(float(x["clock_tail_s"]) for x in rows if x["game"] == g) / 60
    assert abs(m - per_game[g][1]) < 1e-9

# ---- figure -----------------------------------------------------------------------------------------------
apply()
H = 6.4
fig = plt.figure(figsize=(W, H))
axA = fig.add_axes([0.105, 0.345, 0.665, 0.61])
axB = fig.add_axes([0.355, 0.07, 0.625, 0.16])
INK, GRAY, ACC = C["ink"], C["gray"], C["accent"]
labels = []

order = sorted(games, key=lambda g: per_game[g][1])        # smallest tail at the bottom, largest at the top
ypos = {g: i for i, g in enumerate(order)}
for g in order:
    c, t, lv, nl = per_game[g]
    axA.barh(ypos[g], c, color=C["light"], edgecolor=GRAY, linewidth=0.4, height=0.72, zorder=3)
    axA.barh(ypos[g], t, left=c, color=ACC, height=0.72, zorder=3)
    labels.append(axA.text(126.5, ypos[g], f"{lv:.1f} of {nl}", ha="left", va="center", fontsize=6.3, color=C["gray_text"]))
axA.set_yticks(range(25))
axA.set_yticklabels([g.split("-")[0] for g in order], fontsize=6.3)
axA.tick_params(axis="y", length=0, pad=2)
axA.set_ylim(-0.7, 24.7)
axA.set_xlim(0, 125)
axA.set_xticks([0, 30, 60, 90, 120])
axA.grid(axis="x", visible=True)
axA.spines["left"].set_visible(False)
axA.set_xlabel("mean minutes of game clock per run (six runs)", labelpad=2)
labels.append(axA.text(126.5, 25.35, "levels cleared\n(mean of 6)", ha="left", va="bottom", fontsize=6.3,
                       color=C["gray_text"], linespacing=1.05))
leg = [plt.Rectangle((0, 0), 1, 1, fc=C["light"], ec=GRAY, lw=0.4), plt.Rectangle((0, 0), 1, 1, fc=ACC)]
axA.legend(leg, ["on levels it cleared", f"on the level open at the end ({tail_share:.0%} of all clock)"], loc="lower right",
           bbox_to_anchor=(1.0, 0.0), handlelength=1.0, handletextpad=0.4, labelspacing=0.3, borderaxespad=0.3, fontsize=6.5,
           frameon=True, facecolor="white", edgecolor="none", framealpha=0.95)
panel(fig, 0.005, 0.995, "(a)", "Clock per game, 25 games")

# (b) share of the clock by actions executed in the turn; bar = six runs pooled, dots = each run
ys = list(range(len(CATS)))
for y, k in zip(ys, range(5)):
    axB.barh(y, pooled[k], color=ACC if k < 2 else C["light"], edgecolor=GRAY if k >= 2 else ACC, linewidth=0.4, height=0.62,
             zorder=2, alpha=1.0 if k == 0 else 0.55 if k == 1 else 1.0)
    for i, ps in enumerate(per_run):
        axB.plot(ps[k], y, "o" if i < 3 else "^", ms=3.4, mfc=INK if i < 3 else "white", mec=INK, mew=0.6, zorder=4)
    labels.append(axB.text(max(pooled[k], max(p[k] for p in per_run)) + 0.014, y,
                           f"{pooled[k]:.0%}", ha="left", va="center", fontsize=6.8, color=INK))
axB.set_yticks(ys)
axB.set_yticklabels(CATS, fontsize=6.5)
axB.tick_params(axis="y", length=0, pad=2)
axB.set_ylim(len(CATS) - 0.4, -0.6)
axB.set_xlim(0, 0.45)
axB.set_xticks([0, 0.1, 0.2, 0.3, 0.4])
axB.set_xticklabels(["0%", "10%", "20%", "30%", "40%"])
axB.grid(axis="x", visible=True)
axB.spines["left"].set_visible(False)
axB.set_xlabel("share of the clock in turns that executed this many actions", labelpad=2)
hl = [axB.plot([], [], "o", ms=3.4, mfc=INK, mec=INK, ls="")[0], axB.plot([], [], "^", ms=3.4, mfc="white", mec=INK, mew=0.6, ls="")[0]]
axB.legend(hl, ["stall check, 3 runs", "+ object groups, 3 runs"], loc="upper right", bbox_to_anchor=(1.0, 0.82), handletextpad=0.1, labelspacing=0.25,
           borderaxespad=0.1, fontsize=6.3, frameon=True, facecolor="white", edgecolor="none", framealpha=0.95)
panel(fig, 0.005, 0.262, "(b)", f"Clock by actions per turn: {pooled[0] + pooled[1]:.0%} sits in idle turns")

check_no_overlap(fig, labels)
save(fig, "fig_clock", writeup=True)
