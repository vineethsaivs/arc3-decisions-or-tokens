#!/usr/bin/env python3
"""F12 "Five gates, all inside the noise": levels cleared (a) and game score (b) of each 25-game replica run, by arm.

Run from the repo root:
  uv run --no-project --with matplotlib --with numpy python -I -B paper/v3/analysis/figs/fig_gates.py

Reads benchmark.json of every run directory listed below, in solution/kaggle/generated/kernel-outputs/ (levels = sum of
levels_completed, score = mean of final_score; whichever directories exist are used, so a later run is picked up on the
next build), and paper/v2/analysis/out/e4_noise.json for the run-to-run SDs of identical code.
Writes paper/v3/analysis/figs/data/gate_runs.csv (one row per run).
"""
import csv
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt  # noqa: E402

from figstyle import SCALE_V3 as SCALE, C, HERE, REPO, V2OUT, W, apply, fs, check_no_overlap, panel, save  # noqa: E402

KO = REPO / "solution/kaggle/generated/kernel-outputs"
ARMS = [  # (label, run directories); the directories that exist are used
    ("control", ["vertex-kva", "vertex-s15w-1", "vertex-s15w-2", "vertex-s15w-3", "vertex-s15w-4", "kaggle-lf15-1029",
             "kaggle-lf15-1029-2"]),
    ("+ stall\ncheck", [f"kaggle-lf15uns2-1030-{i}" for i in (1, 2, 3)]),
    ("+ corner-\njoined\nobjects", [f"kaggle-lf15us8-1031-{i}" for i in (1, 2, 3)]),
    ("+ scroll\ndetector", [f"kaggle-lf15usc-1031-{i}" for i in (1, 2, 3)]),
    ("+ tool\nfixes", [f"kaggle-lf15ust-1031-{i}" for i in (1, 2, 3)]),   # extends object groups, not scroll
    ("+ replay\ncheck", [f"kaggle-lf15ustv-1101-{i}" for i in (1, 2, 3)]),
]
runs = []   # (arm index, dir, levels, score)
for ai, (_, dirs) in enumerate(ARMS):
    for d in dirs:
        p = KO / d / "benchmark.json"
        if not p.exists():
            continue
        g = json.loads(p.read_text())["game_runs"]
        assert len(g) == 25, d
        runs.append((ai, d, sum(x["levels_completed"] for x in g), st.mean(x["final_score"] for x in g)))
with open(HERE / "data/gate_runs.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["arm", "run_dir", "levels_25", "score_pct"])
    for ai, d, lv, sc in runs:
        w.writerow([ARMS[ai][0], d, lv, f"{sc:.2f}"])

L = [[r[2] for r in runs if r[0] == i] for i in range(len(ARMS))]
S = [[r[3] for r in runs if r[0] == i] for i in range(len(ARMS))]
N = [len(x) for x in L]
mean_l = [st.mean(x) for x in L]
mean_s = [st.mean(x) for x in S]

# ---- checks against the run records and the paper's noise numbers ---------------------------------------
noise = json.loads((V2OUT / "e4_noise.json").read_text())
SD_L = noise["franzen_levels"]["sd"]
SD_S = noise["franzen_score"]["sd"]
assert round(SD_L, 2) == 8.25 and round(SD_S, 2) == 4.98
assert sorted(L[0]) == sorted([129, 120, 125, 122, 135, 123, 110]) and round(mean_l[0], 1) == 123.4
assert sorted(L[1]) == [118, 126, 141] and round(mean_l[1], 1) == 128.3
assert sorted(L[2]) == [127, 128, 132] and round(mean_l[2], 1) == 129.0
assert sorted(L[3]) == [111, 122, 130]
assert sorted(L[4]) == [115, 123, 129] and round(mean_l[4], 1) == 122.3
assert sorted(L[5]) == [124, 128, 128] and round(mean_l[5], 1) == 126.7
print("n per arm", N, "mean levels", [round(m, 1) for m in mean_l], "mean score", [round(m, 2) for m in mean_s])
print("control SD of its own 7 runs", round(st.stdev(L[0]), 2), "score", round(st.stdev(S[0]), 2))

# ---- figure -----------------------------------------------------------------------------------------------
apply(scale=SCALE)
fig = plt.figure(figsize=(W, 4.2))
axA = fig.add_axes([0.105, 0.405, 0.885, 0.50])
axB = fig.add_axes([0.105, 0.175, 0.885, 0.175])    # same arms, same order, no second set of names
INK, GRAY, ACC = C["ink"], C["gray"], C["accent"]
labels = []


def draw(ax, vals, means, sd, ylim, ytick, ylabel, names):
    base = means[0]
    ax.axhspan(base - sd, base + sd, color=C["faint"], zorder=0, lw=0)
    ax.axhline(base, color=GRAY, lw=1.0, ls=(0, (4, 2)), zorder=1)
    for i, v in enumerate(vals):
        n = len(v)
        offs = [0.0] if n == 1 else [(-0.5 + k / (n - 1)) * min(0.42, 0.11 * (n - 1) + 0.06) * 2 for k in range(n)]
        # fixed, run-order jitter (no random numbers): a sorted spread, alternated so neighbours do not touch
        order = sorted(range(n), key=lambda k: v[k])
        xs = [0.0] * n
        for rank, k in enumerate(order):
            xs[k] = offs[(rank * 3) % n if n > 1 else 0]
        col = GRAY if i == 0 else ACC
        ax.plot([i + x for x in xs], v, "o", ms=4.8, mfc=col, mec="white", mew=0.5, ls="", zorder=3, alpha=0.95)
        ax.plot([i - 0.36, i + 0.36], [means[i]] * 2, color=INK, lw=1.8, solid_capstyle="butt", zorder=4)
    ax.set_xlim(-0.6, len(vals) - 0.4)
    ax.set_ylim(*ylim)
    ax.set_yticks(ytick)
    ax.grid(axis="y", visible=True)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(names, linespacing=1.05, fontsize=fs(6.6))
    ax.tick_params(axis="x", length=0, pad=3)
    ax.set_ylabel(ylabel, labelpad=2)


names = [f"{ARMS[i][0]}\nn = {N[i]}" for i in range(len(ARMS))]
draw(axA, L, mean_l, SD_L, (100, 150), [100, 110, 120, 130, 140], "levels cleared per run", names)
axA.set_xticklabels([])
for i in range(len(ARMS)):
    d = mean_l[i] - mean_l[0]
    txt = f"mean {mean_l[i]:.1f}" + ("" if i == 0 else f"\n({d:+.1f})")
    labels.append(axA.text(i, 148.6, txt, ha="center", va="top", fontsize=fs(6.2), color=INK, linespacing=1.05))
labels.append(axA.text(-0.55, 100.9, f"shaded: control mean {chr(177)}1 SD of identical runs ({SD_L:.2f}); dashed: control mean;\n"
                       "in brackets: mean minus control mean; 25 demonstration games",
                       ha="left", va="bottom", fontsize=fs(6.4), color=C["gray_text"], linespacing=1.05))
panel(fig, 0.005, 0.995, "(a)", "Levels cleared per run, by harness change")

draw(axB, S, mean_s, SD_S, (44, 70), [45, 55, 65], "game score\n(percent)", names)
axB.tick_params(axis="x", pad=3)
labels.append(axB.text(len(ARMS) - 0.45, 44.8, f"shaded: {chr(177)}1 SD ({SD_S:.2f})", ha="right", va="bottom", fontsize=fs(6.4),
                       color=C["gray_text"]))
panel(fig, 0.005, 0.383, "(b)", "Mean game score per run")

check_no_overlap(fig, labels)
save(fig, "fig_gates", writeup=True)
