#!/usr/bin/env python3
"""F8 (appendix) "How much of 40.98 is selection": the family's draws, its running best, and the expected best of n draws.

Run from the repo root:
  uv run --no-project --with matplotlib --with numpy python -I -B paper/v3/analysis/figs/fig_selection.py

Reads paper/v3/evidence/data/board_draws.csv. Family A (finals rule v4, V3-B9): the shipped config, its lossless
draft-map variant and its two boot-keepalive variants (r3kp, kp12); valid draws only (the 10-18 0.00 is excluded, V3-B7),
n = 12 to 2026-10-28, mean 34.59 (V3-W16; n = 7, mean 34.27 on 10-21, V3-B5). The family's record closes on 10-28: from
10-29 the pin is a different agent (kp12 plus two harness overlays).
Expected maximum of n i.i.d. draws from the empirical distribution of the N = 12 valid draws (Dodge et al. 2019, "Show
Your Work", expected validation performance): with v_(1) <= ... <= v_(N),
    E[max_n] = sum_i v_(i) * ((i / N)^n - ((i - 1) / N)^n),
checked here against a simulation (200,000 sets of n draws with replacement per n).
10th to 90th percentiles of max_n under the same distribution, P(max_n <= v_(i)) = (i / N)^n. A normal model (the
draws' mean and SD) is drawn for comparison: unlike the empirical form it can exceed the observed best.
"""
import csv
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from figstyle import C, DATA, W, apply, check_no_overlap, panel, save  # noqa: E402

FAMILY = ("franzen-l1a1y8sv2r3p", "franzen-l1a1y8sv2r3hm32p", "franzen-l1a1y8sv2r3kp", "franzen-l1a1y8sv2r3kp12")
rows = [r for r in csv.DictReader(open(DATA / "board_draws.csv")) if r["config"] in FAMILY]
rows = sorted((r for r in rows if r["status"] == "complete" and float(r["public_score"]) > 0), key=lambda r: r["submitted_utc"])
D = [float(r["public_score"]) for r in rows]                        # date order
V = sorted(D)
N = len(V)
NMAX = 23                                                           # the live draws since the fork, to 10-28
ns = np.arange(1, NMAX + 1)


def dodge(n):
    return sum(v * ((i / N) ** n - ((i - 1) / N) ** n) for i, v in enumerate(V, start=1))


E = np.array([dodge(n) for n in ns])
rng = np.random.default_rng(20261021)
SIM = np.array([rng.choice(V, size=(200_000, n)).max(axis=1).mean() for n in ns])
err = float(np.max(np.abs(SIM - E)))
mu, sd = st.mean(V), st.stdev(V)
z = rng.standard_normal((200_000, NMAX))
NORM = np.array([mu + sd * z[:, :n].max(axis=1).mean() for n in ns])

assert N == 12 and round(mu, 2) == 34.59 and max(V) == 40.98 and D[0] == 40.09 and D[-1] == 33.45
assert abs(E[0] - mu) < 1e-9 and err < 0.03, err
print(f"N {N}, mean {mu:.2f}, SD {sd:.2f}; Dodge E[max_n] n=1..{NMAX}: {np.round(E, 2).tolist()}")
print(f"simulation check: max |sim - closed form| = {err:.4f} over n = 1..{NMAX} (200,000 sets each)")
print(f"E[max_7] = {E[6]:.2f} (normal {NORM[6]:.2f}); E[max_12] = {E[11]:.2f} (normal {NORM[11]:.2f}); "
      f"E[max_23] = {E[22]:.2f} (normal {NORM[22]:.2f})")

# ---- figure -----------------------------------------------------------------------------------------------
apply()
H = 2.3
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0.085, 0.17, 0.895, 0.79])
INK, GRAY, ACC = C["ink"], C["gray"], C["accent"]
labels = []
ax.plot(ns, E, color=INK, lw=1.6, marker="o", ms=2.6, zorder=3)
ax.plot(ns, NORM, color=INK, lw=0.9, ls=(0, (3, 2)), zorder=3)
k = np.arange(1, N + 1)
ax.plot(k, D, "o", ms=4.0, mfc=C["light"], mec=INK, mew=0.6, zorder=4)
ax.step(k, np.maximum.accumulate(D), where="post", color=ACC, lw=1.2, zorder=4)
KB = int(np.argmax(D)) + 1                                          # the best draw, in date order
ax.plot(KB, D[KB - 1], "o", ms=5.0, mfc=ACC, mec=ACC, zorder=5)
ax.axhline(mu, color=GRAY, lw=0.7, ls=(0, (1, 2)), zorder=1)
xb = N + 0.35
ax.annotate("", xy=(xb, E[N - 1]), xytext=(xb, mu), arrowprops=dict(arrowstyle="<->", color=INK, lw=0.9, shrinkA=0, shrinkB=0,
            mutation_scale=6), zorder=5)
labels.append(ax.text(xb + 0.2, (mu + E[N - 1]) / 2, f"+{E[N - 1] - mu:.1f}: the expected\nbest of {N} over the mean",
                      fontsize=7, ha="left", va="center", color=INK, linespacing=1.05))
labels.append(ax.text(KB - 0.25, D[KB - 1] + 0.35, f"40.98: best of {N}", color=ACC, fontsize=7.5, fontweight="bold", ha="right", va="bottom"))
labels.append(ax.text(22.9, E[-1] - 0.45, "expected best of n:\nresampled draws", fontsize=7, ha="right", va="top", color=INK,
                      linespacing=1.05))
labels.append(ax.text(22.9, NORM[-1] + 0.35, "normal model", fontsize=7, ha="right", va="bottom", color=INK))
labels.append(ax.text(22.9, mu - 0.3, f"mean of the {N} draws, {mu:.2f}", fontsize=7, ha="right", va="top", color=C["gray_text"]))
ax.set_xlim(0.5, 23.5)
ax.set_xticks([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23])
ax.set_ylim(27, 45)
ax.set_yticks([28, 32, 36, 40, 44])
ax.grid(axis="y", visible=True)
ax.set_xlabel("number of draws, n (points: the family's draws in date order)", labelpad=2)
ax.set_ylabel("public score", labelpad=2)

leg = [plt.Line2D([], [], marker="o", ls="", ms=4.0, mfc=C["light"], mec=INK, mew=0.6),
       plt.Line2D([], [], color=ACC, lw=1.2)]
ax.legend(leg, ["one board draw", "best so far"], loc="lower right", bbox_to_anchor=(1.0, 0.02), handletextpad=0.4,
          labelspacing=0.3, borderaxespad=0.0)
check_no_overlap(fig, labels)
save(fig, "fig_selection")
