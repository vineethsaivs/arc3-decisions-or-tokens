#!/usr/bin/env python3
"""Writeup cover, "Which changes move the agent?": one schematic panel, four labels, no numbers (FRAMING_v3 s9).

Run from the repo root:
  uv run --no-project --with matplotlib --with numpy python -I -B paper/v3/analysis/figs/cover.py

Writes paper/v3/figs/cover_1200x675.png, cover_1600x900.png and cover.pdf, plus cover_thumb_200.png (the 1200 x 675
render scaled to 200 px tall) for the legibility check. Schematic only: the curve is a generic saturating shape, not data.
The arrows carry only questions or tested facts (FRAMING_v3 s9 to s11). The noise band is drawn at the measured ratio: its
half-width (one run SD, 8.25 levels) against the vertical arrows (the across-day yield gap, +12.5 levels), so the lift is
not drawn larger than the noise it sits in. The up arrow is dashed: interleaving removed that gain (FRAMING_v3 s10).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402

from figstyle import C, OUT, apply, check_inside, check_no_overlap  # noqa: E402

apply()
FW, FH = 12.0, 6.75                       # inches: 1200 x 675 at 100 dpi, 1600 x 900 at 133.3 dpi
fig = plt.figure(figsize=(FW, FH))
ax = fig.add_axes([0.075, 0.10, 0.88, 0.66])
INK, ACC, BLUE = C["ink"], C["accent"], C["blue"]

x = np.linspace(0, 10, 400)
f = lambda t: 1 - np.exp(-t / 1.9)        # noqa: E731  (generic saturating curve)
y = f(x)
ARROW = 0.40                              # vertical arrow length = the +12.5-level yield gap
BAND = ARROW * 8.25 / 12.5                # one run SD of identical code, same scale
ax.fill_between(x, y - BAND, y + BAND, color="#E8E8E8", lw=0, zorder=1)
ax.plot(x, y, color=INK, lw=4.0, zorder=3, solid_capstyle="round")
x0 = 4.6
ax.plot(x0, f(x0), "o", ms=17, mfc="white", mec=INK, mew=3.5, zorder=5)
ax.set_xlim(0, 10.6)
ax.set_ylim(-0.08, 1.42)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_linewidth(2.2)
    ax.spines[s].set_color("#6A6A6A")
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel("tokens", fontsize=28, color="#4A4A4A", labelpad=6, loc="right")
ax.set_ylabel("score", fontsize=28, color="#4A4A4A", labelpad=8, loc="top")


def arrow(p, q, color, rad=0.0, lw=5.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>", mutation_scale=34, color=color,
                                 lw=lw, ls=ls, zorder=6, shrinkA=14, shrinkB=0))


x1 = 6.6
arrow((x0, f(x0) + 0.0), (x1, f(x1) + 0.0), "#6A6A6A", rad=-0.04, lw=4.5)
arrow((x0, f(x0)), (x0, f(x0) + ARROW), ACC, ls=(0, (2.2, 1.6)))
arrow((x0, f(x0)), (x0, f(x0) - ARROW), BLUE)
labels = [
    ax.text(x0 - 0.35, f(x0) + 0.24, "context re-send:\nno gain when\ninterleaved", fontsize=30, color=ACC,
            fontweight="bold", ha="right", va="center", linespacing=1.0),
    ax.text(x0 + 0.35, f(x0) - 0.30, "imitating easy wins:\nfewer levels,\nnot interleaved", fontsize=30, color=BLUE, fontweight="bold",
            ha="left", va="center", linespacing=1.0),
    ax.text(x1 + 0.10, f(x1) + 0.06, "faster serving:\nunresolved", fontsize=27, color="#4A4A4A", ha="left",
            va="bottom", linespacing=1.0),
    ax.text(0.35, 0.17, "noise of one run", fontsize=26, color="#6F6F6F", ha="left", va="top", fontstyle="italic"),
]
title = fig.text(0.075, 0.955, "Which changes move the agent?", fontsize=50, fontweight="bold", color=INK, ha="left",
                 va="top")
check_no_overlap(fig, labels + [title])
check_inside(fig)
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "cover_1200x675.png", dpi=100)
fig.savefig(OUT / "cover_1600x900.png", dpi=1600 / FW)
fig.savefig(OUT / "cover.pdf")
fig.savefig(OUT / "cover_thumb_200.png", dpi=200 / FH)
plt.close(fig)
print("wrote", OUT / "cover_1200x675.png", OUT / "cover_1600x900.png", OUT / "cover_thumb_200.png")
