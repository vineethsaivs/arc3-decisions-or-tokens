#!/usr/bin/env python3
"""F3 "Two runs per arm resolve about 23 levels": detectable shift against runs per arm, with our levers placed on it.

Run from the repo root:
  uv run --no-project --with matplotlib --with numpy --with scipy python -I -B paper/v3/analysis/figs/fig_power.py

Detectable shift (two-sided 5%, 80% power, two arms of n runs): d(n) = (z_.975 + z_.80) * SD * sqrt(2 / n) (FRAMING B4).
Reads
  paper/v2/analysis/out/e4_noise.json        SD 8.25 from six identical pairs (6 df) and its 95% CI 5.32 to 18.17 (B1)
  paper/v3/evidence/data/mde_by_runs.csv     the dossier's MDE table (V3-X12), asserted equal to the curve
  paper/v3/evidence/data/vertex_vs_kaggle_runs.csv   eight concurrent identical runs, SD 5.95 (V3-X3)
  paper/v2/analysis/out/yield_stats.json     yield 8,192 minus 2,048: +12.5 levels (D1)
  paper/v2/analysis/out/e5_detect.json       +11% serving tokens priced at 4.19 levels by the budget curve (C4)
  paper/v3/evidence/data/lora_replicas.csv   adapter runs 109, 115, 117 against the pin mean 124.0 (V3-L9, V3-L10)
Each lever is drawn where its size meets the SD 8.25 curve: the runs per arm a two-arm test would need to see it.
"""
import csv
import json
import math
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator  # noqa: E402
from scipy.stats import chi2, norm  # noqa: E402

from figstyle import C, DATA, V2OUT, W, apply, check_no_overlap, panel, save  # noqa: E402

noise = json.loads((V2OUT / "e4_noise.json").read_text())
SD, DF = noise["franzen_levels"]["sd"], noise["franzen_levels"]["df"]
ZSUM = norm.ppf(0.975) + norm.ppf(0.80)


def mde(n, sd):
    return ZSUM * sd * np.sqrt(2.0 / np.asarray(n, dtype=float))


def runs_for(d, sd):
    return 2.0 * (ZSUM * sd / d) ** 2


def sd_ci(level):
    a = (1 - level) / 2
    return SD * math.sqrt(DF / chi2.ppf(1 - a, DF)), SD * math.sqrt(DF / chi2.ppf(a, DF))


vtx = [int(r["levels_25"]) for r in csv.DictReader(open(DATA / "vertex_vs_kaggle_runs.csv")) if r["run"].startswith("vertex-rq1020-")]
SD_VTX = st.stdev(vtx)
YIELD = json.loads((V2OUT / "yield_stats.json").read_text())["tests"]["Y8 all (9) vs all 2048"]["diff"]
det = json.loads((V2OUT / "e5_detect.json").read_text())["D3"]["within-pair sd"]["rows"]
SERV = next(r["expected_levels"] for r in det if r["token_gain"] == 0.11)
rep = list(csv.DictReader(open(DATA / "lora_replicas.csv")))
PIN = st.mean(int(r["levels"]) for r in rep if r["group"] == "pin_y8_family")
LORA = sorted(PIN - int(r["levels"]) for r in rep if r["group"] == "lora_r16")       # losses, positive numbers

# ---- checks --------------------------------------------------------------------------------------------------
lo95, hi95 = sd_ci(0.95)
assert abs(lo95 - noise["franzen_levels"]["ci95"][0]) < 1e-6 and abs(hi95 - noise["franzen_levels"]["ci95"][1]) < 1e-6
row = next(r for r in csv.DictReader(open(DATA / "mde_by_runs.csv")) if r["sd"] == "8.25")
for n in (1, 2, 4, 6, 8, 12):
    assert abs(float(mde(n, SD)) - float(row[f"mde_n{n}"])) < 0.01, n
assert round(float(mde(2, SD)), 1) == 23.1 and round(float(mde(12, SD)), 1) == 9.4
assert round(SD_VTX, 2) == 5.95 and round(YIELD, 1) == 12.5 and round(SERV, 2) == 4.19
assert PIN == 124.0 and LORA == [7, 9, 15]
print(f"SD {SD:.2f} (95% {lo95:.2f} to {hi95:.2f}); runs needed at SD {SD:.2f}: serving {runs_for(SERV, SD):.1f}, "
      f"yield {runs_for(YIELD, SD):.1f}, LoRA {runs_for(LORA[-1], SD):.1f} to {runs_for(LORA[0], SD):.1f}")

# ---- figure -----------------------------------------------------------------------------------------------------
apply()
H = 2.7
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0.105, 0.155, 0.60, 0.815])
INK, GRAY, ACC = C["ink"], C["gray"], C["accent"]
BANDS = ((0.95, "#EDEDED"), (0.80, "#DDDDDD"), (0.50, "#CACACA"))
ns = np.logspace(0, 2, 200)
for level, shade in BANDS:
    lo, hi = sd_ci(level)
    ax.fill_between(ns, mde(ns, lo), mde(ns, hi), color=shade, lw=0, zorder=1)
ax.plot(ns, mde(ns, SD), color=INK, lw=1.6, zorder=3)
ax.plot(ns, mde(ns, SD_VTX), color=INK, lw=0.9, ls=(0, (3, 2)), zorder=3)
labels = []
HALO = dict(boxstyle="square,pad=0.1", fc="white", ec="none")

# the two bench designs: dots on the SD 8.25 line, labels at the top of the guide lines
for n, name in ((2, "2 vs 2"), (12, "12 vs 12")):
    ax.axvline(n, color=GRAY, lw=0.6, ls=(0, (2, 2)), zorder=2)
    y = float(mde(n, SD))
    ax.plot(n, y, "o", ms=3.8, color=INK, zorder=5)
    labels.append(ax.text(n * 1.06, 92, f"{name}:\nsees {y:.1f}", fontsize=7, ha="left", va="top", color=INK,
                          linespacing=1.05, bbox=HALO, zorder=7))

# levers, each where its size meets the SD 8.25 line; labels sit off the band with a leader line
nL = [runs_for(d, SD) for d in LORA]
seg = (ns >= min(nL)) & (ns <= max(nL))
ax.plot(ns[seg], mde(ns[seg], SD), color=ACC, lw=4.0, alpha=0.4, solid_capstyle="round", zorder=4)
ax.plot(nL, LORA, "v", ms=5, mfc="white", mec=ACC, mew=1.2, zorder=6)
nY, nS = runs_for(YIELD, SD), runs_for(SERV, SD)
ax.plot(nY, YIELD, "o", ms=6, mfc=ACC, mec=ACC, zorder=6)
ax.plot(nS, SERV, "s", ms=5, mfc=GRAY, mec=INK, mew=0.7, zorder=6)


def callout(text, xy, xytext, color, ha):
    t = ax.annotate(text, xy=xy, xytext=xytext, fontsize=7.5, ha=ha, va="center", color=color, fontweight="bold",
                    linespacing=1.05, arrowprops=dict(arrowstyle="-", color=color, lw=0.7, shrinkA=1, shrinkB=4), zorder=7)
    labels.append(t)


callout(f"yield change: +{YIELD:.1f}\nneeds about {nY:.0f} runs per arm", (nY, YIELD), (1.15, 48), ACC, "left")
callout(f"fine-tuning: -{LORA[0]} to -{LORA[-1]}\nneeds {nL[-1]:.0f} to {nL[0]:.0f}", (nL[1], LORA[1]), (1.15, 4.2), ACC, "left")
callout(f"+11% serving tokens: +{SERV:.1f}\nneeds about {nS:.0f}", (nS, SERV), (95, 26), INK, "right")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(1, 100)
ax.set_ylim(2, 100)
ax.xaxis.set_major_locator(FixedLocator([1, 2, 5, 12, 20, 50, 100]))
ax.xaxis.set_major_formatter(plt.FixedFormatter(["1", "2", "5", "12", "20", "50", "100"]))
ax.yaxis.set_major_locator(FixedLocator([2, 5, 10, 20, 50, 100]))
ax.yaxis.set_major_formatter(plt.FixedFormatter(["2", "5", "10", "20", "50", "100"]))
ax.xaxis.set_minor_locator(NullLocator())
ax.yaxis.set_minor_locator(NullLocator())
ax.set_xlabel("runs per arm (log scale)", labelpad=2)
ax.set_ylabel("smallest level shift the test sees\n(80% power, two-sided 5%; log scale)", labelpad=2)

# key, right of the axes: the two lines, then the graded band
kx = 0.735
fig.text(kx, 0.955, "run-to-run SD of levels", fontsize=7, va="top", ha="left", fontweight="bold")
for i, (lab, ls, lw) in enumerate(((f"{SD:.2f}: six identical\npairs (used for markers)", "-", 1.6),
                                    (f"{SD_VTX:.2f}: eight concurrent\nruns on one day", (0, (3, 2)), 0.9))):
    y = 0.87 - i * 0.12
    fig.add_artist(plt.Line2D([kx, kx + 0.035], [y, y], transform=fig.transFigure, color=INK, ls=ls, lw=lw))
    fig.text(kx + 0.045, y, lab, fontsize=7, va="center", ha="left", linespacing=1.05)
fig.text(kx, 0.66, "uncertainty in the SD 8.25\n(chi-square, 6 df)", fontsize=7, va="top", ha="left", fontweight="bold",
         linespacing=1.05)
for i, (level, shade) in enumerate(BANDS[::-1]):
    lo, hi = sd_ci(level)
    y = 0.545 - i * 0.075
    fig.add_artist(plt.Rectangle((kx, y - 0.022), 0.035, 0.044, transform=fig.transFigure, fc=shade, ec="none"))
    fig.text(kx + 0.045, y, f"{level:.0%}: SD {lo:.1f} to {hi:.1f}", fontsize=7, va="center", ha="left")
fig.text(kx, 0.29, "markers: each effect placed\nwhere it meets the SD 8.25\nline; its x is the runs per\narm a test needs to see it",
         fontsize=7, va="top", ha="left", linespacing=1.1, color=INK)
check_no_overlap(fig, labels)
save(fig, "fig_power")
