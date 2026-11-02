#!/usr/bin/env python3
"""F4 "The agent sits near its token ceiling": the budget curve, its run-bootstrap band, the operating point and the
+11% serving gain drawn as a slide along the curve.

Run from the repo root (after budget_band.py has written data/budget_band.json):
  uv run --no-project --with matplotlib --with numpy python -I -B paper/v3/analysis/figs/fig_budget.py

Reads
  paper/v2/analysis/out/f1_curves_n2.json      a_eight_run.S_of_rho: the curve of v2 Fig. 1(a) (480 seeds, 890 tok/s)
  paper/v2/analysis/out/new_plateau_pool.json  curve.pin.8 (plateau P, S(1)) and tokens_plus11.pin.8 (+11% in levels)
  paper/v2/analysis/out/e4_noise.json          franzen_score.sd: SD of one 25-game run of identical code, in points
  paper/v3/analysis/figs/data/budget_band.json run-level bootstrap of the same curve (budget_band.py)
Simulated from recorded level attempts on the 25 demonstration games, in the hidden-run regime (110 games, 14 streams,
31,920 s); "at this budget". The v2 panel (b) (two three-run pools) and the constructed 2.5x world go to the appendix.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator  # noqa: E402

from figstyle import C, V2OUT, W, apply, check_no_overlap, save  # noqa: E402

f1 = json.loads((V2OUT / "f1_curves_n2.json").read_text())
npp = json.loads((V2OUT / "new_plateau_pool.json").read_text())
SD_RUN = json.loads((V2OUT / "e4_noise.json").read_text())["franzen_score"]["sd"]
band = json.loads((HERE / "data/budget_band.json").read_text())
cur = f1["a_eight_run"]["S_of_rho"]
R = np.array(sorted(map(float, cur)))
S = np.array([cur[k] for k in sorted(cur, key=float)])
P = npp["curve"]["pin"]["8"]["P"]["mean"]
S1 = npp["curve"]["pin"]["8"]["S1"]["mean"]
LEV = npp["tokens_plus11"]["pin"]["8"]["gain"]
LEV_ALL = npp["tokens_plus11"]["pin"]["8"]["levels"]
LIM = npp["limit_estimates"]                               # the ceiling extrapolated to more pooled runs (claim C2)
CEIL_LO, CEIL_HI = LIM["low"], LIM["high"]
SHARE_LO, SHARE_HI = LIM["share_realized"]["low"], LIM["share_realized"]["high"]
at = lambda x: float(np.interp(np.log(x), np.log(R), S))  # noqa: E731
GAIN = at(1.11) - at(1.0)
B = np.array(band["boot_S"])                               # resamples x rhos
assert band["rhos"] == R.tolist() and B.shape == (band["B"], len(R))

# ---- checks: the numbers v2 and FRAMING print ------------------------------------------------------------------------
assert (round(P, 1), round(S1, 1), round(S1 / P, 2)) == (60.2, 56.7, 0.94)
assert abs(at(1.0) - S1) < 1e-6 and round(GAIN, 1) == 1.4 and round(LEV) == 4 and round(SD_RUN, 1) == 5.0
assert (round(SHARE_LO, 2), round(S1 / P, 2)) == (0.85, 0.94)        # C2: "85 to 94%, so we say about 90%"
assert (round(CEIL_LO, 1), round(CEIL_HI, 1)) == (62.1, 66.4)
q = {p: np.percentile(B, p, axis=0) for p in (2.5, 10, 25, 75, 90, 97.5)}
i1 = R.tolist().index(1.0)
print(f"P {P:.2f}  S(1) {S1:.2f} ({S1 / P:.1%})  +11% tokens: +{GAIN:.2f} points, {LEV:.2f} of {LEV_ALL:.1f} levels; "
      f"SD of one run {SD_RUN:.2f} points")
print(f"bootstrap B={band['B']} seeds={band['seeds']}: S(1) 95% {q[2.5][i1]:.1f} to {q[97.5][i1]:.1f}, 50% {q[25][i1]:.1f} to "
      f"{q[75][i1]:.1f}; full pool at {band['seeds']} seeds {band['full_pool_S'][i1]:.2f}; distinct runs per resample "
      f"mean {np.mean(band['boot_distinct_runs']):.2f}")
print("band at rho:", {r: (round(q[2.5][j], 1), round(q[97.5][j], 1)) for j, r in enumerate(R)})

# ---- figure ------------------------------------------------------------------------------------------------------
apply()
H = 2.7
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0.10, 0.16, 0.60, 0.80])
INK, GRAY, ACC = C["ink"], C["gray"], C["accent"]
for (lo, hi), shade in (((2.5, 97.5), "#EDEDED"), ((10, 90), "#DCDCDC"), ((25, 75), "#C9C9C9")):
    ax.fill_between(R, q[lo], q[hi], color=shade, lw=0, zorder=1)
ax.axhspan(CEIL_LO, CEIL_HI, color=C["faint"], hatch="////", ec="#C8C8C8", lw=0, zorder=0.5)
ax.axhline(P, color=GRAY, lw=0.9, ls=(0, (4, 2)), zorder=2)
ax.plot(R, S, color=INK, lw=1.6, marker="o", ms=2.4, zorder=3)
ax.plot(1.0, S1, "o", ms=6.5, mfc=ACC, mec="white", mew=0.9, zorder=6)
ax.add_patch(FancyArrowPatch((1.0, S1), (1.11, at(1.11)), arrowstyle="-|>", mutation_scale=8, color=ACC, lw=1.6,
                             shrinkA=4, shrinkB=0, zorder=5))
labels = [
    ax.text(0.26, P - 0.8, "plateau of these 8 runs", fontsize=7, color=C["gray_text"], va="top", ha="left"),
    ax.text(0.26, (CEIL_LO + CEIL_HI) / 2, "ceiling with more runs pooled (extrapolated)", fontsize=7, color=C["gray_text"],
            va="center", ha="left", bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none")),
]
# one run's SD of identical code, as a scale bar
xr, y0 = 2.55, 32.0
ax.plot([xr, xr], [y0, y0 + SD_RUN], color=INK, lw=1.6, solid_capstyle="butt", zorder=4)
for yy in (y0, y0 + SD_RUN):
    ax.plot([xr * 0.95, xr * 1.05], [yy, yy], color=INK, lw=0.9, zorder=4)
labels.append(ax.text(xr * 0.9, y0 + SD_RUN / 2, "SD of one run of\nidentical code", fontsize=7, ha="right", va="center",
                      linespacing=1.05))
ax.set_xscale("log")
ax.set_xlim(0.24, 3.1)
ax.set_ylim(15, 69)
ax.xaxis.set_major_locator(FixedLocator([0.25, 0.5, 1, 2, 3]))
ax.xaxis.set_major_formatter(FixedFormatter(["0.25", "0.5", "1", "2", "3"]))
ax.xaxis.set_minor_locator(NullLocator())
ax.set_yticks([20, 30, 40, 50, 60])
ax.grid(axis="y", visible=True)
ax.set_xlabel(r"token budget relative to today's serving speed, $\rho$ (log scale)", labelpad=2)
ax.set_ylabel("simulated score (points of 100)", labelpad=2)

kx = 0.73
fig.add_artist(plt.Line2D([kx + 0.017], [0.93], marker="o", ms=6, mfc=ACC, mec="white", mew=0.8, transform=fig.transFigure))
fig.text(kx + 0.045, 0.93, "the agent today: about\n90% of the extrapolated\nceiling", fontsize=7.5, color=ACC, fontweight="bold", va="center",
         linespacing=1.05)
fig.add_artist(FancyArrowPatch((kx, 0.815), (kx + 0.035, 0.815), transform=fig.transFigure, arrowstyle="-|>",
                               mutation_scale=8, color=ACC, lw=1.6))
fig.text(kx + 0.045, 0.815, f"+11% tokens: about {LEV:.0f}\nlevels per run (+{GAIN:.1f} points)", fontsize=7.5, color=INK,
         va="center", linespacing=1.05)
fig.add_artist(plt.Line2D([kx, kx + 0.035], [0.705, 0.705], transform=fig.transFigure, color=INK, lw=1.6))
fig.text(kx + 0.045, 0.705, "all 8 runs pooled\n(480 seeds)", fontsize=7, va="center", linespacing=1.05)
fig.text(kx, 0.625, f"band: {band['B']} resamples of\nthe 8 runs, {band['seeds']} seeds each", fontsize=7, va="top",
         linespacing=1.05)
for i, (lab, shade) in enumerate((("middle 50%", "#C9C9C9"), ("middle 80%", "#DCDCDC"), ("middle 95%", "#EDEDED"))):
    y = 0.50 - i * 0.07
    fig.add_artist(plt.Rectangle((kx, y - 0.02), 0.035, 0.04, transform=fig.transFigure, fc=shade, ec="none"))
    fig.text(kx + 0.045, y, lab, fontsize=7, va="center")
fig.text(kx, 0.29, "110 games, 14 streams,\n31,920 s, as on the hidden\nrun; attempts from the 25\ndemonstration games",
         fontsize=7, va="top", color=C["gray_text"], linespacing=1.1)
check_no_overlap(fig, labels)
save(fig, "fig_budget")
