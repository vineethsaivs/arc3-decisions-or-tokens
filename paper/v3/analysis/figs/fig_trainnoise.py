#!/usr/bin/env python3
"""F9 (appendix) "The training forward pass is not deterministic": raw against clipped k3 for identical base passes.

Run from the repo root:
  uv run --no-project --with matplotlib --with numpy python -I -B paper/v3/analysis/figs/fig_trainnoise.py

Reads paper/v3/evidence/data/grpo_base_noise_pairs.csv (V3-G5): five base-only passes over the same 24 val windows
(66,949 tokens; three VM processes, two Vertex hosts), all 10 pairs, k3 read in both directions (20 readings each).
Lines: the predeclared stop (raw k3 above 0.02, GATES.md) and the amended stop (clipped k3 more than 0.02 above the
run's own step-0 floor; amendment 2, V3-G7), drawn at the mean floor of these readings.
"""
import csv
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator  # noqa: E402

from figstyle import C, DATA, W, apply, check_no_overlap, panel, save  # noqa: E402

rows = list(csv.DictReader(open(DATA / "grpo_base_noise_pairs.csv")))


def kind(pair):
    a, b = pair.split(" vs ")
    return "-".join(sorted({"vm" if "_vm_" in x else "vertex" for x in (a, b)}, key=lambda s: s != "vm")) if \
        ("_vm_" in a) != ("_vm_" in b) else ("vm" if "_vm_" in a else "vertex")


KINDS = {"vm": "VM with VM", "vm-vertex": "VM with Vertex", "vertex": "Vertex with Vertex"}
order = sorted(rows, key=lambda r: (list(KINDS).index(kind(r["pair"])), r["pair"]))
raw = [float(r[k]) for r in rows for k in ("k3_raw", "k3_raw_rev")]
clip = [float(r[k]) for r in rows for k in ("k3_clip2", "k3_clip2_rev")]
FLOOR = st.mean(clip)
STOP_RAW, MARGIN = 0.02, 0.02

assert len(rows) == 10 and [sum(kind(r["pair"]) == k for r in rows) for k in KINDS] == [3, 6, 1]
assert (round(min(raw), 3), round(max(raw))) == (0.099, 1525)                       # "0.099 to 1,525"
assert (round(min(clip), 4), round(max(clip), 4)) == (0.0629, 0.0658)             # "0.0629 to 0.0658"
assert round(FLOOR, 4) == 0.0644 and round(st.stdev(clip), 4) == 0.0008             # "mean 0.0644, SD 0.0008, n = 20"
assert min(float(r["exact_equal"]) for r in rows) >= 0.121 and max(float(r["exact_equal"]) for r in rows) <= 0.124
print(f"raw k3 {min(raw):.3f} to {max(raw):.1f}; clipped {min(clip):.4f} to {max(clip):.4f}, mean {FLOOR:.4f}, SD {st.stdev(clip):.4f}")

# ---- figure ------------------------------------------------------------------------------------------------------
apply()
H = 2.55
fig = plt.figure(figsize=(W, H))
axA = fig.add_axes([0.125, 0.19, 0.53, 0.70])
axB = fig.add_axes([0.80, 0.19, 0.18, 0.70])
INK, GRAY, ACC = C["ink"], C["gray"], C["accent"]
MK = {"vm": "o", "vm-vertex": "s", "vertex": "D"}
xs = list(range(len(order)))
labels = []

# (a) raw and clipped k3 for every pair, log axis
for x, r in zip(xs, order):
    m = MK[kind(r["pair"])]
    for k, dx in (("k3_raw", -0.14), ("k3_raw_rev", 0.14)):
        axA.plot(x + dx, float(r[k]), m, ms=4.0, mfc="white", mec=INK, mew=0.8, zorder=3)
    for k, dx in (("k3_clip2", -0.14), ("k3_clip2_rev", 0.14)):
        axA.plot(x + dx, float(r[k]), m, ms=3.8, mfc=ACC, mec=ACC, mew=0.6, zorder=4)
axA.axhline(STOP_RAW, color=GRAY, lw=1.0, ls=(0, (4, 2)), zorder=2)
labels.append(axA.text(9.5, STOP_RAW * 0.82, "predeclared stop: k3 above 0.02", fontsize=7, ha="right", va="top",
                       color=C["gray_text"]))
for b in (2.5, 8.5):
    axA.axvline(b, color=C["light"], lw=0.8, zorder=1)
for k, (a, b) in zip(KINDS, ((0, 2), (3, 8), (9, 9))):
    labels.append(axA.text((a + b) / 2, 3000, KINDS[k].replace(" with ", "\nwith "), fontsize=7, ha="center", va="top",
                           color=C["gray_text"], linespacing=1.0))
axA.set_yscale("log")
axA.set_ylim(0.005, 4000)
axA.yaxis.set_major_locator(FixedLocator([0.01, 0.1, 1, 10, 100, 1000]))
axA.yaxis.set_major_formatter(plt.FixedFormatter(["0.01", "0.1", "1", "10", "100", "1,000"]))
axA.yaxis.set_minor_locator(NullLocator())
axA.set_xlim(-0.6, 9.6)
axA.set_xticks(xs)
axA.set_xticklabels([str(i + 1) for i in xs])
axA.tick_params(axis="x", length=0)
axA.set_xlabel("pair of identical base-only passes (each read both ways)", labelpad=3)
axA.set_ylabel("val k3 between identical passes\n(log scale)", labelpad=2)
axA.grid(axis="y", visible=True)
leg = [plt.Line2D([], [], marker="o", ls="", ms=4.0, mfc="white", mec=INK, mew=0.8),
       plt.Line2D([], [], marker="o", ls="", ms=3.8, mfc=ACC, mec=ACC)]
axA.legend(leg, ["raw k3", "clipped k3 (log ratio\nclamped to [-2, 2])"], loc="upper left", bbox_to_anchor=(0.0, 0.80),
           handletextpad=0.3, labelspacing=0.4, borderaxespad=0.0, frameon=True, facecolor="white", edgecolor="none",
           framealpha=1.0)
panel(fig, 0.005, 0.995, "(a)", "Raw k3 spans four decades; clipped k3 does not")

# (b) the clipped readings on a linear axis, against the amended stop
cx = {"vm": -0.3, "vm-vertex": 0.0, "vertex": 0.3}
for kd in KINDS:
    vals = [float(r[k]) for r in order if kind(r["pair"]) == kd for k in ("k3_clip2", "k3_clip2_rev")]
    offs = [0.0] if len(vals) == 1 else [-0.1 + 0.2 * i / (len(vals) - 1) for i in range(len(vals))]
    for o, v in zip(offs, vals):
        axB.plot(cx[kd] + o, v, MK[kd], ms=3.2, mfc=ACC, mec="white", mew=0.4, zorder=3)
axB.axhline(FLOOR, color=INK, lw=0.8, ls=(0, (1, 1.5)), zorder=2)
axB.axhline(FLOOR + MARGIN, color=ACC, lw=1.0, ls=(0, (4, 2)), zorder=2)
labels.append(axB.text(-0.45, FLOOR + MARGIN + 0.0008, "amended stop:\nfloor + 0.02", fontsize=7, ha="left", va="bottom",
                       color=ACC, linespacing=1.0))
labels.append(axB.text(-0.45, FLOOR + 0.004, f"floor {FLOOR:.4f}\nSD {st.stdev(clip):.4f}", fontsize=7, ha="left",
                       va="bottom", color=INK, linespacing=1.0))
axB.set_xlim(-0.5, 0.5)
axB.set_xticks([cx[k] for k in KINDS])
axB.set_xticklabels(["both\nVM", "mixed", "both\nVertex"], fontsize=6.5, linespacing=0.95)
axB.tick_params(axis="x", length=0)
axB.set_ylim(0.055, 0.095)
axB.set_yticks([0.06, 0.07, 0.08, 0.09])
axB.yaxis.set_major_formatter(plt.FixedFormatter(["0.06", "0.07", "0.08", "0.09"]))
axB.set_ylabel("clipped k3 (linear)", labelpad=2)
axB.grid(axis="y", visible=True)
panel(fig, 0.70, 0.995, "(b)", "Clipped, zoomed")
check_no_overlap(fig, labels)
save(fig, "fig_trainnoise")
