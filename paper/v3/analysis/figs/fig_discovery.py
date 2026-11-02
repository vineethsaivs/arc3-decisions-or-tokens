#!/usr/bin/env python3
"""F15 "Where the clock goes, by level": a quarter of the cleared levels hold two thirds of their clock (a); on long levels
the code that wins first appears late (b); few levels check a model against the recorded history (c).

Run from the repo root (after python3 -I -B paper/v3/analysis/discovery_read.py):
  uv run --no-project --with matplotlib --with numpy python -I -B paper/v3/analysis/figs/fig_discovery.py

Reads paper/v3/analysis/figs/data/discovery_levels.csv (one row per level play in an analyzer transcript) and
discovery_summary.json, both written by discovery_read.py. Groups: baseline = 11 runs of the 15-stream harness family
before the replay check ("stall-check family"), ustv = 3 runs with the replay-check helpers, pre_unstick2 = 3 earlier runs
before the stall check. Every number drawn is recomputed from the CSV and asserted against the summary.
"""
import csv
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt  # noqa: E402

from figstyle import C, HERE, W, apply, check_no_overlap, panel, save  # noqa: E402

rows = list(csv.DictReader(open(HERE / "data/discovery_levels.csv")))
S = json.loads((HERE / "data/discovery_summary.json").read_text())
LONG = 900   # seconds: a long level, the summary's 15 minute cut


def cleared_l2(g):
    # 'cleared' is the summary's rule: the transcript segment is cleared (a later turn shows a higher level) and the level
    # is 2 or more. It misses a level cleared on the game's last turn, which reads 0 in the CSV.
    return [r for r in rows if r["group"] == g and r["cleared"] == "1" and r["level"] and int(r["level"]) >= 2]


def lorenz(g):
    """Levels sorted longest first: cumulative share of levels, cumulative share of their clock, and the 15 minute point."""
    d = sorted((int(r["seconds_on_level"]) for r in cleared_l2(g)), reverse=True)
    tot, n, xs, ys, acc = sum(d), len(d), [0.0], [0.0], 0
    for i, s in enumerate(d):
        acc += s
        xs.append((i + 1) / n)
        ys.append(acc / tot)
    k = sum(s > LONG for s in d)
    return xs, ys, k / n, sum(d[:k]) / tot, n


def win_frac(g):
    return sorted(float(r["win_code_elapsed_frac"]) for r in cleared_l2(g)
                  if int(r["seconds_on_level"]) > LONG and r["win_code_elapsed_frac"])


def checks(g):
    cl = cleared_l2(g)
    hand = sum(r["sim_checked"] == "1" for r in cl)
    helper = sum(r["helper_called"] == "1" for r in cl)
    both = sum(r["sim_checked"] == "1" and r["helper_called"] == "1" for r in cl)
    return hand, helper, both, len(cl)


# ---- numbers, asserted against discovery_summary.json and the headlines -----------------------------------
LZ = {g: lorenz(g) for g in ("baseline", "ustv")}
WF = {g: win_frac(g) for g in ("baseline", "ustv")}
CK = {g: checks(g) for g in ("pre_unstick2", "baseline", "ustv")}
for g in ("baseline", "ustv"):
    s, (_, _, px, py, n) = S[g], LZ[g]
    assert n == s["cleared_l2_segments"] and round(px, 4) == s["long_share_of_clears"], (g, n, px)
    assert round(py, 4) == s["long_share_of_clock"], (g, py)
    assert len(WF[g]) == s["win_code_matched"], g
    assert abs((1 - st.median(WF[g])) - s["win_code_left_median"]) < 5e-4, g   # CSV fractions are rounded to 4 places
for g in CK:
    hand, helper, both, n = CK[g]
    s = S[g]
    assert n == s["cleared_l2_segments"] and round(hand / n, 4) == s["sim_checked_cleared"], g
    assert round(helper / n, 4) == s["helper_called_cleared"] and round(both / n, 4) == s["sim_checked_and_helper_cleared"]
    assert round((hand + helper - both) / n, 4) == s["sim_checked_or_helper_cleared"], g
b, u = S["baseline"], S["ustv"]
assert b["cleared_l2_segments"] == 1016 and u["cleared_l2_segments"] == 279
assert round(b["long_share_of_clears"], 2) == 0.27 and round(b["long_share_of_clock"], 2) == 0.66
assert (b["win_code_matched"], b["win_code_eligible"]) == (94, 272) and (u["win_code_matched"], u["win_code_eligible"]) == (27, 83)
assert round(1 - st.median(WF["baseline"]), 2) == 0.13
assert CK["pre_unstick2"] == (11, 0, 0, 275) and CK["baseline"] == (61, 0, 0, 1016) and CK["ustv"] == (13, 20, 5, 279)
print("15 min point", {g: (round(LZ[g][2], 3), round(LZ[g][3], 3)) for g in LZ},
      "win code median elapsed", {g: round(st.median(WF[g]), 3) for g in WF}, "checks", CK)

# ---- figure -----------------------------------------------------------------------------------------------
apply()
fig = plt.figure(figsize=(W, 2.3))
axA = fig.add_axes([0.09, 0.17, 0.22, 0.715])
axB = fig.add_axes([0.395, 0.17, 0.225, 0.715])
axC = fig.add_axes([0.71, 0.17, 0.15, 0.715])
INK, GRAY, ACC, BLUE = C["ink"], C["gray"], C["accent"], C["blue"]
labels = []
PCT = ["0%", "25%", "50%", "75%", "100%"]

# (a) clock concentration
axA.plot([0, 1], [0, 1], color=C["light"], lw=0.8, zorder=1)
for g, col, lw, ls in (("ustv", BLUE, 0.9, (0, (3, 1.5))), ("baseline", ACC, 1.4, "-")):
    xs, ys, px, py, n = LZ[g]
    axA.plot(xs, ys, color=col, lw=lw, ls=ls, zorder=3 if g == "baseline" else 2,
             label="stall-check family" if g == "baseline" else "replay-check build")
xs, ys, px, py, n = LZ["baseline"]
axA.plot(px, py, "o", ms=4, mfc=ACC, mec="white", mew=0.6, zorder=4)
labels.append(axA.text(px + 0.05, py - 0.04, f"over 15 min: {px:.0%} of\nlevels, {py:.0%} of clock", ha="left", va="top",
                       fontsize=6.3, color=INK, linespacing=1.05))
axA.set_xlim(0, 1)
axA.set_ylim(0, 1.02)
for ax in (axA, axB):
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xticklabels(PCT)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticklabels(PCT)
    ax.grid(visible=True)
axA.set_xlabel("share of cleared levels, longest first", labelpad=2)
axA.set_ylabel("share of their clock", labelpad=2)
h, t = axA.get_legend_handles_labels()
axA.legend(h[::-1], t[::-1], loc="lower right", bbox_to_anchor=(1.02, 0.0), handlelength=1.6, handletextpad=0.4, labelspacing=0.3,
           borderaxespad=0.2, fontsize=6.0, frameon=True, facecolor="white", edgecolor="none", framealpha=0.95)
panel(fig, 0.005, 0.995, "(a)", "Clock piles up on long levels")

# (b) when the winning code first appears, long levels
med = st.median(WF["baseline"])
axB.axvline(med, color=GRAY, lw=0.8, ls=(0, (2, 1.5)), zorder=1)
for g, col, lw, ls in (("ustv", BLUE, 0.9, (0, (3, 1.5))), ("baseline", ACC, 1.4, "-")):
    f = WF[g]
    s = S[g]
    axB.step([0] + f + [1], [0] + [(i + 1) / len(f) for i in range(len(f))] + [1], where="post", color=col, lw=lw, ls=ls,
             zorder=3 if g == "baseline" else 2,
             label=f"n = {s['win_code_matched']} of {s['win_code_eligible']}")   # line styles as in (a)
labels.append(axB.text(med - 0.03, 0.6, f"median: {med:.0%} elapsed,\n{1 - med:.0%} left", ha="right", va="center",
                       fontsize=6.3, color=INK, linespacing=1.05))
axB.set_xlim(0, 1)
axB.set_ylim(0, 1.02)
axB.set_xlabel("share of the level's clock elapsed", labelpad=2)
axB.set_ylabel("share of matched long levels", labelpad=2)
h, t = axB.get_legend_handles_labels()
axB.legend(h[::-1], t[::-1], loc="upper left", bbox_to_anchor=(-0.02, 1.02), handlelength=1.6, handletextpad=0.4, labelspacing=0.3,
           borderaxespad=0.2, fontsize=6.0, frameon=True, facecolor="white", edgecolor="none", framealpha=0.95)
panel(fig, 0.325, 0.995, "(b)", "Winning code appears late")

# (c) checks against the recorded history, cleared levels 2 and up (cleared_l2: the summary's rule)
plt.rcParams["hatch.linewidth"] = 0.4
XL = ["before\nstall\ncheck", "stall-\ncheck\nfamily", "replay-\ncheck\nbuild"]
hand, _, _, n = CK["pre_unstick2"]
axC.bar(0, hand / n, width=0.62, color=C["light"], edgecolor=GRAY, lw=0.5, zorder=2)
labels.append(axC.text(0, hand / n + 0.003, f"{hand / n:.1%}\nof {n:,}", ha="center", va="bottom", fontsize=6.0,
                       linespacing=1.05))
hand, _, _, n = CK["baseline"]
axC.bar(1, hand / n, width=0.62, color=ACC, lw=0, zorder=2)
labels.append(axC.text(1, hand / n + 0.003, f"{hand / n:.1%}\nof {n:,}", ha="center", va="bottom", fontsize=6.0,
                       linespacing=1.05))
hand, helper, both, n = CK["ustv"]
segs = [(hand - both, "hand-written\nonly", dict(color=BLUE, lw=0)),
        (both, "both", dict(color=C["sky"], edgecolor=BLUE, lw=0.5, hatch="////")),
        (helper - both, "helper only", dict(color=C["sky"], lw=0))]
bottom = 0
for k, name, kw in segs:
    axC.bar(2, k / n, bottom=bottom, width=0.62, zorder=2, **kw)
    labels.append(axC.text(2.38, bottom + k / n / 2, f"{name}, {k / n:.1%}", ha="left", va="center", fontsize=5.8,
                           color=INK, linespacing=1.0))
    bottom += k / n
labels.append(axC.text(2, bottom + 0.003, f"{bottom:.1%}\nof {n:,}", ha="center", va="bottom", fontsize=6.0,
                       linespacing=1.05))
axC.set_xlim(-0.5, 2.5)
axC.set_ylim(0, 0.13)
axC.set_yticks([0, 0.04, 0.08, 0.12])
axC.set_yticklabels(["0%", "4%", "8%", "12%"])
axC.grid(axis="y", visible=True)
axC.set_xticks(range(3))
axC.set_xticklabels(XL, fontsize=5.8, linespacing=1.05)
axC.tick_params(axis="x", length=0, pad=2)
axC.set_ylabel("cleared levels with a check", labelpad=2)
panel(fig, 0.645, 0.995, "(c)", "Replay checks are rare")

check_no_overlap(fig, labels)
save(fig, "fig_discovery", writeup=True)
