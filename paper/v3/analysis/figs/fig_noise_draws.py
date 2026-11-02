#!/usr/bin/env python3
"""F2 "Identical code, different scores": identical-run pairs and concurrent runs (left), the family's board draws (right).

Run from the repo root:
  uv run --no-project --with matplotlib --with numpy python -I -B paper/v3/analysis/figs/fig_noise_draws.py

Reads
  paper/v2/analysis/out/e4_noise.json                 six byte-identical replica pairs (claims B1, B2)
  paper/v3/evidence/data/vertex_vs_kaggle_runs.csv    eight concurrent byte-identical Vertex runs, 10-20 (V3-X3)
  paper/v3/analysis/figs/data/board_draws_all_2026-11-02.csv   board draws to 11-02: the family r3p, hm32p, r3kp, kp12 (finals
                                                      rule v4, V3-B9, V3-W16) plus five later builds, 17 valid draws (V3-W30)
"""
import csv
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt  # noqa: E402

from figstyle import SCALE_V3 as SCALE, C, DATA, HERE as FHERE, V2OUT, W, apply, fs, check_no_overlap, panel, save  # noqa: E402

# ---- data ------------------------------------------------------------------------------------------------
noise = json.loads((V2OUT / "e4_noise.json").read_text())
PAIRS = [tuple(v["levels"]) for v in noise["franzen_pairs"].values()]          # (run 1, run 2)
SD_PAIRS = noise["franzen_levels"]["sd"]

vtx = [r for r in csv.DictReader(open(DATA / "vertex_vs_kaggle_runs.csv")) if r["run"].startswith("vertex-rq1020-")]
VTX = sorted(int(r["levels_25"]) for r in vtx)

# Draws of one agent: the 12 of the family (finals rule v4: r3p, hm32p, r3kp, kp12) closed on 2026-10-28; five later draws of the
# 15-stream and harness builds (kp12lf15, kp12lf15us8, kp12lf15ustv) bring the valid count to 17 (V3-W30).
GROUP = {"franzen-l1a1y8sv2r3p": "r3p", "franzen-l1a1y8sv2r3hm32p": "hm32p", "franzen-l1a1y8sv2r3kp": "kp",
         "franzen-l1a1y8sv2r3kp12": "kp", "franzen-l1a1y8sv2r3kp12lf15": "later", "franzen-l1a1y8sv2r3kp12lf15us8": "later",
         "franzen-l1a1y8sv2r3kp12lf15ustv": "later"}
all_draws = [r for r in csv.DictReader(open(FHERE / "data/board_draws_all_2026-11-02.csv")) if r["config"] in GROUP]
valid = [r for r in all_draws if r["status"] == "complete" and float(r["public_score"]) > 0]
zeros = [r for r in all_draws if float(r["public_score"]) == 0]
DRAWS = sorted((float(r["public_score"]), GROUP[r["config"]], r["submitted_utc"][:10]) for r in valid)
scores = [d[0] for d in DRAWS]
first12 = [s for s, g, d in DRAWS if g != "later"]

# ---- checks against the dossier and the framing ---------------------------------------------------------
assert sorted(abs(a - b) for a, b in PAIRS) == [3, 4, 4, 6, 16, 22]
assert round(SD_PAIRS, 2) == 8.25
assert VTX == [115, 117, 124, 125, 126, 128, 130, 132] and round(st.stdev(VTX), 2) == 5.95      # V3-X3
assert len(first12) == 12 and round(st.mean(first12), 2) == 34.59                                  # V3-W16
assert len(DRAWS) == 17 and round(st.mean(scores), 2) == 34.59 and round(st.stdev(scores), 2) == 3.39   # V3-W30
assert max(scores) == 40.98 and min(scores) == 28.97
assert len(zeros) == 1 and zeros[0]["submitted_utc"].startswith("2026-10-18")                    # V3-B7, plotted open at 0
assert sorted(s for s, c, _ in DRAWS if c == "r3p") == [34.22, 40.09, 40.98]
assert sorted(s for s, c, _ in DRAWS if c == "later") == [29.63, 33.56, 35.25, 36.38, 38.17]
print(f"pairs {PAIRS}; vertex {VTX} sd {st.stdev(VTX):.2f}; draws {scores} mean {st.mean(scores):.3f} (first 12: {st.mean(first12):.3f}) sd {st.stdev(scores):.2f}")

# ---- figure -----------------------------------------------------------------------------------------------
apply(scale=SCALE)
H = 3.25
fig = plt.figure(figsize=(W, H))
axL = fig.add_axes([0.150, 0.300, 0.415, 0.585])
axR = fig.add_axes([0.745, 0.300, 0.245, 0.585])
INK, GRAY, ACC = C["ink"], C["gray"], C["accent"]
labels = []

# (a) six identical pairs (sorted by gap), then the eight concurrent runs as one strip row
order = sorted(PAIRS, key=lambda p: -abs(p[0] - p[1]))
rows = list(range(len(order)))
y_vtx = len(order) + 0.55
for i, (a, b) in zip(rows, order):
    axL.plot([a, b], [i, i], color=ACC, lw=2.0, solid_capstyle="butt", zorder=2)
    axL.plot(a, i, "o", ms=4.8, mfc=INK, mec=INK, zorder=3)
    axL.plot(b, i, "o", ms=4.8, mfc="white", mec=INK, mew=1.0, zorder=3)
    labels.append(axL.text(max(a, b) + 1.0, i, f"{abs(a - b)}", ha="left", va="center", fontsize=fs(7.5),
                           color=ACC, fontweight="bold"))
axL.plot([min(VTX), max(VTX)], [y_vtx, y_vtx], color=GRAY, lw=0.8, zorder=1)
axL.plot(VTX, [y_vtx] * len(VTX), "D", ms=3.8, mfc=C["light"], mec=INK, mew=0.7, zorder=3)
labels.append(axL.text(max(VTX) + 1.0, y_vtx, f"{max(VTX) - min(VTX)}", ha="left", va="center", fontsize=fs(7.5),
                       color=GRAY, fontweight="bold"))
axL.set_yticks(rows + [y_vtx])
axL.set_yticklabels([f"pair {i + 1}" for i in rows] + ["8 runs,\nsame day"], linespacing=1.0)
axL.tick_params(axis="y", length=0, pad=3)
axL.set_ylim(y_vtx + 0.6, -0.8)
axL.set_xlim(99, 138)
axL.set_xticks([100, 110, 120, 130])
axL.spines["left"].set_visible(False)
axL.grid(axis="x", visible=True)
axL.set_xlabel("levels cleared in one run", labelpad=2)
leg = [axL.plot([], [], "o", ms=4.8, mfc=INK, mec=INK, ls="")[0], axL.plot([], [], "o", ms=4.8, mfc="white", mec=INK, ls="")[0],
       axL.text(0, 0, "22", color=ACC, fontweight="bold", fontsize=fs(7.5), visible=False)]
leg[2] = plt.Line2D([], [], color=ACC, lw=2.0)
axL.legend(leg, ["first run", "second run", "spread"], loc="lower left", bbox_to_anchor=(-0.30, 1.0), ncol=3,
           handletextpad=0.3, columnspacing=0.8, borderaxespad=0.0, handlelength=1.4)
panel(fig, 0.005, 0.995, "(a)", "Identical code, rerun")

# (b) all 17 valid draws, one strip per kind of build; the 0.00 draw of 10-18 is an open dot on the axis, off scale
mean = st.mean(scores)
KINDS = [("r3p", "o", "shipped agent"), ("hm32p", "s", "draft-model variant"), ("kp", "^", "boot-keepalive variant"),
         ("later", "D", "later builds, 10-29 on")]
x_of = {k: -0.30 + 0.20 * i for i, (k, _, _) in enumerate(KINDS)}
mk = {k: m for k, m, _ in KINDS}
Y0 = 26.6                                            # where the off-scale zero sits
axR.plot([-0.45, 0.47], [mean, mean], color=GRAY, lw=1.0, ls=(0, (4, 2)), zorder=1)
seen = {}
for sc, kind, day in DRAWS:
    best = sc == max(scores)
    k = seen.get(kind, 0)
    seen[kind] = k + 1
    dx = 0.045 * (-1) ** k * (1 if k else 0)         # nearby draws of one kind fan out a little
    axR.plot(x_of[kind] + dx, sc, mk[kind], ms=5.4 if best else 4.6, mfc=ACC if best else C["light"],
             mec=ACC if best else INK, mew=0.8, zorder=3)
axR.plot(0.0, Y0, "o", ms=5.0, mfc="white", mec=INK, mew=1.0, zorder=3, clip_on=False)
labels.append(axR.text(0.12, Y0, "0.00", ha="left", va="center", fontsize=fs(6.2), color=INK))
labels.append(axR.text(0.55, max(scores), "40.98\nbest", ha="left", va="center", fontsize=fs(6.6), color=ACC,
                       fontweight="bold", linespacing=1.05))
labels.append(axR.text(0.53, mean, f"mean\n{mean:.2f}", ha="left", va="center", fontsize=fs(6.6),
                       color=C["gray_text"], linespacing=1.05))
axR.set_xlim(-0.45, 0.95)
axR.set_ylim(28, 43)
axR.set_xticks([])
axR.spines["bottom"].set_visible(False)
axR.set_yticks([30, 35, 40])
axR.grid(axis="y", visible=True)
axR.set_ylabel("public score of one draw", labelpad=2)
axR.plot([-0.45, 0.95], [28.0, 28.0], color=INK, lw=0.7, ls=(0, (1, 2)), clip_on=False, zorder=1)  # the scale ends here
leg = [axR.plot([], [], m, ms=4.6, mfc=C["light"], mec=INK, ls="")[0] for _, m, _ in KINDS]
leg.append(axR.plot([], [], "o", ms=4.8, mfc="white", mec=INK, mew=1.0, ls="")[0])
counts = {k: sum(1 for _, g, _ in DRAWS if g == k) for k, _, _ in KINDS}
fig.legend(leg, [f"{n} ({counts[k]})" for k, _, n in KINDS] + ["scored 0.00 on 10-18 (1), off scale"], loc="lower left", ncol=3,
           bbox_to_anchor=(0.005, 0.075), handletextpad=0.3, labelspacing=0.25, columnspacing=1.0, borderaxespad=0.0)
panel(fig, 0.655, 0.995, "(b)", f"Same agent, {len(DRAWS)} valid draws")
fig.text(0.005, 0.005, f"Mean of the first 12 draws (10-13 to 10-28) and of all {len(DRAWS)}: both {mean:.2f}.", ha="left", va="bottom",
         fontsize=fs(6.2), color=C["gray_text"])

check_no_overlap(fig, labels)
save(fig, "fig_noise_draws", writeup=True)
