#!/usr/bin/env python3
"""F7 (appendix) "What we tried": one dot per arm, level change against its control, inside the noise of one run.

Run from the repo root:
  uv run --no-project --with matplotlib --with numpy --with scipy python -I -B paper/v3/analysis/figs/fig_tried.py

Reads
  paper/v3/analysis/figs/data/tried_arms.csv   one row per measured arm: run directories of the arm and its control,
      lever class, whether its mechanism counter fired (from the arm's gate record), shipped or not. Built by hand from
      docs/research/2026-10-19-top/ledger.md table (a), paper/v2/analysis/out/e4_gates.txt (pooled controls) and the
      gate-results records named in its source column; v2's tried-and-not-shipped table plus the dossier's new arms.
  paper/v2/analysis/out/yield_runs.json        levels of every Franzen-stack replica run (asserted against the CSV)
  paper/v3/evidence/data/lora_replicas.csv     levels of the adapter runs
  paper/v2/analysis/out/e4_noise.json          SD 8.25 (six identical pairs) for the band
Control = the pooled byte-identical pair where one exists, else the run the arm was gated against.
Not drawn: BF16 draft KV (0 levels, out of memory) and arms without a level reading (NVFP4 KV probe, D-prime smoke run).
"""
import csv
import json
import math
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import norm  # noqa: E402

from figstyle import C, DATA, V2OUT, W, apply, save  # noqa: E402

LV = {r["dir"]: r["levels"] for r in json.loads((V2OUT / "yield_runs.json").read_text())}
LV.update({r["run"]: int(r["levels"]) for r in csv.DictReader(open(DATA / "lora_replicas.csv")) if r["group"] == "lora_r16"})
SD = json.loads((V2OUT / "e4_noise.json").read_text())["franzen_levels"]["sd"]
rows = list(csv.DictReader(open(HERE / "data/tried_arms.csv")))
for r in rows:
    a, c = r["arm_runs"].split(), r["control_runs"].split()
    r["na"], r["nc"] = len(a), len(c)
    r["a"], r["c"] = st.mean(LV[x] for x in a), st.mean(LV[x] for x in c)
    assert abs(r["a"] - float(r["arm_levels"])) < 1e-9 and abs(r["c"] - float(r["control_levels"])) < 1e-9, r["lever"]
    r["d"] = r["a"] - r["c"]
    assert r["mechanism"] in ("fired", "not fired", "no counter") and r["shipped"] in ("yes", "no")
CLASSES = [("serving", "Serving: more tokens per second"), ("context", "Context: what is kept or re-sent"),
           ("control", "Agent control: budgets, guards, prompts"), ("weights", "Weights: fine-tuning")]
assert {r["cls"] for r in rows} == {k for k, _ in CLASSES} and len(rows) == 27
yv = next(r for r in rows if r["lever"].startswith("yield after"))
assert yv["d"] == 15 and next(r for r in rows if r["lever"].startswith("fine-tune on own wins, full"))["d"] == -12
H95 = norm.ppf(0.975) * SD * math.sqrt(2)          # one run against one run: 95% of identical-code gaps fall inside
H50 = norm.ppf(0.75) * SD * math.sqrt(2)
print(f"{len(rows)} arms; band one run vs one run at SD {SD:.2f}: 50% +/-{H50:.1f}, 95% +/-{H95:.1f}; "
      f"outside 95%: {[r['lever'] for r in rows if abs(r['d']) > H95]}")

# ---- figure -----------------------------------------------------------------------------------------------------
apply()
ROW, HEAD = 0.145, 0.24
BOT = 0.78
H = 0.2 + BOT + ROW * len(rows) + HEAD * len(CLASSES)
fig = plt.figure(figsize=(W, H))
X0, X1 = 0.43, 0.86
ax = fig.add_axes([X0, BOT / H, X1 - X0, (H - 0.2 - BOT) / H])
INK, GRAY, ACC = C["ink"], C["gray"], C["accent"]
ax.axvspan(-H95, H95, color="#EEEEEE", lw=0, zorder=0)
ax.axvspan(-H50, H50, color="#DDDDDD", lw=0, zorder=0)
ax.axvline(0, color=GRAY, lw=0.8, zorder=1)
MK = {"fired": dict(marker="o"), "not fired": dict(marker="X"), "no counter": dict(marker="o")}
y = 0.0
ticks, tlabels = [], []
for key, title in CLASSES:
    fig.text(0.01, 1 - (0.2 + y + HEAD * 0.55) / H, title, fontsize=7.5, fontweight="bold", va="center", ha="left")
    y += HEAD
    for r in [r for r in rows if r["cls"] == key]:
        yc = y + ROW / 2
        col = ACC if r["shipped"] == "yes" else INK
        face = "white" if r["mechanism"] == "no counter" else col
        ax.plot(r["d"], yc, ls="", ms=5.0 if r["mechanism"] != "not fired" else 5.6, mfc=face, mec=col, mew=1.0, zorder=3,
                **MK[r["mechanism"]])
        ticks.append(yc)
        tlabels.append(r["lever"])
        fig.text(X1 + 0.012, 1 - (0.2 + yc) / H, f"{r['na']} v {r['nc']}", fontsize=6.5, va="center", ha="left",
                 color=C["gray_text"])
        y += ROW
ax.set_ylim(y, 0)
ax.set_yticks(ticks)
ax.set_yticklabels(tlabels, fontsize=6.8)
for t, r in zip(ax.get_yticklabels(), [r for k, _ in CLASSES for r in rows if r["cls"] == k]):
    if r["shipped"] == "yes":
        t.set_fontweight("bold")
ax.tick_params(axis="y", length=0, pad=3)
ax.spines["left"].set_visible(False)
ax.set_xlim(-26, 26)
ax.set_xticks([-20, -10, 0, 10, 20])
ax.set_xticklabels(["-20", "-10", "0", "+10", "+20"])
ax.grid(axis="x", visible=True)
ax.set_xlabel("levels, arm minus control (25 demonstration games)", labelpad=2)
# rescale: axes y runs in inches of rows from the top of the first header
ax.set_position([X0, BOT / H, X1 - X0, y / H])
fig.text(X1 + 0.012, 1 - 0.12 / H, "runs", fontsize=6.5, va="center", ha="left", color=C["gray_text"])

# key, under the axis
ky = 0.30 / H
items = [(dict(marker="o", mfc=INK, mec=INK), "mechanism counter fired"),
         (dict(marker="X", mfc=INK, mec=INK, ms=5.6), "did not fire"),
         (dict(marker="o", mfc="white", mec=INK), "no counter recorded"),
         (dict(marker="o", mfc=ACC, mec=ACC), "shipped (bold label)")]
x = 0.03
for kw, lab in items:
    kw = dict(ms=5.0, mew=1.0, **kw) if "ms" not in kw else dict(mew=1.0, **kw)
    fig.add_artist(plt.Line2D([x], [ky], ls="", transform=fig.transFigure, **kw))
    t = fig.text(x + 0.015, ky, lab, fontsize=7, va="center", ha="left")
    fig.canvas.draw()
    x += 0.015 + t.get_window_extent().width / fig.bbox.width + 0.03
fig.text(0.03, 0.12 / H, f"Shaded: where 50% and 95% of the gaps between two single runs of identical code fall "
         f"(SD {SD:.2f}, six pairs).\nRight column: arm runs v control runs.", fontsize=6.8, va="center", ha="left",
         color=C["gray_text"], linespacing=1.1)
save(fig, "fig_tried")
