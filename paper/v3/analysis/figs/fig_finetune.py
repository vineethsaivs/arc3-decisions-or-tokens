#!/usr/bin/env python3
"""F6 "All three adapter runs scored low": levels and four play counters against LoRA scale (0 = no adapter).

Run from the repo root:
  uv run --no-project --with matplotlib --with numpy python -I -B paper/v3/analysis/figs/fig_finetune.py

Reads
  paper/v3/evidence/data/lora_replicas.csv    levels per run: 9 pin runs (Kaggle), adapter at merge scale 2.0 (2 runs)
                                              and 1.0 (1 run), GCP G4 VM (V3-L9, V3-L10)
  paper/v3/evidence/data/lora_controls_vertex.csv  two same-card no-adapter controls of the adapter's own base arm
                                              (hm32) on Vertex AI G4 Spot, 2026-11-03 (V3-W44); levels re-summed from
                                              solution/kaggle/generated/kernel-outputs/vertex-adctl-{1,2}/benchmark.json
  paper/v3/evidence/data/lora_behaviour.csv   per-run counters (V3-L13, V3-L14): reasoning characters per action and
                                              actions per reply on cleared levels (8 pin runs with transcripts),
                                              RESET actions per run (9 pin runs), GAME OVER echoes per run (8 pin runs)
  docs/research/2026-10-19-top/lora.md        the RE-ARC-3 training-play column (datagen attempts 02 and 04), read and
                                              not recomputed (dossier s3, "Unverified numbers"); only the two counters
                                              with a per-reply or per-action form have a comparable training-play value
Adapter strength = merge scale / 2.0 (the trained alpha 32 over r 16): 0 = the pin, 0.5 = scale 1.0, 1.0 = scale 2.0.
"""
import csv
import json
import re
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt  # noqa: E402

from figstyle import SCALE_V3 as SCALE, C, DATA, REPO, W, apply, fs, check_no_overlap, panel, save  # noqa: E402

rep = list(csv.DictReader(open(DATA / "lora_replicas.csv")))
beh = {r["run"]: r for r in csv.DictReader(open(DATA / "lora_behaviour.csv"))}
STRENGTH = {"pin_y8_family": None, "lora_r16": None}

# same-card controls (Vertex): levels are re-summed from the benchmark files and asserted against the csv
CTL = []
for r in csv.DictReader(open(DATA / "lora_controls_vertex.csv")):
    bm = json.load(open(REPO / f"solution/kaggle/generated/kernel-outputs/vertex-{r['run']}/benchmark.json"))
    n = sum(g["levels_completed"] for g in bm["game_runs"])
    assert n == int(r["levels"]) and len(bm["game_runs"]) == 25, (r["run"], n)
    CTL.append(n)
assert sorted(CTL) == [130, 132]


def strength(r):
    return 0.0 if r["group"] == "pin_y8_family" else float(r["merge_scale"]) / 2.0


RUNS = [(r["run"], strength(r), int(r["levels"])) for r in rep]

# RE-ARC-3 training play (lora.md section 2, columns for attempts 02 / 04); asserted to be the text of that table
LORA_MD = (REPO / "docs/research/2026-10-19-top/lora.md").read_text()
TRAIN = {"rchars_per_action_cleared": (966, 1347), "acts_per_reply_cleared": (4.52, 4.28)}
assert "| reasoning chars per action, cleared levels | 2,021 to 2,346 | 1,729 / 1,772 | 966 / 1,347 |" in LORA_MD
assert "| actions per reply, cleared levels | 2.35 to 2.67 | 2.69 / 2.75 | 4.52 / 4.28 |" in LORA_MD

PANELS = [  # key, axis label, ylim, ticks
    ("rchars_per_action_cleared", "reasoning characters\nper action", (800, 2500), [1000, 1500, 2000, 2500]),
    ("acts_per_reply_cleared", "actions per reply", (2.0, 4.8), [2, 3, 4]),
    ("reset_actions", "RESET actions per run", (40, 150), [50, 100, 150]),
    ("gameover_echoes", "deaths (GAME OVER)\nper run", (0, 32), [0, 10, 20, 30]),
]


def series(key):
    out = []
    for name, s, _ in RUNS:
        v = beh[name][key]
        if v not in ("", None):
            out.append((name, s, float(v)))
    return out


# ---- checks against the dossier ------------------------------------------------------------------------------
lv = {s: sorted(l for _, x, l in RUNS if x == s) for s in (0.0, 0.5, 1.0)}
assert lv == {0.0: [119, 120, 120, 122, 124, 126, 128, 128, 129], 0.5: [117], 1.0: [109, 115]}
assert st.mean(lv[0.0]) == 124.0 and round(st.stdev(lv[0.0]), 1) == 3.9
chk = {k: {s: sorted(v for _, x, v in series(k) if x == s) for s in (0.0, 0.5, 1.0)} for k, *_ in PANELS}
assert chk["rchars_per_action_cleared"][1.0] == [1729, 1772] and chk["rchars_per_action_cleared"][0.5] == [1979]
assert (min(chk["rchars_per_action_cleared"][0.0]), max(chk["rchars_per_action_cleared"][0.0])) == (2021, 2346)
assert chk["acts_per_reply_cleared"][1.0] == [2.69, 2.75] and chk["acts_per_reply_cleared"][0.5] == [2.67]
assert (min(chk["acts_per_reply_cleared"][0.0]), max(chk["acts_per_reply_cleared"][0.0])) == (2.35, 2.67)
assert chk["reset_actions"][1.0] == [129, 138] and chk["reset_actions"][0.5] == [100]
assert (min(chk["reset_actions"][0.0]), max(chk["reset_actions"][0.0]), len(chk["reset_actions"][0.0])) == (52, 87, 9)
assert chk["gameover_echoes"][1.0] == [26, 27] and chk["gameover_echoes"][0.5] == [10]
assert (min(chk["gameover_echoes"][0.0]), max(chk["gameover_echoes"][0.0]), len(chk["gameover_echoes"][0.0])) == (8, 13, 8)
for k, *_ in PANELS:
    print(k, {s: (round(st.mean(v), 2), len(v)) for s, v in chk[k].items()})

# ---- figure ----------------------------------------------------------------------------------------------------
apply(scale=SCALE)
H = 3.55
fig = plt.figure(figsize=(W, H))
INK, GRAY, ACC = C["ink"], C["gray"], C["accent"]
XT = [0.0, 0.5, 1.0]
XL = ["0\n(none)", "0.5", "1.0"]
labels = []


def jitter(vals, step):
    """Deterministic side-by-side offsets for equal values (pin runs share a column)."""
    seen, cnt, out = {}, {}, []
    for v in vals:
        cnt[v] = cnt.get(v, 0) + 1
    for v in vals:
        k = seen.get(v, 0)
        seen[v] = k + 1
        out.append((k - (cnt[v] - 1) / 2) * step)
    return out


def draw(ax, pts, roundto):
    pin = [v for _, s, v in pts if s == 0.0]
    offs = jitter([round(v / roundto) for v in pin], 0.045)
    ax.scatter([o for o in offs], pin, s=11, c=C["light"], edgecolors=INK, linewidths=0.5, zorder=3, marker="o")
    for s in (0.5, 1.0):
        vv = [v for _, x, v in pts if x == s]
        ax.scatter([s + o for o in jitter([0] * len(vv), 0.07)], vv, s=20, c=ACC, edgecolors=ACC, linewidths=0.5,
                   zorder=4, marker="D")
    means = [st.mean(v for _, x, v in pts if x == s) for s in XT]
    ax.plot(XT, means, color=ACC, lw=0.9, ls=(0, (2.5, 1.5)), zorder=2)
    ax.plot([-0.13, 0.13], [means[0]] * 2, color=INK, lw=1.4, zorder=2.5, solid_capstyle="butt")
    ax.set_xlim(-0.22, 1.18)
    ax.set_xticks(XT)


# levels: the outcome, on the left, full height
axL = fig.add_axes([0.105, 0.20, 0.235, 0.665])
draw(axL, RUNS, 1)
axL.scatter([0.0, 0.0], sorted(CTL), s=17, c=C["blue"], edgecolors=C["blue"], linewidths=0.5, zorder=4, marker="s")
labels.append(axL.text(0.13, 131.0, "same-card controls\n(Vertex, no adapter)", fontsize=fs(6.0), ha="left", va="center",
                       color=C["blue"], linespacing=1.0))
axL.set_ylim(98, 135)
axL.set_yticks([100, 105, 110, 115, 120, 125, 130, 135])
axL.set_ylabel("levels cleared per run\n(25 demonstration games)", labelpad=2, linespacing=1.05)
axL.set_xticklabels(XL, linespacing=1.0)
axL.grid(axis="y", visible=True)
panel(fig, 0.005, 0.995, "(a)", "Levels per run")

# the four play counters, 2 x 2 on the right; shared x
pos = [(0.505, 0.585), (0.815, 0.585), (0.505, 0.20), (0.815, 0.20)]
for (key, ylab, yl, yt), (x0, y0) in zip(PANELS, pos):
    ax = fig.add_axes([x0, y0, 0.175, 0.28])
    if key in TRAIN:
        a, b = sorted(TRAIN[key])
        ax.axhspan(a, b, color=C["faint"], lw=0, zorder=0)
        for v in (a, b):
            ax.axhline(v, color=C["gray_text"], lw=0.8, ls=(0, (3, 2)), zorder=1)
        above = key == "rchars_per_action_cleared"
        labels.append(ax.text(1.16, b + 40 if above else a - 0.12, "agent's own play\non training games", fontsize=fs(6.0), ha="right",
                              va="bottom" if above else "top", color=C["gray_text"], linespacing=1.0))
    draw(ax, series(key), 50 if key == "rchars_per_action_cleared" else (0.03 if key == "acts_per_reply_cleared" else 2))
    ax.set_ylim(*yl)
    ax.set_yticks(yt)
    ax.set_ylabel(ylab, labelpad=2, fontsize=fs(6.6), linespacing=1.0)
    ax.grid(axis="y", visible=True)
    ax.set_xticklabels(XL if y0 < 0.5 else [], linespacing=1.0)
fig.text(0.74, 0.03, "LoRA scale (0 = no adapter, 1 = as trained)", ha="center", va="bottom", fontsize=fs(8))
fig.text(0.22, 0.03, "LoRA scale", ha="center", va="bottom", fontsize=fs(8))
panel(fig, 0.40, 0.995, "(b)", "Style counters per run")

# key
leg = [plt.Line2D([], [], marker="o", ls="", ms=3.6, mfc=C["light"], mec=INK, mew=0.5),
       plt.Line2D([], [], color=INK, lw=1.4),
       plt.Line2D([], [], marker="D", ls="", ms=4.0, mfc=ACC, mec=ACC),
       plt.Line2D([], [], color=ACC, lw=0.9, ls=(0, (2.5, 1.5)))]
axL.legend(leg, ["no adapter, Kaggle (9 of 18)", "their mean", "with adapter, VM", "mean by scale"],
           loc="lower left", fontsize=fs(6.2), bbox_to_anchor=(0.0, 0.0), handletextpad=0.4, labelspacing=0.3, borderaxespad=0.2, handlelength=1.6,
           frameon=True, facecolor="white", edgecolor="none", framealpha=1.0)
check_no_overlap(fig, labels)
save(fig, "fig_finetune", writeup=True)
