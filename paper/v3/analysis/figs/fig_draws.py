#!/usr/bin/env python3
"""F10 "The live board history": every daily draw by configuration family (a), and our best against the bars (b).

Run from the repo root:
  uv run --no-project --with matplotlib --with numpy python -I -B paper/v3/analysis/figs/fig_draws.py [--anon]

Reads (build them with board_history_build.py)
  paper/v3/analysis/figs/data/board_draws_all_2026-11-02.csv   one row per submission (83), from Kaggle's submission list
  paper/v3/analysis/figs/data/board_snapshots_2026-11-02.csv   last leaderboard snapshot of each UTC day (our best, bar scores)
"""
import csv
import statistics as st
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from figstyle import C, HERE, W, apply, check_no_overlap, panel, save  # noqa: E402

D = HERE / "data"
ANON = "--anon" in sys.argv   # the anonymised PDF: no rank text in (b); saved as fig_draws_anon
dt = lambda s: datetime.fromisoformat(s)  # noqa: E731

FAMILY = {  # configuration name (from the submission description) -> family, in plain words
    "franzen-p448s14": "fork", "franzen-p448s14dp": "fork", "franzen-l1a1p": "fork", "franzen-l1a1fc2p": "fork",
    "fork-franzen": "fork",
    "franzen-l1a1y8sv2p": "yield", "franzen-l1a1y8sv2r3p": "yield", "franzen-l1a1y8sv2r3hm32p": "yield",
    "franzen-l1a1y8sv2r3kp": "later", "franzen-l1a1y8sv2r3wesrp": "later", "franzen-l1a1y8sv2r3kp12": "later",
    "franzen-l1a1y8sv2r3kp12lf15": "newest", "franzen-l1a1y8sv2r3kp12lf15us8": "harness",
    "franzen-l1a1y8sv2r3kp12lf15ustv": "harness",
}
STYLE = {  # family: (label, colour, marker)
    "early": ("early stack (before the fork)", C["gray"], "o"),
    "fork": ("Franzen fork, first pins", C["sky"], "s"),
    "yield": ("yield 8K family", C["accent"], "o"),
    "later": ("later pins", C["blue"], "^"),
    "newest": ("15-stream pin", C["green"], "D"),
    "harness": ("harness builds (Sec. 6)", C["pink"], "P"),
}

# ---- data -------------------------------------------------------------------------------------------------
sub = [r for r in csv.DictReader(open(D / "board_draws_all_2026-11-02.csv"))]
draws = []
for r in sub:
    if r["status"] != "complete" or r["public_score"] == "":
        continue    # errored and pending submissions have no score and do not spend the slot
    cfg = r["config"]
    fam = FAMILY[cfg] if cfg in FAMILY else "early"
    assert fam != "early" or not cfg.startswith(("franzen", "fork")), cfg
    draws.append((dt(r["submitted_utc"]), float(r["public_score"]), fam, r["ref"]))
draws.sort()
assert len(sub) == 83 and len(draws) == 81
assert all(d[0] < dt("2026-10-03") for d in draws if d[2] == "early")
assert all(d[0] >= dt("2026-10-03") for d in draws if d[2] != "early")
n_fam = {f: sum(d[2] == f for d in draws) for f in STYLE}
assert n_fam == {"early": 50, "fork": 9, "yield": 9, "later": 8, "newest": 3, "harness": 2}, n_fam
best = max(d[1] for d in draws)
assert best == 40.98
finals = {d[3]: d for d in draws if d[3] in ("57417773", "57127096")}     # Final Submissions: 2026-10-20, 2026-10-13
assert finals["57417773"][1] == 40.98 and finals["57127096"][1] == 40.09
assert finals["57417773"][0].date().isoformat() == "2026-10-20" and finals["57127096"][0].date().isoformat() == "2026-10-13"
zeros = [d for d in draws if d[1] == 0]
assert [d[0].date().isoformat() for d in zeros] == ["2026-10-03", "2026-10-10", "2026-10-18"]
fork_on = [d for d in draws if d[0] >= dt("2026-10-03")]
print("draws", len(draws), n_fam, "since the fork:", len(fork_on), "valid since the fork:", sum(d[1] > 0 for d in fork_on))

snap = list(csv.DictReader(open(D / "board_snapshots_2026-11-02.csv")))
S_T = [dt(r["date_utc"]) for r in snap]
col = lambda k: [float(r[k]) if r[k] != "" else None for r in snap]  # noqa: E731
ours, r1, r3, r5 = col("our_best"), col("rank1"), col("rank3"), col("rank5")
last = snap[-1]
assert last["date_utc"] == "2026-11-02" and last["our_rank"] == "20" and last["teams"] == "5043"
assert (ours[-1], r1[-1], r3[-1], r5[-1]) == (40.98, 75.56, 58.59, 56.02)
# our best in the snapshots never exceeds the best draw on that date (draws are banked by the time a snapshot is taken)
for t, o in zip(S_T, ours):
    assert o <= max([d[1] for d in draws if d[0] <= t.replace(hour=23, minute=59)] + [0]) + 1e-9, (t, o)

# ---- figure -----------------------------------------------------------------------------------------------
apply()
fig = plt.figure(figsize=(W, 4.9))
axA = fig.add_axes([0.095, 0.52, 0.72, 0.39])
axB = fig.add_axes([0.095, 0.085, 0.72, 0.34], sharex=axA)
INK, GRAY, ACC = C["ink"], C["gray"], C["accent"]
labels = []

# (a) one dot per draw
for fam, (name, colr, mk) in STYLE.items():
    pts = [(d[0], d[1]) for d in draws if d[2] == fam and d[1] > 0]
    axA.plot([p[0] for p in pts], [p[1] for p in pts], mk, ms=3.6, mfc=colr, mec="white", mew=0.4, ls="", label=name, zorder=3)
axA.plot([], [], "o", ms=4.4, mfc="white", mec=INK, mew=0.9, ls="", label="0.00 draw")
for d in zeros:
    f = d[2]
    axA.plot(d[0], 0, STYLE[f][2], ms=4.8, mfc="white", mec=STYLE[f][1], mew=1.1, zorder=5, clip_on=False)
for ref, (t, s, f, _) in finals.items():
    axA.plot(t, s, "o", ms=9.5, mfc="none", mec=INK, mew=1.1, zorder=6)
t20, t13 = finals["57417773"][0], finals["57127096"][0]
labels.append(axA.text(dt("2026-11-02"), 46.3, "Final Submissions: 40.09 (10-13), 40.98 (10-20)", ha="right", va="bottom",
                       fontsize=7, color=INK))
for t, s_ in ((t20, 40.98), (t13, 40.09)):
    axA.plot([t, t], [s_ + 1.4, 46.0], color=INK, lw=0.6, zorder=2)
axA.set_ylim(-1.5, 50)
axA.set_yticks([0, 10, 20, 30, 40])
axA.grid(axis="y", visible=True)
axA.set_ylabel("public score of one draw", labelpad=2)
axA.tick_params(labelbottom=False)
h, l = axA.get_legend_handles_labels()
order = [l.index(STYLE[f][0]) for f in STYLE] + [l.index("0.00 draw")]
axA.legend([h[i] for i in order], [l[i] if l[i] != "0.00 draw" else "0.00 draw (open, at the axis)" for i in order],
           loc="upper left", bbox_to_anchor=(0.0, 0.84), ncol=1, handletextpad=0.1, labelspacing=0.25, borderaxespad=0.0,
           markerscale=1.0, fontsize=6.5)
panel(fig, 0.005, 0.995, "(a)", f"Daily draws, n = {len(draws)} scored submissions")

# (b) our best against the bars
axB.step(S_T, ours, where="post", color=ACC, lw=1.6, zorder=4)
for series, ls_ in ((r1, "-"), (r3, (0, (4, 2))), (r5, (0, (1.5, 1.5)))):
    pts = [(t, v) for t, v in zip(S_T, series) if v is not None]
    axB.plot([p[0] for p in pts], [p[1] for p in pts], color=GRAY, lw=1.0, ls=ls_, zorder=2)
x_end = S_T[-1]
off = (x_end - S_T[0]) * 0.03
for v, name, colr in ((r1[-1], "rank 1: 75.56", C["gray_text"]), (r3[-1], "rank 3: 58.59", C["gray_text"]),
                      (r5[-1], "rank 5: 56.02", C["gray_text"]), (ours[-1], "ours: 40.98" if ANON else f"ours: 40.98\n({last['our_rank']}th of {int(last['teams']):,})", ACC)):
    dy = {"rank 3": 2.6, "rank 5": -2.6}.get(name[:6], 0)      # keep the close bars' labels apart
    labels.append(axB.text(x_end + off, v + dy, name, ha="left", va="center", fontsize=7, color=colr,
                           fontweight="bold" if colr == ACC else "normal", linespacing=1.05))
axB.set_ylim(0, 85)
axB.set_yticks([0, 20, 40, 60, 80])
axB.grid(axis="y", visible=True)
axB.set_ylabel("public score", labelpad=2)
labels.append(axB.text(dt("2026-07-17") + (dt("2026-08-25") - dt("2026-07-17")) / 2, 42,
                       "no leaderboard\nsnapshots before 08-25", ha="center", va="center", fontsize=7, color=C["gray_text"],
                       linespacing=1.1))
panel(fig, 0.005, 0.475, "(b)", "Our best public score against the bars (one snapshot per day)")

axB.set_xlim(dt("2026-07-12"), dt("2026-11-02T20:00"))
axB.xaxis.set_major_locator(mdates.MonthLocator())
axB.xaxis.set_major_formatter(mdates.DateFormatter("%b 1"))
axB.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=0))
axB.set_xlabel("date (2026, UTC)", labelpad=2)

check_no_overlap(fig, labels)
save(fig, "fig_draws_anon" if ANON else "fig_draws", writeup=not ANON)
