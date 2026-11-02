#!/usr/bin/env python3
"""F5 "Re-sending less context coincides with more levels across days, and not when interleaved": per-game paired differences
(a), resume openers per game at each threshold, observed and replayed (b), the 12 interleaved runs and the two
differences with their intervals (c).

Run from the repo root:
  uv run --no-project --with matplotlib --with numpy python -I -B paper/v3/analysis/figs/fig_yield.py [--rebuild]

Reads
  paper/v2/analysis/out/yield_runs.json    per-game levels of the 9 yield-8,192 and 21 yield-2,048 replica runs
  paper/v2/analysis/out/yield_stats.json   family lists, the run-bootstrap CI of the difference (D1), the per-game sign count
  paper/v2/analysis/out/yield_mech.json    per-run resume openers in clean turns, observed and replayed (yield.md s4)
  paper/v3/analysis/out/yield_wave/read_2026-10-24.json    the interleaved wave read by its predeclared rules (V3-W36): 12 valid
      runs, 6 at yield 8,192 against 6 at 2,048; the four launched runs that never ran are not in it
  paper/v3/analysis/figs/data/yield_resume_by_game.csv   per-game resume openers; built with --rebuild (or when missing)
      from the raw transcripts and usage logs in solution/kaggle/generated/kernel-outputs/, with the replay functions of
      paper/v2/analysis/yield_mech.py, and asserted to sum to yield_mech.json run by run.
"""
import csv
import json
import random
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import matplotlib.pyplot as plt  # noqa: E402

from figstyle import SCALE_V3 as SCALE, C, REPO, V2OUT, W, apply, check_no_overlap, fs, panel, save  # noqa: E402

RUNS = {r["dir"]: r for r in json.loads((V2OUT / "yield_runs.json").read_text())}
STATS = json.loads((V2OUT / "yield_stats.json").read_text())
MECH = {m["run"]: m for m in json.loads((V2OUT / "yield_mech.json").read_text())}
RESUME_CSV = HERE / "data/yield_resume_by_game.csv"
REPLAY_RUNS = [("franzen-p448s14-v1", 2048), ("franzen-p448s14rep-v1", 2048), ("franzen-l1a1tbfc-v1", 2048),
               ("franzen-l1a1y8-v1", 8192)]      # the four runs with usage logs, so a clean-turn replay exists


# family lists: the exact level vectors stored in yield_stats.json pick out the runs (same as paper/v2 fig_yield.py)
Y8 = ["franzen-l1a1y8-v1", "franzen-l1a1y8r-v1", "franzen-l1a1y8sv2-v1", "franzen-l1a1y8sv2o16-v1", "franzen-l1a1y8sv2r1-v1",
      "franzen-l1a1y8sv2r2-v1", "franzen-l1a1y8sv2r3-v1", "franzen-l1a1y8sv2r3s4-v1", "franzen-l1a1y8sv2r3hm32-v1"]
CTL = ["gcp-dpctl-v1", "gcp-dpctl-v2", "gcp-h16dp-v1", "gcp-p448s14dpwfrc-v1", "gcp-p448s14dpwfrc-v2", "gcp-l1a1-v1",
       "gcp-l1a1-v2", "gcp-l1tb-v1", "franzen-l1a1tbfc-v1", "franzen-p448-v1", "franzen-p448rep-v1", "franzen-p448s14-v1",
       "franzen-p448s14rep-v1", "franzen-p448lm-v1", "franzen-p448au-v1", "franzen-p448ho-v1", "franzen-p448sb-v1",
       "franzen-p448h14-v1", "franzen-p448h16-v1", "franzen-p448s14h71-v1", "franzen-p448s14ra-v1"]
assert [RUNS[n]["levels"] for n in Y8] == STATS["families"]["Y8 all"]["levels"]
assert [RUNS[n]["levels"] for n in CTL] == STATS["families"]["2048 all"]["levels"]
assert all(RUNS[n]["yield_tokens"] == 8192 for n in Y8) and all(RUNS[n]["yield_tokens"] == 2048 for n in CTL)


# ---- (a) per-game paired differences ----------------------------------------------------------------------------
GAMES = sorted(RUNS[Y8[0]]["per_game_levels"])
gm = lambda names, g: st.mean(RUNS[n]["per_game_levels"][g] for n in names)  # noqa: E731
DIFF = {g: gm(Y8, g) - gm(CTL, g) for g in GAMES}
SAT = [g for g in GAMES if all(RUNS[n]["per_game_levels"][g] == RUNS[n]["per_game_nlevels"][g] for n in Y8 + CTL)]
mean_d = st.mean(DIFF.values())
test = STATS["tests"]["Y8 all (9) vs all 2048"]


def boot_ci(a, b, n=20000, seed=20261013):        # paper/v2/analysis/yield_stats.py boot_ci, same seed
    rng = random.Random(seed)
    ds = sorted(st.mean(rng.choices(a, k=len(a))) - st.mean(rng.choices(b, k=len(b))) for _ in range(n))
    return ds[int(0.025 * n)], ds[int(0.975 * n)]


lo, hi = boot_ci([RUNS[n]["levels"] for n in Y8], [RUNS[n]["levels"] for n in CTL])
assert (lo, hi) == tuple(test["ci"]) and abs(25 * mean_d - test["diff"]) < 1e-9       # +12.5, CI 9.19 to 15.81
w = sum(d > 1e-9 for d in DIFF.values())
l = sum(d < -1e-9 for d in DIFF.values())
assert (w, l, 25 - w - l) == (22, 2, 1) == tuple(STATS["per_game"]["Y8 all (9) vs all 2048"][k] for k in ("wins", "losses", "ties"))
lo_g, hi_g = lo / 25, hi / 25                      # the same CI on the per-game scale
a8, a2 = [RUNS[n]["levels"] for n in Y8], [RUNS[n]["levels"] for n in CTL]
P_AB = sum((x > y) + 0.5 * (x == y) for x in a8 for y in a2) / (len(a8) * len(a2))   # Bouthillier et al. P(A > B)


# ---- (b) per-game resume openers --------------------------------------------------------------------------------
def rebuild():
    sys.path.insert(0, str(REPO / "paper/v2/analysis"))
    import yield_mech as YM
    rows = []
    for name, y in REPLAY_RUNS:
        d = YM.KO / name
        for t in sorted((d / "transcripts-dl").glob("*_p0.txt")):
            g = t.name[: -len("_p0.txt")]
            game = YM.parse_game(t, d / "usage" / f"{g}_p0_usage.jsonl")
            assert game["n_usage"] > 0
            invs, reqs = game["invs"], game["reqs"]
            obs = rep2 = rep8 = 0
            for tr in YM.turns_of(game):
                comps = [reqs[ri].get("completion") or 0 for ii in tr for ri in invs[ii]["req_idx"]]
                ok = (all(invs[ii]["end"] in ("yield", "executed") for ii in tr)
                      and all(invs[ii]["end"] == "yield" for ii in tr[:-1]))
                if ok:
                    obs += len(tr) - 1
                    rep2 += YM.resegment(comps, 2048)
                    rep8 += YM.resegment(comps, 8192)
            rows.append(dict(run=name, own_yield=y, game=g.split("-")[0], observed=obs, replay_2048=rep2, replay_8192=rep8))
    with open(RESUME_CSV, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print("wrote", RESUME_CSV)


if "--rebuild" in sys.argv or not RESUME_CSV.exists():
    rebuild()
RES = [dict(r, own_yield=int(r["own_yield"]), observed=int(r["observed"]), replay_2048=int(r["replay_2048"]),
            replay_8192=int(r["replay_8192"])) for r in csv.DictReader(open(RESUME_CSV))]
for name, y in REPLAY_RUNS:                         # the per-game rows add up to the per-run record, exactly
    rr = [r for r in RES if r["run"] == name]
    m = MECH[name]["replay_resume_in_clean_turns"]
    assert len(rr) == 25
    assert sum(r["observed"] for r in rr) == MECH[name]["observed_resume_in_clean_turns"]
    assert sum(r["replay_2048"] for r in rr) == m["2048"] and sum(r["replay_8192"] for r in rr) == m["8192"]
    assert sum(r[f"replay_{y}"] for r in rr) == sum(r["observed"] for r in rr)        # the replay reproduces its own run
ctl_runs = [n for n, y in REPLAY_RUNS if y == 2048]
games_b = sorted({r["game"] for r in RES})
X2 = {g: st.mean(r["observed"] for r in RES if r["game"] == g and r["run"] in ctl_runs) for g in games_b}
Y2 = {g: st.mean(r["replay_8192"] for r in RES if r["game"] == g and r["run"] in ctl_runs) for g in games_b}
X8 = {g: next(r["replay_2048"] for r in RES if r["game"] == g and r["run"] == "franzen-l1a1y8-v1") for g in games_b}
Y8o = {g: next(r["observed"] for r in RES if r["game"] == g and r["run"] == "franzen-l1a1y8-v1") for g in games_b}
cut_ctl = 1 - sum(Y2.values()) / sum(X2.values())
cut_y8 = 1 - sum(Y8o.values()) / sum(X8.values())
print(f"P(a random 8,192 run beats a random 2,048 run) = {P_AB:.3f}")
print(f"(a) mean per-game diff {mean_d:+.3f} (x25 = {25 * mean_d:+.2f}), CI per game {lo_g:+.3f} to {hi_g:+.3f}; up {w}, down {l}; "
      f"saturated {len(SAT)}: {SAT}")
print(f"(b) 2,048 runs: observed {sum(X2.values()):.1f} -> replay at 8,192 {sum(Y2.values()):.1f} per run ({cut_ctl:.1%} fewer); "
      f"8,192 run: replay at 2,048 {sum(X8.values())} -> observed {sum(Y8o.values())} ({cut_y8:.1%} fewer)")


# ---- (c) the interleaved wave (V3-W36) --------------------------------------------------------------------------
WAVE = json.loads((REPO / "paper/v3/analysis/out/yield_wave/read_2026-10-24.json").read_text())
iw = {arm: [r["levels"] for r in WAVE["runs"] if r["arm"] == arm] for arm in ("y8", "y2")}
assert sorted(iw["y8"]) == [113, 114, 120, 121, 127, 129] and sorted(iw["y2"]) == [119, 123, 123, 124, 125, 125]
m8, m2 = st.mean(iw["y8"]), st.mean(iw["y2"])
assert round(m8, 2) == 120.67 and round(m2, 2) == 123.17 and round(m8 - m2, 2) == -2.50
assert all(not r["invalid"] for r in WAVE["runs"])
SD8 = json.loads((V2OUT / "e4_noise.json").read_text())["franzen_levels"]["sd"]      # 8.25, six identical pairs, 6 df
T6 = 2.446912                                                                          # t quantile 0.975 at 6 df
hw = T6 * SD8 * (2 / 6) ** 0.5                                                         # half width, 6 against 6 runs
I_LO, I_HI = (m8 - m2) - hw, (m8 - m2) + hw
assert (round(I_LO, 1), round(I_HI, 1)) == (-14.2, 9.2)
ACROSS = (12.5, 4.5, 20.6)                                                             # D1: +12.5, at SD 8.25 +4.5 to +20.6
print(f"(c) interleaved 8,192 {iw['y8']} mean {m8:.2f} against 2,048 {iw['y2']} mean {m2:.2f}; diff {m8 - m2:+.2f}, "
      f"interval at SD {SD8:.2f} (6 df) {I_LO:+.1f} to {I_HI:+.1f}")

# ---- figure ------------------------------------------------------------------------------------------------------
apply(scale=SCALE)
H = 4.4
fig = plt.figure(figsize=(W, H))
axA = fig.add_axes([0.125, 0.545, 0.335, 0.335])
axB = fig.add_axes([0.655, 0.545, 0.325, 0.245])
axC = fig.add_axes([0.125, 0.105, 0.265, 0.250])
axD = fig.add_axes([0.605, 0.105, 0.385, 0.250])
INK, GRAY, ACC = C["ink"], C["gray"], C["accent"]
labels = []

# (a) one dot per game, sorted; zero line gray; the mean and its run-bootstrap CI in the accent
order = sorted(GAMES, key=lambda g: DIFF[g])
xs = list(range(len(order)))
axA.axhline(0, color=GRAY, lw=0.9, zorder=1)
axA.axhspan(lo_g, hi_g, color=ACC, alpha=0.18, lw=0, zorder=1)
axA.axhline(mean_d, color=ACC, lw=1.4, zorder=2)
for x, g in zip(xs, order):
    sat = g in SAT
    axA.plot(x, DIFF[g], "o", ms=3.8, mfc="white" if sat else INK, mec=INK, mew=0.8, zorder=3)
labels.append(axA.text(0.3, 1.38, f"mean +{25 * mean_d:.1f} levels per run\nband: run-bootstrap 95% CI",
                       fontsize=fs(6.6), color=ACC, fontweight="bold", va="bottom", ha="left", linespacing=1.05))
labels.append(axA.text(24.4, -0.2, f"{w} games up, {l} down, {25 - w - l} tie", fontsize=fs(6.2), color=INK, ha="right", va="center"))
axA.set_xlim(-1, 25)
axA.set_xticks([])
axA.spines["bottom"].set_visible(False)
axA.set_ylim(-0.75, 1.95)
axA.set_yticks([-0.5, 0, 0.5, 1.0, 1.5])
axA.set_ylabel("levels per game, 8,192 minus 2,048", labelpad=2, fontsize=fs(7.2))
axA.set_xlabel("25 games, sorted", labelpad=3)
leg = [axA.plot([], [], "o", ms=3.8, mfc="white", mec=INK, ls="")[0]]
axA.legend(leg, ["all runs cleared it all"], loc="lower right", bbox_to_anchor=(1.0, 0.0),
           handletextpad=0.2, labelspacing=0.25, borderaxespad=0.0, fontsize=fs(6.0))
panel(fig, 0.005, 0.995, "(a)", "Across days, 9 against 21 runs")

# (b) resume openers per game: x at 2,048, y at 8,192; the 2,048 runs are replayed at 8,192, the 8,192 run at 2,048
mx = 60
axB.plot([0, mx], [0, mx], color=GRAY, lw=0.8, ls=(0, (3, 2)), zorder=1)
labels.append(axB.text(10.0, 11.2, "no change", rotation=45, rotation_mode="anchor", fontsize=fs(6.2),
                       bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"),
                       color=C["gray_text"], ha="left", va="bottom"))
axB.plot(list(X2.values()), list(Y2.values()), "o", ms=3.6, mfc=C["light"], mec=INK, mew=0.6, zorder=3)
axB.plot(list(X8.values()), list(Y8o.values()), "^", ms=4.4, mfc=ACC, mec=ACC, mew=0.6, zorder=4)
axB.set_xlim(0, mx)
axB.set_ylim(0, 0.62 * mx)
axB.set_aspect("equal", adjustable="box")
axB.set_xticks([0, 20, 40, 60])
axB.set_yticks([0, 10, 20, 30])
axB.set_xlabel("openers per game at 2,048", labelpad=2)
axB.set_ylabel("at 8,192", labelpad=2)
leg = [axB.plot([], [], "o", ms=3.6, mfc=C["light"], mec=INK, ls="")[0], axB.plot([], [], "^", ms=4.4, mfc=ACC, mec=ACC, ls="")[0]]
axB.legend(leg, [f"2,048 runs: {cut_ctl:.0%} fewer", f"8,192 run: {cut_y8:.0%} fewer"],
           loc="lower left", bbox_to_anchor=(-0.28, 1.03), handletextpad=0.3, labelspacing=0.3, borderaxespad=0.0,
           fontsize=fs(6.2))
panel(fig, 0.535, 0.995, "(b)", "Resume openers, per game")

# (c) the interleaved runs: six against six, same build, alternating regions
axC.axhline(m2, color=GRAY, lw=0.8, ls=(0, (4, 2)), zorder=1)
for xi, (arm, col) in enumerate((("y2", GRAY), ("y8", ACC))):
    v = sorted(iw[arm])
    cnt = {}
    for lv_ in v:
        cnt[lv_] = cnt.get(lv_, 0) + 1
    seen = {}
    for lv_ in v:
        k = seen.get(lv_, 0)
        seen[lv_] = k + 1
        axC.plot(xi + (k - (cnt[lv_] - 1) / 2) * 0.12, lv_, "o", ms=4.8, mfc=col, mec="white", mew=0.5, zorder=3)
    axC.plot([xi - 0.3, xi + 0.3], [st.mean(v)] * 2, color=INK, lw=1.8, solid_capstyle="butt", zorder=4)
    labels.append(axC.text(xi, 132.2, f"mean {st.mean(v):.2f}", ha="center", va="top", fontsize=fs(6.2), color=INK))
axC.set_xlim(-0.6, 1.6)
axC.set_ylim(108, 133)
axC.set_xticks([0, 1])
axC.set_xticklabels(["yield 2,048", "yield 8,192"])
axC.tick_params(axis="x", length=0, pad=3)
axC.set_yticks([110, 120, 130])
axC.grid(axis="y", visible=True)
axC.set_ylabel("levels per run", labelpad=2)
panel(fig, 0.005, 0.405, "(c)", "Interleaved: 6 against 6 runs")

# the two differences, one interval each, both at the six identical pairs' SD (8.25, 6 df)
rows = [(1.0, "across days\n(9 against 21)", ACROSS[0], ACROSS[1], ACROSS[2], GRAY),
        (0.0, "interleaved\n(6 against 6)", m8 - m2, I_LO, I_HI, ACC)]
axD.axvline(0, color=GRAY, lw=0.9, zorder=1)
for y, name, d, a_, b_, col in rows:
    axD.plot([a_, b_], [y, y], color=col, lw=2.2, solid_capstyle="butt", zorder=2)
    axD.plot(d, y, "o", ms=5.6, mfc=col, mec="white", mew=0.6, zorder=3)
    labels.append(axD.text(d, y + 0.25, f"{d:+.1f}", ha="center", va="bottom", fontsize=fs(6.6), color=col, fontweight="bold"))
    labels.append(axD.text(a_, y - 0.2, f"{a_:+.1f}", ha="center", va="top", fontsize=fs(6.2), color=C["gray_text"]))
    labels.append(axD.text(b_, y - 0.2, f"{b_:+.1f}", ha="center", va="top", fontsize=fs(6.2), color=C["gray_text"]))
axD.set_yticks([r[0] for r in rows])
axD.set_yticklabels([r[1] for r in rows], linespacing=1.0, fontsize=fs(6.4))
axD.tick_params(axis="y", length=0, pad=3)
axD.set_ylim(-0.75, 1.75)
axD.set_xlim(-26, 31)
axD.set_xticks([-20, -10, 0, 10, 20, 30])
axD.spines["left"].set_visible(False)
axD.set_xlabel("levels per run, 8,192 minus 2,048", labelpad=2)
axD.grid(axis="x", visible=True)
fig.text(0.485, 0.405, "Difference, 95% interval (SD 8.25)", fontsize=fs(8.5), ha="left", va="top")
check_no_overlap(fig, labels)
save(fig, "fig_yield", writeup=True)
