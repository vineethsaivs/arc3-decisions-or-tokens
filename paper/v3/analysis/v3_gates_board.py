"""Gate false-ship simulation and board-selection analysis. numpy + scipy. Run: python3 -I v3_gates_board.py
Writes out/gates_board.json and out/gates_board.md. Reads figs/data/harness_runs.csv and
figs/data/board_draws_all_2026-11-02.csv (both in the repo)."""
import csv, json, os
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
rng = np.random.default_rng(20261104)
R = {}

# ---------- (1) gate false-ship ----------
# Rule (gate-predeclared-2026-10-30..11-01 *.json, V3-W19): no_harm = arm mean levels >= control mean - 2.5
# AND arm mean score >= control mean - 1.5. Control 7 runs, arm 3 runs.
rows = list(csv.DictReader(open(os.path.join(HERE, "figs/data/harness_runs.csv"))))
lv = np.array([float(r["levels"]) for r in rows]); sc = np.array([float(r["score"]) for r in rows])
slope, icpt, rho, _, _ = stats.linregress(lv, sc)
resid_sd = float(np.std(sc - (icpt + slope * lv), ddof=2))
R["score_vs_levels"] = dict(n_runs=len(rows), slope_score_per_level=slope, corr=rho, resid_sd=resid_sd)

def p_ship_levels(d, sd, na, nc, thr=-2.5):
    se = sd * np.sqrt(1 / na + 1 / nc)
    return float(stats.norm.sf((thr - d) / se))

def p_ship_joint(d, sd, na, nc, n=400000):
    """levels and score together: score = slope*levels + noise(resid_sd); a true level effect d moves score by slope*d."""
    la = rng.normal(d, sd / np.sqrt(na), n); lc = rng.normal(0, sd / np.sqrt(nc), n)
    sa = slope * la + rng.normal(0, resid_sd / np.sqrt(na), n)
    sctl = slope * lc + rng.normal(0, resid_sd / np.sqrt(nc), n)
    return float(np.mean((la - lc >= -2.5) & (sa - sctl >= -1.5)))

effects = [-6, -3, 0, 3, 6]
sds = [4.7, 5.95, 8.25]
tab = {}
for na, nc in [(3, 7), (2, 7), (3, 3)]:
    for sd in sds:
        tab[f"{na}v{nc}_sd{sd}"] = dict(
            levels_only={str(d): round(p_ship_levels(d, sd, na, nc), 3) for d in effects},
            levels_and_score={str(d): round(p_ship_joint(d, sd, na, nc), 3) for d in effects},
            lead_ge_0_rule={str(d): round(p_ship_levels(d, sd, na, nc, 0.0), 3) for d in effects},
            true_plus6_missed_levels_only=round(1 - p_ship_levels(6, sd, na, nc), 3),
            true_plus6_missed_joint=round(1 - p_ship_joint(6, sd, na, nc), 3))
R["gate_table"] = tab

# ---------- (2) board ----------
draws = list(csv.DictReader(open(os.path.join(HERE, "figs/data/board_draws_all_2026-11-02.csv"))))
fork = [d for d in draws if d["submitted_utc"] >= "2026-10-03" and d["status"] == "complete"]
fork.sort(key=lambda d: d["submitted_utc"])
def cfg(d): return d["config"].replace("franzen-l1a1y8sv2r3", "").replace("franzen-", "")
live = [d for d in fork if float(d["public_score"]) > 0 and d["submitted_utc"] >= "2026-10-04"]
R["n_fork_completed"] = len(fork); R["n_live_nonzero_since_1004"] = len(live)
post = [d for d in live if d["submitted_utc"] >= "2026-10-12"]            # from the yield change on
adapter = [d for d in post if cfg(d) == "wesrp"]
valid = [d for d in post if cfg(d) != "wesrp" and d["submitted_utc"] >= "2026-10-13"]
sc_ = lambda ds: np.array([float(d["public_score"]) for d in ds])
def summ(ds):
    x = sc_(ds); return dict(n=len(x), mean=round(float(x.mean()), 3), sd=round(float(x.std(ddof=1)), 3), min=float(x.min()), max=float(x.max()))
R["sets"] = dict(
    live_nonzero_1004_on=summ(live),
    post_yield_nonzero_1012_on=summ(post),
    valid17=summ(valid),
    adapter3=summ(adapter),
    pre_yield7=summ([d for d in live if d["submitted_utc"] < "2026-10-12"]),
    valid17_plus_1018_zero=dict(n=18, mean=round(float(np.append(sc_(valid), 0).mean()), 3)))
R["valid17_dates_configs"] = [(d["submitted_utc"][:10], d["public_score"], cfg(d)) for d in valid]
R["reconcile_17_vs_21"] = dict(
    definition_17="all nonzero draws 10-13 to 11-02 (r3p, hm32p, kp, kp12, kp12lf15, two harness builds) minus the 3 wesrp adapter draws; the 10-12 y8sv2p draw (34.34, pre-r3 build) and the 0.00 of 10-18 are excluded",
    n21="the 21 nonzero post-yield draws (10-12 to 11-02) = the 17 + 3 adapter draws (35.90, 39.79, 35.98) + the 10-12 draw (34.34)",
    check=round(float((sc_(valid).sum() + sc_(adapter).sum() + 34.34) / 21), 3))

# within-build SD (builds with >=2 valid draws), ANOVA / method of moments between-build
by = {}
for d in valid: by.setdefault(cfg(d), []).append(float(d["public_score"]))
ss_w = sum(((np.array(v) - np.mean(v)) ** 2).sum() for v in by.values()); df_w = sum(len(v) - 1 for v in by.values())
sd_within = float(np.sqrt(ss_w / df_w))
grp = [np.array(v) for v in by.values()]
F, pF = stats.f_oneway(*grp)
N = sum(len(g) for g in grp); k = len(grp)
n0 = (N - sum(len(g) ** 2 for g in grp) / N) / (k - 1)
msb = sum(len(g) * (g.mean() - sc_(valid).mean()) ** 2 for g in grp) / (k - 1)
msw = ss_w / df_w
tau2 = max(0.0, (msb - msw) / n0)
R["builds"] = {c: dict(n=len(v), mean=round(float(np.mean(v)), 2), draws=v) for c, v in by.items()}
R["within_build_sd"] = round(sd_within, 3); R["anova_F"] = round(float(F), 3); R["anova_p"] = round(float(pF), 3)
R["between_build_sd_tau"] = round(float(np.sqrt(tau2)), 3)

x17 = sc_(valid); mu, sd17 = x17.mean(), x17.std(ddof=1)
def emax(n, mean, sd, reps=200000, bound=None):
    m = rng.normal(mean, sd, (reps, n)).max(1)
    return dict(n=n, mean=mean, sd=sd, E_max=round(float(m.mean()), 2), sd_max=round(float(m.std()), 2),
                p_max_ge_40_98=round(float((m >= 40.98).mean()), 3), p95=round(float(np.percentile(m, 95)), 2))
R["expected_max"] = dict(
    n17_sd_family=emax(17, mu, sd17), n17_sd_within=emax(17, mu, sd_within),
    n21_sd_family=emax(21, sc_(post).mean(), sc_(post).std(ddof=1)), n21_sd_within=emax(21, sc_(post).mean(), sd_within),
    p_single_draw_ge_40_98_family=round(float(stats.norm.sf((40.98 - mu) / sd17)), 4),
    z_40_98_family=round(float((40.98 - mu) / sd17), 2))
# all 28 live draws: 7 pre-yield at their own mean, 21 post-yield
pre = sc_([d for d in live if d["submitted_utc"] < "2026-10-12"])
m28 = np.concatenate([rng.normal(pre.mean(), pre.std(ddof=1), (200000, 7)), rng.normal(sc_(post).mean(), sd_within, (200000, 21))], 1).max(1)
R["expected_max"]["n28_mixture_7pre_21post"] = dict(E_max=round(float(m28.mean()), 2), p_ge_40_98=round(float((m28 >= 40.98).mean()), 3), p95=round(float(np.percentile(m28, 95)), 2))
# bootstrap of the 17 valid draws
bm = rng.choice(x17, (200000, 17)).max(1)
R["expected_max"]["n17_bootstrap"] = dict(E_max=round(float(bm.mean()), 2), p_ge_40_98=round(float((bm >= 40.98).mean()), 3))
bm = rng.choice(sc_(post), (200000, 21)).max(1)
R["expected_max"]["n21_bootstrap"] = dict(E_max=round(float(bm.mean()), 2), p_ge_40_98=round(float((bm >= 40.98).mean()), 3))

# private expectation: shrink the r3p build mean toward the family mean (random-effects), plus draw noise
r3p = np.array(by["p"]); n_r = len(r3p)
rel = tau2 / (tau2 + sd_within ** 2 / n_r)
shrunk = mu + rel * (r3p.mean() - mu)
R["private_expectation"] = dict(
    r3p_mean=round(float(r3p.mean()), 2), reliability=round(float(rel), 3), shrunk_true_mean_r3p=round(float(shrunk), 2),
    family_mean=round(float(mu), 2), per_draw_sd=round(float(sd17), 2),
    note="Public and private halves hold different games, so a config's true mean can differ between halves by a game-split term not estimable from these data. Expected private for the best two public draws: between the family mean and the shrunk r3p mean, with the draw SD around it.")

os.makedirs(OUT, exist_ok=True)
json.dump(R, open(os.path.join(OUT, "gates_board.json"), "w"), indent=1, default=float)

# ---------- markdown ----------
L = ["# Gates and board (computed by analysis/v3_gates_board.py)", "",
     "## 1. False-ship simulation (computed)", "",
     "Rule read from `solution/kaggle/generated/gate-predeclared-2026-10-30..11-01-*.json` and V3-W19: ship if arm mean levels >= control mean - 2.5 AND arm mean score >= control mean - 1.5 (plus mechanism and action checks, not simulated). Control 7 runs, arm 3 runs. Normal run noise; control independent.",
     f"Score model for the joint column, fitted on {len(rows)} harness runs: {slope:.3f} score points per level, r = {rho:.2f}, residual SD {resid_sd:.2f}.",
     "I judge SD 5.95 (eight same-day runs) the central frame: 8.25 includes a shared session shift (pair SD about the shift is about 5.6, diag_skeptic), 4.7 is the 18-run pool across days and platforms. Control SD in the 7-run set itself is 7.76.", "",
     "P(ship), 3 arm runs vs 7 control, levels rule only / levels and score:", "",
     "| SD | true -6 | -3 | 0 | +3 | +6 | true +6 missed |", "|---|---|---|---|---|---|---|"]
for sd in sds:
    t = tab[f"3v7_sd{sd}"]
    cell = lambda d: f"{t['levels_only'][str(d)]:.2f} / {t['levels_and_score'][str(d)]:.2f}"
    L.append(f"| {sd} | " + " | ".join(cell(d) for d in effects) + f" | {t['true_plus6_missed_levels_only']:.2f} / {t['true_plus6_missed_joint']:.2f} |")
L += ["", "Other designs (levels rule only), SD 5.95: " + "; ".join(
    f"{k.split('_')[0]}: -6 {v['levels_only']['-6']:.2f}, 0 {v['levels_only']['0']:.2f}, +6 {v['levels_only']['6']:.2f}" for k, v in tab.items() if k.endswith("5.95")), "",
    "The gate screens out large losers only. It ships a null change about 7 times in 10 and a true -6 about 1 in 5 (SD 5.95), and it misses a true +6 rarely; it cannot tell +3 from 0. The score condition (-1.5) lowers every column a few points (see the second number) but does not change this reading.", "",
     "## 2. Board selection (computed)", "",
     f"Definition of the Writeup's 17: nonzero draws 2026-10-13 to 2026-11-02 of the agent's family (r3p 3, hm32p 4, kp 3, kp12 2, kp12lf15 3, two harness builds 1 each) = 17; excludes the 3 wesrp adapter draws, the 10-12 draw of the pre-r3 build y8sv2p (34.34) and the 10-18 zero. Mean {R['sets']['valid17']['mean']}, SD {R['sets']['valid17']['sd']}, range 28.97 to 40.98.",
     f"The insight lens's 21 = the same 17 + 3 adapter draws + the 10-12 draw: mean {R['sets']['post_yield_nonzero_1012_on']['mean']}, SD {R['sets']['post_yield_nonzero_1012_on']['sd']} (check {R['reconcile_17_vs_21']['check']}). Both numbers are right; they are different sets. 'Best of 28 live draws' = those 21 + 7 pre-yield nonzero draws (10-04 to 10-11, mean {R['sets']['pre_yield7']['mean']}).",
     f"Within-build SD (builds with 2 or more draws) {R['within_build_sd']}; one-way ANOVA across builds F {R['anova_F']}, p {R['anova_p']}; between-build SD (method of moments) {R['between_build_sd_tau']}.", "",
     "Expected maximum of n draws and chance of reaching 40.98:", "", "| model | n | E[max] | P(max >= 40.98) |", "|---|---|---|---|"]
e = R["expected_max"]
for name, key, n in [("normal, family SD", "n17_sd_family", 17), ("normal, within-build SD", "n17_sd_within", 17), ("bootstrap", "n17_bootstrap", 17),
                     ("normal, 21 post-yield, family SD", "n21_sd_family", 21), ("normal, 21, within-build SD", "n21_sd_within", 21), ("bootstrap, 21", "n21_bootstrap", 21),
                     ("7 pre-yield + 21 post-yield", "n28_mixture_7pre_21post", 28)]:
    v = e[key]; L.append(f"| {name} | {n} | {v['E_max']} | {v.get('p_max_ge_40_98', v.get('p_ge_40_98'))} |")
L += ["", f"A single draw of the family reaches 40.98 with p {e['p_single_draw_ge_40_98_family']} (z {e['z_40_98_family']}). Against an expected best of about {e['n17_sd_family']['E_max']} to {e['n21_sd_family']['E_max']}, 40.98 is inside selection noise: it is a typical best of the draws, not evidence that r3p is better than the family.",
      f"r3p's three draws average {R['private_expectation']['r3p_mean']}; with reliability {R['private_expectation']['reliability']} the shrunk true mean is {R['private_expectation']['shrunk_true_mean_r3p']}, {R['private_expectation']['shrunk_true_mean_r3p']-R['private_expectation']['family_mean']:.1f} above the family mean {R['private_expectation']['family_mean']}; the build effect is not established (ANOVA p {R['anova_p']}).",
      f"Expected private score for the two auto-selected draws: between {R['private_expectation']['family_mean']} (family mean) and {R['private_expectation']['shrunk_true_mean_r3p']} (shrunk r3p mean), per-draw SD {R['private_expectation']['per_draw_sd']}, not 40.98. {R['private_expectation']['note']}"]
open(os.path.join(OUT, "gates_board.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
