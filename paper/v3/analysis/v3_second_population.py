#!/usr/bin/env python3
"""Second population for the noise recipe: MDE = 2.8016 * SD * sqrt(2/n) at n = 1, 2, 3, 5 runs per arm.

Computed here from repo files:
  - Tufa levels per pass (paper/v2/analysis/out/new_repeat_sets.json, 20 passes, read-only)
  - Franzen v3 four passes (solution/kaggle/generated/kernel-outputs/dfranzen-m2-v3/benchmark.json, via the v2 json)
  - one-shot Franzen-class draws rebuilt from daily/board-*.csv (the Pham-style cross-section)
Read from public pages (typed in below, source named): Bjarnason et al. Table 1, guoyi12311125 README scores.
Run: python3 -I -B paper/v3/analysis/v3_second_population.py   (stdlib only)
"""
import csv, json, math, os, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(HERE, "out")
Z = 2.8016  # 1.95996 + 0.84162, two-sided 5%, 80% power, SD known (the paper's multiplier)
NS = (1, 2, 3, 5)


def mde(sd):
    return {f"n{n}": round(Z * sd * math.sqrt(2 / n), 2) for n in NS}


def block(vals):
    sd = st.stdev(vals)
    m = st.mean(vals)
    return dict(n=len(vals), mean=round(m, 3), sd=round(sd, 3), cv=round(sd / m, 4), mde=mde(sd))


res = {}

# 1. Tufa (computed from the cached per-pass totals)
v2 = json.load(open(os.path.join(REPO, "paper/v2/analysis/out/new_repeat_sets.json")))
tufa = v2["tufa"]["levels_per_pass"]
tufa = tufa if isinstance(tufa, list) else list(tufa.values())
b = block(tufa)
assert b["n"] == 20 and abs(b["sd"] - 2.44) < 0.01
b["unit"] = "levels of 183 (25 public games x 1 pass), one job, Qwen3.6-27B"
b["mde_pct_of_mean"] = {k: round(100 * x / b["mean"], 1) for k, x in b["mde"].items()}
res["tufa_20_passes"] = b

# 2. Franzen v3 four passes (levels 102,106,104,105)
fz = block([102, 106, 104, 105])
fz["unit"] = "levels of 183, 4 passes in one job (values read from v2 note and benchmark.json summary)"
fz["mde_pct_of_mean"] = {k: round(100 * x / fz["mean"], 1) for k, x in fz["mde"].items()}
res["franzen_v3_4_passes"] = fz

# 3. Field cross-section from snapshots (not repeats): exactly one new submission, prior best < 10, new best >= floor
def load(tag):
    rows = list(csv.DictReader(open(os.path.join(REPO, f"solution/kaggle/generated/daily/board-2026-{tag}.csv"), encoding="utf-8-sig")))
    return {r["TeamId"]: (float(r["Score"]), int(r["SubmissionCount"])) for r in rows}

A, B = load("10-01"), load("10-02")
field = {}
for thr in (10, 15, 20):
    x = [s for t, (s, c) in B.items() if c - A.get(t, (0.0, 0))[0:2][1] == 1 and A.get(t, (0.0, 0))[0] < 10 and s >= thr]
    d = block(x)
    d["floor"] = thr
    d["unit"] = "public board score points, different teams, first scored draw, window 10-01 to 10-02"
    d["median"] = st.median(x)
    field[f"floor{thr}"] = d
assert field["floor15"]["n"] == 215 and abs(field["floor15"]["sd"] - 3.04) < 0.01
res["field_cross_section"] = field

# 4. Read from public pages
# Bjarnason, Silva, Monperrus, arXiv 2602.07150, Table 1 (HTML version), m = 10 runs per cell, N = 500 tasks
bj_rows = [
    ("Qwen3-32B t0.6 nano-agent", 16.4, 0.7), ("Qwen3-32B t0.6 r2e-gym", 23.9, 1.4),
    ("DeepSWE t1.0 nano-agent", 31.4, 1.0), ("DeepSWE t1.0 r2e-gym", 34.4, 1.5),
    ("Devstral-2 t0.2 nano-agent", 63.5, 1.1), ("Devstral-2 t0.2 r2e-gym", 34.9, 1.5),
    ("Qwen3-32B t0 nano-agent", 16.4, 1.2), ("Qwen3-32B t0 r2e-gym", 22.3, 1.8),
    ("DeepSWE t0 nano-agent", 20.4, 1.0), ("DeepSWE t0 r2e-gym", 19.2, 1.5),
    ("Devstral-2 t0 nano-agent", 63.8, 1.6), ("Devstral-2 t0 r2e-gym", 35.4, 1.7),
]
bj = []
for name, m, sd in bj_rows:
    bj.append(dict(cell=name, mean_pp=m, sd_pp=sd, cv=round(sd / m, 3), mde_pp=mde(sd)))
sds = [r[2] for r in bj_rows]
res["bjarnason_swebench_verified"] = dict(
    source="arXiv 2602.07150 Table 1 (read via arxiv.org/html/2602.07150): pass@1 mean +/- std over 10 runs, N=500 tasks",
    cells=bj, sd_min=min(sds), sd_median=st.median(sds), sd_max=max(sds),
    mde_at_median_sd=mde(st.median(sds)), mde_at_max_sd=mde(max(sds)), mde_at_min_sd=mde(min(sds)),
    abstract_quotes=["standard deviations exceeding 1.5 percentage points even at temperature 0",
                     "2.2 to 6.0 percentage points depending on which single run is selected"],
    note="SD is within a fixed config over 10 runs; the 2.2 to 6.0 range is the spread of pass@1 over runs per the abstract (read), not a gain.")

# guoyi12311125 README (arc3-repeat-controlled-eval), ARC-AGI-3 Kaggle public score, same code
g = [24.38, 29.41, 29.60]
gb = block(g)
gb["unit"] = "Kaggle public score points, 3 submissions labelled identical code"
gb["with_byte_identical_pair_29.89_30.21_added"] = block(g + [29.89, 30.21])
gb["note"] = "README gives scores only; SD computed here. Unit not stated by the README; conditions (days, hardware) unknown."
res["guoyi_readme_identical_code"] = gb

# Anthropic infrastructure noise post (read): no SD or run count published
res["anthropic_infra_noise"] = dict(
    source="anthropic.com/engineering/infrastructure-noise (read)",
    published=dict(terminal_bench_extremes_gap_pp=6, spread_moderate_configs_pp="just below 2", leaderboard_skepticism_below_pp=3,
                   swebench_ram_5x_effect_pp=1.54, terminal_bench_3x_vs_1x_p=0.40),
    sd_published=False, mde_computable=False,
    note="Naive binomial CI roughly 1 to 2 pp (read). No SD, no runs per config, so the recipe cannot be applied.")
res["miller_error_bars"] = dict(source="arXiv 2411.00640 abstract only (read)", sd_published=False, mde_computable=False)
res["kapoor_agents_that_matter"] = dict(source="arXiv 2407.01502 abstract only (read)", sd_published=False, mde_computable=False)

# our own comparators (read from paper/v3/evidence/data/mde_by_runs.csv)
res["ours_for_comparison"] = dict(sd_levels=[3.9, 5.5, 5.95, 8.25], mean_levels_approx=[111.5, 124.3],
                                  mde_n1_levels=[15.45, 21.79, 23.58, 32.69], cv_range="3.3% to 7.2%")
json.dump(res, open(os.path.join(OUT, "second_population.json"), "w"), indent=1)
print(json.dumps({k: (v if k in ("tufa_20_passes", "franzen_v3_4_passes", "guoyi_readme_identical_code") else "...") for k, v in res.items()}, indent=1)[:3000])
print("field", {k: (v["n"], v["sd"], v["mde"]) for k, v in field.items()})
print("bj sd median/min/max", st.median(sds), min(sds), max(sds), res["bjarnason_swebench_verified"]["mde_at_median_sd"], res["bjarnason_swebench_verified"]["mde_at_max_sd"])
print("bj cv min/max", min(r["cv"] for r in bj), max(r["cv"] for r in bj))
