"""Skeptic-lens verification numbers. python3 -I -B paper/v3/analysis/v3_skeptic_checks.py
Reads paper/v2/analysis/out/yield_runs.json, evidence/data/{board_draws.csv, bench_pool18_levels.json},
analysis/figs/data/board_draws_all_2026-11-02.csv. Writes analysis/out/v3_skeptic_checks.json. stdlib + scipy."""
import csv, itertools, json, math, statistics as st
from pathlib import Path
from scipy import stats
R = Path(__file__).resolve().parents[3]
runs = json.load(open(R/"paper/v2/analysis/out/yield_runs.json"))
by = {r["dir"]: r for r in runs}
out = {}
# 1. six identical pairs (first, second), from V3-W37 and yield_runs.json
pairs = {"p448": ("franzen-p448-v1","franzen-p448rep-v1"), "p448s14": ("franzen-p448s14-v1","franzen-p448s14rep-v1"),
 "l1a1y8": ("franzen-l1a1y8-v1","franzen-l1a1y8r-v1"), "dpctl": ("gcp-dpctl-v1","gcp-dpctl-v2"),
 "L1(p448s14dpwfrc)": ("gcp-p448s14dpwfrc-v1","gcp-p448s14dpwfrc-v2"), "l1a1": ("gcp-l1a1-v1","gcp-l1a1-v2")}
g = []; rec = []
for k,(a,b) in pairs.items():
    A,B = by[a],by[b]; g.append(A["levels"]-B["levels"])
    prior = [r["levels"] for r in runs if r["start_time"] < A["start_time"]]
    rec.append({"pair":k,"first":A["levels"],"second":B["levels"],"gap":A["levels"]-B["levels"],"first_was_record_high_at_its_time":A["levels"]>max(prior) if prior else None,
                "hours_apart":round((__import__("datetime").datetime.fromisoformat(B["start_time"])-__import__("datetime").datetime.fromisoformat(A["start_time"])).total_seconds()/3600,1)})
m = st.mean(g); n = len(g)
sd_incl = math.sqrt(sum(x*x for x in g)/(2*n)); sd_about = st.stdev(g)/math.sqrt(2)
t = m/(st.stdev(g)/math.sqrt(n)); p_t = 2*stats.t.sf(abs(t), n-1)
out["pairs"] = {"rows":rec,"gaps":g,"mean_shift":m,"sd_including_shift":sd_incl,"sd_about_shift":sd_about,"paired_t":t,"paired_t_p_two_sided":p_t,
  "sign_test_two_sided":2*0.5**n,"n_pairs_first_was_record_high":sum(1 for r in rec if r["first_was_record_high_at_its_time"])}
g2 = g[1:]; out["pairs_without_p448"] = {"gaps":g2,"mean":st.mean(g2),"sd_about_shift":st.stdev(g2)/math.sqrt(2)}
# 2. MDE for 2 runs per arm, 2.8*SD*sqrt(2/n)
out["mde_two_runs"] = {f"sd_{s}":round(2.8*s,1) for s in (3.9,4.7,5.6,5.95,8.25)}
# 3. across-days (9 treated vs 21 controls) vs interleaved
tr = [r["levels"] for r in runs if r["yield_tokens"]==8192]
ct = [r["levels"] for r in runs if r["yield_tokens"]==2048 and r["start_time"]>="2026-10-03" and r["start_time"]<"2026-10-11"]
out["across_days"] = {"n_treated":len(tr),"mean_treated":st.mean(tr),"sd_treated":st.stdev(tr),"n_ctrl_2048_oct3_to_oct10":len(ct),"mean_ctrl":st.mean(ct),"sd_ctrl":st.stdev(ct)}
# the writeup's 21 controls: all 2048 runs
ct21 = [r["levels"] for r in runs if r["yield_tokens"]==2048 and r["start_time"]>="2026-10-03" and r["platform"]!="x"]
ct21 = [r["levels"] for r in runs if r["yield_tokens"]==2048 and r["start_time"]>="2026-10-03" and r["dir"]!="franzen-start-v1"]  # excluding start-v1 (92) reproduces the Writeup's 21 controls and +12.5
out["across_days"].update({"n_ctrl_all_2048_from_oct3":len(ct21),"mean_ctrl_all":st.mean(ct21),"sd_ctrl_all":st.stdev(ct21)})
d1 = st.mean(tr)-st.mean(ct21)
for s in (8.25,5.95):
    se = s*math.sqrt(1/len(tr)+1/len(ct21)); out["across_days"][f"diff_se_at_sd_{s}"]=round(se,2); out["across_days"][f"z_at_sd_{s}"]=round(d1/se,2)
out["across_days"]["diff"]=d1
# interleaved vs across-days difference
se_int = lambda s: s*math.sqrt(2/6)
for s in (8.25,5.95):
    se_tot = math.sqrt((s*math.sqrt(1/9+1/21))**2 + se_int(s)**2)
    out["across_days"][f"across_minus_interleaved_{s}"]={"diff":d1-(-2.5),"se":round(se_tot,2),"z":round((d1+2.5)/se_tot,2)}
# 4. same-day eight runs: later pair vs earlier six
e8 = [115,117,124,125,126,128,130,132]; later=[130,128]; early=[115,117,124,125,126,132]
out["same_day_eight"] = {"mean":st.mean(e8),"sd":st.stdev(e8),"later_two_(rq5,rq6)":later,"mean_later":st.mean(later),"mean_early_six":st.mean(early),"later_minus_early":st.mean(later)-st.mean(early)}
# 5. pooled-18 and adapter rank tests
pool = json.load(open(R/"paper/v3/evidence/data/bench_pool18_levels.json"))
c18 = [v["all25"] for v in pool.values() if "all25" in v]
ad = [109,115,117]
out["pool18"] = {"n":len(c18),"mean":st.mean(c18),"sd":st.stdev(c18)}
def rank_p(ctrl, adp):
    allv = ctrl+adp; k=len(adp); obs=sum(adp); cnt=0; tot=0
    for idx in itertools.combinations(range(len(allv)),k):
        s=sum(allv[i] for i in idx); tot+=1
        if s<=obs: cnt+=1   # ties (including the observed split) counted as extreme
    return cnt/tot, tot
out["rank_tests"] = {"adapters_vs_18":rank_p(c18,ad),"adapters_vs_2_same_card_(132,130)":rank_p([132,130],ad),"min_possible_p_n3_vs_n2":1/math.comb(5,3),
   "adapters_vs_18_plus_2":rank_p(c18+[132,130],ad),"adapter_only_scale2_two_runs_vs_18":rank_p(c18,[109,115])}
# 6. board draws
rows = list(csv.DictReader(open(R/"paper/v3/analysis/figs/data/board_draws_all_2026-11-02.csv")))
d = [(r["submitted_utc"][:16],float(r["public_score"]),r["config"],r["ref"]) for r in rows if r["submitted_utc"]>="2026-10-03" and r["status"]=="complete"]
live = [x for x in d if x[1]>0]
adapter = [x for x in d if "wesrp" in x[2]]
out["board"] = {"n_completed_since_fork":len(d),"n_live":len(live),"adapter_draws":[x[:2] for x in adapter],"adapter_mean":st.mean(x[1] for x in adapter),
  "zeros":[x[:2] for x in d if x[1]==0]}
valid = [x for x in live if "wesrp" not in x[2] and x[0]>="2026-10-13"]
# the Writeup's 17 valid: r3p/hm32p/r3kp/kp12/harness builds, excludes adapter, 10-12 y8sv2p and earlier
out["board"]["valid_since_10-13_no_adapter"] = {"n":len(valid),"mean":round(st.mean(x[1] for x in valid),2),"cfgs":sorted(set(x[2] for x in valid))}
v2 = valid+[x for x in d if x[1]==0 and x[0]=="2026-10-18T06:56"]
out["board"]["with_10-18_zero"] = {"n":len(v2),"mean":round(st.mean(x[1] for x in v2),2)}
out["board"]["mean_adapter_plus_valid"] = round(st.mean([x[1] for x in valid]+[x[1] for x in adapter]),2)
json.dump(out, open(R/"paper/v3/analysis/out/v3_skeptic_checks.json","w"), indent=1, default=str)
print(json.dumps(out, indent=1, default=str))
