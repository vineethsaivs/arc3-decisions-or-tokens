#!/usr/bin/env python3
"""Per-run table of the harness program (paper Sec. 6 and Appendix J), read from the Kaggle and Vertex run outputs.

Run from the repo root (stdlib only, no network):
    python3 -I -B paper/v3/analysis/harness_read.py

Reads solution/kaggle/generated/kernel-outputs/<run>/benchmark.json and the *_p0.txt transcripts (counters come from
solution/analysis/unstick_counters.py). Writes paper/v3/analysis/figs/data/harness_runs.csv and prints, per arm, the mean
levels and score, the lead over the pooled control and each predeclared gate item (the same rules as the
gate-predeclared-2026-10-30/31 files; a scroll arm with fewer than 3 runs is INCOMPLETE).
"""
import csv
import json
import re
import statistics as st
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
K = ROOT / "solution" / "kaggle" / "generated" / "kernel-outputs"
OUT = Path(__file__).resolve().parent / "figs" / "data" / "harness_runs.csv"
CONTROL = ["vertex-kva", "vertex-s15w-1", "vertex-s15w-2", "vertex-s15w-3", "vertex-s15w-4", "kaggle-lf15-1029",
           "kaggle-lf15-1029-2"]
ARMS = {  # arm -> (description, run directories)
    "unst": ("stall check, first build (killed on m1)", [f"kaggle-lf15unst-1030-{i}" for i in (1, 2)]),
    "uns2": ("stall check and coverage", [f"kaggle-lf15uns2-1030-{i}" for i in (1, 2, 3)]),
    "us8": ("plus corner-joined regions", [f"kaggle-lf15us8-1031-{i}" for i in (1, 2, 3)]),
    "usc": ("plus scroll detector", [f"kaggle-lf15usc-1031-{i}" for i in (1, 2, 3)]),
}
GROUP = re.compile(r"""\[\s*['"]group['"]\s*\]|\.get\(\s*['"]group['"]""")
SCROLL = "The view scrolled during that sequence"


def play(d):
    games = json.loads((K / d / "benchmark.json").read_text())["game_runs"]
    ratios = [a / b for g in games for i, (a, b) in enumerate(zip(g["actions_per_level"], g["base_actions_per_level"]))
              if i < g["levels_completed"] and b]
    return dict(levels=sum(g["levels_completed"] for g in games), score=st.mean(g["final_score"] for g in games),
                actions=sum(sum(g["actions_per_level"]) for g in games), median_ratio=st.median(ratios),
                games={g["game_id"][:4]: g["levels_completed"] for g in games})


def counters(d):
    out = subprocess.run([sys.executable, "-I", "-B", str(ROOT / "solution/analysis/unstick_counters.py"), str(K / d)],
                         capture_output=True, text=True, check=True).stdout
    n = dict(re.findall(r"(at-stall turns|acted|stall lines|untested lines|newtype lines|streak>=5) (\S+)", out))
    at = int(n["at-stall turns"])
    return dict(stall=int(n["stall lines"]), untested=int(n["untested lines"]), newtype=int(n["newtype lines"]),
                at_stall=at, acted=round(float(n["acted"]) * at), streak5=int(n["streak>=5"]))


def transcripts(d):
    codes = scroll_in = scroll_out = 0
    for f in (K / d).glob("*_p0.txt"):
        t = f.read_text(errors="replace")
        codes += sum(bool(GROUP.search(c)) for c in re.findall(r"<parameter=code>\n(.*?)</parameter>", t, re.S))
        n = t.count(SCROLL)
        if f.name.split("-")[0] in ("bp35", "lf52"):
            scroll_in += n
        else:
            scroll_out += n
    return dict(group_blocks=codes, scroll_in=scroll_in, scroll_out=scroll_out)


def main():
    rows, arm_runs = [], {}
    ctrl = {d: play(d) for d in CONTROL}
    for d, r in ctrl.items():
        rows.append(dict(arm="control", run=d, **{k: v for k, v in r.items() if k != "games"}))
    ca = {k: st.mean(r[k] for r in ctrl.values()) for k in ("levels", "score", "actions", "median_ratio")}
    print("control: levels %s mean %.1f SD %.1f; score mean %.2f SD %.2f; actions %.0f; median ratio %.3f"
          % ([r["levels"] for r in ctrl.values()], ca["levels"], st.stdev(r["levels"] for r in ctrl.values()),
             ca["score"], st.stdev(r["score"] for r in ctrl.values()), ca["actions"], ca["median_ratio"]))
    for arm, (desc, dirs) in ARMS.items():
        dirs = [d for d in dirs if (K / d / "benchmark.json").exists()]
        runs = []
        for d in dirs:
            r = play(d)
            r.update(counters(d))
            r.update(transcripts(d))
            rows.append(dict(arm=arm, run=d, **{k: v for k, v in r.items() if k != "games"}))
            runs.append(r)
        arm_runs[arm] = runs
        ml, ms = st.mean(r["levels"] for r in runs), st.mean(r["score"] for r in runs)
        at, ac = sum(r["at_stall"] for r in runs), sum(r["acted"] for r in runs)
        print("\n%s (%s): %d of %d runs; levels %s mean %.1f (%+.1f vs control); score mean %.2f (%+.2f)"
              % (arm, desc, len(runs), len(ARMS[arm][1]), [r["levels"] for r in runs], ml, ml - ca["levels"], ms,
                 ms - ca["score"]))
        print("  idle turns that act at the 3rd/5th/7th: %d of %d = %.2f; streaks>=5 per run %.2f; stall %s untested %s "
              "newtype %s; group blocks %s; scroll openers in bp35+lf52 %s, elsewhere %s"
              % (ac, at, ac / max(1, at), st.mean(r["streak5"] for r in runs), [r["stall"] for r in runs],
                 [r["untested"] for r in runs], [r["newtype"] for r in runs], [r["group_blocks"] for r in runs],
                 [r["scroll_in"] for r in runs], [r["scroll_out"] for r in runs]))
        print("  m3 actions %.0f <= %.0f: %s; median ratio %.3f <= %.3f: %s; no-harm levels %.1f >= %.1f: %s; score "
              "%.2f >= %.2f: %s"
              % (st.mean(r["actions"] for r in runs), ca["actions"] * 1.1, st.mean(r["actions"] for r in runs) <= ca["actions"] * 1.1,
                 st.mean(r["median_ratio"] for r in runs), ca["median_ratio"] + .05,
                 st.mean(r["median_ratio"] for r in runs) <= ca["median_ratio"] + .05, ml, ca["levels"] - 2.5,
                 ml >= ca["levels"] - 2.5, ms, ca["score"] - 1.5, ms >= ca["score"] - 1.5))
        if arm == "usc" and len(runs) < 3:
            need = 3 * (ca["levels"] - 2.5) - sum(r["levels"] for r in runs)
            print("  INCOMPLETE: %d of 3 runs read; a third run would need %.1f levels for the no-harm mean" % (len(runs), need))
    fields = ["arm", "run", "levels", "score", "actions", "median_ratio", "stall", "untested", "newtype", "at_stall",
              "acted", "streak5", "group_blocks", "scroll_in", "scroll_out"]
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print("\nwrote", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
