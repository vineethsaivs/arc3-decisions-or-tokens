"""Recompute the v3 dossier numbers for the board (topic 1), hm32 (2) and trimtest plus K0 (5).

    python3 -I -B paper/v3/analysis/v3_board_trim.py

Reads only local files: paper/v3/evidence/data/board_draws.csv (Kaggle submissions API, fetched 2026-10-21),
solution/kaggle/generated/daily/board-*.csv, the r3 and hm32 replica outputs in solution/kaggle/generated/
kernel-outputs/, and the trimtest replies in ~/.cache/arc3/trimtest-2026-10-20/. Writes rank_history.csv and
trimtest_by_state.csv next to board_draws.csv. Prints every number the dossier quotes for these topics.
"""
import csv
import itertools
import json
import math
import random
import re
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "paper/v3/evidence/data"
GEN = ROOT / "solution/kaggle/generated"
TRIM = Path.home() / ".cache/arc3/trimtest-2026-10-20"
sys.path.insert(0, str(ROOT / "solution/analysis"))
import trim_compare as tc  # noqa: E402  (reuse its action parser so the numbers match the gate)


def exact_p_top(a, b):
    """One-sided exact permutation p that group a's mean is this far above b's, all splits counted."""
    pool, k = a + b, len(a)
    obs = st.mean(a) - st.mean(b)
    hits = tot = 0
    for idx in itertools.combinations(range(len(pool)), k):
        x = [pool[i] for i in idx]
        y = [pool[i] for i in range(len(pool)) if i not in idx]
        tot += 1
        hits += st.mean(x) - st.mean(y) >= obs - 1e-12
    return hits, tot


def board():
    rows = list(csv.DictReader(open(DATA / "board_draws.csv")))
    for r in rows:
        r["s"] = float(r["public_score"])
    live = [r for r in rows if r["s"] > 0]
    print("== board: fork-era draws", len(rows), "completed;", len(rows) - len(live), "zeros;", len(live), "live")
    zeros = [(r["submitted_utc"], r["config"], r["total_bytes"]) for r in rows if r["s"] == 0]
    print("zeros", zeros)
    print("live bytes range", min(int(r["total_bytes"]) for r in live), max(int(r["total_bytes"]) for r in live))
    since = [r for r in rows if r["submitted_utc"] >= "2026-10-13"]
    for r in sorted(since, key=lambda r: r["submitted_utc"]):
        print("  ", r["submitted_utc"], r["config"], r["public_score"], r["total_bytes"])
    by = {}
    for r in live:
        by.setdefault(r["config"], []).append(r["s"])
    for c, v in by.items():
        sd = st.stdev(v) if len(v) > 1 else float("nan")
        print(f"  {c:32s} n {len(v)} mean {st.mean(v):.2f} sd {sd:.2f}")
    r3p, hm = by["franzen-l1a1y8sv2r3p"], by["franzen-l1a1y8sv2r3hm32p"]
    h, t = exact_p_top(r3p, hm)
    print(f"r3p vs hm32p: diff {st.mean(r3p) - st.mean(hm):.2f}; exact one-sided p {h}/{t} = {h / t:.4f}")
    # Welch t
    va, vb = st.variance(r3p) / len(r3p), st.variance(hm) / len(hm)
    print(f"  Welch t {(st.mean(r3p) - st.mean(hm)) / math.sqrt(va + vb):.2f}")
    # first two r3p draws only (what the 10-19 research saw)
    h2, t2 = exact_p_top(r3p[-2:] if False else [40.09, 34.22], hm)
    print(f"  as of 10-19 (r3p 40.09, 34.22 vs hm32p 4 valid): p {h2}/{t2} = {h2 / t2:.4f}")
    yieldfam = [r["s"] for r in live if r["submitted_utc"] >= "2026-10-12"]
    pre = [r["s"] for r in live if r["submitted_utc"] < "2026-10-12"]
    print(f"yield-family live draws n {len(yieldfam)} mean {st.mean(yieldfam):.2f} sd {st.stdev(yieldfam):.2f}; "
          f"pre-yield n {len(pre)} mean {st.mean(pre):.2f} sd {st.stdev(pre):.2f}")
    h3, t3 = exact_p_top(yieldfam, pre)
    print(f"  exact one-sided p yield vs pre {h3}/{t3} = {h3 / t3:.2e}")
    # rank-sum: how many pre draws exceed each yield draw
    u = sum(1 for a in yieldfam for b in pre if a > b) + 0.5 * sum(1 for a in yieldfam for b in pre if a == b)
    print(f"  Mann-Whitney U {u} of {len(yieldfam) * len(pre)}")
    print(f"  best live draw {max(r['s'] for r in live)} of {len(live)} live draws since 10-04")
    # expected max of n iid normals (Blom approximation is enough for a hedge)
    for n in (8, 15):
        rng = random.Random(0)
        em = st.mean(max(rng.gauss(0, 1) for _ in range(n)) for _ in range(200000))
        print(f"  E[max of {n} normals] = {em:.3f} SD")


def ranks():
    out = []
    for f in sorted((GEN / "daily").glob("board-2026-10-*.csv")):
        day = f.stem.replace("board-", "")
        if day < "2026-10-12":
            continue
        rows = list(csv.DictReader(open(f, encoding="utf-8-sig")))
        me = next(r for r in rows if "vineethsaivs" in r["TeamMemberUserNames"])
        s = lambda k: float(rows[k - 1]["Score"])  # noqa: E731
        out.append(dict(snapshot_utc=day + "T00:06Z", teams=len(rows), our_rank=int(me["Rank"]),
                        our_best=float(me["Score"]), r1=s(1), r3=s(3), r5=s(5), r10=s(10), r20=s(20)))
    with open(DATA / "rank_history.csv", "w", newline="") as g:
        w = csv.DictWriter(g, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print("== rank history (daily 00:06Z snapshots)")
    for r in out:
        print("  ", r)


def hm32():
    run = re.compile(r"#running-req: (\d+)")
    acc = re.compile(r"accept len: ([0-9.]+)")
    gen = re.compile(r"gen throughput \(token/s\): ([0-9.]+)")
    print("== hm32 replica vs r3 (serve.log Decode batch lines)")
    res = {}
    for arm in ("franzen-l1a1y8sv2r3-v1", "franzen-l1a1y8sv2r3hm32-v1"):
        d = GEN / "kernel-outputs" / arm
        g14, a8 = [], []
        for line in open(d / "serve.log", errors="replace"):
            if "Decode batch" not in line:
                continue
            r, a, g = run.search(line), acc.search(line), gen.search(line)
            if r and g and int(r[1]) == 14:
                g14.append(float(g[1]))
            if r and a and int(r[1]) >= 8:
                a8.append(float(a[1]))
        b = json.load(open(d / "benchmark.json"))
        lv = sum(x["levels_completed"] for x in b["game_runs"])
        sc = st.mean(x["final_score"] or 0 for x in b["game_runs"])
        tok = sum(h["generated_tokens"] for x in b["game_runs"] for h in x["history"])
        res[arm] = st.mean(g14)
        print(f"  {arm}: gen@14 {st.mean(g14):.1f} (n {len(g14)}), accept@8+ {st.mean(a8):.4f} (n {len(a8)}), "
              f"levels {lv}/183, mean score {sc:.2f}, generated {tok / 1e6:.3f}M")
    a, b = res.values()
    print(f"  decode ratio {b / a:.4f}")


def trim():
    by = {}
    for line in open(TRIM / "trimtest_replies.jsonl"):
        r = json.loads(line)
        if r.get("error") is None:
            by.setdefault(r["state"], {})[(r["variant"], r["sample"])] = r
    states = [s for s, d in by.items() if len(d) == 4]
    rows = []
    for s in states:
        fa = [tc.first_action(tc.code_of(by[s][("full", k)])) for k in (0, 1)]
        la = [tc.first_action(tc.code_of(by[s][("lean", k)])) for k in (0, 1)]
        rows.append(dict(state=s, game=str(s).split(":")[0] if ":" in str(s) else "",
                         full_acts=sum(a is not None for a in fa), lean_acts=sum(a is not None for a in la),
                         same_ff=int(fa[0] == fa[1]), cross_fl=st.mean(a == b for a in fa for b in la),
                         full0=fa[0], full1=fa[1], lean0=la[0], lean1=la[1]))
    with open(DATA / "trimtest_by_state.csv", "w", newline="") as g:
        w = csv.DictWriter(g, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    d = [r["cross_fl"] - r["same_ff"] for r in rows]
    print(f"== trimtest: {len(rows)} states; mean paired (cross - same) {st.mean(d):+.3f}")
    neg, pos = sum(x < 0 for x in d), sum(x > 0 for x in d)
    print(f"  states with cross < same {neg}, cross > same {pos}, equal {len(d) - neg - pos}")
    # sign-flip permutation on the paired per-state differences (two-sided)
    rng = random.Random(0)
    obs = abs(st.mean(d))
    hits = sum(abs(st.mean(x * rng.choice((1, -1)) for x in d)) >= obs - 1e-12 for _ in range(20000))
    print(f"  sign-flip permutation p (two-sided, 20,000) {hits / 20000:.4f}")
    both_inv = sum(1 for r in rows if r["full0"] is None and r["full1"] is None)
    print(f"  states where both full replies only investigate (no action call): {both_inv}")
    # K0 headroom
    import k0_headroom as k0
    cs = [k0.p_max(top) for line in open(TRIM / "trimtest_logprobs.jsonl") for row in [json.loads(line)]
          if row.get("error") is None for _, top in row["tokens"] if top]
    cs = [c for c in cs if c == c]
    c = st.mean(cs)
    print(f"== K0: {len(cs)} positions, c {c:.4f}, oracle accept K3 {1 + c + c * c + c ** 3:.3f}, "
          f"K4 {1 + c + c * c + c ** 3 + c ** 4:.3f}; headroom over 2.830 {1 + c + c * c + c ** 3 - 2.830:+.3f}")
    per_reply = [st.mean([k0.p_max(top) for _, top in row["tokens"] if top])
                 for line in open(TRIM / "trimtest_logprobs.jsonl") for row in [json.loads(line)]]
    print(f"  per-reply c: min {min(per_reply):.3f} median {st.median(per_reply):.3f} max {max(per_reply):.3f}")
    print(f"  A_1 measured 0.7734 / c = {0.7734 / c:.3f}")


if __name__ == "__main__":
    board()
    ranks()
    hm32()
    trim()
