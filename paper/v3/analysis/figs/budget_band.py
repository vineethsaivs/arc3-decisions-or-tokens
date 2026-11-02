#!/usr/bin/env python3
"""Run-level bootstrap band for the budget curve S(rho) of fig_budget (data step; about 15 minutes on 8 workers).

Run from the repo root:
  uv run --no-project --with numpy python -I -B paper/v3/analysis/figs/budget_band.py [B] [SEEDS]

Same simulator, regime and game picks as paper/v2/analysis/f1_curves_n2.py panel (a): the eight-run pool e5_lib.Y8X at
890 tok/s, 110 games (stratified picks), 14 streams, 31,920 s, D-prime priority. Each resample draws 8 of the 8 runs
with replacement and pools their recorded level attempts (a run drawn twice counts twice); every resample uses the same
seeds 0..SEEDS-1, so the band shows which runs fed the simulator, not seed noise. Writes
paper/v3/analysis/figs/data/budget_band.json.
"""
import json
import multiprocessing as mp
import random
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "paper/v2/analysis"))
import e5_lib as L  # noqa: E402
import new_plateau_pool as N  # noqa: E402

OUT = Path(__file__).resolve().parent / "data/budget_band.json"
RHOS = [0.25, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.11, 1.25, 1.5, 2.0, 3.0]   # the grid of f1_curves_n2 panel (a)
B = int(sys.argv[1]) if len(sys.argv) > 1 else 200
SEEDS = int(sys.argv[2]) if len(sys.argv) > 2 else 10
G = {}


def setup():
    meta = {}
    for r in L.Y8X:
        N.RUN[r] = L.load_pool([r])
        for g, m in N.RUN[r][1].items():
            meta.setdefault(g, m)
    G["meta"] = meta
    G["D"] = L.dprime_priority(L.dprime())


def task(t):
    b, runs = t
    pool = N.merged(runs)
    out = []
    for rho in RHOS:
        s = [L.simulate(N.strat_games(sd, pool, G["meta"]), L.HIDDEN["slots"], L.HIDDEN["window_s"],
                        L.RATE_Y8 * rho / L.HIDDEN["slots"], G["D"]) for sd in range(SEEDS)]
        out.append((statistics.mean(x[0] for x in s), statistics.mean(x[1] for x in s)))
    return b, runs, out


def main():
    t0 = time.time()
    setup()
    rng = random.Random(20261021)
    tasks = [(-1, list(L.Y8X))] + [(b, [rng.choice(L.Y8X) for _ in L.Y8X]) for b in range(B)]
    res = {}
    with mp.get_context("fork").Pool(8) as pl:
        for i, (b, runs, out) in enumerate(pl.imap_unordered(task, tasks, chunksize=1), 1):
            res[b] = (runs, out)
            if i % 20 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)}  {time.time() - t0:.0f}s", flush=True)
    full = res.pop(-1)[1]
    obj = {"protocol": "f1_curves_n2 panel (a) regime; 8-of-8 run resamples with replacement, attempts pooled with "
                       "multiplicity; common seeds 0..SEEDS-1 for every resample",
           "rhos": RHOS, "B": B, "seeds": SEEDS, "rate_tok_s": L.RATE_Y8,
           "full_pool_S": [s for s, _ in full], "full_pool_levels": [lv for _, lv in full],
           "boot_S": [[s for s, _ in res[b][1]] for b in range(B)],
           "boot_levels": [[lv for _, lv in res[b][1]] for b in range(B)],
           "boot_distinct_runs": [len(set(res[b][0])) for b in range(B)]}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(obj, indent=0))
    print(f"wrote {OUT}  {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
