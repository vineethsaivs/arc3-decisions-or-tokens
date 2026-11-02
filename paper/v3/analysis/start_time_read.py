#!/usr/bin/env python3
"""When play started against when the server was ready, per 15-stream-family Kaggle replica run (read-only, stdlib).

Franzen's notebook releases the benchmark when the server reports READY, or at a deadline about 720 s after notebook start
with the server still loading ("DEADLINE at 720s from notebook start; releasing the benchmark with the server still loading"; 720 to 722 s). Early games then wait on a dead
endpoint while their clock runs.

Run from the repo root:
  python3 -I -B paper/v3/analysis/start_time_read.py

Reads  solution/kaggle/generated/kernel-outputs/kaggle-lf15*/kernel.log   kernel time of "Starting benchmark", the
                                                                          deadline line, and the sglang launch
       solution/kaggle/generated/kernel-outputs/kaggle-lf15*/serve.log    wall time of "The server is fired up"
Writes paper/v3/analysis/figs/data/start_times.csv

Server ready (kernel seconds) = launch time + (ready wall time - first serve.log wall time); about a second of slack.
"""
import csv
import glob
import os
import re
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
K = ROOT / "solution/kaggle/generated/kernel-outputs"
OUT = Path(__file__).resolve().parent / "figs/data/start_times.csv"


def ktime(log, pat):
    m = re.search(r'"time":([0-9.]+),"data":"[^"]*' + pat, log)
    return float(m.group(1)) if m else None


rows = []
for run in sorted(glob.glob(str(K / "kaggle-lf15*"))):
    k = open(os.path.join(run, "kernel.log"), errors="replace").read()
    s = open(os.path.join(run, "serve.log"), errors="replace").read()
    start, deadline, launch = ktime(k, "Starting benchmark"), ktime(k, "DEADLINE at "), ktime(k, "sglang serve")
    stamps = re.findall(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\]", s, re.M)
    m = re.search(r"^\[([^\]]+)\] The server is fired up", s, re.M)
    ready = launch + (datetime.fromisoformat(m.group(1)) - datetime.fromisoformat(stamps[0])).total_seconds()
    rows.append({"run": os.path.basename(run), "play_start_s": round(start), "deadline_release": int(deadline is not None),
                 "server_ready_s": round(ready), "early_min": round((ready - start) / 60, 1)})

with open(OUT, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

big = [r for r in rows if r["early_min"] > 10]
assert len(rows) == 23 and sum(r["deadline_release"] for r in rows) == 13
assert len(big) == 6 and min(r["early_min"] for r in big) == 11.5 and max(r["early_min"] for r in big) == 21.3
assert all(r["early_min"] <= 1.0 for r in rows if r["run"].startswith("kaggle-lf15ustv"))
print(f"{len(rows)} runs; released at the deadline {sum(r['deadline_release'] for r in rows)}; "
      f"started over 10 min before the server was ready: {len(big)} ({', '.join(r['run'] for r in big)}), "
      f"{min(r['early_min'] for r in big)} to {max(r['early_min'] for r in big)} min; ustv runs at most 1.0 min")
