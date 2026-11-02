#!/usr/bin/env python3
"""Start-time confound for the six byte-identical pairs (read-only, stdlib). Companion to start_time_read.py (V3-W32 reads the
15-stream Kaggle family; this reads the 12 runs of the six pairs in paper/v2/analysis/out/e4_noise.json).

Run from the repo root:
  python3 -I -B paper/v3/analysis/start_time_pairs.py

Per run: kernel time of "Starting benchmark" (play start), server READY mapped to kernel time as launch time + (ready wall
time - first stamped serve.log line), early = ready - play start (minutes; positive means games played before the server was
ready). Handles Kaggle (compact json) and GCP (spaced json) kernel.log. Prints levels, early minutes and the deadline flag.
"""
import json
import re
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
K = ROOT / "solution/kaggle/generated/kernel-outputs"
NOISE = json.loads((ROOT / "paper/v2/analysis/out/e4_noise.json").read_text())["franzen_pairs"]
PAIRS = {  # pair -> (platform, run 1 dir, run 2 dir); same as paper/v2/analysis/e4_noise.py FRANZEN
    "p448": ("kaggle", "franzen-p448-v1", "franzen-p448rep-v1"),
    "p448s14": ("kaggle", "franzen-p448s14-v1", "franzen-p448s14rep-v1"),
    "l1a1y8": ("kaggle", "franzen-l1a1y8-v1", "franzen-l1a1y8r-v1"),
    "dpctl": ("gcp", "gcp-dpctl-v1", "gcp-dpctl-v2"),
    "L1": ("gcp", "gcp-p448s14dpwfrc-v1", "gcp-p448s14dpwfrc-v2"),
    "l1a1": ("gcp", "gcp-l1a1-v1", "gcp-l1a1-v2"),
}


def ktime(log, pat):
    m = re.search(r'"time": ?([0-9.]+), ?"data": ?"[^"]*' + pat, log)
    return float(m.group(1)) if m else None


def early(run):
    k = (K / run / "kernel.log").read_text(errors="replace")
    s = (K / run / "serve.log").read_text(errors="replace")
    start, deadline, launch = ktime(k, "Starting benchmark"), ktime(k, "DEADLINE at "), ktime(k, "sglang serve")
    stamps = re.findall(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\]", s, re.M)
    m = re.search(r"^\[([^\]]+)\] The server is fired up", s, re.M)
    ready = launch + (datetime.fromisoformat(m.group(1)) - datetime.fromisoformat(stamps[0])).total_seconds()
    return round((ready - start) / 60, 1), deadline is not None


rows = []
for name, (plat, a, b) in PAIRS.items():
    (e1, d1), (e2, d2) = early(a), early(b)
    l1, l2 = NOISE[name]["levels"]
    rows.append((name, plat, l1, l2, e1, e2, d1, d2))
    print(f"{name:8s} {plat:6s} levels {l1:3d} {l2:3d} (gap {l1 - l2:+3d})  early min {e1:6.1f} {e2:6.1f}  deadline {int(d1)} {int(d2)}")

n_early = sum(1 for r in rows for e in r[4:6] if e > 10)
second_later = sum(1 for r in rows if r[5] < r[4])
print(f"runs with games playing over 10 min before READY: {n_early} of 12; pairs whose second run started less early: {second_later} of 6")
