#!/usr/bin/env python3
"""Build the board-history CSVs for fig_draws from the repo's recorded Kaggle snapshots (read-only, no network).

Run from the repo root:
  uv run --no-project python -I -B paper/v3/analysis/figs/board_history_build.py

Reads  solution/kaggle/generated/board-submissions-*.json   every submission-list snapshot (50 rows each); the union
                                                            by submission id covers 2026-07-17 to 2026-11-02
       solution/kaggle/generated/daily/*.json                one leaderboard snapshot per daily job (rank, bar scores)
Writes paper/v3/analysis/figs/data/board_draws_all_2026-11-02.csv      one row per submission
       paper/v3/analysis/figs/data/board_snapshots_2026-11-02.csv      last leaderboard snapshot of each UTC day

The newest submission snapshot equals `solution/kaggle/kapi.sh subs` run on 2026-11-02 (its rows overlay the stored ones); `kapi.sh subs` itself returns only the newest 50 submissions.
"""
import csv
import glob
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
GEN = REPO / "solution/kaggle/generated"
OUT = Path(__file__).resolve().parent / "data"


def config_of(note):
    m = re.search(r"rotation config (\S+?)[ ,]", note or "")
    if m:
        return m.group(1)
    if "fork-franzen" in (note or ""):
        return "fork-franzen"
    return ""


rows = {}
for f in sorted(glob.glob(str(GEN / "board-submissions-*.json"))):
    for r in json.load(open(f))["rows"]:
        rows[r["ref"]] = r     # later snapshots overwrite: a pending row becomes complete
# overlay the newest `kapi.sh subs` output (saved as data/kapi_subs_2026-11-02.txt): the newest draws may still be
# pending in the last stored snapshot
for line in open(OUT / "kapi_subs_2026-11-02.txt"):
    ref, ts, score, status, note = line.rstrip("\n").split(None, 4)
    sc = score.split("=")[1]
    assert ref in rows or ts[:10] >= "2026-10-31", ref
    rows.setdefault(ref, {"ref": ref, "date": ts, "note": note})
    rows[ref].update(status=status, score=None if sc == "None" else float(sc))
OUT.mkdir(exist_ok=True)
with open(OUT / "board_draws_all_2026-11-02.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["ref", "submitted_utc", "status", "public_score", "config", "note"])
    for r in sorted(rows.values(), key=lambda r: r["date"]):
        w.writerow([r["ref"], r["date"][:19], r["status"], "" if r["score"] is None else f"{r['score']:.2f}",
                    r.get("config") or config_of(r["note"]), r["note"]])
print(len(rows), "submissions")

snap = {}
for f in sorted(glob.glob(str(GEN / "daily/*.json"))):
    d = json.load(open(f))
    if d.get("rank") is None or not d.get("created_at"):
        continue
    snap[d["created_at"][:10]] = d      # sorted by name: the last of each day wins
with open(OUT / "board_snapshots_2026-11-02.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["date_utc", "created_at", "our_best", "our_rank", "teams", "rank1", "rank3", "rank5", "rank10"])
    for k in sorted(snap):
        d = snap[k]
        w.writerow([k, d["created_at"][:19], d["best"], d["rank"], d["field"], d.get("rank1") if d.get("rank1") is not None else "",
                    d["rank3"], d["rank5"], d["rank10"]])
print(len(snap), "daily snapshots")
