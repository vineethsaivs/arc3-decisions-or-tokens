"""Split each game's clock from its analyzer transcript (`<game>_p0.txt`), cleaned up from the first tail.py.

A turn header reads `--- analysis_step=N | action=M | HH:MM:SS | tool-agent ---` (M is the action counter and the time
is wall clock when the turn starts); the user prompt of each turn carries `Current state: step N, level L.`.
The clock of turn i is t[i+1] - t[i] and the actions it executed are action[i+1] - action[i]; the last turn of a game has
no end stamp and is dropped. Time stamps never wrap past midnight in these runs (asserted).
"""
import glob
import json
import os
import re

HEADER = re.compile(r"^--- analysis_step=(\d+) \| action=(\d+) \| (\d\d:\d\d:\d\d) \| tool-agent ---$", re.M)
STATE = re.compile(r"Current state: step (\d+), level (\d+)\.")


def secs(hms):
    return int(hms[:2]) * 3600 + int(hms[3:5]) * 60 + int(hms[6:])


def turns(path):
    """[(seconds, action counter, level)] for each analyzer turn of one game."""
    txt = open(path, errors="replace").read()
    hs = list(HEADER.finditer(txt))
    out = []
    for i, h in enumerate(hs):
        body = txt[h.end(): hs[i + 1].start() if i + 1 < len(hs) else len(txt)]
        out.append((secs(h.group(3)), int(h.group(2)), int(STATE.search(body).group(2))))
    assert all(b[0] >= a[0] and b[1] >= a[1] for a, b in zip(out, out[1:])), path
    return out


def bucket(n):
    """Actions executed in a turn -> 0, 1, 2 (two or three) or 3 (four or more)."""
    return 0 if n == 0 else 1 if n == 1 else 2 if n <= 3 else 3


def game_split(pts, n_levels, levels_completed):
    """(clock on levels cleared, clock on the level open when the window closed, seconds by bucket, first-idle seconds).

    A game that cleared every level has no stuck tail: the clock up to its last turn was spent on cleared levels."""
    t0, tend, lv_end = pts[0][0], pts[-1][0], pts[-1][2]
    done = levels_completed == n_levels
    assert done or lv_end - pts[0][2] == levels_completed, (lv_end, levels_completed)
    first_final = tend if done else next(t for t, _, lv in pts if lv == lv_end)
    by = [0, 0, 0, 0]
    first_idle = 0
    prev_idle = False
    for a, b in zip(pts, pts[1:]):
        d, n = b[0] - a[0], b[1] - a[1]
        by[bucket(n)] += d
        if n == 0 and not prev_idle:
            first_idle += d
        prev_idle = n == 0
    return first_final - t0, tend - first_final, by, first_idle


def run_split(kernel_dir):
    """{game id: (cleared s, tail s, by-bucket s, first-idle s, levels cleared, levels in game)} for one run."""
    bench = {g["game_id"]: g for g in json.load(open(os.path.join(kernel_dir, "benchmark.json")))["game_runs"]}
    out = {}
    for f in sorted(glob.glob(os.path.join(kernel_dir, "*_p0.txt"))):
        gid = os.path.basename(f)[: -len("_p0.txt")]
        g = bench[gid]
        c, t, by, fi = game_split(turns(f), g["number_of_levels"], g["levels_completed"])
        out[gid] = (c, t, by, fi, g["levels_completed"], g["number_of_levels"])
    assert len(out) == 25 == len(bench)
    return out
