"""Untested-list variants on real frames: 4- vs 8-connectivity, count-desc vs count-asc order.
python3 cov.py GAME NLEV ACTS.json  (replay an agent's level)   or   python3 cov.py --starts  (every level start, human replays)"""
import sys, json, re, glob, os
sys.path.insert(0, '<repo>/solution/analysis/keyfx')
from envload import arcade, BY_ID
from arcengine import GameAction
CH = "WwgGcBMPRbSYOrNp"
N4 = ((-1, 0), (1, 0), (0, -1), (0, 1)); N8 = N4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))
def comp(g, r, c, nb):
    if not (0 <= r < len(g) and 0 <= c < len(g[r])): return set()
    col = g[r][c]; seen = {(r, c)}; st = [(r, c)]
    while st:
        y, x = st.pop()
        for dy, dx in nb:
            ny, nx = y + dy, x + dx
            if (ny, nx) not in seen and 0 <= ny < len(g) and 0 <= nx < len(g[ny]) and g[ny][nx] == col:
                seen.add((ny, nx)); st.append((ny, nx))
    return seen
def box(cells):
    rs = [r for r, _ in cells]; cs = [c for _, c in cells]; return max(rs) - min(rs) + 1, max(cs) - min(cs) + 1
def types(g, nb):
    rows = len(g); cols = max(len(r) for r in g); T = {}; seen = set()
    for r in range(rows):
        for c in range(len(g[r])):
            if (r, c) in seen: continue
            cells = comp(g, r, c, nb); seen |= cells
            r0 = min(y for y, _ in cells); r1 = max(y for y, _ in cells); c0 = min(x for _, x in cells); c1 = max(x for _, x in cells)
            if len(cells) * 10 >= rows * cols or (r1 - r0 + 1) * 4 >= rows * 3 or (c1 - c0 + 1) * 4 >= cols * 3: continue
            if (r0 <= 1 or c0 <= 1 or r1 >= rows - 2 or c1 >= cols - 2) and min(r1 - r0, c1 - c0) <= 1: continue
            k = T.setdefault((g[r][c], (r1 - r0 + 1, c1 - c0 + 1)), {"count": 0, "at": (r0, c0), "pixels": len(cells), "cells": []})
            k["count"] += 1; k["cells"].append(cells)
    return T
def untested(cur, hist, nb, order):
    clicked, keys = set(), set()
    for prev, name in hist:
        m = re.match(r"MOUSE\(row=(\d+), col=(\d+)\)", name)
        if m:
            r, c = int(m.group(1)), int(m.group(2)); clicked.add((r, c)); cells = comp(prev, r, c, nb)
            if cells: keys.add((prev[r][c], box(cells)))
    near = {(r + dr, c + dc) for r, c in clicked for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
    out = [(k, v) for k, v in types(cur, nb).items() if k not in keys and all(cs.isdisjoint(near) for cs in v["cells"])]
    key = (lambda kv: (-kv[1]["count"], -kv[1]["pixels"])) if order == 'desc' else (lambda kv: (kv[1]["count"], -kv[1]["pixels"]))
    out.sort(key=key)
    return "; ".join(f"{CH[k[0]]} {k[1][0]}x{k[1][1]}" + (f" x{v['count']}" if v['count'] > 1 else "") for k, v in out[:6]) + (f"; +{len(out)-6}" if len(out) > 6 else "")
grid = lambda o: [[int(v) for v in r] for r in o.frame[-1]]
VARS = [(N4, 'desc', '4/desc'), (N8, 'desc', '8/desc'), (N8, 'asc', '8/asc')]
NAME = {"UP": 1, "DOWN": 2, "LEFT": 3, "RIGHT": 4, "SPACE": 5, "UNDO": 7}
def parse(a):
    m = re.match(r"MOUSE\(row=(\d+), col=(\d+)\)", a)
    return (6, {"x": int(m.group(2)), "y": int(m.group(1))}) if m else (NAME[a], {})
R = '<repo>/reference/re-arc-3/re_arc/dsl/official_human_replays'
if sys.argv[1:2] == ['--starts']:
    for f in sorted(glob.glob(R + '/*.json')):
        d = json.load(open(f)); gid = d['game_id']
        if not any(a == 6 for seg in d['levels'] for a, _ in seg): continue
        env = arcade.make(gid); obs = env.reset()
        for i, seg in enumerate(d['levels']):
            g = grid(obs); print(f"{gid[:4]} L{i+1}: " + " | ".join(f"{n}: {untested(g, [], nb, o)}" for nb, o, n in VARS))
            for a, data in seg: obs = env.step(BY_ID[a], data=data or {})
    sys.exit()
gid, nlev, acts = sys.argv[1], int(sys.argv[2]), json.load(open(sys.argv[3]))
d = json.load(open(f'{R}/{gid}.json'))
env = arcade.make(d['game_id']); obs = env.reset()
for seg in d['levels'][:nlev]:
    for a, data in seg: obs = env.step(BY_ID[a], data=data or {})
level = obs.levels_completed; cur = grid(obs); hist = []
for i, a in enumerate(acts):
    aid, data = parse(a); prev = cur; obs = env.step(BY_ID[aid], data=data); cur = grid(obs); hist.append((prev, a))
    if 'GAME_OVER' in str(obs.state):
        obs = env.step(GameAction.RESET, data={}); cur = grid(obs)
        for nb, o, n in VARS: print(f'#{i} death, {n}: {untested(cur, hist, nb, o)}')
    elif obs.levels_completed > level: print(f'#{i} LEVEL COMPLETE'); break
for nb, o, n in VARS: print(f'END {n}: {untested(cur, hist, nb, o)}')
