"""Budget-bar census on human replays: per attempt, edge strips (2 rows/cols at each border) whose cell counts change
monotonically with actions. Report per game: strips that change on >= 70% of actions within an attempt."""
import json, glob, collections
def strips(g):
    h, w = len(g), len(g[0])
    out = {}
    for name, cells in (("top", [(r, c) for r in (0, 1) for c in range(w)]), ("bottom", [(r, c) for r in (h - 2, h - 1) for c in range(w)]),
                        ("left", [(r, c) for c in (0, 1) for r in range(h)]), ("right", [(r, c) for c in (w - 2, w - 1) for r in range(h)])):
        out[name] = collections.Counter(g[r][c] for r, c in cells)
    return out
for f in sorted(glob.glob('frames/h_*.json')):
    d = json.load(open(f)); seq = d['seq']; gid = d['gid'][:4]
    stats = collections.defaultdict(lambda: [0, 0, 0])   # strip,color -> [actions, decreased, increased]
    for prev, cur in zip(seq, seq[1:]):
        if cur['lv'] != prev['lv'] or cur.get('auto'): continue
        a, b = strips(prev['g']), strips(cur['g'])
        for s in a:
            for col in set(a[s]) | set(b[s]):
                k = (s, col); st = stats[k]; st[0] += 1
                if b[s][col] < a[s][col]: st[1] += 1
                elif b[s][col] > a[s][col]: st[2] += 1
    n = max(v[0] for v in stats.values()) if stats else 0
    hits = [(k, v) for k, v in stats.items() if v[0] >= 10 and max(v[1], v[2]) >= 0.5 * v[0]]
    print(gid, n, sorted(((s, c, f"{v[1]}-/{v[2]}+ of {v[0]}") for (s, c), v in hits))[:4])
