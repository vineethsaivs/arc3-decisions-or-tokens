"""Level starts where 4-connectivity splits a glyph: an 8-connected object of >=3 cells, in a bbox >= 3x3, made of
>= 2 4-connected pieces, not a dot texture (its color has < 10 such objects)."""
import json, glob, sys
sys.path.insert(0, '.')
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
tot = hit = 0; games = {}
for f in sorted(glob.glob('frames/h_*.json')):
    d = json.load(open(f)); gid = d['gid'][:4]; last = None
    for e in d['seq']:
        if e['lv'] == last: continue
        last = e['lv']; g = e['g']; tot += 1
        rows, cols = len(g), len(g[0]); seen = set(); found = []
        bycol = {}
        for r in range(rows):
            for c in range(cols):
                if (r, c) in seen: continue
                cells = comp(g, r, c, N8); seen |= cells
                if len(cells) * 10 >= rows * cols: continue
                rs = [y for y, _ in cells]; cs = [x for _, x in cells]
                h, w = max(rs) - min(rs) + 1, max(cs) - min(cs) + 1
                s4 = set(); n4 = 0
                for p in cells:
                    if p not in s4: s4 |= comp(g, p[0], p[1], N4); n4 += 1
                if n4 >= 2 and h >= 3 and w >= 3 and len(cells) >= 3:
                    bycol.setdefault(g[r][c], []).append((h, w, n4))
        found = [(k, v) for k, v in bycol.items() if len(v) < 10]
        if found:
            hit += 1; games.setdefault(gid, []).append((e['lv'], [(k, v[:3]) for k, v in found][:3]))
print(f"{hit} of {tot} level starts have a split glyph; games {len(games)}")
for k, v in games.items(): print(k, v[:3])
