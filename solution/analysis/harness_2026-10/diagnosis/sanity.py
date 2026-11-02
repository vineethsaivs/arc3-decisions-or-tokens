import sys, json, os, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from uns2_lines import replay, comp, CH, TYPE, HEADER, SEC
run = sys.argv[1]; ok = bad = 0; badex = []
for g in json.load(open(os.path.join(run, "benchmark.json")))["game_runs"]:
    gid = g["game_id"]; grids, lv, st = replay(gid, g["history"])
    t = open(os.path.join(run, gid + "_p0.txt"), errors="replace").read(); parts = HEADER.split(t)
    for i in range(1, len(parts) - 3, 4):
        M = int(parts[i + 1]); sp = SEC.split(parts[i + 3]); user = next((sp[j + 1] for j in range(1, len(sp) - 1, 2) if sp[j] == "USER PROMPT"), "")
        for ln in user.splitlines():
            if ln.startswith(("Untested on this level:", "New on this level")):
                grid = grids[M - 1]
                for m in TYPE.finditer(ln):
                    col, h, w, r, c = CH.index(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(5)), int(m.group(6))
                    cells = comp(grid, r, c)
                    rs = [y for y, _ in cells]; cs = [x for _, x in cells]
                    if grid[r][c] == col and (max(rs) - min(rs) + 1, max(cs) - min(cs) + 1) == (h, w): ok += 1
                    else: bad += 1; badex.append((gid[:4], M, m.group(0)))
print(run[-6:], "listed types matching the replayed board:", ok, "mismatch:", bad, badex[:5])
