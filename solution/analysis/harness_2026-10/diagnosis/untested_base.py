"""Base rate for the Untested list: after each RESET (model or post-game-over) in a click game, compute the overlay's
untested types (top 6) and check whether the next K clicks on the level touch one. Same metric for all runs."""
import sys, json, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from uns2_lines import comp
from newtype_base import census  # noqa: E402  (module runs its main on import, so guard below)
import arc_agi
from arcengine.enums import GameAction
ENV_DIR = "ARC-AGI-3-Kaggle-Starter/environment_files"
run = sys.argv[1]; out = []
for g in json.load(open(os.path.join(run, "benchmark.json")))["game_runs"]:
    gid = g["game_id"]; hist = g["history"]
    env = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=ENV_DIR).make(gid)
    f = env.reset()
    if 6 not in f.available_actions: continue
    grids = [[list(map(int, r)) for r in f.frame[-1]]]; lv = [f.levels_completed]; st = [str(f.state)]
    for h in hist:
        a = h["action"]
        f = env.reset() if a["id"] == "RESET" else env.step(GameAction[a["id"]], a.get("data") or None)
        grids.append([list(map(int, r)) for r in f.frame[-1]] if f.frame else grids[-1]); lv.append(f.levels_completed); st.append(str(f.state))
    for t, h in enumerate(hist):
        if h["action"]["id"] != "RESET" or lv[t + 1] != lv[t]: continue
        s = t + 1; L = lv[s]
        start = max(i for i in range(s + 1) if i == 0 or lv[i] > lv[i - 1] or (lv[i] == L and lv[i - 1] < L))
        clicked, keys = set(), set()
        for i in range(start, s):
            a = hist[i]["action"]
            if a["id"] == "ACTION6" and lv[i] == L:
                r, c = a["data"]["y"], a["data"]["x"]; cells = comp(grids[i], r, c)
                rs = [y for y, _ in cells]; cs = [x for _, x in cells]
                clicked.add((r, c)); keys.add((grids[i][r][c], (max(rs) - min(rs) + 1, max(cs) - min(cs) + 1)))
        near = {(r + dy, c + dx) for r, c in clicked for dy in (-1, 0, 1) for dx in (-1, 0, 1)}
        ty = census(grids[s], False)
        unt = sorted(((k, v) for k, v in ty.items() if k not in keys and all(cl.isdisjoint(near) for cl in v["cells"])),
                     key=lambda kv: (-kv[1]["count"], -kv[1]["pixels"]))[:6]
        if not unt: continue
        ukeys = {k for k, _ in unt}
        nxt = [i for i in range(s, len(hist)) if lv[i] == L][:60]
        nxt = [i for i in nxt if hist[i]["action"]["id"] == "ACTION6"]
        first = None
        for n, i in enumerate(nxt[:10]):
            r, c = hist[i]["action"]["data"]["y"], hist[i]["action"]["data"]["x"]; grid = grids[i]
            cells = comp(grid, r, c); rs = [y for y, _ in cells]; cs = [x for _, x in cells]
            ck = (grid[r][c], (max(rs) - min(rs) + 1, max(cs) - min(cs) + 1)); tyi = census(grid, False)
            nr = {(r + dy, c + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)}
            if ck in ukeys or any(not cl.isdisjoint(nr) for k in ukeys for cl in tyi.get(k, {}).get("cells", [])):
                first = n; break
        out.append({"game": gid[:4], "level": L + 1, "at": s, "after_gameover": "GAME_OVER" in st[t], "first_hit": first, "nclicks": len(nxt)})
json.dump(out, open(sys.argv[2], "w"))
