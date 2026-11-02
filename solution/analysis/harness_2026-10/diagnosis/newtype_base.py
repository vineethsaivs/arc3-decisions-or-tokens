"""Base rate: at each level start (level>=2) of a click game, would the first K clicks touch a type the overlay lists as New?
Same metric for uns2 (line shown) and controls (line not shown)."""
import sys, json, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from uns2_lines import comp, CH
import arc_agi
from arcengine.enums import GameAction
ENV_DIR = "ARC-AGI-3-Kaggle-Starter/environment_files"

def census(grid, novel):
    rows, cols = len(grid), len(grid[0]); types, seen = {}, set()
    for r in range(rows):
        for c in range(cols):
            if (r, c) in seen: continue
            cells = comp(grid, r, c); seen |= cells
            r0 = min(y for y, _ in cells); r1 = max(y for y, _ in cells); c0 = min(x for _, x in cells); c1 = max(x for _, x in cells)
            if len(cells) * 10 >= rows * cols or (r1 - r0 + 1) * 4 >= rows * 3 or (c1 - c0 + 1) * 4 >= cols * 3: continue
            if (r0 <= 1 or c0 <= 1 or r1 >= rows - 2 or c1 >= cols - 2) and min(r1 - r0, c1 - c0) <= 1: continue
            box = (r1 - r0 + 1, c1 - c0 + 1)
            if not novel: key = (grid[r][c], box)
            elif max(box) <= 8: key = (grid[r][c], frozenset((y - r0, x - c0) for y, x in cells))
            else: key = (grid[r][c], "region")
            k = types.setdefault(key, {"count": 0, "box": box, "pixels": len(cells), "cells": []})
            k["count"] += 1; k["cells"].append(cells)
    return {k: v for k, v in types.items() if v["pixels"] > 1 or v["count"] < 10}

def main():
  run = sys.argv[1]; Ks = (3, 5, 10); out = []
  for g in json.load(open(os.path.join(run, "benchmark.json")))["game_runs"]:
      gid = g["game_id"]; hist = g["history"]
      env = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=ENV_DIR).make(gid)
      f = env.reset()
      if 6 not in f.available_actions: continue
      grids = [[list(map(int, r)) for r in f.frame[-1]]]; lv = [f.levels_completed]
      for h in hist:
          a = h["action"]
          f = env.reset() if a["id"] == "RESET" else env.step(GameAction[a["id"]], a.get("data") or None)
          grids.append([list(map(int, r)) for r in f.frame[-1]] if f.frame else grids[-1]); lv.append(f.levels_completed)
      earlier, boards, upto = set(), set(), 0
      for s in range(1, len(grids)):
          if lv[s] <= lv[s - 1]: continue
          L = lv[s] + 1
          for i in range(upto, s):
              b = hash(str(grids[i]))
              if b not in boards: boards.add(b); earlier |= set(census(grids[i], True))
          upto = s
          merged = {}
          for k, v in census(grids[s], True).items():
              if k in earlier: continue
              key = (k[0], v["box"])
              if key in merged: merged[key]["count"] += v["count"]; merged[key]["cells"] += v["cells"]
              else: merged[key] = dict(v, cells=list(v["cells"]))
          new = sorted(merged.items(), key=lambda kv: (-kv[1]["count"], -kv[1]["pixels"]))[:6]
          if not new: continue
          keys = {k for k, _ in new}
          clicks = [i for i in range(s, len(hist)) if lv[i] == lv[s]]
          clicks = [i for i in clicks if hist[i]["action"]["id"] == "ACTION6"]
          first_hit = None
          for n, i in enumerate(clicks):
              r, c = hist[i]["action"]["data"]["y"], hist[i]["action"]["data"]["x"]; grid = grids[i]
              cells = comp(grid, r, c); rs = [y for y, _ in cells]; cs = [x for _, x in cells]
              ck = (grid[r][c], (max(rs) - min(rs) + 1, max(cs) - min(cs) + 1))
              ty = census(grid, False); near = {(r + dy, c + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)}
              if ck in keys or any(not cl.isdisjoint(near) for k in keys for cl in ty.get(k, {}).get("cells", [])):
                  first_hit = n; break
              if n >= max(Ks): break
          out.append({"game": gid[:4], "level": L, "nclicks": len(clicks), "first_hit": first_hit, "cleared": max(lv) > lv[s]})
  json.dump(out, open(sys.argv[2], "w"))

if __name__ == "__main__":
    main()
