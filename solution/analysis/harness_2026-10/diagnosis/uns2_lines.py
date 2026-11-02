"""Replay each uns2 game through the offline engine; for every opener line (Untested / New / unused actions / stall),
check whether the next 3 turns clicked a listed type or used a listed action. Writes JSON rows to argv[2]."""
import json, re, sys, os, pickle
import arc_agi
from arcengine.enums import GameAction

ENV_DIR = "ARC-AGI-3-Kaggle-Starter/environment_files"
CH = "WwgGcBMPRbSYOrNp"
HEADER = re.compile(r"^--- analysis_step=(\d+) \| action=(\d+) \| (\S+) \| tool-agent ---$", re.M)
SEC = re.compile(r"^\[(SYSTEM PROMPT|USER PROMPT|THINKING|TOOL CALL: python|TOOL RESULT: python|MODEL RESPONSE META|ANALYZER STATUS|ASSISTANT)\]\n", re.M)
LEVEL = re.compile(r"Current state: step (\d+), level (\d+)\.")
TYPE = re.compile(r"([A-Za-z]) (\d+)x(\d+)(?: x(\d+))? \(e\.g\. row (\d+), col (\d+)\)")


def comp(grid, r, c):
    if not (0 <= r < len(grid) and 0 <= c < len(grid[r])):
        return set()
    color = grid[r][c]; seen = {(r, c)}; st = [(r, c)]
    while st:
        y, x = st.pop()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                ny, nx = y + dy, x + dx
                if (ny, nx) not in seen and 0 <= ny < len(grid) and 0 <= nx < len(grid[ny]) and grid[ny][nx] == color:
                    seen.add((ny, nx)); st.append((ny, nx))
    return seen


def types(grid):
    rows, cols = len(grid), len(grid[0]); out, seen = {}, set()
    for r in range(rows):
        for c in range(cols):
            if (r, c) in seen: continue
            cells = comp(grid, r, c); seen |= cells
            r0 = min(y for y, _ in cells); r1 = max(y for y, _ in cells)
            c0 = min(x for _, x in cells); c1 = max(x for _, x in cells)
            box = (r1 - r0 + 1, c1 - c0 + 1)
            out.setdefault((grid[r][c], box), []).append(cells)
    return out


def replay(gid, hist):
    env = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=ENV_DIR).make(gid)
    env.include_frame_data = True
    f = env.reset()
    grids = [[list(map(int, r)) for r in f.frame[-1]]]; lv = [f.levels_completed]; st = [str(f.state)]
    for h in hist:
        a = h["action"]; name = a["id"]
        if name == "RESET":
            f = env.reset()
        else:
            d = a.get("data") or None
            f = env.step(GameAction[name], d if d else None)
        g = f.frame[-1] if f.frame else None
        grids.append([list(map(int, r)) for r in g] if g is not None else grids[-1])
        lv.append(f.levels_completed); st.append(str(f.state))
    return grids, lv, st


NAMES = {"ACTION1": "UP", "ACTION2": "DOWN", "ACTION3": "LEFT", "ACTION4": "RIGHT", "ACTION5": "SPACE",
         "ACTION6": "MOUSE", "ACTION7": "UNDO", "RESET": "RESET"}


def main(run, out_path):
    b = json.load(open(os.path.join(run, "benchmark.json")))["game_runs"]
    rows = []
    for g in b:
        gid = g["game_id"]; hist = g["history"]
        grids, lv, st = replay(gid, hist)
        t = open(os.path.join(run, gid + "_p0.txt"), errors="replace").read()
        parts = HEADER.split(t); turns = []
        for i in range(1, len(parts) - 3, 4):
            body = parts[i + 3]; sp = SEC.split(body); user = ""; think = []
            for j in range(1, len(sp) - 1, 2):
                if sp[j] == "USER PROMPT" and not user: user = sp[j + 1]
                if sp[j] == "THINKING": think.append(sp[j + 1])
            m = LEVEL.search(user)
            turns.append({"step": int(parts[i]), "M": int(parts[i + 1]), "time": parts[i + 2],
                          "level": int(m.group(2)) if m else None, "user": user, "think": "\n".join(think)})
        mis = sum(1 for tu in turns if tu["level"] is not None and tu["M"] - 1 < len(lv) and lv[tu["M"] - 1] + 1 != tu["level"])
        for k, tu in enumerate(turns):
            lines = {}
            for ln in tu["user"].splitlines():
                for key in ("Stall check:", "Untested on this level:", "Valid actions not yet used on this level:", "New on this level"):
                    if ln.startswith(key): lines[key] = ln
            if not lines: continue
            a0 = tu["M"] - 1
            a1 = turns[k + 3]["M"] - 1 if k + 3 < len(turns) else len(hist)
            L = tu["level"]
            acts = []
            for i in range(a0, min(a1, len(hist))):
                if lv[i] + 1 != L: break  # level changed (completed) inside the window
                acts.append(i)
            va = re.search(r"Valid actions right now: (.*)\.", tu["user"])
            row = {"valid": va.group(1) if va else "", "game": gid[:4], "turn": tu["step"], "M": tu["M"], "level": L, "time": tu["time"],
                   "lines": list(lines), "n_actions_3turns": len(acts), "misaligned": mis,
                   "level_done_in_window": any(lv[i] + 1 > L for i in range(a0, min(a1, len(hist)) + 1) if i < len(lv)),
                   "level_ever_done": max(lv) + 1 > L}
            for key in ("Untested on this level:", "New on this level"):
                if key not in lines: continue
                listed = [(CH.index(m.group(1)), (int(m.group(2)), int(m.group(3))), int(m.group(5)), int(m.group(6)))
                          for m in TYPE.finditer(lines[key])]
                hits = []
                for i in acts:
                    a = hist[i]["action"]
                    if a["id"] != "ACTION6": continue
                    r, c = a["data"]["y"], a["data"]["x"]; grid = grids[i]
                    cells = comp(grid, r, c)
                    if not cells: continue
                    rs = [y for y, _ in cells]; cs = [x for _, x in cells]
                    ck = (grid[r][c], (max(rs) - min(rs) + 1, max(cs) - min(cs) + 1))
                    ty = types(grid); near = {(r + dy, c + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)}
                    for col, box, er, ec in listed:
                        inst = ty.get((col, box), [])
                        if ck == (col, box) or any(not cl.isdisjoint(near) for cl in inst):
                            changed = any(grids[i + 1][y][x] != grid[y][x] for y in range(4, 60) for x in range(4, 60))
                            hits.append({"i": i, "type": f"{CH[col]} {box[0]}x{box[1]}", "rc": [r, c],
                                         "changed": changed, "state_after": st[i + 1], "lv_after": lv[i + 1]})
                            break
                row[key[:8] + "_listed"] = [f"{CH[c]} {b[0]}x{b[1]}" for c, b, _, _ in listed]
                row[key[:8] + "_hits"] = hits
                row[key[:8] + "_clicks"] = sum(1 for i in acts if hist[i]["action"]["id"] == "ACTION6")
                row[key[:8] + "_mentioned"] = any(
                    s in turns[k]["think"] for s in ("Untested", "untested", "New on this level", "new on this level", "New element", "new type"))
            if "Valid actions not yet used on this level:" in lines:
                un = [x.strip(" .") for x in lines["Valid actions not yet used on this level:"].split(":", 1)[1].split(",")]
                used = [NAMES.get(hist[i]["action"]["id"]) for i in acts]
                row["unused_listed"] = un
                row["unused_used"] = sorted(set(u for u in un if u in used))
                row["unused_gameover"] = [i for i in acts if NAMES.get(hist[i]["action"]["id"]) in un and "GAME_OVER" in st[i + 1]]
            row["gameover_in_window"] = [i for i in acts if "GAME_OVER" in st[i + 1]]
            rows.append(row)
        print(gid[:4], "turns", len(turns), "actions", len(hist), "level-misaligned turns", mis, file=sys.stderr)
    json.dump(rows, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
