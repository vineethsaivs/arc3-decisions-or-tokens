"""Replay human runs (and two agent levels) on the local engine; dump every frame as hex rows for the overlay check."""
import sys, json, glob, re
sys.path.insert(0, '<repo>/solution/analysis/keyfx')
from envload import arcade, BY_ID
from arcengine import GameAction
R = '<repo>/reference/re-arc-3/re_arc/dsl/official_human_replays'
NAMES = {1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT", 5: "SPACE", 7: "UNDO"}
hexg = lambda o: ["".join("0123456789abcdef"[int(v)] for v in r) for r in o.frame[-1]]
def disp(a, data): return f"MOUSE(row={data['y']}, col={data['x']})" if a == 6 else NAMES[a]
def run(gid, steps, out):
    env = arcade.make(gid); obs = env.reset()
    seq = [{"a": "", "g": hexg(obs), "lv": obs.levels_completed + 1, "auto": False}]
    valid = sorted({a for a, _ in steps})
    for a, data in steps:
        obs = env.step(BY_ID[a], data=data or {})
        seq.append({"a": disp(a, data or {}), "g": hexg(obs), "lv": obs.levels_completed + 1, "auto": False})
        if 'GAME_OVER' in str(obs.state):
            obs = env.step(GameAction.RESET, data={})
            seq.append({"a": "RESET", "g": hexg(obs), "lv": obs.levels_completed + 1, "auto": True, "over": True})
    json.dump({"gid": gid, "valid": valid, "seq": seq}, open(out, 'w'))
    return len(seq)
for f in sorted(glob.glob(R + '/*.json')):
    d = json.load(open(f))
    n = run(d['game_id'], [(a, data) for seg in d['levels'] for a, data in seg], f"{sys.argv[1]}/h_{d['game_id'][:4]}.json")
    print(d['game_id'][:4], n)
def parse(s):
    m = re.match(r"MOUSE\(row=(\d+), col=(\d+)\)", s)
    return (6, {"x": int(m.group(2)), "y": int(m.group(1))}) if m else ({v: k for k, v in NAMES.items()}[s], {})
for tag, gid, nlev, path in (("u2bp35", "bp35", 2, "u2_bp35_L3.json"), ("u1su15", "su15", 4, "u1_su15_L5.json")):
    d = json.load(open(glob.glob(f"{R}/{gid}*.json")[0]))
    steps = [(a, data) for seg in d['levels'][:nlev] for a, data in seg] + [parse(s) for s in json.load(open(f"{sys.argv[2]}/{path}"))]
    print(tag, run(d['game_id'], steps, f"{sys.argv[1]}/a_{tag}.json"))
