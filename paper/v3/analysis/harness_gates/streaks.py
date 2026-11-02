"""Per run: analyzer turns, idle streaks, response to a 3rd/5th/7th idle turn, stall/untested lines.
python3 streaks.py RUN_DIR [RUN_DIR ...]   (dirs hold *_p0.txt and benchmark.json)"""
import re, glob, json, os, sys, collections
HEADER = re.compile(r"^--- analysis_step=(\d+) \| action=(\d+) \| (\S+) \| tool-agent ---$", re.M)
LEVEL = re.compile(r"Current state: step (\d+), level (\d+)\.")
for d in sys.argv[1:]:
    files = [f for f in glob.glob(os.path.join(d, '*_p0.txt')) + glob.glob(os.path.join(d, 'transcripts-dl', '*_p0.txt'))]
    seen = set(); out = collections.Counter()
    for f in files:
        g = os.path.basename(f)
        if g in seen: continue
        seen.add(g)
        t = open(f, errors='replace').read()
        if not t: continue
        out['stall_lines'] += t.count('Stall check: your last')
        out['untested_lines'] += t.count('Untested on this level:')
        parts = HEADER.split(t); steps = []
        for i in range(1, len(parts) - 3, 4):
            m = LEVEL.search(parts[i + 3]); steps.append((int(parts[i + 1]), int(m.group(2)) if m else 0))
        idle = 0; lvl = None; streak = 0
        for k, (a, L) in enumerate(steps):
            if L != lvl:
                if lvl is not None: out['streak5'] += streak >= 5; out['streak4'] += streak >= 4
                lvl, idle, streak = L, 0, 0
            nxt = steps[k + 1] if k + 1 < len(steps) else None
            acted = nxt is not None and nxt[0] > a
            out['turns'] += 1
            if idle in (3, 5, 7):
                out['at_stall'] += 1; out['at_stall_acted'] += acted
            idle = 0 if acted else idle + 1
            streak = max(streak, idle)
        out['streak5'] += streak >= 5; out['streak4'] += streak >= 4; out['levels'] += len(set(L for _, L in steps))
    b = json.load(open(os.path.join(d, 'benchmark.json')))['game_runs']
    lv = sum(g['levels_completed'] for g in b); acts = sum(sum(g['actions_per_level']) for g in b)
    resp = out['at_stall_acted'] / max(1, out['at_stall'])
    print(f"{os.path.basename(d.rstrip('/'))}: levels {lv} actions {acts} turns {out['turns']} level-plays {out['levels']} "
          f"streak>=4 {out['streak4']} streak>=5 {out['streak5']} at-stall turns {out['at_stall']} acted {resp:.2f} "
          f"stall lines {out['stall_lines']} untested lines {out['untested_lines']}")
