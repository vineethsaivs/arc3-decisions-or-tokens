"""Deaths (GAME_OVER) per run from benchmark.json histories replayed on the offline engine; and actions spent in
attempts that ended in a death. python deaths.py RUN_DIR..."""
import json, sys, os, collections
sys.path.insert(0, '<scratch>/fb519d0b-5ca4-49bf-b4d4-4ffb21661429/scratchpad/scripts')
import uns2_lines as U
K = 'solution/kaggle/generated/kernel-outputs/'
for run in sys.argv[1:]:
    b = json.load(open(K + run + '/benchmark.json'))['game_runs']
    deaths = wasted = acts = 0; per = collections.Counter()
    for g in b:
        if not g.get('history'): continue
        grids, lv, st = U.replay(g['game_id'], g['history'])
        since = 0
        for i in range(1, len(st)):
            acts += 1; since += 1
            if 'GAME_OVER' in st[i] and 'GAME_OVER' not in st[i - 1]:
                deaths += 1; wasted += since; per[g['game_id'][:4]] += 1
            if lv[i] != lv[i - 1] or g['history'][i - 1]['action']['id'] == 'RESET' or 'GAME_OVER' in st[i]:
                since = 0
    print(f"{run}: actions {acts} deaths {deaths} actions in died attempts {wasted} ({wasted / max(1, acts):.0%}) top {per.most_common(5)}")
