"""M3: outcomes of multi-action executions (one python response executing >=3 actions, batch or loop) and what
follows a stop. Stop = the no-op/stale guard halted it, or GAME OVER. Follow-up: seconds and tokens until the next
executed action, whether the next turn is idle (yield), and whether the level was cleared within the next 3 turns."""
from levels import *
import re, statistics as st, collections

B = re.compile(r'batch of (\d+) actions, (\d+) executed')
def outcome(res):
    lines = re.findall(r'^\[action\] (.*)$', res, re.M)
    n = 0; o = 'ran_out'
    for a in lines:
        m = B.search(a)
        if m: n += int(m.group(2))
        elif 'not executed' not in a and ' -> ' in a: n += 1
        if 'LEVEL COMPLETED' in a or 'stop_reason: level_completed' in a or 'RUN COMPLETE' in a or 'run_complete' in a: o = 'level_completed'
        elif 'GAME OVER' in a or 'game_over' in a: o = 'game_over'
        elif 'stale_state' in a or 'no_op_action' in a: o = 'guard_stop' if o == 'ran_out' else o
    if o == 'ran_out' and 'StaleStateActionError' in res: o = 'guard_stop'
    return n, o

cnt = collections.Counter(); follow = collections.defaultdict(list)
for r, g, ts in games():
    for t in ts: flags(t)
    for i, t in enumerate(ts):
        for x in t['resp']:
            n, o = outcome(x['result'])
            if n < 3: continue
            cnt[o] += 1
            if o in ('guard_stop', 'game_over', 'ran_out'):
                # next turn(s)
                if i + 1 < len(ts):
                    nx = ts[i + 1]
                    follow[o].append(dict(next_idle=nx['nexec'] == 0, next_yield=nx['yield'],
                                          next_dur=nx['dur'],
                                          cleared3=any(ts[k]['level'] != t['level'] for k in range(i + 1, min(len(ts), i + 4)))))
tot = sum(cnt.values())
print('responses executing >=3 actions:', tot, {k: f"{v} ({v/tot:.0%})" for k, v in cnt.most_common()})
for o, F in follow.items():
    print(f"after {o} (n={len(F)}): next turn idle {sum(f['next_idle'] for f in F)/len(F):.0%}, yield {sum(f['next_yield'] for f in F)/len(F):.0%}, "
          f"level cleared within 3 turns {sum(f['cleared3'] for f in F)/len(F):.0%}")
