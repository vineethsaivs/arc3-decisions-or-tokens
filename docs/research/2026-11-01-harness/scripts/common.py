import pickle, re, os
from datetime import datetime

D = pickle.load(open(os.environ.get('PKL', '<scratch>/fb519d0b-5ca4-49bf-b4d4-4ffb21661429/scratchpad/turns.pkl'), 'rb'))
RUNS = sorted({r for r, g in D})
GAMES = sorted({g for r, g in D if g != '_bench'})


def secs(ts):
    t = datetime.strptime(ts, '%H:%M:%S')
    return t.hour * 3600 + t.minute * 60 + t.second


def games():
    for (r, g), turns in sorted(D.items()):
        if g == '_bench' or not turns:
            continue
        # timestamps: seconds since first turn, unwrap midnight
        base = secs(turns[0]['ts']); prev = 0; off = 0
        for t in turns:
            s = secs(t['ts']) - base + off
            if s < prev - 3600:
                off += 86400; s += 86400
            t['t'] = s; prev = s
        # turn duration = next turn start - this start (last turn: unknown, use 0)
        for i, t in enumerate(turns):
            t['dur'] = turns[i + 1]['t'] - t['t'] if i + 1 < len(turns) else 0
            t['nexec'] = nexec(t['actions'])
        yield r, g, turns


def bench(r):
    return D[(r, '_bench')]


ACT_NAME = re.compile(r'^(\w+(?:\(row=\d+, col=\d+\))?)')

_B = re.compile(r'batch of (\d+) actions, (\d+) executed')
def nexec(lines):
    n = 0
    for a in lines:
        m = _B.search(a)
        if m: n += int(m.group(2))
        elif 'not executed' in a: pass
        elif ' -> ' in a: n += 1
    return n
