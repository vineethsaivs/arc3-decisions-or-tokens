"""M2: phase split of each cleared level (L>=2 and L1) into orientation (before first action),
failed attempts (up to the last GAME OVER / deliberate RESET), and the final attempt; plus when the
function the winning turn called was first defined on the level."""
from levels import *
import statistics as st, collections

def q(xs):
    xs = sorted(xs)
    if not xs: return 'n=0'
    return f"n={len(xs)} med {st.median(xs):.0f} mean {st.mean(xs):.0f} p75 {xs[int(.75*len(xs))]:.0f} p90 {xs[int(.9*len(xs))]:.0f}"

rows = []
for r, g, L, seg, cleared, t0, t1, ts, i0 in segments():
    if not cleared or L is None:
        continue
    dur = t1 - t0
    # first turn that executed an action (its end time approximates the first action)
    fa = next((k for k, t in enumerate(seg) if t['nexec']), None)
    t_fa = seg[fa]['t'] if fa is not None else t1
    # failures: turn whose user prompt reports a GAME OVER or deliberate reset -> new attempt starts at that turn
    fails = [t['t'] for t in seg if t['gameover'] or t['dreset']]
    t_last_fail = fails[-1] if fails else t_fa
    win = seg[-1]
    wincode = '\n'.join(x['code'] for x in win['resp'])
    called = set(CALL.findall(wincode))
    defined_on_level = {}
    for t in seg:
        for n in t['defs']:
            defined_on_level.setdefault(n, t['t'])
    key = [defined_on_level[n] for n in called if n in defined_on_level]
    # earliest definition (on this level) of a function the winning code calls
    t_key = min(key) if key else None
    nturns = len(seg)
    idle = sum(1 for t in seg if not t['nexec'])
    wins_by_search = bool(SEARCH_DEF.search(wincode)) or any(re.search(r'(bfs|search|solve|astar|plan)', n, re.I) for n in called if n in defined_on_level)
    win_n = win['nexec']
    rows.append(dict(r=r, g=g, L=L, dur=dur, orient=t_fa - t0, fails=len(fails), final=t1 - t_last_fail,
                     t_key=None if t_key is None else t1 - t_key, nturns=nturns, idle=idle, search=wins_by_search,
                     win_n=win_n, yields=sum(t['yield'] for t in seg), stops=sum(t['stopped'] for t in seg),
                     searchdef_any=any(t['searchdef'] for t in seg)))

import pickle; pickle.dump(rows, open('phases.pkl', 'wb'))
for name, sel in [('L1', lambda x: x['L'] == 1), ('L>=2', lambda x: x['L'] >= 2)]:
    R = [x for x in rows if sel(x)]
    D_ = sum(x['dur'] for x in R)
    print(f"== {name} cleared levels n={len(R)} total {D_/3600:.1f} h")
    print('  duration s', q([x['dur'] for x in R]))
    print(f"  orientation share {sum(x['orient'] for x in R)/D_:.1%}; final-attempt share {sum(x['final'] for x in R)/D_:.1%}")
    nf = [x for x in R if x['fails']]
    print(f"  levels with >=1 failure (GAME OVER or deliberate RESET): {len(nf)} ({len(nf)/len(R):.0%}), their clock share {sum(x['dur'] for x in nf)/D_:.0%}")
    print(f"  winning turn used a search/solver defined on the level: {sum(x['search'] for x in R)} ({sum(x['search'] for x in R)/len(R):.0%})")
    print('  turns per level', q([x['nturns'] for x in R]), ' idle-turn share', f"{sum(x['idle'] for x in R)/sum(x['nturns'] for x in R):.0%}")
    # long levels
    long = [x for x in R if x['dur'] > 900]
    if long:
        DL = sum(x['dur'] for x in long)
        print(f"  levels >15 min: n={len(long)} ({len(long)/len(R):.0%} of levels, {DL/D_:.0%} of clock)")
        print(f"    orientation {sum(x['orient'] for x in long)/DL:.1%}, final attempt {sum(x['final'] for x in long)/DL:.1%}, failures/level {st.mean(x['fails'] for x in long):.1f}")
        kk = [x for x in long if x['t_key'] is not None]
        print(f"    winning code calls a function first defined on the level: {len(kk)}/{len(long)}; clock from that definition to clear: {q([x['t_key'] for x in kk])}; as share of level {sum(x['t_key'] for x in kk)/sum(x['dur'] for x in kk):.0%}")
        print(f"    search used in win: {sum(x['search'] for x in long)}/{len(long)}; any search defined: {sum(x['searchdef_any'] for x in long)}/{len(long)}")
        print('    turns', q([x['nturns'] for x in long]), 'yields', q([x['yields'] for x in long]), 'stopped batches', q([x['stops'] for x in long]))
    short = [x for x in R if x['dur'] <= 900]
    print(f"  levels <=15 min: n={len(short)}; search used in win {sum(x['search'] for x in short)}/{len(short)}")
