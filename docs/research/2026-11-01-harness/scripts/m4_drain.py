"""M4: context drains (history_messages drops). How many turns survive a drain, whether the current
level's opening turns survive, how much level clock runs with the level's opening out of context, and
how behaviour changes in the two turns after a drain vs the two before."""
from levels import *
import statistics as st, collections, re

COORD = re.compile(r'\(\s*\d+\s*,\s*\d+\s*\)|\[\s*\d+\s*,\s*\d+|rows? \d+|cols? \d+')

def q(xs):
    xs = sorted(xs)
    return f"n={len(xs)} med {st.median(xs):.0f} mean {st.mean(xs):.1f} p25 {xs[len(xs)//4]} p75 {xs[3*len(xs)//4]}" if xs else 'n=0'

drains = []; plays = 0
lvl_stats = []  # per level: cleared, dur, drains, amnesic clock
for r, g, ts in games():
    plays += 1
    for t in ts: flags(t)
    # level start index for each turn
    start_idx = []; cur = None; s = 0
    for i, t in enumerate(ts):
        if t['level'] != cur: cur = t['level']; s = i
        start_idx.append(s)
    amnesic_from = {}  # level start idx -> turn index from which the level opening is out of context
    for i in range(1, len(ts)):
        a, b = ts[i - 1]['hist'], ts[i]['hist']
        if a is None or b is None or b >= a: continue
        # walk back to find earliest turn fully retained
        tot = 0; k = i
        while k >= 0 and tot + 1 + 2 * len(ts[k]['resp']) <= b:
            tot += 1 + 2 * len(ts[k]['resp']); k -= 1
        first_kept = k + 1
        ls = start_idx[i]
        lost_open = first_kept > ls
        drains.append(dict(r=r, g=g, i=i, kept_turns=i - first_kept + 1, lost_open=lost_open,
                           turns_into_level=i - ls, before=a, after=b))
        if lost_open and ls not in amnesic_from: amnesic_from[ls] = i
    # per level
    i = 0
    while i < len(ts):
        L = ts[i]['level']; j = i
        while j < len(ts) and ts[j]['level'] == L: j += 1
        cleared = j < len(ts) and ts[j]['level'] is not None and L is not None and ts[j]['level'] > L
        t_end = ts[j]['t'] if j < len(ts) else ts[-1]['t'] + ts[-1]['dur']
        nd = sum(1 for d in drains if d['r'] == r and d['g'] == g and i <= d['i'] < j)
        am = amnesic_from.get(i)
        amn = (t_end - ts[am]['t']) if am is not None and am < j else 0
        lvl_stats.append(dict(r=r, g=g, L=L, cleared=cleared, dur=t_end - ts[i]['t'], drains=nd, amn=amn, nturns=j - i))
        i = j

print(f"drains: {len(drains)} over {plays} game plays = {len(drains)/plays:.1f} per play")
print('turns retained after a drain:', q([d['kept_turns'] for d in drains]))
print(f"drains that drop the current level's opening turn: {sum(d['lost_open'] for d in drains)} ({sum(d['lost_open'] for d in drains)/len(drains):.0%})")
for name, sel in [('cleared L>=2', lambda x: x['cleared'] and x['L'] and x['L'] >= 2), ('stuck (final) levels', lambda x: not x['cleared'])]:
    R = [x for x in lvl_stats if sel(x)]
    Dd = sum(x['dur'] for x in R)
    print(f"{name}: n={len(R)}; with >=1 drain {sum(x['drains']>0 for x in R)} ({sum(x['drains']>0 for x in R)/len(R):.0%}); "
          f"clock run with the level opening out of context: {sum(x['amn'] for x in R)/3600:.1f} h of {Dd/3600:.1f} h ({sum(x['amn'] for x in R)/Dd:.0%})")
    long = [x for x in R if x['dur'] > 900]
    if long:
        print(f"   of which >15 min: n={len(long)}, with a drain {sum(x['drains']>0 for x in long)/len(long):.0%}, amnesic clock share {sum(x['amn'] for x in long)/sum(x['dur'] for x in long):.0%}, drains/level {st.mean(x['drains'] for x in long):.1f}")

# behaviour around drains that lost the level opening: 2 turns before vs 2 after
def feats(t):
    th = ' '.join(x['think'] for x in t['resp'])
    seg_reads = sum(len(re.findall(r'segmentation', x['code'])) for x in t['resp'])
    return dict(idle=int(t['nexec'] == 0), rchars=t['rchars'], calls=t['ncalls'], segreads=seg_reads,
                coordrate=len(COORD.findall(th)) / max(1, len(th)) * 1000, newdefs=len(t['defs']), errs=t['errs'])
idx = {(r, g): ts for r, g, ts in games()}
for t_ in idx.values():
    for t in t_: flags(t)
for lab, sel in [('drains losing level opening', lambda d: d['lost_open']), ('drains keeping it', lambda d: not d['lost_open'])]:
    B = collections.defaultdict(list); A = collections.defaultdict(list)
    for d in drains:
        if not sel(d): continue
        ts = idx[(d['r'], d['g'])]; i = d['i']
        if i - 2 < 0 or i + 2 >= len(ts): continue
        # same level throughout
        if len({ts[k]['level'] for k in range(i - 2, i + 3)}) > 1: continue
        for k in (i - 2, i - 1):
            for f, v in feats(ts[k]).items(): B[f].append(v)
        for k in (i + 1, i + 2):
            for f, v in feats(ts[k]).items(): A[f].append(v)
    print(f"{lab} (turn pairs n={len(B['idle'])}):", ', '.join(f"{f} {st.mean(B[f]):.2f}->{st.mean(A[f]):.2f}" for f in B))
