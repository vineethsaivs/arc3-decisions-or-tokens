"""M2b: on long cleared levels, when does the winning approach first appear? Proxy: the first turn on the
level whose code identifiers overlap the winning turn's code (Jaccard >= 0.5), or whose thinking shares
>= 40% of the winning thinking's distinctive words. Report clock after that point (execution/refinement)."""
from levels import *
import statistics as st, keyword, builtins, re, collections

STOP = set(keyword.kwlist) | set(dir(builtins)) | {'current_frame', 'segmentation', 'nodes', 'ascii', 'action', 'history',
        'transitions', 'previous_frame', 'last_transition', 'frame_diff', 'print', 'len', 'range', 'i', 'j', 'k', 'r', 'c',
        'n', 'x', 'y', 'a', 'b', 'f', 'e', 's', 't', 'p', 'd', 'm', 'v', 'g', 'color', 'bbox', 'pixels', 'id', 'append',
        'items', 'get', 'split', 'join', 'last_action_call_result', 'level_completed', 'no_op', 'row', 'col', 'MOUSE',
        'UP', 'DOWN', 'LEFT', 'RIGHT', 'SPACE', 'RESET', 'UNDO', 'fr', 'res', 'out'}
ID = re.compile(r'\b[A-Za-z_]\w+\b')
W = re.compile(r'\b[a-z]{5,}\b')
COMMON = None

def ids(code):
    return {w for w in ID.findall(code) if w not in STOP and not w.isdigit()}

rows = []
docfreq = collections.Counter(); ndocs = 0
segs = list(segments())
for r, g, L, seg, cleared, t0, t1, ts, i0 in segs:
    for t in seg:
        ndocs += 1
        docfreq.update(set(W.findall(' '.join(x['think'] for x in t['resp']).lower())))
rare = lambda w: docfreq[w] < 0.02 * ndocs

def q(xs):
    xs = sorted(xs)
    return f"n={len(xs)} med {st.median(xs):.2f} mean {st.mean(xs):.2f} p25 {xs[len(xs)//4]:.2f} p75 {xs[3*len(xs)//4]:.2f}" if xs else 'n=0'

for r, g, L, seg, cleared, t0, t1, ts, i0 in segs:
    if not cleared or L is None or L < 2 or len(seg) < 4: continue
    dur = t1 - t0
    win = seg[-1]
    wc = ids('\n'.join(x['code'] for x in win['resp']))
    wt = {w for w in W.findall(' '.join(x['think'] for x in win['resp']).lower()) if rare(w)}
    t_code = t_think = None
    for k, t in enumerate(seg[:-1]):
        c = ids('\n'.join(x['code'] for x in t['resp']))
        if t_code is None and wc and c and len(wc & c) / len(wc | c) >= 0.5: t_code = k
        th = {w for w in W.findall(' '.join(x['think'] for x in t['resp']).lower()) if rare(w)}
        if t_think is None and len(wt) >= 5 and len(wt & th) / len(wt) >= 0.4: t_think = k
    def after(k):
        if k is None: return None
        return (t1 - seg[k]['t']) / dur, len(seg) - k, sum(1 for t in seg[k:] if t['gameover'] or t['dreset']), sum(1 for t in seg[k:] if t['nexec'] == 0)
    rows.append(dict(dur=dur, code=after(t_code), think=after(t_think), n=len(seg)))

for lab, sel in [('all cleared L>=2 (>=4 turns)', lambda x: True), ('>15 min', lambda x: x['dur'] > 900), ('>30 min', lambda x: x['dur'] > 1800)]:
    R = [x for x in rows if sel(x)]
    for key in ('code', 'think'):
        K = [x for x in R if x[key]]
        if not K: continue
        frac = [x[key][0] for x in K]
        clock_after = sum(x[key][0] * x['dur'] for x in K) / sum(x['dur'] for x in K)
        print(f"{lab} [{key} proxy] found in {len(K)}/{len(R)}; share of level clock after approach first appears: {q(frac)}; clock-weighted {clock_after:.0%}; "
              f"turns after: med {st.median(x[key][1] for x in K)}; failures after: mean {st.mean(x[key][2] for x in K):.2f}; idle turns after: mean {st.mean(x[key][3] for x in K):.1f}")
