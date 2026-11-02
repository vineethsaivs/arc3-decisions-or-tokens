"""M6: simulator validation habit. A snippet 'validates' when it loops over recorded transitions/history and
compares a prediction from a model-written function with the recorded after-frame. Per level (L>=2 and L1):
does it ever validate, when (share of level clock elapsed), and clear rates; on long levels and stuck levels."""
from levels import *
import re, statistics as st, collections

LOOP = re.compile(r'for\s+[^\n]*\bin\s+[^\n]*(transitions|history)\b')
AFTER = re.compile(r'after_frame|\.frame\b|after\b')
CMP = re.compile(r'!=|==|mismatch|match|ok\b|assert', re.I)
SIMCALL = re.compile(r'\b(\w*(sim|predict|step|apply|model|move|simulate|transition|forward)\w*)\s*\(', re.I)

def validates(code, defined):
    if not LOOP.search(code) or not AFTER.search(code) or not CMP.search(code): return False
    for m in SIMCALL.finditer(code):
        name = m.group(1)
        if name in defined: return True
    return False

rows = []
for r, g, ts in games():
    for t in ts: flags(t)
    defined = set()
    i = 0
    while i < len(ts):
        L = ts[i]['level']; j = i
        while j < len(ts) and ts[j]['level'] == L: j += 1
        cleared = j < len(ts) and ts[j]['level'] is not None and L is not None and ts[j]['level'] > L
        t0 = ts[i]['t']; t1 = ts[j]['t'] if j < len(ts) else ts[-1]['t'] + ts[-1]['dur']
        first_val = None; nval = 0; first_sim = None
        for k in range(i, j):
            for x in ts[k]['resp']:
                defined |= set(DEF.findall(x['code']))
                if first_sim is None and SIM_DEF.search(x['code']): first_sim = ts[k]['t']
                if validates(x['code'], defined):
                    nval += 1
                    if first_val is None: first_val = ts[k]['t']
        rows.append(dict(L=L, cleared=cleared, dur=t1 - t0, val=first_val is not None, nval=nval,
                         val_at=None if first_val is None else (first_val - t0) / max(1, t1 - t0),
                         sim=first_sim is not None, sim_at=None if first_sim is None else (first_sim - t0) / max(1, t1 - t0)))
        i = j

for lab, sel in [('L>=2 all', lambda x: x['L'] and x['L'] >= 2), ('L>=2 >15 min', lambda x: x['L'] and x['L'] >= 2 and x['dur'] > 900)]:
    R = [x for x in rows if sel(x)]
    C = [x for x in R if x['cleared']]; S = [x for x in R if not x['cleared']]
    def rate(X, k): return f"{sum(x[k] for x in X)}/{len(X)} ({sum(x[k] for x in X)/max(1,len(X)):.0%})"
    print(f"{lab}: validates cleared {rate(C,'val')}, stuck {rate(S,'val')}; sim-like def cleared {rate(C,'sim')}, stuck {rate(S,'sim')}")
    va = [x['val_at'] for x in C if x['val_at'] is not None]
    if va: print(f"   cleared: first validation at median {st.median(va):.0%} of level clock (n={len(va)})")
    vs = [x['val_at'] for x in S if x['val_at'] is not None]
    if vs: print(f"   stuck: first validation at median {st.median(vs):.0%} of the time played (n={len(vs)})")
    V = [x for x in R if x['val']]; N = [x for x in R if not x['val']]
    print(f"   clear rate with validation {sum(x['cleared'] for x in V)/max(1,len(V)):.0%} (n={len(V)}) vs without {sum(x['cleared'] for x in N)/max(1,len(N)):.0%} (n={len(N)})")
