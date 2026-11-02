"""M1: per level, delay from the level's start (previous level's completing action) to its first action,
in seconds and generated tokens, from benchmark.json action histories. Also level durations."""
from common import *
import statistics as st, collections

rows = []  # (run, game, level_idx, cleared, start, first, end, gap_s, gap_tok, dur_s, nact, tok)
for r in RUNS:
    for g in bench(r)['game_runs']:
        h = g['history']; apl = g['actions_per_level']; lc = g['levels_completed']
        cum = 0; prev_end = 0.0
        for k, n in enumerate(apl):
            if n == 0:
                break
            acts = h[cum:cum + n]; cum += n
            cleared = k < lc
            first = acts[0]['wallclock_seconds']
            end = acts[-1]['wallclock_seconds']
            gap_tok = acts[0]['generated_tokens']
            tok = sum(a['generated_tokens'] for a in acts)
            rows.append(dict(run=r, game=g['game_id'][:4], k=k + 1, cleared=cleared, start=prev_end, first=first,
                             end=end, gap=first - prev_end, gap_tok=gap_tok, dur=end - prev_end, nact=n, tok=tok,
                             base=g['base_actions_per_level'][k], nlev=g['number_of_levels']))
            prev_end = end
        # stuck tail: from last cleared end to window end
import pickle
pickle.dump(rows, open('levels.pkl', 'wb'))

def q(xs):
    xs = sorted(xs)
    if not xs: return 'n=0'
    return f"n={len(xs)} median {st.median(xs):.0f} mean {st.mean(xs):.0f} p75 {xs[int(.75*len(xs))]:.0f} p90 {xs[int(.9*len(xs))]:.0f}"

l1 = [x for x in rows if x['k'] == 1]
ln = [x for x in rows if x['k'] >= 2]
print('L1 first-action delay s:', q([x['gap'] for x in l1]))
print('L>=2 first-action delay s:', q([x['gap'] for x in ln]))
print('L>=2 first-action delay tokens:', q([x['gap_tok'] for x in ln]))
cl = [x for x in ln if x['cleared']]
print('L>=2 cleared level duration s:', q([x['dur'] for x in cl]))
tot_gap = sum(x['gap'] for x in cl); tot_dur = sum(x['dur'] for x in cl)
print(f'share of cleared L>=2 clock before first action: {tot_gap/tot_dur:.1%} ({tot_gap/60:.0f} of {tot_dur/60:.0f} min)')
stuck = [x for x in ln if not x['cleared']]
print('stuck levels (L>=2) first-action delay s:', q([x['gap'] for x in stuck]))
# by level index
by = collections.defaultdict(list)
for x in ln: by[x['k']].append(x['gap'])
print('median gap by level:', {k: round(st.median(v)) for k, v in sorted(by.items())}, {k: len(v) for k, v in sorted(by.items())})
# gap share of per-run window: sum gaps (all levels>=2) / (25 games * 7200)
for r in RUNS:
    gs = sum(x['gap'] for x in ln if x['run'] == r)
    print(r, f'sum new-level gaps {gs/60:.0f} min = {gs/(25*7200):.1%} of window; levels cleared', sum(x['cleared'] for x in rows if x['run'] == r))
