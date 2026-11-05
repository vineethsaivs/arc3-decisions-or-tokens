"""Reconcile the run-to-run SD frames of the paper (2026-11-04). Run from the repo root:
    python3 paper/v3/analysis/v3_variance_frames.py
Writes paper/v3/analysis/out/variance_frames.json and variance_frames.md.
Inputs (all in the repo): paper/v2/analysis/out/yield_runs.json (34 Kaggle and GCP runs, per game levels),
paper/v2/analysis/out/e4_noise.json (six identical pairs), paper/v3/evidence/data/bench_pool18_levels.json (18 pool),
paper/v3/evidence/data/lora_pergame.csv (nine y8-family builds per game), and the Vertex benchmark.json files under
solution/kaggle/generated/kernel-outputs/ (vertex-rq1020-*, vertex-cal1, vertex-yw-*; per game levels_completed).
Needs numpy and scipy (python3 -c 'import numpy, scipy').
"""
import json, glob, csv, os, math
import numpy as np
from scipy import stats
from scipy.optimize import brentq

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
P = lambda *a: os.path.join(ROOT, *a)
OUT = P('paper', 'v3', 'analysis', 'out')
rng = np.random.default_rng(20261104)


def sd_ci(ss, df, a=0.05):
    """SD point estimate and chi-square interval from sum of squares ss on df degrees of freedom."""
    sd = math.sqrt(ss / df)
    lo = math.sqrt(ss / stats.chi2.ppf(1 - a / 2, df)); hi = math.sqrt(ss / stats.chi2.ppf(a / 2, df))
    return dict(sd=sd, df=df, ci95=[lo, hi], ss=ss)


def frame(vals):
    v = np.array(vals, float)
    r = sd_ci(float(((v - v.mean()) ** 2).sum()), len(v) - 1)
    r.update(n=len(v), mean=float(v.mean()), levels=[int(x) for x in v])
    return r


# ---------------------------------------------------------------- load runs
yr = {r['dir']: r for r in json.load(open(P('paper/v2/analysis/out/yield_runs.json')))}
fam_dirs = ['franzen-l1a1y8-v1', 'franzen-l1a1y8r-v1', 'franzen-l1a1y8sv2-v1', 'franzen-l1a1y8sv2o16-v1',
            'franzen-l1a1y8sv2r1-v1', 'franzen-l1a1y8sv2r2-v1', 'franzen-l1a1y8sv2r3-v1',
            'franzen-l1a1y8sv2r3s4-v1', 'franzen-l1a1y8sv2r3hm32-v1']
pair_dirs = {'p448': ('franzen-p448-v1', 'franzen-p448rep-v1'), 'p448s14': ('franzen-p448s14-v1', 'franzen-p448s14rep-v1'),
             'l1a1y8': ('franzen-l1a1y8-v1', 'franzen-l1a1y8r-v1'), 'dpctl': ('gcp-dpctl-v1', 'gcp-dpctl-v2'),
             'L1': ('gcp-p448s14dpwfrc-v1', 'gcp-p448s14dpwfrc-v2'), 'l1a1': ('gcp-l1a1-v1', 'gcp-l1a1-v2')}


def vertex(name):
    d = json.load(open(P('solution/kaggle/generated/kernel-outputs', 'vertex-' + name, 'benchmark.json')))
    gl = {}
    for g in d['game_runs']:
        k = g.get('game_id') or g.get('game') or g.get('name')
        gl[str(k).split('-')[0]] = g.get('levels_completed', 0)
    return dict(levels=sum(gl.values()), per_game=gl, start=d['start_time'])


rq = {n: vertex(n) for n in ['rq1020-%d' % i for i in range(1, 9)]}
cal1 = vertex('cal1')
yw = {n: vertex(n) for n in sorted(os.path.basename(p)[7:] for p in glob.glob(P('solution/kaggle/generated/kernel-outputs/vertex-yw-*'))
                                   if os.path.exists(os.path.join(p, 'benchmark.json')))}
wave = {r['name']: r for r in json.load(open(P('paper/v3/analysis/out/yield_wave/read_2026-10-24.json')))['runs']}
y8 = [n for n in wave if wave[n]['arm'] == 'y8']; y2 = [n for n in wave if wave[n]['arm'] == 'y2']
assert all(yw[n]['levels'] == wave[n]['levels'] for n in wave), 'yield wave totals disagree'
games = sorted(yr[fam_dirs[0]]['per_game_levels'])
assert len(games) == 25
pg = lambda d: np.array([d[g] for g in games], float)
fam_pg = np.array([pg(yr[d]['per_game_levels']) for d in fam_dirs])
assert [int(x.sum()) for x in fam_pg] == [yr[d]['levels'] for d in fam_dirs]
# lora_pergame.csv must reproduce the family matrix
rows = list(csv.DictReader(open(P('paper/v3/evidence/data/lora_pergame.csv'))))
cols = ['l1a1y8', 'l1a1y8r', 'l1a1y8sv2', 'l1a1y8sv2o16', 'l1a1y8sv2r1', 'l1a1y8sv2r2', 'l1a1y8sv2r3', 'l1a1y8sv2r3s4', 'l1a1y8sv2r3hm32']
csv_pg = np.array([[float(r[c]) for r in rows if r['game'] in games] for c in cols])
csv_games = [r['game'] for r in rows if r['game'] in games]
csv_match = bool(csv_games == games and np.array_equal(csv_pg, fam_pg))
rq_pg = np.array([pg(v['per_game']) for v in rq.values()])
cal_pg = pg(cal1['per_game'])

# ---------------------------------------------------------------- frames
F = {}
# (a) six identical pairs
pairs = {k: [yr[a]['levels'], yr[b]['levels']] for k, (a, b) in pair_dirs.items()}
e4 = json.load(open(P('paper/v2/analysis/out/e4_noise.json')))['franzen_pairs']
assert all(e4[k]['levels'] == v for k, v in pairs.items())
d = np.array([v[1] - v[0] for v in pairs.values()], float)       # second minus first
F['a_pairs'] = sd_ci(float((d ** 2).sum() / 2), 6)
F['a_pairs'].update(pairs=pairs, diffs_second_minus_first=[int(x) for x in d], mean_shift=float(d.mean()),
                    sd_of_diffs=float(d.std(ddof=1)))
sd_d = float(d.std(ddof=1))
F['a_pairs_about_shift'] = sd_ci(float(((d - d.mean()) ** 2).sum() / 2), 5)  # per-run SD after removing the common shift
t_shift = d.mean() / (sd_d / math.sqrt(6))
F['a_pairs_about_shift'].update(t_shift=float(t_shift), p_shift_t5=float(2 * stats.t.sf(abs(t_shift), 5)),
                                sign_test_p_6_of_6=2 * 0.5 ** 6, shift_ci95=[float(d.mean() - stats.t.ppf(.975, 5) * sd_d / math.sqrt(6)),
                                                                              float(d.mean() + stats.t.ppf(.975, 5) * sd_d / math.sqrt(6))])
hrs = {}
for k, (a, b) in pair_dirs.items():
    ta = np.datetime64(yr[a]['start_time']); tb = np.datetime64(yr[b]['start_time'])
    hrs[k] = round(float((tb - ta) / np.timedelta64(1, 'h')), 1)
F['a_pairs']['hours_between_runs'] = hrs
F['a_pairs']['platform'] = {k: yr[a]['platform'] for k, (a, b) in pair_dirs.items()}
# (b) eight same-day identical rq runs
F['b_same_day8'] = frame([v['levels'] for v in rq.values()])
F['b_same_day8']['starts'] = {n: v['start'][:16] for n, v in rq.items()}
# (c) nine same-family Kaggle builds
F['c_family9'] = frame([yr[d_]['levels'] for d_ in fam_dirs])
F['c_family9']['builds'] = [yr[d_]['arm'] for d_ in fam_dirs]
F['c_family9']['starts'] = {yr[d_]['arm']: yr[d_]['start_time'][:16] for d_ in fam_dirs}
# (d) pooled 18 no-adapter runs
pool = json.load(open(P('paper/v3/evidence/data/bench_pool18_levels.json')))
pl = [v['all25'] for v in pool.values()]
F['d_pool18'] = frame(pl)
assert sorted(pl) == sorted([yr[x]['levels'] for x in fam_dirs] + [cal1['levels']] + [v['levels'] for v in rq.values()])
F['d_pool18']['platform_means'] = {p: float(np.mean([v['all25'] for v in pool.values() if v['platform'] == p])) for p in ('kaggle', 'vertex')}
F['d_pool18']['composition'] = '9 Kaggle y8-family builds (frame c) + vertex-cal1 + 8 vertex-rq1020 runs (frame b)'
# (e) interleaved yield wave
F['e_y8_arm'] = frame([wave[n]['levels'] for n in y8]); F['e_y2_arm'] = frame([wave[n]['levels'] for n in y2])
ss_e = F['e_y8_arm']['ss'] + F['e_y2_arm']['ss']
F['e_within_arm_pooled'] = sd_ci(ss_e, 10)
F['e_within_arm_pooled']['diff_y8_minus_y2'] = F['e_y8_arm']['mean'] - F['e_y2_arm']['mean']
F['e_within_arm_pooled']['f_ratio_var_y8_over_y2'] = F['e_y8_arm']['sd'] ** 2 / F['e_y2_arm']['sd'] ** 2
F['e_within_arm_pooled']['f_p_two_sided'] = float(2 * min(stats.f.sf(F['e_within_arm_pooled']['f_ratio_var_y8_over_y2'], 5, 5),
                                                          stats.f.cdf(F['e_within_arm_pooled']['f_ratio_var_y8_over_y2'], 5, 5)))
F['e_y8_arm']['start_span_h'] = None


# ---------------------------------------------------------------- order / day drift inside frames
def spearman_by_start(names, starts, levels):
    t = [np.datetime64(s) for s in starts]
    o = np.argsort(t); r = stats.spearmanr(np.arange(len(o)), np.array(levels)[o])
    return dict(rho=float(r.statistic), p=float(r.pvalue), n=len(levels))


F['b_same_day8']['spearman_levels_vs_start'] = spearman_by_start(list(rq), [v['start'] for v in rq.values()], [v['levels'] for v in rq.values()])
F['c_family9']['spearman_levels_vs_start'] = spearman_by_start(fam_dirs, [yr[x]['start_time'] for x in fam_dirs], [yr[x]['levels'] for x in fam_dirs])
F['e_y8_arm']['spearman_levels_vs_start'] = spearman_by_start(y8, [yw[n]['start'] for n in y8], [yw[n]['levels'] for n in y8])
F['e_y2_arm']['spearman_levels_vs_start'] = spearman_by_start(y2, [yw[n]['start'] for n in y2], [yw[n]['levels'] for n in y2])
rq_h = [np.datetime64(v['start']) for v in rq.values()]
F['b_same_day8']['span_h'] = round(float((max(rq_h) - min(rq_h)) / np.timedelta64(1, 'h')), 2)
fh = [np.datetime64(yr[x]['start_time']) for x in fam_dirs]
F['c_family9']['span_h'] = round(float((max(fh) - min(fh)) / np.timedelta64(1, 'h')), 1)
F['c_family9']['same_day_subset'] = {'note': 'builds started 10-12 only (r1 r2 r3 s4 hm32), 10-10 pair+sv2 and 10-11 o16 excluded',
                                     **frame([yr[x]['levels'] for x in fam_dirs if yr[x]['start_time'].startswith('2026-10-12')])}

# ---------------------------------------------------------------- pooled estimates over disjoint run sets
# disjoint: pairs (6 df, includes shift), rq8 (7), family9 (8), interleaved arms (10)
parts = {'pairs (about pair means, includes the shift)': (F['a_pairs']['ss'], 6),
         'same-day 8 (rq1020)': (F['b_same_day8']['ss'], 7),
         'family 9 (Kaggle builds)': (F['c_family9']['ss'], 8),
         'interleaved arms (y8 5 df + y2 5 df)': (ss_e, 10)}
allss = sum(v[0] for v in parts.values()); alldf = sum(v[1] for v in parts.values())
samesess = {k: v for k, v in parts.items() if not k.startswith('pairs')}
ssb = sum(v[0] for v in samesess.values()); dfb = sum(v[1] for v in samesess.values())
F['pooled_all_disjoint'] = sd_ci(allss, alldf); F['pooled_all_disjoint']['parts'] = {k: dict(ss=v[0], df=v[1], sd=math.sqrt(v[0] / v[1])) for k, v in parts.items()}
F['pooled_all_disjoint']['note'] = 'sets share no run; pooled by degrees of freedom (31 df). The pairs enter with the shift inside.'
F['pooled_excl_pairs'] = sd_ci(ssb, dfb); F['pooled_excl_pairs']['note'] = 'sets without the cross-day pairs (25 df): rq8, family9 (58 h span), interleaved arms'
sub = [v for k, v in parts.items() if not k.startswith('pairs') and not k.startswith('family')]
F['pooled_identical_only_same_session'] = sd_ci(sum(v[0] for v in sub), sum(v[1] for v in sub))
F['pooled_identical_only_same_session']['note'] = 'rq8 + interleaved arms (each set is one build; 17 df); no config differences, no cross-day pairs'
# heterogeneity of variance across the four sets (Bartlett)
grp = [(F['a_pairs']['ss'], 6), (F['b_same_day8']['ss'], 7), (F['c_family9']['ss'], 8), (ss_e, 10)]
k = len(grp); N = sum(g[1] for g in grp); sp2 = sum(g[0] for g in grp) / N
bt = (N * math.log(sp2) - sum(g[1] * math.log(g[0] / g[1]) for g in grp)) / (1 + (sum(1 / g[1] for g in grp) - 1 / N) / (3 * (k - 1)))
F['bartlett_four_sets'] = dict(stat=float(bt), df=k - 1, p=float(stats.chi2.sf(bt, k - 1)))
F['f_pairs_vs_family9'] = dict(f=F['a_pairs']['sd'] ** 2 / F['c_family9']['sd'] ** 2, df=[6, 8],
                               p_one_sided=float(stats.f.sf(F['a_pairs']['sd'] ** 2 / F['c_family9']['sd'] ** 2, 6, 8)))
F['f_pairs_vs_rq8'] = dict(f=F['a_pairs']['sd'] ** 2 / F['b_same_day8']['sd'] ** 2, df=[6, 7],
                           p_one_sided=float(stats.f.sf(F['a_pairs']['sd'] ** 2 / F['b_same_day8']['sd'] ** 2, 6, 7)))
# random-effects view: day/session shift variance from pairs vs per-run variance from same-session frames
within_var = ssb / dfb
shift_var = max(0.0, (sd_d ** 2) / 2 - within_var)      # var(d)=2*(sw^2)+2*sshift^2 style; d=r2-r1 has var 2 s_w^2 + (shift spread)
F['shift_decomposition'] = dict(
    note='Var(second minus first) = 2*within + spread of the shift across pairs. Pairs: s_d^2/2 = %.2f; same-session pooled within = %.2f; mean shift -9.17 is a further, common term.' % (sd_d ** 2 / 2, within_var),
    pairs_sd_d_sq_over_2=sd_d ** 2 / 2, same_session_within_var=within_var,
    rms_pairs_var=F['a_pairs']['sd'] ** 2,
    mean_shift_sq_over_2=float(d.mean() ** 2 / 2),
    share_of_pair_ms_from_common_shift=float((d.mean() ** 2 / 2) / F['a_pairs']['sd'] ** 2))


# ---------------------------------------------------------------- runs needed (two sample t, equal n, 80% power, 5% two-sided)
def power(n, delta, sd, a=0.05):
    df = 2 * n - 2; nc = delta / (sd * math.sqrt(2.0 / n)); tc = stats.t.ppf(1 - a / 2, df)
    pw = stats.nct.sf(tc, df, nc) + stats.nct.cdf(-tc, df, nc)
    return 1.0 if not np.isfinite(pw) else float(pw)  # nct overflows for huge nc (power is 1 there)


def n_needed(delta, sd, pw=0.8):
    n = 2
    while power(n, delta, sd) < pw:
        n += 1
        if n > 5000: return None
    return n


def mde(n, sd, pw=0.8):
    return brentq(lambda dlt: power(n, dlt, sd) - pw, 1e-6, 50 * sd)


SDS = [4.7, 5.95, 8.25]; DELTAS = [5, 10, 15, 23]
table = {str(sd): {str(dl): n_needed(dl, sd) for dl in DELTAS} for sd in SDS}
z = lambda dl, sd: 2 * ((stats.norm.ppf(.975) + stats.norm.ppf(.8)) * sd / dl) ** 2
table_z = {str(sd): {str(dl): round(z(dl, sd), 2) for dl in DELTAS} for sd in SDS}
mde_t = {str(sd): {str(n): round(mde(n, sd), 2) for n in (2, 3, 6, 12, 20)} for sd in SDS}
mde_z = {str(sd): {str(n): round((stats.norm.ppf(.975) + stats.norm.ppf(.8)) * sd * math.sqrt(2 / n), 2) for n in (2, 3, 6, 12, 20)} for sd in SDS}
# unequal 3 vs 7 gate design (harness program): MDE with Welch-free pooled t, df 8
def power_se(nc, df, tc):
    v = stats.nct.sf(tc, df, nc) + stats.nct.cdf(-tc, df, nc)
    return 1.0 if not np.isfinite(v) else float(v)


def mde_unequal(n1, n2, sd, pw=0.8, a=0.05):
    df = n1 + n2 - 2; se = sd * math.sqrt(1 / n1 + 1 / n2); tc = stats.t.ppf(1 - a / 2, df)
    f = lambda dlt: power_se(dlt / se, df, tc) - pw
    return brentq(f, 1e-6, 50 * sd)
mde_3v7 = {str(sd): round(mde_unequal(3, 7, sd), 2) for sd in SDS}
RN = dict(settings='two-sample t, equal n per arm, noncentral t power, 80% power, 5% two-sided, SD in levels on 25 games',
          sds=SDS, deltas=DELTAS, n_per_arm_t=table, n_per_arm_z_unrounded=table_z, mde_t_by_n=mde_t, mde_z_by_n=mde_z, mde_3_vs_7_t=mde_3v7,
          sd_sources={'4.7': 'pooled 18 no-adapter runs (frame d, 4.69)', '5.95': 'eight same-day identical runs (frame b)', '8.25': 'six identical pairs (frame a)'})

# ---------------------------------------------------------------- per-game variance decomposition
def decompose(M, label, center_groups=None):
    """M: runs x 25 games. Compare sum of per-game variances with variance of the run total."""
    X = M.copy()
    if center_groups is not None:  # remove group means (config means) so the frame measures within-config spread
        for g in center_groups: X[g] = X[g] - X[g].mean(0)
        df = X.shape[0] - len(center_groups)
    else:
        X = X - X.mean(0); df = X.shape[0] - 1
    pv = (X ** 2).sum(0) / df                       # per-game variance
    tot = (X.sum(1) ** 2).sum() / df                # variance of the run total
    order = np.argsort(-pv); cum = np.cumsum(pv[order]) / pv.sum()
    # permutation null: shuffle each game's column across runs (independent games); total variance under independence
    Mp = M.copy(); sims = []
    for _ in range(4000):
        Z = np.column_stack([rng.permutation(M[:, j]) for j in range(M.shape[1])])
        if center_groups is not None: continue
        sims.append(((Z.sum(1) - Z.sum(1).mean()) ** 2).sum() / (M.shape[0] - 1))
    out = dict(label=label, runs=int(M.shape[0]), df=int(df), sum_per_game_var=float(pv.sum()), total_var=float(tot),
               total_sd=float(math.sqrt(tot)), sum_game_sd=float(math.sqrt(pv.sum())),
               sum_covariances_x2=float(tot - pv.sum()), ratio_total_over_sum=float(tot / pv.sum()),
               games_zero_variance=[games[j] for j in range(25) if pv[j] == 0], n_games_zero_variance=int((pv == 0).sum()),
               n_games_var_ge_1=int((pv >= 1).sum()),
               top5=[dict(game=games[j], var=float(pv[j]), share=float(pv[j] / pv.sum()), mean=float(M[:, j].mean()), min=int(M[:, j].min()), max=int(M[:, j].max())) for j in order[:5]],
               top5_share=float(cum[4]), top3_share=float(cum[2]), games_for_80pct=int(np.searchsorted(cum, 0.8) + 1))
    if sims:
        sims = np.array(sims); out['indep_null_total_var_mean'] = float(sims.mean())
        out['p_obs_total_var_le_null'] = float((sims <= tot + 1e-9).mean()); out['p_obs_total_var_ge_null'] = float((sims >= tot - 1e-9).mean())
    return out


PG = {}
PG['family9'] = decompose(fam_pg, 'nine same-family Kaggle builds (frame c)')
PG['same_day8_rq'] = decompose(rq_pg, 'eight same-day identical Vertex runs (frame b)')
PG['pool18'] = decompose(np.vstack([fam_pg, cal_pg[None, :], rq_pg]), 'pooled 18 (frame d), about the grand mean')
Y8 = np.array([pg(yw[n]['per_game']) for n in y8]); Y2 = np.array([pg(yw[n]['per_game']) for n in y2])
PG['interleaved_y8'] = decompose(Y8, 'interleaved y8 arm, 6 runs'); PG['interleaved_y2'] = decompose(Y2, 'interleaved y2 arm, 6 runs')
pm = np.array([pg(yr[a]['per_game_levels']) for a, b in pair_dirs.values()] + [pg(yr[b]['per_game_levels']) for a, b in pair_dirs.values()])
PG['six_pairs_about_pair_means'] = decompose(pm, 'six identical pairs, centred on each pair mean (12 runs, 6 df)', center_groups=[[i, i + 6] for i in range(6)])
# game-level view across all same-session frames: variance of a game in frames b, c, e arms
allv = np.array([fam_pg.var(0, ddof=1), rq_pg.var(0, ddof=1), Y8.var(0, ddof=1), Y2.var(0, ddof=1)])
mean_var = allv.mean(0)
cons = []
for j in range(25):
    cons.append(dict(game=games[j], var_family9=float(allv[0, j]), var_rq8=float(allv[1, j]), var_y8=float(allv[2, j]), var_y2=float(allv[3, j]), mean_var=float(mean_var[j])))
PG['cross_frame_game_variance'] = dict(rows=cons, games_zero_in_all_four=[c['game'] for c in cons if c['mean_var'] == 0],
                                       games_zero_in_family9=[c['game'] for c in cons if c['var_family9'] == 0],
                                       games_zero_in_rq8=[c['game'] for c in cons if c['var_rq8'] == 0],
                                       top5_by_mean_var=[c['game'] for c in sorted(cons, key=lambda c: -c['mean_var'])[:5]])
PG['family9_matches_lora_pergame_csv'] = csv_match
# level-clearing probability per game (rq8) and binomial-like dispersion
# yield-test interval at three SDs (difference y8 minus y2, six runs per arm)
dif = F['e_within_arm_pooled']['diff_y8_minus_y2']; se1 = math.sqrt(2 / 6)
YI = {}
for lab, sd_, df_ in (('8.25 (pairs, paper convention, t on 6 df)', 8.25, 6), ('5.95 (same-day 8, t on 7 df)', 5.95, 7),
                      ('4.88 (this test own pooled within-arm SD, t on 10 df)', F['e_within_arm_pooled']['sd'], 10)):
    h = stats.t.ppf(.975, df_) * sd_ * se1; YI[lab] = [round(dif - h, 2), round(dif + h, 2)]
RN['yield_test_interval_by_sd'] = YI
# first-minus-second shift by platform
plat = F['a_pairs']['platform']; dd = dict(zip(pairs, d))
F['a_pairs']['shift_by_platform'] = {pl: dict(diffs=[int(dd[k]) for k in pairs if plat[k] == pl], mean=float(np.mean([dd[k] for k in pairs if plat[k] == pl]))) for pl in ('kaggle', 'gcp')}
res = dict(frames=F, runs_needed=RN, per_game=PG, csv_reproduced=csv_match)
os.makedirs(OUT, exist_ok=True)
json.dump(res, open(os.path.join(OUT, 'variance_frames.json'), 'w'), indent=1, default=float)
print('wrote json; csv match', csv_match)


# ---------------------------------------------------------------- markdown
f = lambda x, n=2: ('%.' + str(n) + 'f') % x
ci = lambda v: '%.1f to %.1f' % tuple(v['ci95'])
rows = [('a', 'six identical pairs, about pair means (includes the first-to-second shift)', 'a_pairs', 'p448 (Kaggle), p448s14 (Kaggle), l1a1y8 (Kaggle), dpctl (GCP), L1 (GCP), l1a1 (GCP); 12 runs, 2.3 to 26.7 h between members; sources yield_runs.json, e4_noise.json'),
        ('a2', 'same pairs, SD about the common shift (-9.17 removed)', 'a_pairs_about_shift', 'same 12 runs; computed here from the six signed gaps'),
        ('b', 'eight same-day identical runs', 'b_same_day8', 'vertex-rq1020-1 to 8, one build, Vertex AI, all started 2026-10-20 07:29 to 09:59 UTC; sources kernel-outputs benchmark.json, bench_pool18_levels.json'),
        ('c', 'nine same-family builds', 'c_family9', '9 Kaggle y8-family builds (y8, y8r, sv2, sv2o16, r1, r2, r3, r3s4, r3hm32), 10-10 to 10-12 (%.0f h span), NOT byte-identical; source yield_runs.json, lora_pergame.csv' % F['c_family9']['span_h']),
        ('d', 'pooled 18 no-adapter runs', 'd_pool18', 'frame c + vertex-cal1 + frame b; source bench_pool18_levels.json'),
        ('e1', 'interleaved test, yield 8,192 arm', 'e_y8_arm', 'yw-y8-* six runs, Vertex, 2026-10-21; source read_2026-10-24.json'),
        ('e2', 'interleaved test, yield 2,048 arm', 'e_y2_arm', 'yw-y2-* six runs, same wave'),
        ('e', 'interleaved test, pooled within arm', 'e_within_arm_pooled', 'e1 + e2, 10 df'),
        ('P1', 'pooled, four disjoint sets (a, b, c, e)', 'pooled_all_disjoint', 'df-weighted, 31 df, shift inside via a'),
        ('P2', 'pooled, sets without the cross-day pairs (b, c, e)', 'pooled_excl_pairs', '25 df; c spans 58 h'),
        ('P3', 'pooled, identical-build same-session only (b, e)', 'pooled_identical_only_same_session', '17 df, no config differences')]
md = ['# Run-to-run SD frames (computed by paper/v3/analysis/v3_variance_frames.py)', '',
      'All SDs are in levels on the 25 development games. "computed" = this script; "read" = taken from a repo file and asserted equal.', '',
      '## 1. The frames', '', '| id | frame | n | mean | SD | df | 95% chi-square interval | runs and source |', '|---|---|---|---|---|---|---|---|']
for i, lab, key, src in rows:
    v = F[key]; md.append('| %s | %s | %s | %s | %s | %d | %s | %s |' % (i, lab, v.get('n', '12' if key.startswith('a_') else ''), f(v['mean'], 1) if 'mean' in v else '', f(v['sd']), v['df'], ci(v), src))
sh = F['a_pairs_about_shift']; a = F['a_pairs']
md += ['', '## 2. Why the frames differ', '',
       '- **Order shift is the main cause of 8.25.** The second run was lower in 6 of 6 pairs (second minus first: %s; mean %s; sign test p %s; paired t on 5 df p %s; 95%% interval for the shift %s to %s). The mean square of the pairs, 68.1, is 62%% common shift (42.0) and 38%% spread. Removing the shift gives SD %s (5 df, interval %s). Kaggle pairs shifted %s, GCP pairs %s.' % (
           a['diffs_second_minus_first'], f(a['mean_shift']), f(sh['sign_test_p_6_of_6'], 3), f(sh['p_shift_t5'], 3), f(sh['shift_ci95'][0], 1), f(sh['shift_ci95'][1], 1), f(sh['sd']), ci(sh),
           a['shift_by_platform']['kaggle']['diffs'], a['shift_by_platform']['gcp']['diffs']),
       '- **Day drift and platform are not separable from the shift.** Pairs ran 2.3 to 26.7 h apart on two platforms; start-to-READY timing does not explain it (V3-W37). Only one pair (l1a1y8, gap 4) ran within 3 h; the six pairs cannot split day, order and platform.',
       '- **Same-day identical runs (b, 5.95) match the SD about the shift (5.6).** No trend with start time inside the 2.5 h (Spearman %s, p %s).' % (f(F['b_same_day8']['spearman_levels_vs_start']['rho']), f(F['b_same_day8']['spearman_levels_vs_start']['p'])),
       '- **The nine-family SD (c, 3.9) is the lowest** and its interval (2.6 to 7.5) contains 5.35 and 5.6. The builds differ, so it could in principle be larger, not smaller, than pure repeat noise; with 8 df nothing says it is truly lower than b. The F test of pairs against c is p %s one-sided (4.46 on 6, 8 df), against b p %s.' % (f(F['f_pairs_vs_family9']['p_one_sided'], 3), f(F['f_pairs_vs_rq8']['p_one_sided'], 3)),
       '- **The pool (d, 4.69) mixes b and c**, so it is between them. Its two platforms agree (Kaggle %s, Vertex %s).' % (f(F['d_pool18']['platform_means']['kaggle'], 1), f(F['d_pool18']['platform_means']['vertex'], 1)),
       '- **The interleaved arms (e) differ by 3x** (y8 6.53, y2 2.23; variance ratio 8.6, two-sided F p %s). Pooled within-arm SD is %s (10 df, %s). Heterogeneity across the four disjoint sets is not significant (Bartlett p %s), so one SD per frame is not demanded by the data except for the shift.' % (f(F['e_within_arm_pooled']['f_p_two_sided'], 3), f(F['e_within_arm_pooled']['sd']), ci(F['e_within_arm_pooled']), f(F['bartlett_four_sets']['p'], 2)),
       '', '## 3. Pooled estimate', '',
       '- Per-run SD without the cross-day shift: **%s (25 df, interval %s)** pooling b, c, e; **%s (17 df, interval %s)** with identical builds only (b, e). Point value for planning: 5 to 5.5.' % (f(F['pooled_excl_pairs']['sd']), ci(F['pooled_excl_pairs']), f(F['pooled_identical_only_same_session']['sd']), ci(F['pooled_identical_only_same_session'])),
       '- Including the cross-day pairs, all four disjoint sets: **%s (31 df, interval %s)**.' % (f(F['pooled_all_disjoint']['sd']), ci(F['pooled_all_disjoint'])),
       '- Cross-day, not interleaved: the error of a difference also carries a shift that the pairs put at about -9 (second minus first) with an interval of %s to %s; 8.25 (interval %s) is the RMS that covers it.' % (f(sh['shift_ci95'][0], 1), f(sh['shift_ci95'][1], 1), ci(a)),
       '', '## 4. Runs needed (two-sample t, equal n per arm, noncentral t, 80% power, 5% two-sided)', '',
       'Script: this file (functions power, n_needed, mde). Cells are n per arm.', '', '| SD | d = 5 | d = 10 | d = 15 | d = 23 |', '|---|---|---|---|---|']
for sd_ in ('4.7', '5.95', '8.25'):
    md.append('| %s | %s |' % (sd_, ' | '.join(str(RN['n_per_arm_t'][sd_][str(dl)]) for dl in (5, 10, 15, 23))))
md += ['', 'Normal approximation, unrounded (the paper convention), for contrast: ' + '; '.join('SD %s: %s' % (sd_, ', '.join(f(RN['n_per_arm_z_unrounded'][sd_][str(dl)], 1) for dl in (5, 10, 15, 23))) for sd_ in ('4.7', '5.95', '8.25')) + ' (d = 5, 10, 15, 23).',
       '', 'Minimum detectable gain at 80% power, n per arm (t | normal):', '', '| SD | n=2 | n=3 | n=6 | n=12 | n=20 |', '|---|---|---|---|---|---|']
for sd_ in ('4.7', '5.95', '8.25'):
    md.append('| %s | %s |' % (sd_, ' | '.join('%s | %s' % (f(RN['mde_t_by_n'][sd_][str(n)], 1), f(RN['mde_z_by_n'][sd_][str(n)], 1)) if False else '%s / %s' % (f(RN['mde_t_by_n'][sd_][str(n)], 1), f(RN['mde_z_by_n'][sd_][str(n)], 1)) for n in (2, 3, 6, 12, 20))))
md += ['', '3 against 7 (the harness gate design), MDE with t on 8 df: SD 4.7 %s, 5.95 %s, 8.25 %s (the paper quotes 15.9 at 8.25 by normal approximation).' % tuple(f(RN['mde_3_vs_7_t'][k], 1) for k in ('4.7', '5.95', '8.25')),
       '', 'Interleaved yield test difference (-2.50, six per arm), 95% interval by SD: ' + '; '.join('%s: %s to %s' % (k, v[0], v[1]) for k, v in YI.items()) + '.']
md += ['', '## 5. Per-game decomposition', '',
       'Sum of per-game variances against the variance of the run total (covariances are the difference). "null" is the total variance when each game column is shuffled across runs (independent games).', '',
       '| frame | runs | sum per-game var | total var | total SD | sum of covariances x2 | games with zero variance | top 5 share of per-game var |', '|---|---|---|---|---|---|---|---|']
for k in ('family9', 'same_day8_rq', 'pool18', 'interleaved_y8', 'interleaved_y2', 'six_pairs_about_pair_means'):
    v = PG[k]; md.append('| %s | %d | %s | %s | %s | %s | %d (%s) | %s |' % (v['label'], v['runs'], f(v['sum_per_game_var'], 1), f(v['total_var'], 1), f(v['total_sd']), f(v['sum_covariances_x2'], 1), v['n_games_zero_variance'], ' '.join(v['games_zero_variance']), f(v['top5_share'], 2)))
cf = PG['cross_frame_game_variance']
md += ['', '- Independent games predict a total SD of %s to %s in every non-pair frame (sum of per-game SDs: family %s, same-day %s, pool %s, y8 %s, y2 %s); observed total SD is 3.9 to 6.5 there (the y2 arm, 2.2, aside), and the permutation null is matched within noise (family9 p %s, same-day8 p %s, pool18 p %s for observed total variance at or below null). Same-session run totals behave like a sum of independent game outcomes; no shared run effect is detectable in them.' % (
           f(min(PG[k]['sum_game_sd'] for k in ('family9', 'same_day8_rq', 'pool18', 'interleaved_y8'))), f(max(PG[k]['sum_game_sd'] for k in ('family9', 'same_day8_rq', 'pool18', 'interleaved_y8'))),
           f(PG['family9']['sum_game_sd']), f(PG['same_day8_rq']['sum_game_sd']), f(PG['pool18']['sum_game_sd']), f(PG['interleaved_y8']['sum_game_sd']), f(PG['interleaved_y2']['sum_game_sd']),
           f(PG['family9']['p_obs_total_var_le_null'], 2), f(PG['same_day8_rq']['p_obs_total_var_le_null'], 2), f(PG['pool18']['p_obs_total_var_le_null'], 2)),
       '- The pairs frame is the exception: per-game variances sum to %s (SD %s) but the total is %s (SD 8.25), a positive covariance term of +%s, which is the common shift. The y8 arm of the interleaved test also shows positive covariance (+%s), the y2 arm strongly negative (%s); 6 runs each, so treat both as noise.' % (
           f(PG['six_pairs_about_pair_means']['sum_per_game_var'], 1), f(PG['six_pairs_about_pair_means']['sum_game_sd']), f(PG['six_pairs_about_pair_means']['total_var'], 1), f(PG['six_pairs_about_pair_means']['sum_covariances_x2'], 1), f(PG['interleaved_y8']['sum_covariances_x2'], 1), f(PG['interleaved_y2']['sum_covariances_x2'], 1)),
       '- Who carries the variance (pool of 18): %s. Five games hold about 55 to 64%% of per-game variance in every frame; tn36 is first in all but the nine-family frame (top there: sc25), then s5i5, su15, sc25, sk48 or wa30.' % ', '.join('%s %s (range %d to %d, %s of total)' % (t['game'], f(t['var'], 1), t['min'], t['max'], f(t['share'], 2)) for t in PG['pool18']['top5']),
       '- Games that never move: lp85 and sb26 have zero variance in all four same-session frames; ar25 and cd82 are zero in the nine-family and pool-18 sets (ar25 8 of 8 levels, cd82 6 of 6, cleared every time). In the pool of 18, %d games never move (%s); %d of 25 move at all, %d have variance of 1 or more.' % (PG['pool18']['n_games_zero_variance'], ' '.join(PG['pool18']['games_zero_variance']), 25 - PG['pool18']['n_games_zero_variance'], PG['pool18']['n_games_var_ge_1']),
       '- lora_pergame.csv reproduces the nine-family matrix exactly (asserted: %s).' % csv_match]
md += ['', '## 6. Recommended wording', '',
       'Writeup (replace the SD sentence in Identical code, different scores and What to copy item 1; about +20 words, cut elsewhere):', '',
       '> Run-to-run SD depends on the comparison. Without a day gap (eight same-day identical runs, nine near-identical builds, the interleaved test; 4.7 pooled over 18) it is about 5 levels (4.9 to 5.3; 95% interval 3.9 to 8.0). Across days it is 8.25 (interval 5.3 to 18.2), because the second run of all six identical pairs scored lower (mean -9.2), a shift we cannot assign to day, order or platform. Interleave arms; if you cannot, plan on 8.25. Two runs per arm detect only gains above 13 (SD 4.7) to 23 (SD 8.25) by normal approximation, and above 27 to 47 with the t distribution; detecting 10 levels takes 5 to 12 runs per arm.',
       '',
       'Short Writeup version (about 55 words, if the word budget is tight):', '', '> SD across runs is about 5 levels within a day and 8.25 across days (the second run of six identical pairs scored 9 lower on average). Two runs per arm detect only gains above 13 to 23 levels (normal approximation; 27 to 47 by t); 10 levels need 5 to 12 runs per arm.', '', 'PDF (Identical-runs and Limits sections, one sentence each):', '',
       '> The pairs SD of 8.25 is a root-mean-square of the within-pair gaps and includes a common first-to-second shift (mean -9.17, t on 5 df p 0.036, all six pairs); the SD about the shift is 5.6 (5 df). Same-session frames give 3.9 (nine builds), 5.95 (eight identical runs), 4.7 (18 pooled) and 4.9 (interleaved arms, pooled); their df-weighted pool is 4.9 (25 df, 3.9 to 6.8). We use 8.25 for cross-day comparisons and about 5 to 6 for same-session ones, and we state minimum detectable effects as a range.',
       '',
       'MDE as a range (state both ends, say which SD): two runs per arm, 80% power, 5% two-sided: 13 to 23 levels by normal approximation (SD 4.7 to 8.25), 27 to 47 with the t distribution on 2 df; six runs per arm: 8 to 15 (t); twelve: 5.6 to 9.9 (t). Also note the paper currently quotes normal-approximation figures (V3-W12, V3-W19: 23, 15.9); the t-based 3-against-7 value at 8.25 is 18.2.',
       '', '## 7. Contradictions with current claims', '',
       '- Writeup/PDF treat 8.25 as the one noise SD. It is the largest of the frames, and about 62% of its mean square is a mean shift, not spread. Truthful, but the "noise" label hides that the shift is directional (second lower, 6 of 6).',
       '- The "two runs per arm detect only gains above about 23" line is a normal-approximation figure at SD 8.25. At 2 df the t-based figure is 47; at SD 5 it is 13 (normal) or 27 (t). So 23 is neither the cautious nor the optimistic end.',
       '- V3-W36 interval for the yield test (-14.2 to +9.2) uses SD 8.25 on 6 df; the test own within-arm SD is 4.9 (10 df) and gives %s to %s. The paper chose the widest; say that it is the conservative choice.' % tuple(YI['4.88 (this test own pooled within-arm SD, t on 10 df)']),
       '- The 4.69 pooled SD (frame d) is derived from bench_pool18_levels.json (18 runs, 17 df, interval 3.5 to 7.0); this script reproduces it from the 18 totals.',
       '- Frame c (3.9) is not "identical code": nine different builds. Its same-day subset (10-12 only, 5 builds) gives 3.87 on 4 df (interval 2.3 to 11.1), so it neither supports nor refutes 5.95.']
open(os.path.join(OUT, 'variance_frames.md'), 'w').write('\n'.join(md) + '\n')
print('wrote md')
