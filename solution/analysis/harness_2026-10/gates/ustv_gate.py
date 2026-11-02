"""Read verify against gate-predeclared-2026-11-01-franzen-l1a1y8sv2r3lf15ustv.json (run from repo root).
python3 us8_gate.py [us8|s8]: W = Kaggle kaggle-lf15<arm>-1031-{1,2,3}; A pooled = the kv2 gate's 7 runs."""
import json, re, statistics as st, subprocess, sys
ARM = 'ustv'
from pathlib import Path
K = Path('solution/kaggle/generated/kernel-outputs')
W_DIRS = [d for d in [f'kaggle-lf15{ARM}-1101-{i}' for i in (1, 2, 3)] if (K / d / 'benchmark.json').exists()]
INCUMBENT = float(sys.argv[1]) if len(sys.argv) > 1 else 129.0  # W mean levels of the build it would replace
W3_SKIPPED = 'W3 skipped' in open(__import__('os').path.expanduser('~/.cache/arc3/ustv_chain.log')).read() if __import__('os').path.exists(__import__('os').path.expanduser('~/.cache/arc3/ustv_chain.log')) else False
A_DIRS = ['vertex-kva', 'vertex-s15w-1', 'vertex-s15w-2', 'vertex-s15w-3', 'vertex-s15w-4', 'kaggle-lf15-1029', 'kaggle-lf15-1029-2']


def play(d):
    games = json.loads((K / d / 'benchmark.json').read_text())['game_runs']
    ratios = [a / b for g in games for i, (a, b) in enumerate(zip(g['actions_per_level'], g['base_actions_per_level']))
              if i < g['levels_completed'] and b]
    return dict(levels=sum(g['levels_completed'] for g in games), acts=sum(sum(g['actions_per_level']) for g in games),
                score=st.mean(g['final_score'] for g in games), med=st.median(ratios))


def counters(d):
    out = subprocess.run([sys.executable, 'solution/analysis/unstick_counters.py', str(K / d)], capture_output=True,
                         text=True, check=True).stdout
    nums = dict(re.findall(r"(at-stall turns|acted|stall lines|untested lines|newtype lines|streak>=5|levels) (\S+)", out))
    print('  ', out.strip())
    return nums, out


W = {d: play(d) for d in W_DIRS}
A = {d: play(d) for d in A_DIRS}
at = acted = 0
stall, untested, s5, newt = [], [], [], []
for d in W_DIRS:
    nums, out = counters(d)
    n = int(nums['at-stall turns']); at += n; acted += round(float(nums['acted']) * n)
    stall.append(int(nums['stall lines'])); untested.append(int(nums['untested lines'])); s5.append(int(nums['streak>=5'])); newt.append(int(nums['newtype lines']))
mean = lambda rs, k: st.mean(r[k] for r in rs.values())  # noqa: E731
for name, rs in (('W', W), ('A', A)):
    for k, r in rs.items():
        print(f"{name} {k}: levels {r['levels']} score {r['score']:.2f} actions {r['acts']} med {r['med']:.3f}")
    print(f"{name} mean: levels {mean(rs, 'levels'):.1f} score {mean(rs, 'score'):.2f} actions {mean(rs, 'acts'):.0f} "
          f"med {mean(rs, 'med'):.3f}")
resp = acted / max(1, at)
GROUP = re.compile(r"""\[\s*['"]group['"]\s*\]|\.get\(\s*['"]group['"]""")
uses = [sum(bool(GROUP.search(code)) for f in (K / d).glob('*_p0.txt')
            for code in re.findall(r"<parameter=code>\n(.*?)</parameter>", f.read_text(errors='replace'), re.S))
        for d in W_DIRS]
print(f"group uses per W run {uses}")
def cnt(d, s):
    return sum(f.read_text(errors='replace').count(s) for f in (K / d).glob('*_p0.txt'))
getsrc = [cnt(d, 'could not get source code') for d in W_DIRS]
snip = [cnt(d, 'SnippetTimeout: the snippet used') for d in W_DIRS]
tto = [cnt(d, 'Tool timed out') for d in W_DIRS]
VCALL = re.compile(r"check_model\(|check_goal\(")
vcalls, vgames = [], []
for d in W_DIRS:
    n, games = 0, set()
    for f in (K / d).glob('*_p0.txt'):
        k = sum(bool(VCALL.search(code)) for code in re.findall(r"<parameter=code>\n(.*?)</parameter>", f.read_text(errors='replace'), re.S))
        n += k
        if k:
            games.add(f.name.split('-')[0])
    vcalls.append(n); vgames.append(len(games))
print(f"verify calls per W run {vcalls}; games using it per run {vgames}")
print(f"could-not-get-source per W run {getsrc} (baseline mean 7.8); SnippetTimeout results {snip}; 'Tool timed out' {tto} (baseline mean 6.0)")
print(f"reported: streaks of 5+ idle turns per run {st.mean(s5):.2f} (reference 2.75)")
checks = {
    f"m1 verify calls per run: mean {st.mean(vcalls):.1f} >= 10, min {min(vcalls)} >= 5, games mean {st.mean(vgames):.1f} >= 4":
        st.mean(vcalls) >= 10 and min(vcalls) >= 5 and st.mean(vgames) >= 4,
    f"m1 seg8 group reads per run: mean {st.mean(uses):.1f} >= 15": st.mean(uses) >= 15,
    f"incumbent: W mean levels {mean(W, 'levels'):.1f} >= {INCUMBENT:.1f} - 5": mean(W, 'levels') >= INCUMBENT - 5,
    f"m1 could-not-get-source per run: mean {st.mean(getsrc):.1f} <= 2.0": st.mean(getsrc) <= 2.0,
    f"m1 SnippetTimeout results per run: mean {st.mean(snip):.1f} >= 2": st.mean(snip) >= 2,
    f"m2 at-stall turns that act {resp:.2f} ({acted}/{at}) >= 0.70{' (s8: not gated)' if ARM == 's8' else ''}":
        resp >= 0.70 or ARM == 's8',
    f"m3 actions {mean(W, 'acts'):.0f} <= A x 1.10 ({mean(A, 'acts') * 1.1:.0f})": mean(W, 'acts') <= mean(A, 'acts') * 1.1,
    f"m3 median actions/baseline {mean(W, 'med'):.3f} <= A + 0.05 ({mean(A, 'med') + 0.05:.3f})":
        mean(W, 'med') <= mean(A, 'med') + 0.05,
    f"no harm levels {mean(W, 'levels'):.1f} >= A - 2.5 ({mean(A, 'levels') - 2.5:.1f})":
        mean(W, 'levels') >= mean(A, 'levels') - 2.5,
    f"no harm score {mean(W, 'score'):.2f} >= A - 1.5 ({mean(A, 'score') - 1.5:.2f})":
        mean(W, 'score') >= mean(A, 'score') - 1.5,
}
for k, v in checks.items():
    print(('PASS ' if v else 'FAIL ') + k)
print(f'W runs read: {len(W_DIRS)} of 3')
complete = len(W_DIRS) == 3 or (len(W_DIRS) == 2 and W3_SKIPPED)
print('VERDICT', 'INCOMPLETE' if not complete else 'PASS' if all(checks.values()) else 'KILL')  # pooled over the W runs the wave ran
