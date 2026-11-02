"""Read unstick2 against gate-predeclared-2026-10-30-franzen-l1a1y8sv2r3lf15uns2.json (run from repo root).
W = Kaggle kaggle-lf15uns2-1030-{1,2,3}; A pooled = the kv2 gate's 7 runs (play numbers only)."""
import json, re, statistics as st, subprocess, sys
from pathlib import Path
K = Path('solution/kaggle/generated/kernel-outputs')
W_DIRS = [d for d in ['kaggle-lf15uns2-1030-1', 'kaggle-lf15uns2-1030-2', 'kaggle-lf15uns2-1030-3'] if (K / d / 'benchmark.json').exists()]
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
bad = sum(len(re.findall(r'Valid actions not yet used on this level:[^\n]*ACTION\d', f.read_text(errors='replace')))
          for d in W_DIRS for f in (K / d).glob('*_p0.txt'))
print(f"reported: streaks of 5+ idle turns per run {st.mean(s5):.2f} (reference 2.75)")
checks = {
    f"m1 stall-check openers per run {st.mean(stall):.1f} >= 5, untested-list openers {st.mean(untested):.1f} >= 5, "
    f"new-types openers {st.mean(newt):.1f} >= 20": st.mean(stall) >= 5 and st.mean(untested) >= 5 and st.mean(newt) >= 20,
    f"m1 unused-action lines naming an engine action {bad} == 0": bad == 0,
    f"m2 at-stall turns that act {resp:.2f} ({acted}/{at}) >= 0.70": resp >= 0.70,
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
print('VERDICT', 'PASS' if all(checks.values()) and len(W_DIRS) == 3 else ('KILL' if not all(checks.values()) else 'INCOMPLETE'))
