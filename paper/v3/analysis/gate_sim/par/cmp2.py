"""SCALE-style shared-control wave: sensitivity to k, nA and the bar. usage: python3 -I cmp2.py ../gate_k/pool.json run|game N"""
import importlib.util as iu, sys
here = __file__.rsplit("/", 1)[0]
spec = iu.spec_from_file_location("cmp", here + "/cmp.py"); sys.argv = sys.argv[:4]
src = open(here + "/cmp.py").read().split("rules = {")[0]
exec(compile(src, "cmp.py", "exec"))
for k, nA, bar in ((2, 8, 1.5), (2, 8, 1.75), (2, 6, 1.75), (3, 8, 1.75), (3, 8, 2.0), (3, 10, 1.75), (3, 10, 2.0)):
    fn = lambda ds: scale(ds, bar=bar, nA=nA)
    null = ev(fn, (0,) * k); six = ev(fn, (6,) + (0,) * (k - 1)); three = ev(fn, (3,) + (0,) * (k - 1)); neg = ev(fn, (-3,) * k)
    print(f"{m.MODEL} k={k} nA={nA} bar {bar}: null {null[0]:.3f} | one +6 right {six[1]:.3f} | one +3 right {three[1]:.3f} | all -3 {neg[0]:.3f} | jobs {null[3]:.0f}-{six[3]:.0f}")
