"""Tune screen + fresh-confirm rules for K adapters on the 17 bench games. stdlib only.
usage: python3 -I sim2.py pool.json MODEL N    MODEL = run (resample whole runs: bench sd 2.6, as measured on 16 runs)
                                                     | game (independent games: bench sd 3.5, stress case)
Effect d on the 25-game scale, applied to the 9 non-saturated bench games (eval.md oc2 model), bench effect ~0.53 d."""
import json, random, statistics as st, sys

pool = json.load(open(sys.argv[1])); MODEL = sys.argv[2]; N = int(sys.argv[3])
runs, caps = list(pool["runs"].values()), pool["caps"]
BENCH = ["ar25", "bp35", "cd82", "cn04", "dc22", "ft09", "g50t", "ka59", "lp85", "m0r0", "r11l", "s5i5", "sb26", "sc25",
         "tr87", "tu93", "vc33"]
SAT = {"lp85", "r11l", "ar25", "sb26", "vc33", "tr87", "ft09", "cd82"}
NSI = [i for i, g in enumerate(BENCH) if g not in SAT]
CAP = [caps[g] for g in BENCH]
VEC = [[r[g] for g in BENCH] for r in runs]
EMP = [[r[g] for r in runs] for g in BENCH]
rng = random.Random(5)


def run(d):
    x = list(rng.choice(VEC)) if MODEL == "run" else [rng.choice(EMP[i]) for i in range(len(BENCH))]
    for i in NSI:
        if d > 0 and x[i] < CAP[i] and rng.random() < d / 17 * 1.12: x[i] += 1
        if d < 0 and x[i] > 0 and rng.random() < -d / 17: x[i] -= 1
    return x


def delta(W, A):
    n, m = len(W), len(A)
    dg = [sum(r[i] for r in W) / n - sum(r[i] for r in A) / m for i in range(len(BENCH))]
    D = sum(dg)
    return D, D - max(dg)


def d0s(ds, bar=1.5):
    """naive: every adapter gets the DECISION rule against shared concurrent A; ship the largest pooled D."""
    K = len(ds); jobs = 6 * K + 6
    w1 = [[run(d) for _ in range(6)] for d in ds]; a1 = [run(0) for _ in range(6)]
    alive = [k for k in range(K) if min(delta(w1[k], a1)) >= 0]
    prom = []
    if alive:
        a2 = a1 + [run(0) for _ in range(6)]; jobs += 6 * len(alive) + 6
        for k in alive:
            D, Dwo = delta(w1[k] + [run(ds[k]) for _ in range(6)], a2)
            if D >= bar and Dwo >= 0: prom.append((D, k))
    if not prom: return None, None, jobs, None
    D, k = max(prom)
    return k, D, jobs, None


def screen(ds, nw, na, floor):
    K = len(ds)
    W = [[run(d) for _ in range(nw)] for d in ds]; A = [run(0) for _ in range(na)]
    res = [delta(W[k], A) for k in range(K)]
    ok = lambda k: (floor == "none" or res[k][0] >= 0) and (floor != "D+wo" or res[k][1] >= 0)  # noqa: E731
    alive = [k for k in range(K) if ok(k)]
    if not alive: return None, W, A, nw * K + na
    return max(alive, key=lambda k: res[k][0]), W, A, nw * K + na


def confirm(d, how, bar):
    """fresh runs only. how: 'gs7' = 7v7 (kill D<0 or Dwo<0) then 7v7, decide on 14v14; 'one12' / 'one14' one shot."""
    if how == "gs7":
        W = [run(d) for _ in range(7)]; A = [run(0) for _ in range(7)]
        if min(delta(W, A)) < 0: return False, None, 14, W, A
        W += [run(d) for _ in range(7)]; A += [run(0) for _ in range(7)]
        D, Dwo = delta(W, A)
        return D >= bar and Dwo >= 0, D, 28, W, A
    n = int(how[3:])
    W = [run(d) for _ in range(n)]; A = [run(0) for _ in range(n)]
    D, Dwo = delta(W, A)
    return D >= bar and Dwo >= 0, D, 2 * n, W, A


def ssc(ds, nw, na, floor, how, bar, pooled_bar=None):
    k, Ws, As, jobs = screen(ds, nw, na, floor)
    if k is None: return None, None, jobs, None
    ok, D, j, W, A = confirm(ds[k], how, bar)
    jobs += j
    if pooled_bar is not None and D is not None:
        Dp, Dpwo = delta(Ws[k] + W, As + A)
        ok = Dp >= pooled_bar and Dpwo >= 0
    return (k, D, jobs, k) if ok else (None, None, jobs, k)


def ev(fn, ds, n=N):
    ship = harm = best = sel_best = 0; jobs, tru = [], []
    top = max(ds); unique = ds.count(top) == 1
    for _ in range(n):
        k, D, j, sel = fn(ds)
        jobs.append(j)
        sel_best += sel is not None and ds[sel] == top
        if k is None: continue
        ship += 1; harm += ds[k] < 0; best += ds[k] == top; tru.append(ds[k])
    return ship / n, harm / n, (sel_best / n if unique else float("nan")), st.fmean(jobs), (st.fmean(tru) if tru else float("nan"))


if __name__ == "__main__":
    print("model", MODEL, "bench run sd", round(st.stdev(sum(run(0)) for _ in range(20000)), 2))
    SC = {3: [(0, 0, 0), (6, 0, 0), (3, 3, 3), (3, 0, -3), (-3, -3, -3)],
          4: [(0, 0, 0, 0), (6, 0, 0, 0), (3, 3, 3, 3), (-3, -3, -3, -3)],
          2: [(0, 0), (6, 0), (3, 3), (-3, -3)],
          1: [(0,), (3,), (6,), (-3,)]}
    alloc = {1: (6, 6), 2: (4, 6), 3: (3, 5), 4: (2, 6)}
    rules = {"D0s naive best-of-K": lambda ds: d0s(ds)}
    for floor in ("D", "none"):
        for how, bar in (("gs7", 1.5), ("gs7", 1.0), ("one12", 1.5), ("one12", 1.75), ("one14", 1.5)):
            rules[f"SSC floor={floor} {how} bar {bar}"] = (lambda f, h, b: lambda ds: ssc(ds, *alloc[len(ds)], f, h, b))(floor, how, bar)
    rules["SSC floor=D gs7 pooled bar 2.0"] = lambda ds: ssc(ds, *alloc[len(ds)], "D", "gs7", 1.5, pooled_bar=2.0)
    only = sys.argv[4:] or None
    for K in (1, 2, 3, 4):
        for ds in SC[K]:
            for name, fn in rules.items():
                if only and not any(o in name for o in only): continue
                if K == 1 and not name.startswith("D0s"): continue
                p, h, sb, j, t = ev(fn, list(ds))
                print(f"K{K} {str(ds):16s} {name:34s} P_ship {p:.3f} P_harm {h:.3f} P_sel_best {sb:.2f} jobs {j:5.1f} E_true|ship {t:+.2f}")
