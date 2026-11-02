"""Compare the three proposals' gate designs on one noise model (sim2's empirical run or game model).
usage: python3 -I cmp.py ../gate_k/pool.json run|game N"""
import importlib.util as iu, random, statistics as st, sys
here = __file__.rsplit("/", 1)[0]
spec = iu.spec_from_file_location("sim2", here + "/../gate_k/sim2.py"); m = iu.module_from_spec(spec)
sys.argv = sys.argv[:4]; spec.loader.exec_module(m)
N = int(sys.argv[3]); run, delta = m.run, m.delta


def single(d, bar=1.5):  # declared (c'): 6v6 kill D<0 or Dwo<0; +6v6; 12v12 D>=bar and Dwo>=0
    W = [run(d) for _ in range(6)]; A = [run(0) for _ in range(6)]
    if min(delta(W, A)) < 0: return False, 12
    W += [run(d) for _ in range(6)]; A += [run(0) for _ in range(6)]
    D, Dwo = delta(W, A)
    return D >= bar and Dwo >= 0, 24


def gi(ds):  # screen 3W each + 5A (k=3) / 4W + 6A (k=2); out if D<0; best -> fresh single rule
    k = len(ds); nw, na = {2: (4, 6), 3: (3, 5)}[k]
    W = [[run(d) for _ in range(nw)] for d in ds]; A = [run(0) for _ in range(na)]
    res = [delta(W[i], A) for i in range(k)]
    alive = [i for i in range(k) if res[i][0] >= 0]
    jobs = nw * k + na
    if not alive: return None, jobs
    b = max(alive, key=lambda i: res[i][0])
    ok, j = single(ds[b])
    return (b if ok else None), jobs + j


def scale(ds, bar=None, nA=None, w2min=0.0, kill_wo=True):  # one concurrent wave, shared A; best survivor -> fresh 6v6
    k = len(ds); bar = bar if bar is not None else {1: 1.5, 2: 1.75, 3: 2.0}[k]; nA = nA or {1: 6, 2: 8, 3: 10}[k]
    W = [[run(d) for _ in range(6)] for d in ds]; A = [run(0) for _ in range(nA)]
    res = [delta(W[i], A) for i in range(k)]
    alive = [i for i in range(k) if res[i][0] >= 0 and (not kill_wo or res[i][1] >= 0)]
    jobs = 6 * k + nA
    if not alive: return None, jobs
    b = max(alive, key=lambda i: res[i][0])
    W2 = [run(ds[b]) for _ in range(6)]; A2 = [run(0) for _ in range(6)]
    D2 = delta(W2, A2)[0]; D, Dwo = delta(W[b] + W2, A + A2)
    return (b if D >= bar and Dwo >= 0 and D2 >= w2min else None), jobs + 12


def signal(ds):  # ds[0]=WA on its own declared rule; others: own 6v6 wave 1 each, best passer -> wave 2, bar 2.0
    okA, jobs = single(ds[0])
    var = ds[1:]; res = []
    for d in var:
        W = [run(d) for _ in range(6)]; A = [run(0) for _ in range(6)]; jobs += 12
        res.append((delta(W, A), W, A))
    alive = [i for i in range(len(var)) if min(res[i][0]) >= 0]
    prom = [0] if okA else []
    if alive:
        b = max(alive, key=lambda i: res[i][0][0])
        W = res[b][1] + [run(var[b]) for _ in range(6)]; A = res[b][2] + [run(0) for _ in range(6)]; jobs += 12
        D, Dwo = delta(W, A)
        if D >= 2.0 and Dwo >= 0: prom.append(b + 1)
    if not prom: return None, jobs
    return max(prom, key=lambda i: ds[i] + random.random() * 1e-9), jobs  # slot: largest pooled D; approx by truth


def ev(fn, ds):
    ship = harm = right = 0; jobs = []; top = max(ds)
    for _ in range(N):
        k, j = fn(list(ds)); jobs.append(j)
        if k is None: continue
        ship += 1; harm += ds[k] < 0; right += ds[k] == top
    return ship / N, right / N, harm / N, st.fmean(jobs)


rules = {"GI screen+fresh": gi, "SCALE M(k)": scale, "SCALE no-wo-kill": lambda ds: scale(ds, kill_wo=False),
         "SCALE bar1.5 k3": lambda ds: scale(ds, bar=1.5), "SIGNAL": signal}
print("model", m.MODEL, "N", N)
print("single (0,)", single.__name__, [round(x, 3) for x in ev(lambda ds: (0 if single(ds[0])[0] else None, 0), (0,))[:3]],
      "(6,)", [round(x, 3) for x in ev(lambda ds: (0 if single(ds[0])[0] else None, 0), (6,))[:3]],
      "(3,)", [round(x, 3) for x in ev(lambda ds: (0 if single(ds[0])[0] else None, 0), (3,))[:3]])
for ds in ((0, 0, 0), (6, 0, 0), (0, 6, 0), (3, 0, 0), (3, 3, 3), (6, 3, 3), (-3, -3, -3), (0, -3, -3), (-3, 0, 0)):
    for name, fn in rules.items():
        p, r, h, j = ev(fn, ds)
        print(f"{str(ds):14s} {name:18s} P_ship {p:.3f} P_best {r:.3f} P_harm {h:.3f} jobs {j:.1f}")
pr = random.Random(9)
draws = [(pr.gauss(0.5, 2.5), [pr.gauss(0, 1.5) for _ in range(3)]) for _ in range(N)]
for name, fn in rules.items():
    g = h = s = 0; jobs = []
    for c, e in draws:
        ds = [c + x for x in e]
        k, j = fn(ds); jobs.append(j)
        if k is not None: s += 1; g += ds[k]; h += ds[k] < 0
    print(f"prior {name:18s} P_ship {s/N:.3f} E_gain {g/N:+.3f} P_harmful {h/N:.3f} jobs {st.fmean(jobs):.1f}")
