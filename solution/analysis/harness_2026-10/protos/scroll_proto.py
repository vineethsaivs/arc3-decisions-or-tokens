import json, glob, os, sys, time, collections

def detect_scroll(grid_a, grid_b, band=4, max_shift=32):
    """(dr, dc) the interior's content moved by between frames, or None. One axis at a time; lines with one color
    are skipped (they match any shift)."""
    h = len(grid_a); w = len(grid_a[0]) if h else 0
    if h != len(grid_b) or h < 2 * band + 8 or w < 2 * band + 8:
        return None
    rows_a = [tuple(r[band:w - band]) for r in grid_a[band:h - band]]
    rows_b = [tuple(r[band:w - band]) for r in grid_b[band:h - band]]
    cols_a = list(zip(*rows_a)); cols_b = list(zip(*rows_b))
    best = None
    for axis, (la, lb) in enumerate(((rows_a, rows_b), (cols_a, cols_b))):
        n = len(lb)
        info = [len(set(x)) > 1 for x in lb]
        def frac(s):
            m = t = 0
            for i in range(max(0, s), min(n, n + s)):
                if info[i]:
                    t += 1; m += la[i - s] == lb[i]
            return (m / t if t else 0.0), t
        f0, _ = frac(0)
        if f0 > 0.5:
            continue
        for s in range(-max_shift, max_shift + 1):
            if s == 0:
                continue
            f, t = frac(s)
            if t >= 8 and f >= 0.7 and (best is None or (f, -abs(s)) > (best[0], -abs(best[1][axis]))):
                best = (f, (s, 0) if axis == 0 else (0, s))
    return list(best[1]) if best else None

if __name__ == "__main__":
    tot = collections.Counter(); hits = collections.Counter(); ex = {}; worst = 0
    for f in sorted(glob.glob(sys.argv[1] + "/h_*.json")):
        g = os.path.basename(f)[2:-5]
        seq = json.load(open(f))["seq"]
        grids = [[[int(ch, 16) for ch in row] for row in e["g"]] for e in seq]
        for (ea, a), (eb, b) in zip(zip(seq, grids), zip(seq[1:], grids[1:])):
            if ea["lv"] != eb["lv"] or a == b:
                continue
            t0 = time.perf_counter(); s = detect_scroll(a, b); worst = max(worst, time.perf_counter() - t0)
            tot[g] += 1
            if s:
                hits[g] += 1; ex.setdefault(g, []).append((eb["lv"], tuple(s)))
    for g in sorted(tot):
        print(f"{g:5s} changed pairs {tot[g]:4d} scroll {hits[g]:4d}", collections.Counter(ex.get(g, [])).most_common(4))
    print(f"worst call {worst*1000:.1f} ms")
