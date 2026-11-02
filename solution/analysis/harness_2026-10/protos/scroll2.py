import json, glob, os, sys, time, collections

def detect_scroll(grid_a, grid_b, band=4, max_shift=32):
    """[dr, dc] the interior's content moved by between two frames, or None (one axis at a time). Candidate shifts come
    from whole-line matches (lines of one color are skipped: they match any shift); a candidate is a scroll when its
    cell mismatch rate is at most 0.1 and at most 0.3 of the unshifted rate."""
    h = len(grid_a); w = len(grid_a[0]) if h else 0
    if h != len(grid_b) or h < 2 * band + 8 or w < 2 * band + 8:
        return None
    rows_a = [tuple(r[band:w - band]) for r in grid_a[band:h - band]]
    rows_b = [tuple(r[band:w - band]) for r in grid_b[band:h - band]]
    cells = len(rows_b) * len(rows_b[0])
    mm0 = sum(x != y for ra, rb in zip(rows_a, rows_b) for x, y in zip(ra, rb))
    if mm0 < 60:
        return None
    best = None
    for axis, (la, lb) in enumerate(((rows_a, rows_b), (list(zip(*rows_a)), list(zip(*rows_b))))):
        n = len(lb)
        info = [len(set(x)) > 1 for x in lb]
        scored = []
        for s in range(-max_shift, max_shift + 1):
            if s == 0:
                continue
            m = t = 0
            for i in range(max(0, s), min(n, n + s)):
                if info[i]:
                    t += 1; m += la[i - s] == lb[i]
            if t >= 8 and m:
                scored.append((m / t, s))
        for _, s in sorted(scored, reverse=True)[:3]:
            lo, hi = max(0, s), min(n, n + s)
            mm = sum(x != y for i in range(lo, hi) for x, y in zip(la[i - s], lb[i]))
            rate = mm / ((hi - lo) * len(lb[0]))
            if rate <= 0.1 and rate <= 0.3 * mm0 / cells and (best is None or rate < best[0]):
                best = (rate, [s, 0] if axis == 0 else [0, s])
    return best[1] if best else None

if __name__ == "__main__":
    tot = collections.Counter(); hits = collections.Counter(); ex = {}; worst = 0; big_none = collections.Counter()
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
                hits[g] += 1; ex.setdefault(g, []).append(tuple(s))
            elif sum(x != y for r1, r2 in zip(a, b) for x, y in zip(r1, r2)) >= 300:
                big_none[g] += 1
    for g in sorted(tot):
        print(f"{g:5s} changed pairs {tot[g]:4d} scroll {hits[g]:4d} big-unexplained {big_none[g]:3d}", collections.Counter(ex.get(g, [])).most_common(5))
    print(f"worst call {worst*1000:.1f} ms")
