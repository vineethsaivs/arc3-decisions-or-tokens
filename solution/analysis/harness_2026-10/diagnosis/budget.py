"""Budget-bar estimate prototype on human replays. budget(frames) -> (side, color, start, now, n, rate, left) or None."""
import json, glob, sys, collections

def strip_counts(g):
    h, w = len(g), len(g[0])
    sides = {"top": [g[0], g[1]], "bottom": [g[h - 2], g[h - 1]],
             "left": [[row[0] for row in g], [row[1] for row in g]], "right": [[row[w - 2] for row in g], [row[w - 1] for row in g]]}
    return {s: collections.Counter(v for line in lines for v in line) for s, lines in sides.items()}

def budget(frames, min_ratio=0.3):
    counts = [strip_counts(g) for g in frames]
    best = None
    for side in ("top", "bottom", "left", "right"):
        for color in set().union(*(c[side] for c in counts)):
            s = [c[side][color] for c in counts]
            start = max((i for i in range(1, len(s)) if s[i] > s[i - 1]), default=0)
            seg = s[start:]
            n = len(seg) - 1
            drops = sum(b < a for a, b in zip(seg, seg[1:]))
            if n < 3 or drops < 3 or drops < min_ratio * n or seg[0] < 8 or seg[0] - seg[-1] < 3:
                continue
            rate = (seg[0] - seg[-1]) / n
            cand = (drops / n, seg[0] - seg[-1], side, color, seg[0], seg[-1], n, rate, int(seg[-1] / rate))
            if best is None or cand[:2] > best[:2]:
                best = cand
    return best[2:] if best else None

if __name__ == "__main__":
    for f in sorted(glob.glob('frames/h_*.json')):
        d = json.load(open(f)); gid = d['gid'][:4]
        seq = [e for e in d['seq'] if not e.get('auto')]
        fired = steps = 0; errs = []; examples = []
        lv_frames = []; lv = None
        for e in seq:
            if e['lv'] != lv:
                lv_frames = []; lv = e['lv']
            lv_frames.append(e['g'])
            steps += 1
            b = budget(lv_frames)
            if b:
                fired += 1
                if len(examples) < 2 and b[4] > 0: examples.append(f"L{lv} {b[0]} {b[1]} {b[2]}->{b[3]} n{b[4]} r{b[5]:.2f} left{b[6]}")
        print(f"{gid}: fired {fired}/{steps}  {examples}")
