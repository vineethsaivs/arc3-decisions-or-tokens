#!/usr/bin/env python3
"""F14 `fig_scroll`: a camera scroll reaches the model as noise (real replay frames, real harness modules).

Run from the repo root:
  uv run --no-project --with matplotlib --with numpy python -I -B paper/v3/analysis/figs/fig_scroll.py
  ... fig_scroll.py --rebuild FRAMES_DIR   (re-reads all h_*.json to recompute data/scroll_corpus.json)

Rows A and B: one frame pair each, bp35 level 1 (replay frames 4 and 5) and lf52 level 3 (frames 58 and 59), from
data/frames/h_*.json. Numbers come from the REAL modules (realmods.py: Franzen's frame_diff.py before and after
overlay scroll; nodes from the unpatched segmentation.py, as test_franzen_scroll.py does): changed_cell_count,
len(moved) and `scroll`. The corrected diff image is drawn with the module's own _shift_view and its cell count is
asserted equal to the module's changed_cell_count. Row C: over every consecutive same-level pair in the 22 human replays
whose frame changed, how often _detect_scroll fires; asserted against test_franzen_scroll.py's report (bp35 103,
lf52 40, 6,170 silent pairs elsewhere in that test's count).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from figstyle import C, HERE, W, apply, check_no_overlap, panel, save  # noqa: E402
from realmods import build, grid, rgb  # noqa: E402

DATA = HERE / "data"
M = build()


def corpus(frames_dir):
    games = {}
    for f in sorted(Path(frames_dir).glob("h_*.json")):
        seq = json.load(open(f))["seq"]
        gs = [grid(e["g"]) for e in seq]
        n = fired = 0
        for ea, eb, ga, gb in zip(seq, seq[1:], gs, gs[1:]):
            if ea["lv"] != eb["lv"] or ga == gb:
                continue
            n += 1
            fired += bool(M.fd1._detect_scroll(ga, gb))
        games[f.stem[2:]] = [n, fired]
    return {"rule": "consecutive pairs, same level, frame changed; fired = _detect_scroll(a, b) is not None", "games": games}


if len(sys.argv) > 2 and sys.argv[1] == "--rebuild":
    (DATA / "scroll_corpus.json").write_text(json.dumps(corpus(sys.argv[2]), indent=1))
CORP = json.loads((DATA / "scroll_corpus.json").read_text())["games"]
fired_by = {k: v[1] for k, v in CORP.items() if v[1]}
silent = sum(n - f for n, f in CORP.values())
assert len(CORP) == 22 and fired_by == {"bp35": 103, "lf52": 40}, fired_by
assert CORP["bp35"][0] == 363 and CORP["lf52"][0] == 772, (CORP["bp35"], CORP["lf52"])
assert silent == 6170, silent
other_pairs = sum(n for k, (n, f) in CORP.items() if k not in fired_by)


def pair(game, i):
    seq = json.load(open(DATA / f"frames/h_{game}.json"))["seq"]
    assert seq[i]["lv"] == seq[i + 1]["lv"]
    a, b = grid(seq[i]["g"]), grid(seq[i + 1]["g"])
    na, nb = M.seg0.segment_layer(a, M.COLORS)["nodes"], M.seg0.segment_layer(b, M.COLORS)["nodes"]
    d0 = M.fd0.compute_frame_diff(a, b, na, nb)
    d1 = M.fd1.compute_frame_diff(a, b, na, nb)
    assert "scroll" not in d0 and d1["scroll"] == M.fd1._detect_scroll(a, b)
    raw = np.array(a) != np.array(b)
    shifted, _ = M.fd1._shift_view(a, na, d1["scroll"])
    shifted = np.array(shifted)
    into = shifted == -1
    fixed = shifted != np.array(b)           # includes the cells scrolled into view
    assert raw.sum() == d0["changed_cell_count"] and fixed.sum() == d1["changed_cell_count"]
    real = fixed & ~into
    return dict(game=game, lv=seq[i]["lv"], i=i, a=a, b=b, raw=raw, real=real, into=into, d0=d0, d1=d1)


P = [pair("bp35", 4), pair("lf52", 58)]
p0, p1 = P
assert (p0["d0"]["changed_cell_count"], len(p0["d0"]["moved"]), p0["lv"]) == (1343, 89, 1)
assert (p1["d0"]["changed_cell_count"], p1["lv"]) == (1633, 3)
for p in P:
    print(p["game"], f"level {p['lv']} frames {p['i']}-{p['i'] + 1}: unpatched {p['d0']['changed_cell_count']} cells, "
          f"{len(p['d0']['moved'])} moved; patched scroll {p['d1']['scroll']}, {p['d1']['changed_cell_count']} cells "
          f"({int(p['into'].sum())} scrolled into view), {len(p['d1']['moved'])} moved")
print("corpus:", fired_by, "silent pairs", silent, "pairs in the 20 other games", other_pairs)

# ---- figure ----------------------------------------------------------------------------------------------------------
apply()
S, GAP, X0 = 1.24, 0.10, 0.12
HEAD = [0.17, 0.03]                         # column headers above row A only
ROWH = [0.24 + h + S + 0.62 for h in HEAD]
CTOP = 0.06 + sum(ROWH) + 0.14             # top of panel C (inches)
CH = 1.2
H = CTOP + 0.26 + CH + 0.62 + 0.14
fig = plt.figure(figsize=(W, H))
texts = []


def fmt(n):
    return f"{n:,}"


def draw(ax, img):
    ax.imshow(img, interpolation="nearest", aspect="equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.5)
        s.set_color(C["gray"])


heads = ["before", "after", "changed cells, unpatched diff", "changed cells after undoing the scroll"]
for r, (p, letter) in enumerate(zip(P, "AB")):
    top = 0.06 + sum(ROWH[:r])
    panel(fig, 0.01, 1 - top / H, letter, f"{p['game']}, level {p['lv']}: replay frames {p['i']} and {p['i'] + 1}")
    ytop = top + 0.24 + HEAD[r]
    imgs = [rgb(p["a"]), rgb(p["b"]), None, None]
    for k in range(4):
        x = X0 + k * (S + GAP)
        ax = fig.add_axes([x / W, 1 - (ytop + S) / H, S / W, S / H])
        if k < 2:
            draw(ax, imgs[k])
        else:
            canvas = np.ones((64, 64, 3))
            if k == 2:
                canvas[p["raw"]] = matplotlib_rgb = plt.matplotlib.colors.to_rgb(C["accent"])
            else:
                canvas[p["into"]] = plt.matplotlib.colors.to_rgb(C["light"])
                canvas[p["real"]] = plt.matplotlib.colors.to_rgb(C["accent"])
            draw(ax, canvas)
        if r == 0:
            texts.append(fig.text((x + S / 2) / W, 1 - (ytop - 0.03) / H, ["before", "after", "unpatched diff", "after undoing the scroll"][k],
                                  fontsize=7, color=C["gray_text"], ha="center", va="bottom"))
        note = {0: f"frame {p['i']}", 1: f"frame {p['i'] + 1}",
                2: f"{fmt(p['d0']['changed_cell_count'])} cells changed\n{len(p['d0']['moved'])} objects 'moved'\nno scroll reported",
                3: f"scroll {p['d1']['scroll']}\n{fmt(p['d1']['changed_cell_count'])} cells changed,\n"
                   f"{fmt(int(p['into'].sum()))} scrolled into view\n{len(p['d1']['moved'])} objects 'moved'"}[k]
        texts.append(fig.text((x + 0.02) / W, 1 - (ytop + S + 0.05) / H, note, fontsize=6.8, ha="left", va="top", linespacing=1.2,
                              color=C["accent"] if k == 3 else C["ink"], fontweight="bold" if k == 3 else "normal"))

texts.append(fig.text(X0 / W, 1 - (CTOP - 0.1) / H, "Diff images: orange = changed cell, gray = cell scrolled into view "
                      "(the patched module counts it as changed).", fontsize=6.5, color=C["gray_text"], ha="left", va="bottom"))

# ---- C: how often the detector fires ------------------------------------------------------------------------------
panel(fig, 0.01, 1 - CTOP / H, "C", "Detector firing on every changed same-level frame pair, 22 human replays")
order = ["bp35", "lf52"] + sorted(k for k in CORP if k not in fired_by)
frac = [100 * CORP[k][1] / CORP[k][0] for k in order]
AX0, AXW = 0.55, 4.8
axc = fig.add_axes([AX0 / W, 1 - (CTOP + 0.26 + CH) / H, AXW / W, CH / H])
xs = np.arange(len(order))
axc.bar(xs[:2], frac[:2], color=C["accent"], width=0.7)
axc.plot(xs[2:], np.zeros(len(order) - 2), "s", color=C["gray"], ms=2.5)
axc.set_ylim(-3, 36)
axc.set_xlim(-0.7, len(order) - 0.3)
axc.set_yticks([0, 10, 20, 30])
axc.set_ylabel("pairs where it fires (%)")
axc.set_xticks(xs)
axc.set_xticklabels(order, rotation=90, fontsize=6.3)
axc.grid(axis="y")
texts.append(axc.text(1.0, frac[0] - 1, f"bp35: {CORP['bp35'][1]} of {CORP['bp35'][0]} pairs ({frac[0]:.0f}%)", fontsize=7,
                      color=C["accent"], ha="left", va="center", fontweight="bold"))
texts.append(axc.text(1.6, frac[1] + 1.5, f"lf52: {CORP['lf52'][1]} of {CORP['lf52'][0]} pairs ({frac[1]:.0f}%)", fontsize=7,
                      color=C["accent"], ha="left", va="center", fontweight="bold"))
texts.append(axc.text(2.2, 14.5, f"the other 20 games: 0 of {fmt(other_pairs)} pairs (squares at 0)", fontsize=7, color=C["gray_text"],
                      ha="left", va="center"))
texts.append(fig.text(X0 / W, 0.1 / H, "Pairs are consecutive replay frames on the same level with a changed frame (6,313 in all; "
                      "the detector stays silent on 6,170).\nTwo frame pairs and one replay per game: this shows what the detector does on "
                      "human play, not that the model plays better.", fontsize=6.5, color=C["gray_text"], ha="left", va="bottom",
                      linespacing=1.2))
assert sum(n for n, f in CORP.values()) == 6313

check_no_overlap(fig, texts)
save(fig, "fig_scroll", writeup=True)
