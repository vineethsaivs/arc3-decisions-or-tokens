#!/usr/bin/env python3
"""F13 `fig_glyph`: what 4-connected objects do to a drawn glyph (real frames, real harness modules).

Run from the repo root:
  uv run --no-project --with matplotlib --with numpy python -I -B paper/v3/analysis/figs/fig_glyph.py
  ... fig_glyph.py --rebuild FRAMES_DIR   (re-reads all h_*.json to recompute data/glyph_corpus.json)

Panels A to C: the start of bp35 level 3 (first replay frame with lv == 3, human replay h_bp35.json, seq index 55), a
7 x 18 cell crop around the three X marks. The nodes and groups drawn come from the REAL patched segmentation.py
(solution/kaggle/franzen_overlays/seg8.py applied to Franzen's harness tree, loaded by realmods.py); the script asserts
that the unpatched and patched modules give the same nodes (ids, hashes, pixels, boundaries) and that the counts it prints equal the module output.
Panel D: recount over every level start of the 22 human replays. Rule (mine, stated on the figure): a `group` of 2 or
more nodes whose bounding box is at least 3 x 3 cells (node bbox ends are inclusive). That gives 76 starts in 13 games;
dropping sparse dot lattices (fill under 0.1) gives 73. The seg8 docstring's 71 in 13 games used an unstated dot-texture
rule and is not reproduced (see paper/v3/figs/README.md).
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from figstyle import C, HERE, W, apply, check_no_overlap, panel, save  # noqa: E402
from realmods import build, grid, rgb  # noqa: E402

DATA = HERE / "data"
M = build()


def nodes_groups(g):
    s0, s1 = M.seg0.segment_layer(g, M.COLORS), M.seg1.segment_layer(g, M.COLORS)
    assert s0["adjacency_list"] == s1["adjacency_list"] and len(s0["nodes"]) == len(s1["nodes"])
    for a, b in zip(s0["nodes"], s1["nodes"]):
        assert (a["id"], a["hash"], a["pixels"], a["boundary"]) == (b["id"], b["hash"], b["pixels"], b["boundary"])
    return s1["nodes"]


def multi_node_glyphs(nodes):
    """(nodes, bbox, fill) of groups of 2+ nodes whose bounding box (node bbox ends are inclusive) is at least 3 x 3 cells."""
    by = defaultdict(list)
    for n in nodes:
        by[n["group"]].append(n)
    out = []
    for ns in by.values():
        r0, c0 = min(n["bbox"][0] for n in ns), min(n["bbox"][1] for n in ns)
        r1, c1 = max(n["bbox"][2] for n in ns), max(n["bbox"][3] for n in ns)
        if len(ns) >= 2 and r1 - r0 + 1 >= 3 and c1 - c0 + 1 >= 3:
            out.append((ns, (r0, c0, r1, c1), sum(n["pixels"] for n in ns) / ((r1 - r0 + 1) * (c1 - c0 + 1))))
    return out


def corpus(frames_dir):
    starts, hit, hit_f, games = 0, 0, 0, {}
    for f in sorted(Path(frames_dir).glob("h_*.json")):
        last, game = None, f.stem[2:]
        games.setdefault(game, [0, 0])
        for e in json.load(open(f))["seq"]:
            if e["lv"] != last:
                last = e["lv"]
                starts += 1
                games[game][0] += 1
                gl = multi_node_glyphs(nodes_groups(grid(e["g"])))
                hit += bool(gl)
                hit_f += any(fill >= 0.1 for _, _, fill in gl)
                games[game][1] += bool(gl)
    return {"rule": "group of >=2 nodes, group bbox >=3x3 cells; hit_filtered also needs fill (cells / bbox area) >= 0.1",
            "starts": starts, "hit": hit, "hit_filtered": hit_f,
            "games": len(games), "hit_games": sum(1 for v in games.values() if v[1]), "by_game": games}


if len(sys.argv) > 2 and sys.argv[1] == "--rebuild":
    (DATA / "glyph_corpus.json").write_text(json.dumps(corpus(sys.argv[2]), indent=1))
CORP = json.loads((DATA / "glyph_corpus.json").read_text())
assert (CORP["starts"], CORP["hit"], CORP["games"], CORP["hit_games"]) == (184, 76, 22, 13), CORP

# ---- the level start ----------------------------------------------------------------------------------------------
seq = json.load(open(DATA / "frames/h_bp35.json"))["seq"]
IDX = next(i for i, e in enumerate(seq) if e["lv"] == 3)
g = grid(seq[IDX]["g"])
nodes = nodes_groups(g)
R0, R1, C0, C1 = 6, 13, 18, 36            # crop, end exclusive: 7 rows x 18 cols
X_COLOR = "O"                              # colour index 12 in COLORS
xn = [n for n in nodes if n["color"] == X_COLOR and n["bbox"][0] >= R0 and n["bbox"][2] < R1
      and n["bbox"][1] >= C0 and n["bbox"][3] < C1]
xcells = sum(1 for r in range(R0, R1) for c in range(C0, C1) if g[r][c] == 12)
n_nodes, n_groups = len(xn), len({n["group"] for n in xn})
assert IDX == 55 and xcells == 15 and all(n["pixels"] == 1 for n in xn)
assert (n_nodes, n_groups) == (15, 3), (n_nodes, n_groups)
glyphs = [(ns, bb) for ns, bb, _ in multi_node_glyphs(nodes) if bb[0] >= R0 and bb[2] < R1 and bb[1] >= C0 and bb[3] < C1]
assert len(glyphs) == 3 and all(bb[2] - bb[0] == 2 and bb[3] - bb[1] == 2 and len(ns) == 5 for ns, bb in glyphs)
print(f"bp35 L3 start (seq {IDX}): X marks = {n_nodes} nodes (4-connected), {n_groups} groups (group field); "
      f"corpus {CORP['hit']} ({CORP['hit_filtered']} filtered) of {CORP['starts']} starts in {CORP['hit_games']} of {CORP['games']} games")

# ---- figure ----------------------------------------------------------------------------------------------------------
apply()
CELL = 0.18                                # inches per board cell
AW, AH = (C1 - C0) * CELL, (R1 - R0) * CELL
H = 6.4
fig = plt.figure(figsize=(W, H))
texts = []
crop = g_crop = [row[C0:C1] for row in g[R0:R1]]
LEFT = 0.12
TXT_X = (LEFT + AW + 0.22) / W


def board(top, fade, title_letter, title):
    panel(fig, 0.01, 1 - (top - 0.02) / H, title_letter, title)
    ax = fig.add_axes([LEFT / W, 1 - (top + 0.24 + AH) / H, AW / W, AH / H])
    ax.imshow(rgb(crop, fade), extent=(C0, C1, R1, R0), interpolation="nearest", aspect="auto")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.5)
        s.set_color(C["gray"])
    ax.set_xlim(C0, C1)
    ax.set_ylim(R1, R0)
    return ax


def side(top, big, small, color):
    y = 1 - (top + 0.24 + AH * 0.5) / H
    texts.append(fig.text(TXT_X, y + 0.012, big, fontsize=11, fontweight="bold", color=color, ha="left", va="bottom"))
    texts.append(fig.text(TXT_X, y + 0.004, small, fontsize=7, color=C["gray_text"], ha="left", va="top", linespacing=1.2))


STEP = 0.24 + AH + 0.22
tops = [0.06, 0.06 + STEP, 0.06 + 2 * STEP]
ax = board(tops[0], 0.0, "(a)", "Board crop: three X marks (bp35 level 3 start)")
side(tops[0], "3 X marks", f"{xcells} cells of colour 12 drawn\non a plain plate", C["ink"])

ax = board(tops[1], 0.55, "(b)", "4-connected nodes, one outline each")
hues = plt.cm.hsv(np.linspace(0, 0.92, n_nodes))
for k, n in enumerate(xn):
    r0, c0, r1, c1 = n["bbox"]
    ax.add_patch(Rectangle((c0 + 0.06, r0 + 0.06), c1 - c0 + 0.88, r1 - r0 + 0.88, fill=False, ec=hues[k], lw=1.1))
side(tops[1], f"{n_nodes} nodes", "every cell of an X is its own\nobject: corners do not touch\nunder 4-connectivity", C["ink"])

ax = board(tops[2], 0.55, "(c)", "8-connected `group`, one outline each")
for (ns, (r0, c0, r1, c1)), col in zip(sorted(glyphs, key=lambda t: t[1][1]), (C["blue"], C["green"], C["pink"])):
    ax.add_patch(Rectangle((c0 + 0.06, r0 + 0.06), c1 - c0 + 0.88, r1 - r0 + 0.88, fill=False, ec=col, lw=1.6))
side(tops[2], f"{n_groups} objects", "each X is one 3 x 3 object\nwith the new `group` field", C["accent"])

# ---- D: corpus -------------------------------------------------------------------------------------------------------
dtop = tops[2] + 0.24 + AH + 0.3
panel(fig, 0.01, 1 - dtop / H, "(d)", "How often: human level starts holding such a glyph")
BW, BH = 5.0, 0.2
bx, by = 0.12, dtop + 0.3
axb = fig.add_axes([bx / W, 1 - (by + BH) / H, BW / W, BH / H])
axb.barh([0], [CORP["hit"]], color=C["accent"], height=1)
axb.barh([0], [CORP["starts"] - CORP["hit"]], left=[CORP["hit"]], color=C["light"], height=1)
axb.set_xlim(0, CORP["starts"])
axb.axis("off")
texts.append(fig.text((bx + 0.05) / W, 1 - (by + BH / 2) / H, f"{CORP['hit']} starts", color="white", fontsize=7.5,
                      fontweight="bold", ha="left", va="center"))
texts.append(fig.text((bx + BW - 0.05) / W, 1 - (by + BH / 2) / H, f"{CORP['starts'] - CORP['hit']} starts without",
                      color=C["gray_text"], fontsize=7.5, ha="right", va="center"))
texts.append(fig.text(bx / W, 1 - (by + BH + 0.1) / H,
                      f"{CORP['hit']} of {CORP['starts']} level starts in {CORP['hit_games']} of {CORP['games']} human replays "
                      "hold a `group` of 2 or more nodes with a\nbounding box of at least 3 x 3 cells "
                      f"({CORP['hit_filtered']} if sparse dot lattices, fill under 0.1, are dropped). Recount with the real module.", fontsize=7, color=C["ink"], ha="left", va="top", linespacing=1.25))
texts.append(fig.text(bx / W, 0.012, "Colours are the game's 16 indices on a fixed palette chosen for this picture; "
                      "(b) and (c) are faded so the outlines read.", fontsize=6.5, color=C["gray_text"], ha="left", va="bottom"))

check_no_overlap(fig, texts)
save(fig, "fig_glyph", writeup=True)
