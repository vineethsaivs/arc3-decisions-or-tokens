#!/usr/bin/env python3
"""F1 "One turn of the agent, and where each lever acts" (schematic, vector).

Run from the repo root:
  uv run --no-project --with matplotlib --with numpy python -I -B paper/v3/analysis/figs/fig_loop.py

Nothing is measured here. Every fact drawn is asserted to appear in paper/v2/evidence/system.md (the component record,
snapshot 2026-10-13) or the v3 dossier, so the schematic cannot drift from the shipped configuration:
  yield 8,192 (Franzen 2,048), 128K window, 448 of 512 experts, 14 streams, 70 Mamba slots, MTP draft x3, 110 hidden
  games, one RTX PRO 6000 96 GB, fail-closed at boot and at the end (O18, O19); the adapter was not shipped (dossier s3).
The three key decisions (coordinator, 2026-10-21) are in the accent colour: the 8,192-token yield, expert pruning that
lets 14 games run at once, and the fail-closed checks. Inherited and supporting parts are gray.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from figstyle import C, REPO, W, apply, check_no_overlap, save  # noqa: E402

SYS = (REPO / "paper/v2/evidence/system.md").read_text()
DOS = (REPO / "paper/v3/evidence/dossier_1013_1021.md").read_text()
for fact in ("yield every 8,192 tokens", "(Franzen 2,048)", "128K window", "448 of 512 experts", "14 streams",
             "70 Mamba slots", "MTP draft x3", "110 hidden games", "RTX PRO 6000 96 GB", "python REPL + action()",
             "10x board and", "admits 14 of 110 games"):
    assert fact in SYS, fact
assert "Not promoted, not uploaded to Kaggle" in DOS
assert "A scored run finishing under 20 levels raises" in SYS and "a server that exits before READY raises at once" in SYS

apply()
H = 3.4
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 110)
ax.set_ylim(0, 68)
ax.axis("off")
INK, GRAY, ACC = C["ink"], C["gray"], C["accent"]
EDGE, FILL = "#9A9A9A", "#F4F4F4"
texts = []


def box(x, y, w, h, title, body=None, accent=False, fill=None, lw=0.9, z=2):
    ec = ACC if accent else EDGE
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc=fill or ("#FBE9DF" if accent else FILL),
                                ec=ec, lw=lw * (1.6 if accent else 1.0), zorder=z))
    if body is None:
        texts.append(ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=7.5, color=ACC if accent else INK,
                             fontweight="bold" if accent else "normal", linespacing=1.15, zorder=z + 1))
    else:
        texts.append(ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=7.8, fontweight="bold",
                             color=ACC if accent else INK, zorder=z + 1))
        texts.append(ax.text(x + w / 2, y + h - 6.6, body, ha="center", va="top", fontsize=7, color=INK, linespacing=1.15,
                             zorder=z + 1))


def arrow(p, q, color=GRAY, lw=1.0, rad=0.0, style="-|>", ls="-", z=1):
    ax.add_patch(FancyArrowPatch(p, q, connectionstyle=f"arc3,rad={rad}", arrowstyle=style, mutation_scale=8, color=color,
                                 lw=lw, ls=ls, zorder=z, shrinkA=0, shrinkB=0))


def label(x, y, s, color=INK, ha="center", va="center", size=7, weight="normal", style="normal"):
    texts.append(ax.text(x, y, s, ha=ha, va=va, fontsize=size, color=color, fontweight=weight, fontstyle=style,
                         linespacing=1.15, zorder=6))


# ---- the turn: game -> prompt -> reasoning -> python/action() -> game ----------------------------------------------
Y, BH = 41, 15
box(1.5, Y, 19, BH, "Game", "one of 110 hidden\ngames, offline")
box(29, Y, 21.5, BH, "Prompt", "opener, board image,\nkept history (128K)")
box(58.5, Y, 21, BH, "Model reasons", "thinking tokens on\nthe local server")
box(87.5, Y, 21, BH, "Python tool", "runs code, calls\naction()")
arrow((20.5, Y + BH * 0.62), (29, Y + BH * 0.62))
label(24.75, Y + BH * 0.62 + 3.4, "board", size=6.8, color=C["gray_text"])
arrow((50.5, Y + BH * 0.62), (58.5, Y + BH * 0.62))
arrow((79.5, Y + BH * 0.62), (87.5, Y + BH * 0.62))
# actions go back to the game across the top: a new turn starts
arrow((98, Y + BH), (11, Y + BH), rad=0.17, lw=1.0)
label(54.5, 66.2, "an action ends the turn; the next turn starts from the new board", size=7, color=C["gray_text"])

# ---- key decision 1: when a turn yields, what is re-sent -----------------------------------------------------------
arrow((66, Y), (42, Y), rad=-0.55, color=ACC, lw=1.6)
label(46, Y - 10.6, "1  Yield after 8,192 tokens without an action (was 2,048)", color=ACC, weight="bold", size=7.5)
label(46, Y - 13.0, "the turn resumes with its opening message (the opener) and the\nboard image sent again; a later yield re-sends them less often",
      color=ACC, size=7, va="top")

# ---- the server ------------------------------------------------------------------------------------------------------
SY, SH = 1.5, 19.5
box(1.5, SY, 107, SH, "", fill="#FAFAFA", lw=0.8, z=0)
label(3.5, SY + SH - 2.4, "Server: SGLang on one RTX PRO 6000 (96 GB); 9 hours, no internet", ha="left", va="top",
      size=7.6, weight="bold")
label(106.5, SY + SH - 2.4, "faster serving acts here", ha="right", va="top", size=7, color=C["gray_text"], style="italic")
CY, CH = SY + 2.0, 10.5
box(3.5, CY, 23.5, CH, "Qwen3.8-Flash-Next,\n4-bit, draft model;\nfine-tuning acts here")
box(29.5, CY, 25, CH, "2  448 of 512 experts:\nmemory for 14 games\nat once", accent=True)
box(57, CY, 28.5, CH, "3  Fail-closed checks: a\ndead server or a run under\n20 levels raises an error", accent=True)
box(88, CY, 18.5, CH, "priority\nscheduler, prefix\ncache, watchdog")
# the model's tokens come from the server
arrow((78, Y), (78, SY + SH), color=GRAY, lw=0.9, style="<|-|>", ls=(0, (2, 1.6)))
label(79.2, (Y + SY + SH) / 2, "tokens", ha="left", size=6.8, color=C["gray_text"])

check_no_overlap(fig, [t for t in texts if t.get_text().strip()], pad=1.0)
save(fig, "fig_loop", writeup=True)
