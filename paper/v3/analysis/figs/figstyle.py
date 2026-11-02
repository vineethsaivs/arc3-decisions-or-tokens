"""Shared style for the v3 figures: one place for fonts, sizes, colours and export.

Every figure script does:
    from figstyle import apply, C, W, save, letter
    apply()
    fig = plt.figure(figsize=(W, h))
    ...
    save(fig, "fig_name", writeup=False)   # paper/v3/figs/fig_name.pdf, fig_name.png (200 dpi)
                                           # writeup=True adds fig_name_writeup.png, exactly 1600 px wide

Figures are saved at their exact figure size (no tight crop), so the PDF is W = 5.5 in wide and drops into
\\linewidth without scaling; every script lays itself out by hand and asserts that no text leaves the page.
"""
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

W = 5.5                      # inches, single-column text width
WRITEUP_PX = 1600            # Kaggle Writeup gallery images
REPO = Path(__file__).resolve().parents[4]
OUT = REPO / "paper/v3/figs"
DATA = REPO / "paper/v3/evidence/data"
V2OUT = REPO / "paper/v2/analysis/out"
HERE = Path(__file__).resolve().parent

# Okabe-Ito. Gray carries baselines and context; ACCENT carries the one thing each figure is about.
C = {
    "accent": "#D55E00",     # vermillion: the lever, the result to look at
    "blue": "#0072B2",       # a second series, only when a figure needs two
    "sky": "#56B4E9",
    "green": "#009E73",
    "orange": "#E69F00",
    "pink": "#CC79A7",
    "gray": "#7F7F7F",       # baselines, controls, context
    "gray_text": "#5F5F5F",  # gray that still reads as text on white
    "light": "#D9D9D9",
    "faint": "#EFEFEF",
    "ink": "#222222",
    "grid": "#E3E3E3",
}


_S = 1.0                     # font scale set by apply(scale=...); 1.0 keeps the older figures as drawn
SCALE_V3 = 1.25              # in-figure text +25% for the Writeup (column width near 700 px); figures opt in


def fs(x):
    """A font size, scaled by the scale given to apply()."""
    return x * _S


def apply(scale=1.0):
    """Set the shared style. scale multiplies every font size (text only; lines and markers keep their size)."""
    global _S
    _S = scale
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": fs(8),
        "axes.labelsize": fs(8),
        "axes.titlesize": fs(8.5),
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "xtick.labelsize": fs(7),
        "ytick.labelsize": fs(7),
        "legend.fontsize": fs(7),
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": C["ink"],
        "axes.labelcolor": C["ink"],
        "axes.linewidth": 0.7,
        "text.color": C["ink"],
        "xtick.color": C["ink"],
        "ytick.color": C["ink"],
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "axes.grid": False,
        "grid.color": C["grid"],
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "lines.linewidth": 1.3,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "standard",
        "figure.dpi": 150,
    })


def panel(fig, x, y, letter, title):
    """Panel letter plus its one-line title, at figure fraction (x, y) of the letter's top left."""
    t1 = fig.text(x, y, letter, fontsize=fs(9), fontweight="bold", ha="left", va="top")
    fig.canvas.draw()
    w = t1.get_window_extent().width / fig.bbox.width
    t2 = fig.text(x + w + 0.008, y, title, fontsize=fs(8.5), ha="left", va="top")
    return t1, t2


def check_inside(fig):
    """Fail if any visible text leaves the figure (the export does not crop, so it would be clipped)."""
    fig.canvas.draw()
    bb = fig.bbox
    for t in fig.findobj(matplotlib.text.Text):
        if t.get_visible() and t.get_text().strip():
            e = t.get_window_extent()
            assert e.x0 >= bb.x0 - 1 and e.x1 <= bb.x1 + 1 and e.y0 >= bb.y0 - 1 and e.y1 <= bb.y1 + 1, (t.get_text(), e, bb)


def check_no_overlap(fig, texts, pad=0.5):
    """Fail if any two of the given text artists overlap (labels the script places by hand)."""
    fig.canvas.draw()
    boxes = [(t.get_text(), t.get_window_extent().expanded(1.0, 1.0)) for t in texts if t.get_text().strip()]
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i][1], boxes[j][1]
            if a.x0 < b.x1 - pad and b.x0 < a.x1 - pad and a.y0 < b.y1 - pad and b.y0 < a.y1 - pad:
                msg = f"labels overlap: {boxes[i][0]!r} and {boxes[j][0]!r}"
                if os.environ.get("FIG_LAX"):   # draft renders only; a final render must pass
                    print("WARNING", msg)
                else:
                    raise AssertionError(msg)


def save(fig, name, writeup=False):
    check_inside(fig)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=200)
    if writeup:
        fig.savefig(OUT / f"{name}_writeup.png", dpi=WRITEUP_PX / fig.get_figwidth())
    plt.close(fig)
    print("wrote", OUT / f"{name}.pdf")
