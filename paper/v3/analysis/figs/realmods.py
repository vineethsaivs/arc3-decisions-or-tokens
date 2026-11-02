"""Load the REAL harness modules for the glyph and scroll figures (no reimplementation).

Builds a temp copy of Franzen's patched bundle (solution/sim/franzen_tree.py), loads segmentation.py and frame_diff.py
before any overlay (`seg0`, `fd0`), applies the overlays of arm l1a1y8sv2r3kp12lf15usc in the order its cell 7 runs them,
and loads the same two files again (`seg1`, `fd1`). Same recipe as solution/sim/test_franzen_seg8.py and
test_franzen_scroll.py.
"""
import contextlib
import io
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

from figstyle import REPO

ARM = "l1a1y8sv2r3kp12lf15usc"


def build():
    sim = str(REPO / "solution/sim")
    if sim not in sys.path:
        sys.path.insert(0, sim)
    from test_franzen_sandbox import TREE, apply  # noqa: E402
    from test_franzen_seg8 import COLORS, load, order  # noqa: E402

    with contextlib.redirect_stdout(io.StringIO()), tempfile.TemporaryDirectory(prefix="figmods-") as tmp:
        names = order(ARM)
        bundle = Path(tmp) / "bundle"
        shutil.copytree(TREE, bundle, ignore=shutil.ignore_patterns("__pycache__"))
        inf = bundle / "src/ARC3-Inference/inference/utils"
        seg0, fd0 = load(inf / "segmentation.py", "fig_seg0"), load(inf / "frame_diff.py", "fig_fd0")
        for name in names:
            apply(name, bundle)
        assert "scroll" in names and "seg8" in names, names
        seg1, fd1 = load(inf / "segmentation.py", "fig_seg1"), load(inf / "frame_diff.py", "fig_fd1")
    assert not hasattr(fd0, "_detect_scroll") and hasattr(fd1, "_detect_scroll")
    return SimpleNamespace(seg0=seg0, seg1=seg1, fd0=fd0, fd1=fd1, COLORS=COLORS)


def grid(row):
    """One replay frame: 64 hex-digit row strings to a list of int rows."""
    return [[int(ch, 16) for ch in r] for r in row]


# Fixed 16-colour map for the pictures (index = the game's colour value); this is the figures' own choice, not the harness's.
PALETTE = ["#FFFFFF", "#CCCCCC", "#999999", "#666666", "#333333", "#000000", "#E53AA3", "#FF7BCC",
           "#F93C31", "#1E93FF", "#88D8F1", "#FFDC00", "#FF851B", "#921231", "#4FCC30", "#A356D6"]


def rgb(g, fade=0.0):
    """A grid of colour indices as an (h, w, 3) float array; fade blends toward white (0 none, 1 all white)."""
    import numpy as np
    pal = np.array([[int(p[i:i + 2], 16) / 255 for i in (1, 3, 5)] for p in PALETTE])
    img = pal[np.asarray(g)]
    return img * (1 - fade) + fade
