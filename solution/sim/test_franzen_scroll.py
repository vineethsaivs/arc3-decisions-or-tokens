"""Offline check for scroll (franzen_overlays/scroll.py) on l1a1y8sv2r3kp12lf15usc.

On a copy of Franzen's patched bundle with the arm's overlays applied in the order its cell 7 runs them:
1. every overlay applies, the files compile, a second scroll refuses; the system prompt documents `scroll`;
2. a synthetic camera scroll (a textured interior shifted 6 rows, HUD bar fixed, one player moving 2 cols) gives
   `scroll: [6, 0]` and `moved` lists the player and nothing but the fixed HUD bar; a synthetic frame pair with no scroll diffs exactly as the
   unpatched module does;
3. the sandbox's own copy of the module (spliced into the bootstrap) carries the scroll code;
4. the opener line: a scrolled sequence names the total shift; a game over, a level change or no scroll give none;
5. on every human replay frame dump (FRAMES_DIR): scroll fires only in bp35 and lf52, and on every frame pair where it
   does not fire the diff equals the unpatched module's.

    python3 solution/sim/test_franzen_scroll.py [FRAMES_DIR]
"""
from __future__ import annotations

import glob
import importlib
import json
import os
import random
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

from test_franzen_seg8 import COLORS, load, order
from test_franzen_sandbox import FORK_ENV, TREE, apply, import_tool_agent

ARM = "l1a1y8sv2r3kp12lf15usc"


def textured(h=64, w=64, seed=3):
    rnd = random.Random(seed)
    g = [[0] * w for _ in range(h)]
    for _ in range(140):   # scattered blocks give every interior line more than one color
        r, c, k = rnd.randrange(h), rnd.randrange(w), rnd.choice((2, 5, 9, 11))
        for dr in range(rnd.randint(1, 3)):
            for dc in range(rnd.randint(1, 3)):
                if r + dr < h and c + dc < w:
                    g[r + dr][c + dc] = k
    return g


def camera(world, top, h=64, w=64):
    """The screen: rows top..top+h-1 of a taller world, with a fixed HUD bar in row 0."""
    g = [list(world[top + r][:w]) for r in range(h)]
    g[0] = [12] * 40 + [0] * (w - 40)
    return g


def main() -> None:
    names = order(ARM)
    assert names[0] == "scroll" and "seg8" in names and "unstick2" in names, names
    with tempfile.TemporaryDirectory(prefix="scroll-") as tmp:
        bundle = Path(tmp) / "bundle"
        shutil.copytree(TREE, bundle, ignore=shutil.ignore_patterns("__pycache__"))
        inf = bundle / "src" / "ARC3-Inference" / "inference"
        before = load(inf / "utils" / "frame_diff.py", "fd_before")
        seg = load(inf / "utils" / "segmentation.py", "seg_plain")
        for name in names:
            apply(name, bundle)
        try:
            apply("scroll", bundle)
        except RuntimeError as exc:
            assert "already applied" in str(exc), exc
        else:
            raise AssertionError("a second scroll did not refuse")
        after = load(inf / "utils" / "frame_diff.py", "fd_after")

        os.environ.update(FORK_ENV)
        for key in [k for k in sys.modules if k.startswith("inference")]:
            del sys.modules[key]
        ta = import_tool_agent(bundle)
        system = ta.ToolAgent(model="stub")._system_prompt
        assert "(a camera scroll), the result also has `scroll`" in system, "the frame_diff hint lacks scroll"
        print(f"ok 1 {ARM}: {len(names)} overlays, scroll first; compiles; a second scroll refuses; prompt documents scroll")

        def diff(mod, a, b):
            return mod.compute_frame_diff(a, b, seg.segment_layer(a, COLORS)["nodes"], seg.segment_layer(b, COLORS)["nodes"])

        world = textured(h=96)
        a, b = camera(world, 20), camera(world, 14)   # the camera moved up 6 rows: content moves 6 rows down
        for g in (a, b):
            for r in range(30, 33):
                g[r][20:23] = [14] * 3
        for r in range(30, 33):                        # the player is drawn after the scroll, 2 cols right
            b[r][20:25] = [world[14 + r][c] for c in range(20, 22)] + [14] * 3
        d = diff(after, a, b)
        assert d.get("scroll") == [6, 0], d.get("scroll")
        player = [m for m in d["moved"] if m["color"] == "N"]   # color 14; its before spot is shown in the new view
        assert len(player) == 1 and player[0]["from"] == [36, 20] and player[0]["to"] == [30, 22], d["moved"]
        assert all(m["to"][0] <= 6 or m["from"][0] <= 6 for m in d["moved"] if m["color"] != "N"), d["moved"]  # HUD only
        assert len(diff(before, a, b)["moved"]) > 5
        still = textured(seed=5)
        moved = [row[:] for row in still]
        for r in range(40, 43):
            moved[r][10:13], moved[r][14:17] = [0] * 3, [14] * 3
        assert diff(after, still, moved) == diff(before, still, moved) and "scroll" not in diff(after, still, moved)
        print(f"ok 2 {ARM}: a 6-row camera scroll reads scroll [6, 0], moved = the player plus the HUD bar "
              f"(unpatched: {len(diff(before, a, b)['moved'])} moved); a plain move diffs as before")

        sys.path.insert(0, str(bundle / "src" / "ARC3-Inference"))
        sb = importlib.import_module("inference.agent.python_tool_sandbox")
        child = next(v for v in vars(sb).values() if isinstance(v, str) and "def compute_frame_diff" in v)
        assert "def _detect_scroll" in child and 'diff["scroll"] = scroll' in child, "the sandbox child lacks scroll"
        print(f"ok 3 {ARM}: the sandbox reads the patched module")

        entries = [SimpleNamespace(frame=SimpleNamespace(grid=camera(world, top), level=2)) for top in (26, 20, 14)]
        line = ta._scroll_summary_line(entries, 2, {})
        assert line and "12 rows down" in line, line
        assert ta._scroll_summary_line(entries, 2, {"game_over": True}) is None
        entries[0].frame.level = 1
        assert ta._scroll_summary_line(entries, 2, {}) is None
        flat = [SimpleNamespace(frame=SimpleNamespace(grid=still, level=2)) for _ in range(3)]
        assert ta._scroll_summary_line(flat, 2, {}) is None and ta._scroll_summary_line(None, 0, {}) is None
        print(f"ok 4 {ARM}: the opener names a 12-row scroll over two steps; none on game over, level change or no scroll")

        frames = sys.argv[1] if len(sys.argv) > 1 else None
        if frames:
            fired, same = {}, 0
            for f in sorted(glob.glob(os.path.join(frames, "h_*.json"))):
                game = os.path.basename(f)[2:-5]
                seq = json.load(open(f))["seq"]
                grids = [[[int(ch, 16) for ch in row] for row in e["g"]] for e in seq]  # rows are hex strings
                for ea, eb, ga, gb in zip(seq, seq[1:], grids, grids[1:]):
                    if ea["lv"] != eb["lv"] or ga == gb:
                        continue
                    if after._detect_scroll(ga, gb):
                        fired[game] = fired.get(game, 0) + 1
                    else:   # where it stays silent the diff must not change
                        assert diff(after, ga, gb) == diff(before, ga, gb), (game, eb)
                        same += 1
            assert set(fired) == {"bp35", "lf52"}, fired
            print(f"ok 5 {ARM}: scroll fires only in {fired}; {same} silent pairs diff exactly as before")
    print("PASS test_franzen_scroll")


if __name__ == "__main__":
    main()
