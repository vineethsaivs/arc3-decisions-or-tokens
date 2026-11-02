"""scroll: say when the view scrolled, and diff objects after undoing the scroll.

Some games move a camera over a world larger than the screen. `frame_diff` then reported the shift as hundreds of
unrelated changes (bp35 s8 step 34: 1,343 cells, 89 "moved" pairs of 1px dots), and the model read it as the board
changing: uns2 built a "water drains" theory on bp35 L2, s8 spent 22 of 27 bp35 L2 turns and most of lf52 L3 and L4 on
offset arithmetic (kaggle-lf15us8-1031-1, kaggle-lf15uns2-1030-3).

1. `utils/frame_diff.py`: `_detect_scroll(a, b)` finds a whole-interior shift along one axis (the 4-cell edge band, where
   HUD bars live, is left out). Candidates come from whole-line matches (lines of one color are skipped since they match
   any shift); a candidate is a scroll when its cell mismatch rate is at most 0.1 and at most 0.3 of the unshifted rate,
   and at least 60 cells changed. On the 22 human-replay games it fires in bp35 (103 frame pairs) and lf52 (40) and in
   no other game (0 of about 6,000 changed pairs); about 1 ms a call. `compute_frame_diff` shifts the whole before
   frame and its objects by a detected scroll (cells scrolled into view count as changed, so a fixed HUD bar shows as
   moved) and adds `scroll: [rows, cols]`. With no scroll, its output is unchanged.
2. The turn opener says when the frames of the last sequence scrolled, and by how much in total.
3. The frame_diff hint documents `scroll`.

Run by build_franzen_arms.py right after Franzen's harness patch, as exec(code, {'BUNDLE_DIR': BUNDLE_DIR}).
Stdlib only. Every edit is asserted to match once; nothing is written unless all three files still compile.
"""
from pathlib import Path

_inf = Path(BUNDLE_DIR) / "src" / "ARC3-Inference" / "inference"  # noqa: F821
_MARK = "overlay scroll"

_DETECT = '''


# overlay scroll: a camera move shifts the whole interior; objects are compared after undoing it
def _detect_scroll(grid_a, grid_b, band=4, max_shift=32):
    # [rows, cols] the interior's content moved by (one axis at a time), or None
    h = len(grid_a)
    w = len(grid_a[0]) if h else 0
    if h != len(grid_b) or h < 2 * band + 8 or w < 2 * band + 8 or w != len(grid_b[0]):
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
        info = [len(set(x)) > 1 for x in lb]  # a one-color line matches any shift
        scored = []
        for s in range(-max_shift, max_shift + 1):
            if s == 0:
                continue
            m = t = 0
            for i in range(max(0, s), min(n, n + s)):
                if info[i]:
                    t += 1
                    m += la[i - s] == lb[i]
            if t >= 8 and m:
                scored.append((m / t, s))
        for _, s in sorted(scored, reverse=True)[:3]:
            lo, hi = max(0, s), min(n, n + s)
            mm = sum(x != y for i in range(lo, hi) for x, y in zip(la[i - s], lb[i]))
            rate = mm / ((hi - lo) * len(lb[0]))
            if rate <= 0.1 and rate <= 0.3 * mm0 / cells and (best is None or rate < best[0]):
                best = (rate, [s, 0] if axis == 0 else [0, s])
    return best[1] if best else None


def _shift_view(grid_a, nodes_a, scroll):
    # the before frame as it would look after the scroll; cells scrolled into view are -1 (changed)
    dr, dc = scroll
    h, w = len(grid_a), len(grid_a[0])
    shifted = [[grid_a[r - dr][c - dc] if 0 <= r - dr < h and 0 <= c - dc < w else -1 for c in range(w)]
               for r in range(h)]
    nodes = [dict(node, boundary=[[p[0] + dr, p[1] + dc] for p in node.get("boundary") or []]) for node in nodes_a]
    return shifted, nodes


def compute_frame_diff(grid_a, grid_b, nodes_a, nodes_b, max_group_match=None):
    scroll = _detect_scroll(grid_a, grid_b)
    if scroll:
        grid_a, nodes_a = _shift_view(grid_a, nodes_a, scroll)
    diff = _compute_frame_diff_unscrolled(grid_a, grid_b, nodes_a, nodes_b, max_group_match)
    if scroll:
        diff["scroll"] = scroll
    return diff
'''

_SUMMARY = '''def _scroll_summary_line(history_entries, count, summary):
    """overlay scroll: the camera shift over the last sequence's frames, as an opener line, or None."""
    try:
        if not count or summary.get("game_over") or summary.get("level_transition"):
            return None
        from inference.utils.frame_diff import _detect_scroll
        frames = [e.frame for e in list(history_entries or [])[-(int(count) + 1):] if e.frame is not None]
        dr = dc = 0
        for a, b in zip(frames, frames[1:]):
            if a.level != b.level:
                return None
            s = _detect_scroll(a.grid, b.grid)
            if s:
                dr, dc = dr + s[0], dc + s[1]
        if not (dr or dc):
            return None
        parts = ([f"{abs(dr)} rows {'down' if dr > 0 else 'up'}"] if dr else []) + (
            [f"{abs(dc)} cols {'right' if dc > 0 else 'left'}"] if dc else [])
        return ("The view scrolled during that sequence: everything inside the board moved " + " and ".join(parts)
                + " on screen. The camera moved, not the objects; the world is larger than the screen, so keep a "
                "running offset. `frame_diff()` reports this as `scroll` and compares objects after undoing it.")
    except Exception:
        return None


'''

_EDITS = {
    _inf / "utils" / "frame_diff.py": [
        ("def compute_frame_diff(grid_a, grid_b, nodes_a, nodes_b, max_group_match=None):\n",
         "def _compute_frame_diff_unscrolled(grid_a, grid_b, nodes_a, nodes_b, max_group_match=None):\n"),
        ('            "changed_color": changed_color, "resized": resized}\n',
         '            "changed_color": changed_color, "resized": resized}\n' + _DETECT),
    ],
    _inf / "agent" / "tool_agent.py": [
        ("def _render_auto_frame_diff(\n", _SUMMARY + "def _render_auto_frame_diff(\n"),
        ('                lines.append("Executed actions: none.")\n',
         '                lines.append("Executed actions: none.")\n'
         '            _scroll_line = _scroll_summary_line(history_entries, normalized_count, previous_step_summary)\n'
         '            if _scroll_line:  # overlay scroll\n'
         '                lines.append(_scroll_line)\n'),
    ],
    _inf / "agent" / "prompts.py": [
        ('    "- Examples: `frame_diff()` compares the latest action',
         '    "- When everything inside the board shifted between the two frames (a camera scroll), the result also has '
         '`scroll`: `[rows, cols]` the content moved by on screen (positive = down / right), and objects are compared '
         'after undoing it, so `moved` lists only real motion. A scrolling view means the world is larger than the '
         'screen.\\n"\n'
         '    "- Examples: `frame_diff()` compares the latest action'),
    ],
}

_plan = []
for _path, _edits in _EDITS.items():
    _text = _path.read_text()
    if _MARK in _text or "_detect_scroll" in _text or "a camera scroll" in _text:
        raise RuntimeError("scroll overlay: already applied to " + str(_path))
    for _old, _new in _edits:
        _count = _text.count(_old)
        if _count != 1:
            raise RuntimeError(f"scroll overlay: anchor matches {_count} times in {_path.name}, expected 1: "
                               f"{_old.strip()[:70]!r}")
        _text = _text.replace(_old, _new)
    compile(_text, str(_path), "exec")
    _plan.append((_path, _text))
for _path, _text in _plan:
    _path.write_text(_text)
print("arm overlay scroll: frame_diff undoes a camera scroll and reports it; the opener says when the view scrolled")
