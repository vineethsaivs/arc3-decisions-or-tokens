"""seg8: every segmentation node names its 8-connected object.

`current_frame.segmentation` is the model's primary view (about two uses per python call in kaggle-lf15unst-1030-1)
and its nodes are 4-connected, so a glyph drawn with corner contacts falls apart: bp35 L3's X marks (the level's key)
read as 15 single pixels and a stuck run called them "pure decoration". On the start boards of the 21 human replays,
71 of 184 level starts in 13 games have such an object (an 8-connected same-color object of 3x3 or more made of 2 or
more nodes, color not a dot texture): bp35 and r11l X marks, s5i5 and su15 pieces, diagonal strokes.

Each node gains `group`: the lowest node id among the same-color nodes joined to it through corner contacts, so nodes
sharing a `group` form one 8-connected object. Ids, hashes, boundaries and adjacency are unchanged; the node line in
the system prompt names the field. The module is spliced into the sandbox, so both views carry it.

Run by build_franzen_arms.py right after Franzen's harness patch, as exec(code, {'BUNDLE_DIR': BUNDLE_DIR}).
Stdlib only. Every edit is asserted to match once; nothing is written unless both files still compile.
"""
from pathlib import Path

_inf = Path(BUNDLE_DIR) / "src" / "ARC3-Inference" / "inference"  # noqa: F821
_MARK = "overlay seg8"

_EDITS = {
    _inf / "utils" / "segmentation.py": [
        ("    n = len(components)\n",
         "    n = len(components)\n"
         "\n"
         "    # overlay seg8: same-color components that touch at a corner share a group (8-connectivity);\n"
         "    # a group's root is its lowest id, since the larger root always joins the smaller\n"
         "    group = list(range(n))\n"
         "\n"
         "    def _root(i):\n"
         "        while group[i] != i:\n"
         "            group[i] = group[group[i]]\n"
         "            i = group[i]\n"
         "        return i\n"
         "\n"
         "    for r in range(height - 1):\n"
         "        for c in range(width):\n"
         "            for nc in (c - 1, c + 1):\n"
         "                if 0 <= nc < width and layer[r + 1][nc] == layer[r][c]:\n"
         "                    ra, rb = _root(comp_id[r][c]), _root(comp_id[r + 1][nc])\n"
         "                    if ra != rb:\n"
         "                        group[max(ra, rb)] = min(ra, rb)\n"),
        ('                "children": children[cid],\n',
         '                "children": children[cid],\n'
         '                "group": _root(cid),  # overlay seg8\n'),
    ],
    _inf / "agent" / "prompts.py": [
        ("`children` (ids of objects fully enclosed by this one).",
         "`children` (ids of objects fully enclosed by this one), and `group` (the lowest id among the same-color "
         "nodes joined to this one through corner contacts: nodes sharing a `group` form one 8-connected object, "
         "such as the strokes of an X or a diagonal line)."),
    ],
}

_plan = []
for _path, _edits in _EDITS.items():
    _text = _path.read_text()
    if _MARK in _text or "8-connected object" in _text:
        raise RuntimeError("seg8 overlay: already applied to " + str(_path))
    for _old, _new in _edits:
        _count = _text.count(_old)
        if _count != 1:
            raise RuntimeError(f"seg8 overlay: anchor matches {_count} times in {_path.name}, expected 1: "
                               f"{_old.strip()[:70]!r}")
        _text = _text.replace(_old, _new)
    compile(_text, str(_path), "exec")
    _plan.append((_path, _text))
for _path, _text in _plan:
    _path.write_text(_text)
print("arm overlay seg8: segmentation nodes carry their 8-connected group")
