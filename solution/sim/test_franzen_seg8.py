"""Offline check for seg8 (franzen_overlays/seg8.py) on l1a1y8sv2r3kp12lf15us8 and l1a1y8sv2r3kp12lf15s8.

On a copy of Franzen's patched bundle with each arm's overlays applied in the order its cell 7 runs them:
1. every overlay applies, the files compile, a second seg8 refuses; the system prompt names `group`;
2. on a board with an X, a diagonal line, a plain square, two same-color squares touching only at a corner and two
   different-color cells touching at a corner: ids, hashes and the rest of each node match the unpatched module,
   and `group` joins exactly the corner-touching same-color nodes, rooted at the lowest id;
3. the sandbox's own copy of the module (spliced into the bootstrap) carries `group` too;
4. on the start boards of every human replay frame dump, a group is always one 8-connected same-color object.

    python3 solution/sim/test_franzen_seg8.py [FRAMES_DIR]
"""
from __future__ import annotations

import glob
import importlib.util
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

from test_franzen_sandbox import FORK_ENV, TREE, apply, import_tool_agent

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kaggle"))
import build_franzen_arms as B  # noqa: E402

COLORS = "WwgGcBMPRbSYOrNp"


def order(arm: str) -> list[str]:
    nb = json.loads((B.BASE / "submission.ipynb").read_text())
    B.apply_edits(nb, B.REPLICA + B.ARMS[arm][1]())
    return re.findall(r"^# arm overlay: (\w+)$", B.cell_text(nb["cells"][B.ENV]), re.M)


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def board():
    g = [[0] * 16 for _ in range(16)]
    for r, c in ((1, 1), (1, 3), (2, 2), (3, 1), (3, 3)):   # X of color 8: five nodes, one group
        g[r][c] = 8
    for k in range(4):                                         # diagonal line of color 11: four nodes, one group
        g[6 + k][1 + k] = 11
    for r in (1, 2):                                           # plain 2x2 square of color 9
        for c in (8, 9):
            g[r][c] = 9
    for r in (5, 6):                                           # two 2x2 squares of color 12 touching at a corner
        for c in (8, 9):
            g[r][c] = 12
    for r in (7, 8):
        for c in (10, 11):
            g[r][c] = 12
    g[12][12] = 14                                             # different colors at a corner: two groups
    g[13][13] = 15
    return g


def groups_ok(seg: dict, grid: list[list[int]]) -> None:
    nodes = seg["nodes"]
    cells = {}
    h, w = len(grid), len(grid[0])
    lab = [[-1] * w for _ in range(h)]
    # rebuild 4-connected labels in reading order, as the module does
    for r in range(h):
        for c in range(w):
            if lab[r][c] != -1:
                continue
            cid = len(cells)
            st = [(r, c)]
            lab[r][c] = cid
            cells[cid] = []
            while st:
                y, x = st.pop()
                cells[cid].append((y, x))
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and lab[ny][nx] == -1 and grid[ny][nx] == grid[y][x]:
                        lab[ny][nx] = cid
                        st.append((ny, nx))
    assert len(cells) == len(nodes)
    by8 = {}
    seen = set()
    for r in range(h):
        for c in range(w):
            if (r, c) in seen:
                continue
            st = [(r, c)]
            seen.add((r, c))
            members = set()
            while st:
                y, x = st.pop()
                members.add(lab[y][x])
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = y + dy, x + dx
                        if (0 <= ny < h and 0 <= nx < w and (ny, nx) not in seen
                                and grid[ny][nx] == grid[y][x]):
                            seen.add((ny, nx))
                            st.append((ny, nx))
            for m in members:
                by8[m] = min(members)
    for node in nodes:
        assert node["group"] == by8[node["id"]], (node["id"], node["group"], by8[node["id"]])


def check_arm(arm: str) -> None:
    names = order(arm)
    assert names[0] == "seg8" and "wastefix" in names, names
    with tempfile.TemporaryDirectory(prefix="seg8-") as tmp:
        bundle = Path(tmp) / "bundle"
        shutil.copytree(TREE, bundle, ignore=shutil.ignore_patterns("__pycache__"))
        seg_path = bundle / "src" / "ARC3-Inference" / "inference" / "utils" / "segmentation.py"
        for name in names:
            if name == "seg8":
                before = load(seg_path, "seg_before_" + arm)
            apply(name, bundle)
        try:
            apply("seg8", bundle)
        except RuntimeError as exc:
            assert "already applied" in str(exc), exc
        else:
            raise AssertionError("a second seg8 did not refuse")
        after = load(seg_path, "seg_after_" + arm)

        os.environ.update(FORK_ENV)
        for key in [k for k in sys.modules if k.startswith("inference")]:   # the other arm's bundle
            del sys.modules[key]
        ta = import_tool_agent(bundle)
        system = ta.ToolAgent(model="stub")._system_prompt
        assert "and `group` (the lowest id among the same-color nodes joined" in system, system[:200]
        print(f"ok 1 {arm}: {len(names)} overlays, seg8 first; compiles; a second seg8 refuses; prompt names group")

        g = board()
        old, new = before.segment_layer(g, COLORS), after.segment_layer(g, COLORS)
        assert old["adjacency_list"] == new["adjacency_list"]
        for a, b in zip(old["nodes"], new["nodes"], strict=True):
            assert {k: v for k, v in b.items() if k not in ("group", "bbox")} == a, (a, b)  # wastefix adds bbox
        groups_ok(new, g)
        by_color = {}
        for node in new["nodes"]:
            by_color.setdefault(node["color"], set()).add(node["group"])
        assert len(by_color["R"]) == 1 and sum(n["color"] == "R" for n in new["nodes"]) == 5
        assert len(by_color["Y"]) == 1 and sum(n["color"] == "Y" for n in new["nodes"]) == 4
        assert len(by_color["O"]) == 1 and sum(n["color"] == "O" for n in new["nodes"]) == 2
        assert len(by_color["N"]) == 1 and len(by_color["p"]) == 1 and by_color["N"] != by_color["p"]
        print(f"ok 2 {arm}: the X (5 nodes), the diagonal (4) and the corner-touching squares (2) are one group each; "
              "different colors stay apart; every other field and the adjacency match the unpatched module")

        sys.path.insert(0, str(bundle / "src" / "ARC3-Inference"))
        sb = importlib.import_module("inference.agent.python_tool_sandbox")
        child = next(v for v in vars(sb).values() if isinstance(v, str) and "def segment_layer" in v)
        assert '"group": _root(cid),  # overlay seg8' in child, "the sandbox child lacks the patched module"
        print(f"ok 3 {arm}: the sandbox reads the patched module")

        frames = sys.argv[1] if len(sys.argv) > 1 else None
        if frames:
            n = 0
            for f in sorted(glob.glob(os.path.join(frames, "h_*.json"))):
                last = None
                for e in json.load(open(f))["seq"]:
                    if e["lv"] != last:
                        last = e["lv"]
                        g = [[int(ch, 16) for ch in row] for row in e["g"]]  # rows are hex strings
                        groups_ok(after.segment_layer(g, COLORS), g)
                        n += 1
            print(f"ok 4 {arm}: groups match the 8-connected objects on {n} human level starts")


def main() -> None:
    for arm in ("l1a1y8sv2r3kp12lf15us8", "l1a1y8sv2r3kp12lf15s8"):
        check_arm(arm)
    print("PASS test_franzen_seg8")


if __name__ == "__main__":
    main()
