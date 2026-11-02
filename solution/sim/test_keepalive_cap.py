"""Offline check of keepalive_cap (build_franzen_arms.py): each cap-12 arm is its base plus one edit, and the keepalive
overlay exactly as cell 7 embeds it installs a solver whose RESET cap is 12. The keepalive itself (spacing, stops,
idle close) is test_keepalive.py's; this checks only the constant. Run: python3 solution/sim/test_keepalive_cap.py"""
import ast
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_keepalive as T  # noqa: E402
from franzen_tree import pristine_tree  # noqa: E402

B = T.builder()
for base in ("l1a1y8sv2r3kp", "l1a1y8sv2r3wesrp", "l1a1y8sv2r3we3srp"):
    assert B.ARMS[base + "12"][1]() == B.ARMS[base][1]() + B.keepalive_cap(12), base
cell7 = T.arm_cells(B, "l1a1y8sv2r3kp12")[B.ENV]
# the string literal that cell 7 compiles and execs as franzen_overlays/keepalive.py (the cell has ! lines, so no ast)
head = "# arm overlay: keepalive\nexec(compile("
start = cell7.index(head) + len(head)
code = ast.literal_eval(cell7[start:cell7.index(", 'franzen_overlays/keepalive.py', 'exec')", start)])
assert "KEEPALIVE_MAX_RESETS = 12  # arm keepalive_cap" in code and "KEEPALIVE_MAX_RESETS = 5 " not in code
with tempfile.TemporaryDirectory() as tmp:
    tree = T.copy_tree(pristine_tree(), Path(tmp) / "tree")
    exec(compile(code, "franzen_overlays/keepalive.py", "exec"), {"BUNDLE_DIR": tree})
    solver = (tree / T.SOLVER).read_text()
    compile(solver, "solver.py", "exec")
    assert "KEEPALIVE_MAX_RESETS = 12  # arm keepalive_cap" in solver
    assert "while not reason and sent < KEEPALIVE_MAX_RESETS:" in solver
print("PASS test_keepalive_cap")
