"""franzen_overlays/asyncmrope_serve.py against our Pennyroyal v2.5.3 wheel: the edit lands once,
the file compiles, a rerun is a no-op, and the CPU fallback path still produces the same deltas.

    python3 solution/sim/test_franzen_asyncmrope.py      # no GPU, no torch
"""
from __future__ import annotations

import py_compile
import sys
import tempfile
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OVERLAY = REPO / "solution/kaggle/franzen_overlays/asyncmrope_serve.py"


def main() -> None:
    sys.path.insert(0, str(REPO / "solution/sim"))
    from franzen_tree import pennyroyal_wheel  # noqa: PLC0415

    ns: dict = {}
    exec(compile(OVERLAY.read_text(), str(OVERLAY), "exec"), ns)
    with tempfile.TemporaryDirectory(prefix="asyncmrope-") as tmp:
        root = Path(tmp) / "sglang"
        target = root / ns["REL"]
        target.parent.mkdir(parents=True)
        with zipfile.ZipFile(pennyroyal_wheel()) as wheel:
            original = wheel.read("sglang/" + ns["REL"]).decode()
        target.write_text(original)
        ns["patch_tree"](root)
        patched = target.read_text()
        assert patched.count(ns["MARK"]) == 1 and ns["OLD"] not in patched
        assert "pin_memory().to(device=device, non_blocking=True)" in patched
        assert "mrope_delta_tensor = mrope_delta_tensor.to(device=device)" in patched  # fallback kept
        py_compile.compile(str(target), doraise=True)
        ns["patch_tree"](root)
        assert target.read_text() == patched, "a rerun changed the file"
        # Only the one line changed: removing the new block restores the original text.
        assert patched.replace(ns["NEW"], ns["OLD"]) == original
    print("ok  asyncmrope lands once on our wheel, compiles, reruns as a no-op, keeps the blocking fallback\n\n1 checks passed")


if __name__ == "__main__":
    main()
