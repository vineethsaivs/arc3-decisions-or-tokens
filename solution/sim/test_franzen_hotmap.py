"""The ARC traffic draft vocabulary (franzen_overlays/hotmap_serve.py, generated/arc-hot-map-2026-10-12.json).

    python3 solution/sim/test_franzen_hotmap.py      # CPU torch via uv; no GPU, no network after the cache

For both maps: the embedded bitmap decodes to exactly n unique ids with all 25 of the served map's ids >= 248044
(</think>, <tool_call>, </tool_call>, <|im_end|>, <|endoftext|> among them), every id inside the vocabulary; a wrong
sha256 or count refuses. install() writes the .pt with a real torch, and SGLang's loader (load_token_map:
torch.load(weights_only=True), then an int64 tensor) reads back the same ids. The hm32 arm's cell 15 installs the map
in place of Pennyroyal's and keeps the tokenizer assert.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OVERLAY = REPO / "solution/kaggle/franzen_overlays/hotmap_serve.py"
MANIFEST = REPO / "solution/kaggle/generated/arc-hot-map-2026-10-12.json"
SPECIAL = list(range(248044, 248060)) + list(range(248068, 248077))


def _reexec_under_uv() -> None:
    if os.environ.get("_HOTMAP_CHILD"):
        return
    env = dict(os.environ, _HOTMAP_CHILD="1")
    raise SystemExit(subprocess.run(["uv", "run", "--no-project", "--with", "torch", "python",
                                     str(Path(__file__).resolve())], env=env).returncode)


def run(cmd, *, check=True, env=None):  # cell 15's run(), minus the printing
    result = subprocess.run(list(map(str, cmd)), text=True, capture_output=True, env=env)
    if check and result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}): {result.stderr[-2000:]}")
    return result


def main() -> None:
    _reexec_under_uv()
    import torch

    hm = {}
    exec(compile(OVERLAY.read_text(), str(OVERLAY), "exec"), hm)
    manifest = json.loads(MANIFEST.read_text())
    assert manifest["special_ids"] == SPECIAL
    for size, m in manifest["maps"].items():
        n = int(size)
        ids = hm["decode"](m["bitmap_zlib_b64"], m["bitmap_sha256"], n)
        assert len(ids) == len(set(ids)) == n and set(SPECIAL) <= set(ids) and max(ids) < 248077, size
        for bad in ((m["bitmap_zlib_b64"], "0" * 64, n), (m["bitmap_zlib_b64"], m["bitmap_sha256"], n + 1)):
            try:
                hm["decode"](*bad)
            except RuntimeError:
                pass
            else:
                raise AssertionError(f"{size}: a wrong sha256 or count did not refuse")
        with tempfile.TemporaryDirectory(prefix="hotmap-") as tmp:
            path = hm["install"](m["bitmap_zlib_b64"], m["bitmap_sha256"], n, sys.executable, run, dict(os.environ),
                                 work=tmp)
            loaded = torch.tensor(torch.load(str(path), weights_only=True), dtype=torch.int64)  # load_token_map
            assert loaded.tolist() == ids and loaded.shape == (n,), size
        print(f"ok   {size}: {n} unique ids, 25 specials, sha256 checked, SGLang's loader reads the same ids")

    sys.path.insert(0, str(REPO / "solution/kaggle"))
    import build_franzen_arms as b

    edits = [e for e in b.ARMS["l1a1y8sv2r3hm32"][1]() if "arm hotmap" in e[2]]
    assert len(edits) == 1 and edits[0][0] == b.SERVE and edits[0][1] == b.FRSPEC_MAP
    nb = json.loads((b.BASE / "submission.ipynb").read_text())
    b.apply_edits(nb, b.REPLICA + b.ARMS["l1a1y8sv2r3hm32"][1]())
    cell = b.cell_text(nb["cells"][b.SERVE])
    assert "hot_tokens_64k.pt" not in cell and "_hotmap['install'](" in cell and "TOKENIZER_SHA, " in cell
    assert cell.index("_hotmap['install'](") < cell.index('args += ["--speculative-token-map", str(tok)]')
    print("ok   hm32 cell 15: the traffic map replaces Pennyroyal's, the tokenizer assert stays")


if __name__ == "__main__":
    main()
