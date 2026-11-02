"""Arm d3draft: serve ursuciprian's D3 refit of the MTP drafter's 29 dense tensors.

What. Our drafter (albucino, Franzen's Kaggle copy) keeps its dense mtp.* tensors (fc, norms, hyper-connection
mixers, attention, shared expert, router gate; BF16) in mtp-dense.safetensors next to lm_head and the embedding.
ursuciprian retrained exactly those 29 tensors by KL to a served 4-bit Flash-Next's top-20 (D3, Hugging Face
ursuciprian/Qwen3.8-Flash-Next-NVFP4-GDN-MSE@03f4a057). Their live screen: accepted drafts over 3 positions 2.015 to
2.178 (+8.1%). Speculative decoding verifies every token against the target, so outputs cannot change; only the
acceptance rate (tokens per step) can. Our private dataset vineethsaivs/flashnext-mtp-d3 holds the 29 tensors
unchanged (sha256 below); docs/research/2026-10-11-improve/serving.md section 2.1.

How. Run in cell 15 right after prepare_draft_view(): copy the draft view's mtp-dense.safetensors to local disk,
overwrite the 29 tensors' bytes in place (names, dtypes, shapes and byte sizes asserted equal, so the header and
the index stay byte-identical), and repoint the view's symlink at the copy. A rerun on an already spliced view is a
no-op. solution/sim/test_franzen_d3draft.py splices a synthetic pair.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import struct
from pathlib import Path

REFIT = "/kaggle/input/datasets/vineethsaivs/flashnext-mtp-d3/mtp_refit_d3.safetensors"
REFIT_SHA256 = "8d9a42d36a900256cc4f7a24fc3001d7d8a1ca9813236594489af0605d28ac1f"
NAME = "mtp-dense.safetensors"
N_TENSORS = 29
WORK = "/tmp/sgl-intel/d3"


def header(path) -> tuple[dict, int]:
    """(tensor entries without __metadata__, byte offset where the data starts)."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        meta = json.loads(f.read(n))
    meta.pop("__metadata__", None)
    return meta, 8 + n


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def splice(view, refit, work, expect_sha256: str | None = REFIT_SHA256, n_tensors: int = N_TENSORS) -> Path:
    view, refit, work = Path(view), Path(refit), Path(work)
    link, out = view / NAME, work / NAME
    if link.is_symlink() and link.resolve() == out.resolve():
        print(f"d3draft: {link} already points at the spliced copy", flush=True)
        return out
    if not link.is_symlink():
        raise RuntimeError(f"d3draft: {link} is not the draft view's symlink")
    if expect_sha256 is not None and _sha256(refit) != expect_sha256:
        raise RuntimeError(f"d3draft: {refit} does not match sha256 {expect_sha256}")
    src = link.resolve()
    rh, rb = header(refit)
    sh, sb = header(src)
    weight_map = json.loads((view / "model.safetensors.index.json").read_text())["weight_map"]
    if len(rh) != n_tensors or any(weight_map.get(name) != NAME for name in rh):
        raise RuntimeError(f"d3draft: the refit holds {len(rh)} tensors, or some are not mapped to {NAME}")
    for name, t in rh.items():
        s = sh.get(name)
        size = t["data_offsets"][1] - t["data_offsets"][0]
        if s is None or (t["dtype"], t["shape"]) != (s["dtype"], s["shape"]) \
                or size != s["data_offsets"][1] - s["data_offsets"][0]:
            raise RuntimeError(f"d3draft: {name} differs in dtype, shape or size from the drafter's")
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, out)
    with open(refit, "rb") as rf, open(out, "r+b") as df:
        for name, t in rh.items():
            a, b = t["data_offsets"]
            rf.seek(rb + a)
            df.seek(sb + sh[name]["data_offsets"][0])
            df.write(rf.read(b - a))
    if header(out) != (sh, sb):
        raise RuntimeError("d3draft: the spliced file's header changed")
    link.unlink()
    link.symlink_to(out)
    print(f"d3draft: {len(rh)} mtp.* tensors spliced from {refit.name} into {out} ({src.name} untouched)", flush=True)
    return out


def apply(view) -> Path:
    return splice(view, REFIT, WORK)
