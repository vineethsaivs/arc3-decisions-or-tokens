"""Arm asyncmrope: upload speculative M-RoPE deltas without stalling the host.

Runs inside the launcher cell (cell 15) before the server starts; apply(VENV) patches the installed
Pennyroyal SGLang. In ``ForwardBatch.compute_spec_mrope_positions`` a batch with any image input
stacks per-request M-RoPE deltas on the host and copies them with a pageable ``.to(device)``, which
blocks the host until the GPU stream drains, before every target verify and draft extend. Our games
send two or three images per turn, so nearly every batch takes that path. A pinned source and a
``non_blocking`` copy give the same values without the stall; PyTorch's caching host allocator keeps
the pinned block alive until the copy has run. Son Pham's team measured +3% tokens/s at 13 streams on
this exact file (sonpham-org/arc-3 kfuse8, 2026-10-04); their repository has no licence, so this is our
own implementation of the standard pinned-copy idiom, not their code.
"""
from __future__ import annotations

from pathlib import Path

MARK = "arm asyncmrope"
REL = "srt/model_executor/forward_batch_info.py"
OLD = "            mrope_delta_tensor = torch.stack(mrope_deltas, dim=0).to(device=device)\n"
NEW = (
    "            mrope_delta_tensor = torch.stack(mrope_deltas, dim=0)\n"
    f"            if mrope_delta_tensor.device.type == \"cpu\" and torch.cuda.is_available():  # {MARK}\n"
    "                # pinned source + non_blocking copy: same values, no host stall before the launch\n"
    "                mrope_delta_tensor = mrope_delta_tensor.pin_memory().to(device=device, non_blocking=True)\n"
    "            else:\n"
    "                mrope_delta_tensor = mrope_delta_tensor.to(device=device)\n"
)


def patch_tree(sglang_root) -> list[Path]:
    """Apply the edit once to an sglang package directory; a second call is a no-op."""
    path = Path(sglang_root) / REL
    text = path.read_text()
    if MARK in text:
        return [path]
    if text.count(OLD) != 1:
        raise RuntimeError(f"asyncmrope: {REL} does not match the Pennyroyal v2.5.3 source "
                           f"({text.count(OLD)} matches)")
    path.write_text(text.replace(OLD, NEW))
    return [path]


def apply(venv) -> list[Path]:
    roots = sorted(Path(venv).glob("lib/python*/site-packages/sglang"))
    if len(roots) != 1:
        raise RuntimeError(f"asyncmrope: expected one installed sglang under {venv}, found {roots}")
    patched = patch_tree(roots[0])
    print("asyncmrope: speculative M-RoPE deltas upload pinned and non_blocking", flush=True)
    return patched
