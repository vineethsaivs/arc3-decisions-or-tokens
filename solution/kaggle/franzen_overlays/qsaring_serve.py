"""Arm qsaring: widen the QSA pending-ring stride so MTP verify cannot corrupt index keys.

Runs inside the launcher cell (cell 15) after the target is validated and before the server
starts: exec'd into a fresh namespace, then apply(VENV) patches the installed Pennyroyal SGLang.

The defect (our wheel, Pennyroyal v2.5.3, verified by reading it 2026-10-10). Qwen4-Exp's QSA
indexer keeps each request's not-yet-compressed index keys in a ring addressed
``req * ratio + pos % ratio`` (ratio 4). A target-verify forward stores its whole draft window
into the ring before the compress step reads a completed group's members back from it. Our MTP
verifies 4 tokens (SPEC_STEPS 3 + 1). After any partial accept the window starts mid-group, so
its rows past the group boundary land on ``pos % 4`` slots that still hold the boundary group's
earlier members, and the group's compressed key is computed from the next group's tokens. Both
the eager builders (``qsa/metadata.py``) and the CUDA-graph row kernel (``qsa/graph_metadata.py``)
use that addressing. The compressed keys steer which earlier blocks the sparse attention reads,
so the damage is to long-range recall of text the model generated itself.

The fix is lovedheart's sglang commit 427e9380ad (2026-10-06, Apache-2.0, "qsa: widen pending-ring
stride to 2x ratio to stop spec-verify clobbering"; their mutation-checked sweep: 90 of 118 groups
corrupted with the old stride, 0 with the new), ported here as asserted edits because our wheel's
surrounding text differs from their base. Stride 2 * ratio keeps a group's pending members and a
verify window of up to ratio + 1 tokens apart; the ring buffer doubles (a few rows per request) and
the draft-width guard becomes ratio + 1 (upstream's 2 * ratio still wraps; see EDITS).
solution/sim/test_franzen_qsaring.py checks the edits on our wheel and replays verify windows
through the patched builders.
"""
from __future__ import annotations

from pathlib import Path

MARK = "arm qsaring"

STRIDE_FN = '''

def qsa_ring_stride(compress_ratio: int) -> int:
    """Pending-ring slots per request: one group plus one speculation window ({mark})."""
    return 2 * compress_ratio
'''.format(mark=MARK)

# (relative path, old, new): each old text must occur exactly once in the Pennyroyal v2.5.3 source.
EDITS = [
    ("srt/layers/attention/qsa/metadata.py",
     "\n\ndef build_pending_ring_slots(\n",
     STRIDE_FN + "\n\ndef build_pending_ring_slots(\n"),
    ("srt/layers/attention/qsa/metadata.py",
     "    rows = token_to_batch_idx.long()[: logical_positions.numel()]\n",
     f"    stride = qsa_ring_stride(compress_ratio)  # {MARK}\n"
     "    rows = token_to_batch_idx.long()[: logical_positions.numel()]\n"),
    ("srt/layers/attention/qsa/metadata.py",
     "    slots = requests * compress_ratio + positions % compress_ratio\n",
     f"    slots = requests * stride + positions % stride  # {MARK}\n"),
    ("srt/layers/attention/qsa/metadata.py",
     "        slots = torch.where(pending, slots, positions % compress_ratio)\n",
     f"        slots = torch.where(pending, slots, positions % stride)  # {MARK}\n"),
    ("srt/layers/attention/qsa/metadata.py",
     "    return requests[:, None] * compress_ratio + positions % compress_ratio\n",
     f"    stride = qsa_ring_stride(compress_ratio)  # {MARK}\n"
     "    return requests[:, None] * stride + positions % stride\n"),
    ("srt/layers/attention/qsa/graph_metadata.py",
     "    RATIO: tl.constexpr,\n    FULL_PAGE: tl.constexpr,  # full-KV tokens per page\n",
     "    RATIO: tl.constexpr,\n"
     f"    STRIDE: tl.constexpr,  # {MARK}: pending-ring slots per request (2 * RATIO)\n"
     "    FULL_PAGE: tl.constexpr,  # full-KV tokens per page\n"),
    ("srt/layers/attention/qsa/graph_metadata.py",
     "    tl.store(state_slots_ptr + row, req * RATIO + (current % RATIO).to(tl.int64))\n",
     f"    tl.store(state_slots_ptr + row, req * STRIDE + (current % STRIDE).to(tl.int64))  # {MARK}\n"),
    ("srt/layers/attention/qsa/graph_metadata.py",
     "        slot = req * RATIO + (member % RATIO).to(tl.int64)\n",
     f"        slot = req * STRIDE + (member % STRIDE).to(tl.int64)  # {MARK}\n"),
    ("srt/layers/attention/qsa/graph_metadata.py",
     "        RATIO=indexer.compress_ratio,\n        FULL_PAGE=pool.qsa_compressed_page_size * indexer.compress_ratio,\n",
     "        RATIO=indexer.compress_ratio,\n"
     f"        STRIDE=2 * indexer.compress_ratio,  # {MARK}\n"
     "        FULL_PAGE=pool.qsa_compressed_page_size * indexer.compress_ratio,\n"),
    ("srt/mem_cache/qsa_kv_pool.py",
     "        ring_slots = self.qsa_num_request_slots * self.qsa_compress_ratio\n",
     f"        ring_slots = self.qsa_num_request_slots * (2 * self.qsa_compress_ratio)  # {MARK}\n"),
    # Upstream raised this guard to 2 * ratio, which is too loose for its own ring: a window that
    # starts r tokens into a group (r = 1..ratio-1 pending members) wraps onto the oldest pending
    # member once it reaches 2 * ratio - r + 1 tokens, so the safe width is ratio + 1 (5 for us;
    # we verify 4). The test replays both sides of that bound.
    ("srt/layers/attention/qwen_sparse_attn_backend.py",
     "        if draft_tokens > self.compress_ratio:\n",
     f"        if draft_tokens > self.compress_ratio + 1:  # {MARK}: stride 2*ratio holds a window of ratio+1\n"),
]


def patch_tree(sglang_root) -> list[Path]:
    """Apply every edit exactly once to an sglang package directory, validating all of them
    before writing any file; a second call is a no-op and a half-patched tree raises."""
    root = Path(sglang_root)
    files = sorted({rel for rel, _, _ in EDITS})
    texts = {rel: (root / rel).read_text() for rel in files}
    marked = [rel for rel in files if MARK in texts[rel]]
    if marked:
        if len(marked) != len(files):
            raise RuntimeError(f"qsaring: only {marked} carry the patch; the SGLang tree is half-patched")
        return [root / rel for rel in files]
    for rel, old, new in EDITS:
        if texts[rel].count(old) != 1:
            raise RuntimeError(f"qsaring: {rel} does not match the Pennyroyal v2.5.3 source "
                               f"({texts[rel].count(old)} matches for {old.strip()[:60]!r})")
        texts[rel] = texts[rel].replace(old, new)
    for rel in files:
        (root / rel).write_text(texts[rel])
    return [root / rel for rel in files]


def apply(venv) -> list[Path]:
    roots = sorted(Path(venv).glob("lib/python*/site-packages/sglang"))
    if len(roots) != 1:
        raise RuntimeError(f"qsaring: expected one installed sglang under {venv}, found {roots}")
    patched = patch_tree(roots[0])
    print(f"qsaring: QSA pending-ring stride is 2 * ratio in {', '.join(p.name for p in patched)}", flush=True)
    return patched
