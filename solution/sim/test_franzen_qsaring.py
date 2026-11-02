"""QSA pending-ring stride (franzen_overlays/qsaring_serve.py) against our Pennyroyal v2.5.3 wheel.

1. The overlay patches the wheel's four files exactly once, every edited file still compiles, a
   rerun is a no-op and a half-patched tree raises.
2. Replays MTP verify windows through the wheel's own ring builders, extracted from metadata.py
   before and after the patch: store the whole window into the ring, then read each completed
   group's members back, as the verify path does. With the shipped stride a 4-token window that
   starts mid-group hands the boundary group the next group's tokens; with the patch no group is
   corrupted for any window up to ratio + 1, and ratio + 2 wraps (so the guard is ratio + 1).
3. The CUDA-graph row kernel uses STRIDE for both the state slot and the member slots and is
   launched with STRIDE = 2 * ratio (Triton cannot run here, so its arithmetic is replayed in Python).

    python3 solution/sim/test_franzen_qsaring.py      # CPU torch via uv; no GPU, no network after the cache
"""
from __future__ import annotations

import ast
import os
import py_compile
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OVERLAY = REPO / "solution/kaggle/franzen_overlays/qsaring_serve.py"
FILES = ["srt/layers/attention/qsa/metadata.py", "srt/layers/attention/qsa/graph_metadata.py",
         "srt/mem_cache/qsa_kv_pool.py", "srt/layers/attention/qwen_sparse_attn_backend.py"]
RATIO = 4


def _reexec_under_uv() -> None:
    if os.environ.get("_QSARING_CHILD"):
        return
    env = dict(os.environ, _QSARING_CHILD="1")
    raise SystemExit(subprocess.run(["uv", "run", "--no-project", "--with", "torch", "--with", "numpy",
                                     "python", str(Path(__file__).resolve())], env=env).returncode)


def wheel_tree(dest: Path) -> Path:
    sys.path.insert(0, str(REPO / "solution/sim"))
    from franzen_tree import pennyroyal_wheel  # noqa: PLC0415

    root = dest / "sglang"
    with zipfile.ZipFile(pennyroyal_wheel()) as wheel:
        for rel in FILES:
            out = root / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(wheel.read("sglang/" + rel))
    return root


def builders(metadata_py: Path) -> dict:
    """The ring builders (and qsa_ring_stride when present) from metadata.py, with real torch."""
    import torch

    tree = ast.parse(metadata_py.read_text())
    wanted = {"qsa_ring_stride", "build_pending_ring_slots", "build_group_ring_slots"}
    body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(metadata_py), "exec"), namespace)
    return namespace


def corrupted_groups(ns: dict, window: int, prefixes: range, req: int = 3) -> tuple[int, int]:
    """(corrupted, completed) groups over verify windows starting after each prefix length."""
    import torch

    bad = done = 0
    for prefix in prefixes:
        ring: dict[int, int] = {}
        first = prefix - prefix % RATIO
        for pos in range(first, prefix):  # the boundary group's members stored by earlier forwards
            slot = ns["build_pending_ring_slots"](
                token_to_batch_idx=torch.tensor([0]), req_pool_indices=torch.tensor([req]),
                sequence_lengths=torch.tensor([pos + 1]), logical_positions=torch.tensor([pos]),
                compress_ratio=RATIO, is_extend=False)
            ring[int(slot)] = pos
        positions = torch.arange(prefix, prefix + window)
        slots = ns["build_pending_ring_slots"](
            token_to_batch_idx=torch.zeros(window, dtype=torch.long), req_pool_indices=torch.tensor([req]),
            sequence_lengths=torch.tensor([prefix + window]), logical_positions=positions,
            compress_ratio=RATIO, is_extend=False)
        for slot, pos in zip(slots.tolist(), positions.tolist()):  # the whole window lands first
            ring[slot] = pos
        ends = [p for p in positions.tolist() if p % RATIO == RATIO - 1]
        if not ends:
            continue
        members = ns["build_group_ring_slots"](
            req_pool_indices=torch.tensor([req]), group_end_positions=torch.tensor(ends),
            sequence_ids=torch.zeros(len(ends), dtype=torch.long), compress_ratio=RATIO)
        for end, group in zip(ends, members.tolist()):
            done += 1
            if [ring.get(slot) for slot in group] != list(range(end - RATIO + 1, end + 1)):
                bad += 1
    return bad, done


def main() -> None:
    _reexec_under_uv()
    namespace: dict = {}
    exec(compile(OVERLAY.read_text(), str(OVERLAY), "exec"), namespace)
    checks = 0
    with tempfile.TemporaryDirectory(prefix="qsaring-") as tmp:
        root = wheel_tree(Path(tmp))
        original = builders(root / FILES[0])

        # 1. the patch lands once, compiles, reruns cleanly and refuses a half-patched tree
        patched_paths = namespace["patch_tree"](root)
        assert sorted(p.relative_to(root).as_posix() for p in patched_paths) == sorted(FILES)
        for rel in FILES:
            text = (root / rel).read_text()
            assert namespace["MARK"] in text, rel
            py_compile.compile(str(root / rel), doraise=True)
        before = {rel: (root / rel).read_text() for rel in FILES}
        namespace["patch_tree"](root)
        assert before == {rel: (root / rel).read_text() for rel in FILES}, "a rerun changed the tree"
        with tempfile.TemporaryDirectory(prefix="qsaring-half-") as tmp2:
            half = wheel_tree(Path(tmp2))
            (half / FILES[2]).write_text(before[FILES[2]])
            try:
                namespace["patch_tree"](half)
            except RuntimeError as error:
                assert "half-patched" in str(error), error
            else:
                raise AssertionError("a half-patched tree was accepted")
        print("ok  the overlay patches all four wheel files once; they compile; a rerun is a no-op; "
              "a half-patched tree raises")
        checks += 1

        # 2. verify windows through the wheel's own builders, before and after
        patched = builders(root / FILES[0])
        assert patched["qsa_ring_stride"](RATIO) == 2 * RATIO
        prefixes = range(1, 257)
        old_bad, old_done = corrupted_groups(original, 4, prefixes)
        new_bad, new_done = corrupted_groups(patched, 4, prefixes)
        assert old_done == new_done > 0
        assert old_bad > 0, "the shipped stride should corrupt mid-group verify windows"
        assert new_bad == 0, f"the patched stride still corrupts {new_bad} groups"
        no_spec = corrupted_groups(original, 1, prefixes)
        assert no_spec[0] == 0, "plain decode (one token per forward) must be clean either way"
        widest = corrupted_groups(patched, RATIO + 1, prefixes)
        assert widest[0] == 0 and widest[1] > 0, widest
        too_wide = corrupted_groups(patched, RATIO + 2, prefixes)
        assert too_wide[0] > 0, "a window of ratio + 2 should wrap the doubled ring; the guard bound is wrong"
        print(f"ok  4-token verify windows after prefixes 1..256: shipped stride corrupts {old_bad} of "
              f"{old_done} completed groups, patched 0; one-token decode is clean; patched windows up to "
              f"{RATIO + 1} tokens are clean and {RATIO + 2} wrap ({too_wide[0]} of {too_wide[1]}), so the "
              f"guard is ratio + 1")
        checks += 1

        # 3. the CUDA-graph row kernel and its launch
        graph = (root / FILES[1]).read_text()
        assert "tl.store(state_slots_ptr + row, req * STRIDE + (current % STRIDE).to(tl.int64))" in graph
        assert "slot = req * STRIDE + (member % STRIDE).to(tl.int64)" in graph
        assert re.search(r"STRIDE=2 \* indexer\.compress_ratio,", graph)
        assert "STRIDE: tl.constexpr" in graph
        assert not re.search(r"req \* RATIO \+", graph), "a RATIO-strided ring address is left in the kernel"

        def kernel_rows(stride: int, prefix: int, window: int, req: int = 3) -> int:
            """The row kernel's arithmetic: each verify row stores its key, boundary rows read members."""
            ring = {req * stride + pos % stride: pos for pos in range(prefix - prefix % RATIO, prefix)}
            rows = list(range(prefix, prefix + window))
            for current in rows:
                ring[req * stride + current % stride] = current
            bad = 0
            for current in rows:
                if (current + 1) % RATIO == 0:
                    members = [max(current - (RATIO - 1 - k), 0) for k in range(RATIO)]
                    if [ring.get(req * stride + m % stride) for m in members] != members:
                        bad += 1
            return bad

        assert sum(kernel_rows(RATIO, p, 4) for p in prefixes) > 0
        assert sum(kernel_rows(2 * RATIO, p, 4) for p in prefixes) == 0
        pool = (root / FILES[2]).read_text()
        assert "ring_slots = self.qsa_num_request_slots * (2 * self.qsa_compress_ratio)" in pool
        backend = (root / FILES[3]).read_text()
        assert "if draft_tokens > self.compress_ratio + 1:" in backend
        print("ok  the CUDA-graph row kernel stores and reads with STRIDE = 2 * ratio (replayed: clean), "
              "the ring buffer is doubled and the draft guard matches it")
        checks += 1
    # 4. the GPU probe's pure functions (franzen_overlays/qsaparity_serve.py) on synthetic replies
    probe_ns: dict = {}
    probe_src = REPO / "solution/kaggle/franzen_overlays/qsaparity_serve.py"
    exec(compile(probe_src.read_text(), str(probe_src), "exec"), probe_ns)
    reply = lambda ids, lp: {"ids": ids, "lp": lp, "cached": 0}  # noqa: E731
    rows = [
        {"a": 100, "d": 50, "cached_tokens": 149, "fresh_cached_tokens": 0,
         "cached": reply([1, 2, 3], [-0.1, -0.2, -0.3]), "fresh": reply([1, 2, 9], [-0.3, -0.2, -0.3]),
         "again": reply([1, 2, 9], [-0.3, -0.2, -0.3])},
        {"a": 100, "d": 50, "cached_tokens": 149, "fresh_cached_tokens": 0,
         "cached": reply([7, 2, 3], [-1.0, -0.2, -0.3]), "fresh": reply([1, 2, 3], [-0.5, -0.2, -0.3]),
         "again": reply([1, 2, 3], [-0.5, -0.2, -0.3])},
        {"a": 100, "d": 50, "cached_tokens": 100, "fresh_cached_tokens": 0,  # reused only A: not counted
         "cached": reply([5], [-0.1]), "fresh": reply([6], [-0.1]), "again": reply([6], [-0.1])},
    ]
    summary = probe_ns["summarize"](rows)
    assert summary["prompts"] == 3 and summary["prompts_reusing_decode_keys"] == 2, summary
    cvf, fvf = summary["cached_vs_fresh"], summary["fresh_vs_fresh"]
    assert cvf["first_token_agreement"] == 0.5 and cvf["median_tokens_before_divergence"] == 1.0, cvf
    assert abs(cvf["mean_abs_first_logprob_diff"] - 0.35) < 1e-9, cvf
    assert fvf["first_token_agreement"] == 1.0 and fvf["median_tokens_before_divergence"] == 3, fvf
    rendered = probe_ns["_render"]("SYS", "USER")
    assert rendered.endswith("<|im_start|>assistant\n<think>\n") and "SYS" in rendered and "USER" in rendered
    print("ok  the GPU probe's summary counts only prompts whose decode-time keys were reused and reports "
          "agreement and divergence correctly on synthetic replies")
    checks += 1
    print(f"\n{checks} checks passed")


if __name__ == "__main__":
    main()
