"""Arm fastdec: Pennyroyal's online MXFP8 for Flash-Next's BF16 projections, without its two
row-capped paths, with the MXFP8 GEMMs autotuned. Research:
docs/research/2026-10-01-top5-plan/serving-speedups.md.

Runs inside the launcher cell (cell 15): apply(VENV, env) after the install and target validation,
before the server starts; check(LOG, log_offset, url, strict=not TRUE_SUBMISSION) after the cell's
READY wait.

Pennyroyal v2.5.3 already ships online FP8 (SGLANG_SM120_ONLINE_MXFP8, from mratsim's patch
0003; Gabriel Olympie's 0005/0006 and Lord Han Solo's mixed NVFP4/FP8 checkpoint do the same
job). It stores every otherwise-BF16 transformer projection as MXFP8 and runs it through
FlashInfer CUTLASS, and stores the HyperConnection mix weights and lm_head as row-wise FP8.
Franzen's launcher sets the switch to 0.

The switch alone would slow us down. The row-wise FP8 HC mix has a fused kernel only up to 16
rows (hc_mix_triton._FUSED_MIX_MAX_ROWS) and the FP8 lm_head GEMV only up to 32 rows
(sm120_online_fp8._MAX_KERNEL_ROWS). Above that, both dequantize the whole weight to BF16 on
every call: about 12 GB of extra traffic per verify forward for the 97 HC modules alone.
Target verify runs batch x 4 rows (MTP 3 steps), so 40 rows at 10 streams. Pennyroyal
qualified the switch only at 1 to 4 requests (16 rows or fewer).

Three asserted text edits to the installed SGLang:
  srt/layers/hyperconnection.py   GatedResidual never takes the row-wise FP8 mix path.
  srt/models/qwen4_exp.py         post_load_weights returns before converting lm_head.
  srt/model_executor/runner/flashinfer_autotune.py
                                  the target worker autotunes. Our config never does
                                  (quantization auto-round, MoE backend auto), so every MXFP8
                                  GEMM would run FlashInfer's untuned fallback tile (tactic 0,
                                  CtaShape128x32x128B, no swap-AB) at every row count.
                                  Pennyroyal's modelopt_fp4 run tuned them through its FP4 GEMM
                                  backend, and its decode gains were measured that way. Its draft
                                  (unquant) never tuned, so ours (compressed-tensors) does not.
What stays converted is the projections: GDN in_proj_qkvz and out_proj, attention qkv and o,
the QSA indexer and the shared expert, in the 48 target layers and the MTP layer. That is 5.9 GB
of BF16 read per verify forward becoming 3.0 GB, and about 2.7 GiB of VRAM that SGLang gives to
the KV pool at the same --mem-fraction-static. Sampling stays exact; only the projection numerics
change (W8A8, MXFP8 block-32 scales).

The tuning is decode-shaped (up to 40 rows at 10 streams). Speculative runners skip the EXTEND
autotune, so prefill-sized GEMMs stay on the fallback tile, here as in Pennyroyal, whose TTFT got
slower with the switch on. Expect prefill to cost more per token, not less.
"""
from __future__ import annotations

from pathlib import Path

MARK = "arm fastdec"
SWITCH = "SGLANG_SM120_ONLINE_MXFP8"
AUTOTUNE_DONE = "FlashInfer autotune completed."

HC_OLD = "            rowwise_mix = online_fp8 and online_fp8_enabled()\n"
HC_NEW = f"            rowwise_mix = False  # {MARK}: HC mix stays BF16; its FP8 path dequantizes above 16 rows\n"
HEAD_OLD = ("        if self.pp_group.is_last_rank:\n"
            "            replace_linear_weight_rowwise_fp8(self.lm_head)\n")
HEAD_NEW = f"        return  # {MARK}: lm_head stays BF16; its FP8 path dequantizes above 32 rows\n"
TUNE_OLD = "        fp8_gemm_needs_autotune = False\n"
TUNE_NEW = (f"        fp8_gemm_needs_autotune = envs.{SWITCH}.get() and not for_speculative_draft"
            f"  # {MARK}: tune the target's MXFP8 GEMMs, as Pennyroyal's modelopt_fp4 run did\n")


def patch_sglang(venv) -> list[Path]:
    """Apply the edits to the installed SGLang exactly once; a rerun of the cell is a no-op."""
    roots = sorted(Path(venv).glob("lib/python*/site-packages/sglang"))
    if len(roots) != 1:
        raise RuntimeError(f"fastdec: expected one installed sglang under {venv}, found {roots}")
    edits = [(roots[0] / "srt/layers/hyperconnection.py", HC_OLD, HC_NEW),
             (roots[0] / "srt/models/qwen4_exp.py", HEAD_OLD, HEAD_NEW),
             (roots[0] / "srt/model_executor/runner/flashinfer_autotune.py", TUNE_OLD, TUNE_NEW)]
    for path, old, new in edits:
        text = path.read_text()
        if new in text:
            continue
        if MARK in text or text.count(old) != 1:
            raise RuntimeError(f"fastdec: {path.name} does not match the Pennyroyal v2.5.3 source; "
                               "this SGLang build is not the one the arm was written against")
        path.write_text(text.replace(old, new))
    return [path for path, *_ in edits]


def apply(venv, env) -> None:
    patched = patch_sglang(venv)  # raises before the switch is set, so a failed patch never ships the flag
    env[SWITCH] = "true"
    print(f"fastdec: {SWITCH}=true for the projections; HC mix and lm_head kept BF16; target autotune on; "
          f"patched {', '.join(p.name for p in patched)}", flush=True)


def check(log, offset: int, health_url: str, strict: bool) -> None:
    """After cell 15's wait: the server is up inside the notebook's 720 s startup deadline and its log
    shows the autotune finished. A bench run (strict) stops here otherwise, instead of spending the
    play window on a run that could not be read; the scored run only warns, so it still scores."""
    import urllib.request
    try:
        with urllib.request.urlopen(health_url, timeout=5) as r:
            up = r.status == 200
    except Exception:
        up = False
    problem = None
    if not up:
        problem = "server not READY within the notebook's 720 s startup deadline, or it died"
    else:
        with open(log, "rb") as f:
            f.seek(offset)
            if AUTOTUNE_DONE.encode() not in f.read():
                problem = (f"no '{AUTOTUNE_DONE}' in serve.log; the MXFP8 GEMMs run FlashInfer's untuned "
                           "fallback tile, so this run could not be read")
    if problem and strict:
        raise RuntimeError(f"fastdec: {problem}")
    print(f"fastdec WARNING (scored run, continuing): {problem}" if problem else
          "fastdec: READY within the startup deadline and FlashInfer autotune completed", flush=True)
