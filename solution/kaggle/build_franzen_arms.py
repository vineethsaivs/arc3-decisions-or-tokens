"""Build Franzen-fork arms for the 25-game scored-run replica. Zero GPU, no push.

    python3 solution/kaggle/build_franzen_arms.py ARM [ARM ...]   # writes candidates-franzen-ARM
    python3 solution/kaggle/build_franzen_arms.py --list
    python3 solution/kaggle/build_franzen_arms.py --self-check     # every edit lands exactly once

WHY. Franzen's demo mode runs 10 easy games on 10 slots, so his priority scheduler never runs.
Every arm here plays all 25 public games on a 7,200 s shared window with 10 admitted streams:
the same slot-seconds per game as the hidden run (10 slots x 31,920 s / 110 games). Only the
non-submission branch changes; TRUE_SUBMISSION keeps his exact schedule. Plan: docs/PLAN_2026-10-01.md.

Each arm is the replica plus a short list of exact, asserted string edits to the base fork's
cells. Harness-level arms (B track) add a cell that runs an overlay script from
solution/kaggle/franzen_overlays/ against the patched bundle, right after cell 7 applies
Franzen's harness patch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
GEN = HERE / "generated"
BASE = GEN / "candidates-fork-franzen"
OVERLAYS = HERE / "franzen_overlays"

ENV, PRECACHE, SERVE, CUSTOM, RUN = 7, 9, 15, 19, 23  # cell indices in the base fork

REPLICA = [
    (CUSTOM,
     "demo_excluded_games = [] if TRUE_SUBMISSION else ['bp35', 'cd82', 'cn04', 'dc22', 'g50t', 'ka59', 'lf52', 'ls20', 'm0r0', 's5i5', 'sk48', 'sp80', 'su15', 'tn36', 'wa30']",
     "demo_excluded_games = []  # replica: all 25 public games (the submission path was already [])"),
    (CUSTOM,
     "bm.solver.max_runtime_s_per_game = 25*60 #532*60 * bm.solver.concurrency // 110",
     "bm.solver.max_runtime_s_per_game = 7200  # replica: 10 slots x 31,920 s / 110 games, times 25 games / 10 slots"),
]


def streams(n: int) -> list[tuple[int, str, str]]:
    """Admit n games at once: harness gate, server request cap, and decode graphs for 11..n."""
    graphs = ", ".join(str(b) for b in range(11, n + 1))
    return [
        (ENV, "'ARC3_MAX_ACTIVE_STREAMS': 10,", f"'ARC3_MAX_ACTIVE_STREAMS': {n},"),
        (SERVE, "    MAXREQ=10,\n    CUDAGRAPH_MAXBS=10,", f"    MAXREQ={n},\n    CUDAGRAPH_MAXBS={n},"),
        (SERVE, 'graph_bs = sorted({1, 2, 4, 7, 8, 9, 10, CFG["MAXREQ"], CFG["CUDAGRAPH_MAXBS"]})',
         f'graph_bs = sorted({{1, 2, 4, 7, 8, 9, 10, {graphs}, CFG["MAXREQ"], CFG["CUDAGRAPH_MAXBS"]}})'),
    ]


# A2: weight load is gated by the 185 GB background precache (host RAM is 177 GB, so it cannot
# simply run far ahead). Franzen starts it 60 s after the server launch; start it at the launch
# instead, same 3 threads, and drop the 30 s torch/sglang sanity import (cell 0 gates the GPU).
STARTUP = [
    (PRECACHE, "kwargs={'delay': 60, 'threads': 3})", "kwargs={'delay': 0, 'threads': 3})  # arm: no 60 s delay"),
    (SERVE,
     "run([PYTHON, \"-c\", '''import pathlib, sglang, torch\n"
     "print(\"sglang\", sglang.__version__, \"| torch\", torch.__version__)\n"
     "assert torch.cuda.is_available(), \"CUDA unavailable\"\n"
     "print(\"GPU:\", torch.cuda.get_device_name(0), \"capability:\", torch.cuda.get_device_capability(0))\n"
     "'''], env=env)",
     "# arm: 30 s torch/sglang sanity import dropped; cell 0 already gates the GPU"),
]

# A4: keep more history after a trim. Window 128K; drain 58K leaves about 59K. A 46K drain
# leaves about 71K, between Franzen and Lord Han Solo's 81.5K, at the cost of more frequent trims.
HISTORY = [
    (ENV, "'ARC3_CONTEXT_DRAIN_TOKENS': 58*1024,", "'ARC3_CONTEXT_DRAIN_TOKENS': 46*1024,  # arm: keep ~71K after a trim"),
]


# A3: give returning games their cached prefix back. KVLITE frees about 1 GB of VRAM for the
# KV pool (memory fraction 0.96 -> 0.97 leaves ~1.2 GB free at runtime). HICACHE adds SGLang's host-RAM KV tier with rellik13's flags (22.53 on the board; he
# ran 48 GB beside an FP8 PLE). Our PLE is BF16 and the run uses ~114 of 177 GB host RAM, so 32.
KVLITE = [
    # Mamba stays at 60: the 10-01 probe with 48 slots made SGLang cap running requests at 9.
    (SERVE, "    MEMFRAC=0.96,", "    MEMFRAC=0.97,  # arm: +~1 GB KV"),
]


def hicache(gb: int) -> list[tuple[int, str, str]]:
    return [
        (SERVE, "    CHUNK=8192,", "    CHUNK=4096,  # arm: rellik13's chunk size with the host tier"),
        (SERVE, '"--enable-cache-report", "--enable-metrics", "--enable-request-time-stats-logging"]',
         '"--enable-cache-report", "--enable-metrics", "--enable-request-time-stats-logging",\n'
         f'    "--enable-hierarchical-cache", "--hicache-size", "{gb}", "--hicache-write-policy", "write_through",\n'
         '    "--hicache-io-backend", "kernel"]  # arm: host-RAM KV tier'),
    ]


# Yield after 8,192 generated tokens in a turn instead of Franzen's 2,048 (cell 7).
YIELD8K = [(ENV, "'LOCAL_ANALYZER_YIELD_TOKENS': '2048',",
            "'LOCAL_ANALYZER_YIELD_TOKENS': '8192',  # arm y8: yield 4x less often")]
# +0.47 GiB of runtime GPU headroom (the pool loses about 34.7K of 1.53M tokens, 2.3%). Free device memory dips to
# 0.15 GiB in this family's runs and two 10-11 benches died of CUDA OOM after READY, one 1 h 54 min in; the
# verifier puts the 9 h OOM risk at about 5 to 20% per run now, 1 to 5% with this (docs/research/2026-10-11-improve).
MEMFRAC955 = [(SERVE, "    MEMFRAC=0.96,", "    MEMFRAC=0.955,  # arm: +0.47 GiB runtime headroom against CUDA OOM")]
# A 4th MTP draft token: verify window 5 = compress_ratio + 1, the widest the qsaring stride keeps clean (the shipped
# guard refused anything over 4). After D3 the 4th position should add about 0.35 to 0.40 accepted tokens per step for
# about +10% step time (docs/research/2026-10-11-improve/serving.md 2.4). Needs qsaring().
SPEC4 = [(SERVE, "    SPEC_STEPS=3,", "    SPEC_STEPS=4,  # arm s4: verify window 5, inside the qsaring guard")]
# Image hashes computed by the tokenizer process instead of the scheduler. Every request re-sends 7 to 36 history
# images, and without this flag the single-threaded scheduler sha256-hashes each 9.83 MB pixel tensor in
# MultimodalInputs (schedule_batch.py set_pad_value) while all 14 streams wait. Same hash values, so lossless
# (docs/research/2026-10-11-improve/field-web.md; Son Pham's team ships it).
MMHASH = [(SERVE, '"SGLANG_MM_PREPROCESS_DEVICE": "cpu", "NUMPY_MADVISE_HUGEPAGE": "0",',
           '"SGLANG_MM_PREPROCESS_DEVICE": "cpu", "NUMPY_MADVISE_HUGEPAGE": "0",\n'
           '    "SGLANG_MM_PRECOMPUTE_HASH": "1",  # arm mmhash: image hashes off the scheduler thread')]
# Output cap 12K -> 16K with the window raised 4K, so the trim budget (window - cap - 512) stays 118,272 and the
# largest prompt seen (119,970) plus 16,384 still fits the server's 139,264. About 40 replies per run are cut at
# 12,288 while still reasoning; reply lengths halve every ~2K past 4K, so about two thirds would finish under 16K
# (docs/research/2026-10-11-improve/yield-turns.md, L3).
OUTCAP16 = [(ENV, "'LOCAL_ANALYZER_MAX_OUTPUT': 12*1024,",
             "'LOCAL_ANALYZER_MAX_OUTPUT': 16*1024,  # arm o16: fewer replies cut while reasoning"),
            (ENV, "'LOCAL_ANALYZER_CONTEXT_WINDOW': (116+12)*1024,",
             "'LOCAL_ANALYZER_CONTEXT_WINDOW': (116+16)*1024,  # arm o16: trim budget unchanged")]
# No token yield at all (tool_agent.py: <= 0 disables it). A turn then ends only when an action runs, the solver
# asks it to stop, or a request times out; the priority gate is held per game and handed over on trims, never at a
# yield, so scheduling is unchanged. y8 still yields on 196 of 1,317 turns (l1a1y8sv2 v1).
YIELD0 = [(ENV, "'LOCAL_ANALYZER_YIELD_TOKENS': '2048',",
           "'LOCAL_ANALYZER_YIELD_TOKENS': '0',  # arm y0: never yield on tokens")]


# A probe answers "does it boot and stay up under load", never "is it better": 300 s per game.
PROBE = [(CUSTOM, "bm.solver.max_runtime_s_per_game = 7200  # replica",
          "bm.solver.max_runtime_s_per_game = 300  # PROBE ONLY, not a bench; replica")]
# A smoke run long enough for games to reach their first trim (~66K generated tokens) and hand
# their slots over, so scheduler arms exercise both the fresh-admission and handover paths.
SMOKE = [(CUSTOM, "bm.solver.max_runtime_s_per_game = 7200  # replica",
          "bm.solver.max_runtime_s_per_game = 1800  # SMOKE ONLY, not a bench; replica")]
# Long enough for games to outlive a watchdog relaunch (the 10-11 rehearsal's took 1,596 s) and keep playing.
REHEARSE = [(CUSTOM, "bm.solver.max_runtime_s_per_game = 7200  # replica",
             "bm.solver.max_runtime_s_per_game = 2700  # WATCHDOG REHEARSAL ONLY, not a bench; replica")]

DPRIME_SOURCE = HERE.parents[1] / "docs/research/2026-10-04-climb/evidence/kernels/shiiin9_form_priority_dprime.py"


def dprime() -> list[tuple[int, str, str]]:
    """D-prime slot priority (shiiin9, team AFF AI CLUB, public notebook
    affectify-arc-dfranzen-form-d1), embedded verbatim and installed in cell 19 before the run, so it
    is active in the scored run too. Priority A*M*C + B*phi with an additive continuation value, a
    linear stall decay and a 40% tail fade; unstarted games are priced by the formula instead of
    jumping the queue. solution/analysis/scheduler_sim.py puts it about +2.8 public game score above
    Franzen's live rule in the hidden-run regime (110 games, 14 slots); the replica sees half that."""
    code = DPRIME_SOURCE.read_text()
    anchor = "# show config\nprint('benchmark.solver:', bm.solver)"
    block = ("# arm dprime: D-prime slot priority (shiiin9 / AFF AI CLUB public notebook), verbatim; scored run too\n"
             "import sys as _fp_sys, types as _fp_types\n"
             "_fp_mod = _fp_types.ModuleType('ours_form_priority')\n"
             "_fp_sys.modules['ours_form_priority'] = _fp_mod\n"
             "exec(compile(" + repr(code) + ", 'ours_form_priority.py', 'exec'), _fp_mod.__dict__)\n"
             "import inference.agent.tool_agent as _fp_ta\n"
             "import inference.framework.solver as _fp_solver\n"
             "_fp_mod.install_d(_fp_ta, _fp_solver)\n\n")
    return [(CUSTOM, anchor, block + anchor)]


def overlay(name: str, src: Path = OVERLAYS) -> list[tuple[int, str, str]]:
    """Run <src>/<name>.py (franzen_overlays by default) on the patched bundle, right after the harness patch."""
    code = (src / f"{name}.py").read_text()
    marker = "print('harness patch applied successfully')"
    return [(ENV, marker, marker + "\n# arm overlay: " + name + "\nexec(compile(" + repr(code)
             + ", " + repr(f"franzen_overlays/{name}.py") + ", 'exec'), {'BUNDLE_DIR': BUNDLE_DIR})")]


def overlay_last(name: str, src: Path = OVERLAYS) -> list[tuple[int, str, str]]:
    """Run <src>/<name>.py AFTER every other overlay (right after wastefix's exec line), for an overlay whose anchors
    are text that earlier overlays write (overlay() puts the newest overlay first)."""
    code = (src / f"{name}.py").read_text()
    anchor = "'franzen_overlays/wastefix.py', 'exec'), {'BUNDLE_DIR': BUNDLE_DIR})"
    return [(ENV, anchor, anchor + "\n# arm overlay: " + name + "\nexec(compile(" + repr(code)
             + ", " + repr(f"franzen_overlays/{name}.py") + ", 'exec'), {'BUNDLE_DIR': BUNDLE_DIR})")]


def mamba(slots: int) -> list[tuple[int, str, str]]:
    """SGLang caps running requests at max_mamba_cache_size // 5 (10-01 probe: 48 slots -> 9),
    so any arm above 12 streams needs 5 slots per stream. About 54 MB of VRAM per slot."""
    return [(SERVE, "    MAMBA_CACHE=60,", f"    MAMBA_CACHE={slots},  # arm: 5 slots per admitted stream")]


# Cell 15, after the install and target validation and before the server launch: where
# serving arms patch the installed SGLang.
TARGET_ANCHOR = ('print("Target:", MODEL_DIR, "| shards:", len(target_shards), "| PLE dtype:", '
                 'target_cfg.get("text_config", {}).get("ple_embedding_dtype", "checkpoint default (typically BF16)"))')


def relaxed_accept(threshold: float) -> list[tuple[int, str, str]]:
    """Lossy MTP acceptance: a draft token the target rates at or above the threshold is always
    taken. Gabriel Olympie's serve.sh measured +13% decode at 0.5 and +29% at 0.3 (one stream);
    Franzen runs 1.0/1.0 (lossless). Gate it on levels, not on the server log."""
    return [(SERVE, "    SPEC_ACCEPT_SINGLE=1.0,\n    SPEC_ACCEPT_ACC=1.0,",
             f"    SPEC_ACCEPT_SINGLE={threshold},  # arm: relaxed acceptance\n    SPEC_ACCEPT_ACC={threshold},")]


def p448() -> list[tuple[int, str, str]]:
    """C1: REAP keep-448 routed experts. Frees ~7.3 GiB of expert weights, which SGLang gives to
    the KV pool at the same memory fraction (1.01M -> ~1.6M tokens). franzen_overlays/p448_serve.py."""
    code = (OVERLAYS / "p448_serve.py").read_text()
    anchor = TARGET_ANCHOR
    return [(SERVE, anchor, anchor + "\n# arm p448: serve 448 of 512 routed experts (REAP)\n_p448 = {}\n"
             + "exec(compile(" + repr(code) + ", 'franzen_overlays/p448_serve.py', 'exec'), _p448)\n"
             + "MODEL_DIR = _p448['apply'](MODEL_DIR, PREFIX, VENV, env)")]


SERVE_END = '          f"releasing the benchmark with the server still loading")'  # last line of cell 15


def fastdec() -> list[tuple[int, str, str]]:
    """A6: Pennyroyal's own online MXFP8 switch for the BF16 projections, with the HC mix and
    lm_head kept BF16 (their FP8 paths dequantize above 16 / 32 rows; verify runs 40 at 10
    streams) and the MXFP8 GEMMs autotuned. ~2.7 GiB more KV at the same memory fraction; prefill
    gets slower, not faster. After the READY wait a bench run stops unless the server is up inside
    the 720 s startup deadline and serve.log shows the autotune finished (the scored run only warns).
    franzen_overlays/fastdec_serve.py."""
    code = (OVERLAYS / "fastdec_serve.py").read_text()
    return [(SERVE, TARGET_ANCHOR, TARGET_ANCHOR + "\n# arm fastdec: online MXFP8 projections, HC mix and lm_head kept BF16\n"
             + "_fastdec = {}\nexec(compile(" + repr(code) + ", 'franzen_overlays/fastdec_serve.py', 'exec'), _fastdec)\n"
             + "_fastdec['apply'](VENV, env)"),
            (SERVE, SERVE_END, SERVE_END + "\n_fastdec['check'](LOG, log_offset, url, strict=not TRUE_SUBMISSION)"
             "  # arm fastdec: READY by 720 s and autotuned, or a bench run stops")]


def replycache_diag() -> list[tuple[int, str, str]]:
    """B1 serving diagnostic, bench runs only: one serve.log line per finished request with the depth
    it left in the radix cache, and the git blob of the chat template the server gets
    (franzen_overlays/replycache_serve.py; solution/analysis/reply_reuse.py reads the lines). The
    scored run leaves SGLang untouched. The B1 fix itself is overlay("replycache")."""
    code = (OVERLAYS / "replycache_serve.py").read_text()
    return [(SERVE, TARGET_ANCHOR, TARGET_ANCHOR + "\n# arm replycache: one serve.log line per finished request, bench runs only\n"
             + "env['ARC3_REPLYCACHE_DIAG'] = '0' if TRUE_SUBMISSION else '1'\n"
             + "_replycache = {}\nexec(compile(" + repr(code) + ", 'franzen_overlays/replycache_serve.py', 'exec'), _replycache)\n"
             + "_replycache['apply'](VENV, env, MODEL_DIR)")]


def agentfix() -> list[tuple[int, str, str]]:
    """B4 / A1 agent fixes, franzen_overlays/agentfix.py: RESET exposed (EXPOSE_RESET=on) and skipped
    when its level is already at its start (no action spent, the rest of the snippet runs), the stale
    guard's catch contract (6 waived no-ops once per snippet), and level_starts / attempt_starts as REPL
    globals. Scored run too. Gate on the behavior counters in solution/analysis/agentfix_counters.py,
    not on levels."""
    return [(ENV, "    'EXPOSE_UNDO': 'on',",
             "    'EXPOSE_UNDO': 'on',\n    'EXPOSE_RESET': 'on',  # arm agentfix: RESET behind the level-start guard")] + overlay("agentfix")


LAUNCH_ANCHOR = "# ---- launch detached and wait for health ----\n"  # cell 15, after the argv is assembled


def thinkbudget(tokens: int) -> list[tuple[int, str, str]]:
    """B5: SGLang's strict thinking budget. --enable-strict-thinking plus SGLANG_MAX_THINK_TOKENS forces
    </think> at the budget through the grammar mask that NEXTN verify applies (a custom logit processor
    never runs there), and Qwen3Detector's think exclusions are emptied so only the budget changes.
    After the READY wait a bench run sends one request with a 64-token budget, which must stop thinking
    by 80 reasoning tokens and still parse, or the bench stops. A server still loading at the deadline
    skips it rather than holding the games' clocks; the scored run skips it.
    franzen_overlays/thinkbudget_serve.py; counters: solution/analysis/thinking_counters.py."""
    code = (OVERLAYS / "thinkbudget_serve.py").read_text()
    return [(SERVE, LAUNCH_ANCHOR, f"# arm thinkbudget: strict thinking budget of {tokens} tokens, think exclusions emptied\n"
             + "_thinkbudget = {}\nexec(compile(" + repr(code) + ", 'franzen_overlays/thinkbudget_serve.py', 'exec'), _thinkbudget)\n"
             + f"args = _thinkbudget['apply'](VENV, env, args, {tokens})\n" + LAUNCH_ANCHOR),
            (SERVE, SERVE_END, SERVE_END + "\n_thinkbudget['check'](url, proc, CFG['SERVED_NAME'], strict=not TRUE_SUBMISSION)"
             "  # arm thinkbudget: 64-token self-test on a bench run")]


NVFP4KV = OVERLAYS / "nvfp4kv"
NVFP4KV_PATCHES = ("0001-qsa-nvfp4-kv.patch", "0002-optional-mamba-pingpong-fix.patch")


def nvfp4kv(max_total_tokens: int) -> list[tuple[int, str, str]]:
    """B3: Lesilva's QSA NVFP4 KV patches (vendored with provenance in franzen_overlays/nvfp4kv/) and the
    flags they need. Cell 15 passes one KVDTYPE to both KV flags; this splits them: target nvfp4, MTP draft
    fp8_e4m3 (the draft flag rejects nvfp4). A token drops from 14,080 to 8,704 bytes, so the pool goes from
    1.53M FP8 tokens to max_total_tokens (the wheel's own ceiling is about 2.1M at 100 Mamba slots; the cap
    only lowers it). apply() patches SGLang before it returns the dtype, so a failed patch never launches
    nvfp4. In a scored run, an NVFP4 server that exits before READY is relaunched once on FP8 KV inside the
    cell's READY wait, so the draw is not played against a dead server. franzen_overlays/nvfp4kv_serve.py."""
    code = (OVERLAYS / "nvfp4kv_serve.py").read_text()
    patches = {name: (NVFP4KV / name).read_text() for name in NVFP4KV_PATCHES}
    died = "    if proc.poll() is not None:\n        show_log_tail()\n        break"  # cell 15's READY wait
    return [
        (SERVE, '    KVDTYPE="fp8_e4m3",',
         '    KVDTYPE="fp8_e4m3",  # arm nvfp4kv: nvfp4 once apply() below has patched SGLang\n'
         '    DRAFT_KVDTYPE="fp8_e4m3",  # arm nvfp4kv: the MTP draft keeps FP8 (its flag rejects nvfp4)\n'
         '    MAX_TOTAL_TOKENS=None,  # arm nvfp4kv: KV pool cap, set by apply()'),
        (SERVE, '"--speculative-draft-kv-cache-dtype", CFG["KVDTYPE"],',
         '"--speculative-draft-kv-cache-dtype", CFG["DRAFT_KVDTYPE"],'),
        (SERVE, 'if CFG["GDN_MTP_CACHE_MODE"]: args += ["--gdn-mtp-cache-mode", CFG["GDN_MTP_CACHE_MODE"]]',
         'if CFG["GDN_MTP_CACHE_MODE"]: args += ["--gdn-mtp-cache-mode", CFG["GDN_MTP_CACHE_MODE"]]\n'
         'if CFG["MAX_TOTAL_TOKENS"]: args += ["--max-total-tokens", str(CFG["MAX_TOTAL_TOKENS"])]  # arm nvfp4kv'),
        (SERVE, TARGET_ANCHOR, TARGET_ANCHOR + "\n# arm nvfp4kv: QSA NVFP4 KV patches, then the split KV dtypes and the pool cap\n"
         + "_nvfp4kv = {}\nexec(compile(" + repr(code) + ", 'franzen_overlays/nvfp4kv_serve.py', 'exec'), _nvfp4kv)\n"
         + f"CFG.update(_nvfp4kv['apply'](VENV, env, {int(max_total_tokens)}, " + repr(patches) + "))"),
        (SERVE, died,
         "    if proc.poll() is not None:\n        show_log_tail()\n"
         '        if TRUE_SUBMISSION and "nvfp4" in args:  # arm nvfp4kv: NVFP4 died before READY, relaunch once on FP8 KV\n'
         "            proc, args = _nvfp4kv['relaunch_fp8'](proc, args, env, LOG, int(SERVED_MODEL_PORT))\n"
         "            continue\n"
         "        break"),
    ]


def nvfp4kv_parity() -> list[tuple[int, str, str]]:
    """Probe only, never in a scored rerun: cell 15 boots the FP8 reference first; parity() scores 32 recorded
    prompts on it, restarts the server on NVFP4, scores them again and stops the run unless teacher-forced
    logprobs agree: NVFP4 NLL at most 0.02 nats/token above FP8 (the signed change Lesilva reported; mean
    |dlogp| is printed, not gated), top-1 agreement at least 95%, greedy decode agreeing for 2+ tokens."""
    prompts = (NVFP4KV / "parity_prompts.json").read_text()
    launch = "precache_model_thread.start()"
    return [(SERVE, launch, "args_nvfp4 = args\nif not TRUE_SUBMISSION:  # arm nvfp4kv probe: the FP8 reference boots first\n"
                            "    args = _nvfp4kv['reference_args'](args)\n" + launch),
            (SERVE, SERVE_END, SERVE_END + "\nif not TRUE_SUBMISSION:  # arm nvfp4kv probe: FP8 vs NVFP4 logprob parity, raises on failure\n"
             "    proc, log_offset = _nvfp4kv['parity'](proc, args_nvfp4, env, LOG, url, json.loads(" + repr(prompts) + "))\n"
             "    args = args_nvfp4")]


RUN_ANCHOR = "await bm.run(soft_end_time=soft_end, runtime_environment=target, minimal_diagnostics=TRUE_SUBMISSION)\n"


def failclosed() -> list[tuple[int, str, str]]:
    """Fail closed (franzen_overlays/failclosed.py): a scored run that ends with fewer than 20 levels
    across all games raises, so Kaggle records an error (which does not spend the daily slot) instead
    of banking a near-zero score; daily_submit.sh then retries. Bench and probe runs only print the
    summary. Cell 23, right after bm.run returns."""
    code = (OVERLAYS / "failclosed.py").read_text()
    return [(RUN, RUN_ANCHOR, RUN_ANCHOR
             + "# arm failclosed: a dead scored run raises (an error does not spend the slot); bench runs print only\n"
             + "_failclosed = {}\nexec(compile(" + repr(code) + ", 'franzen_overlays/failclosed.py', 'exec'), _failclosed)\n"
             + "if TRUE_SUBMISSION:\n    _failclosed['check'](bm.game_runs)\n"
             + "else:\n    print('failclosed (not enforced off the scored path):', _failclosed['summarize'](bm.game_runs), flush=True)\n")]


def watchdog(kill_at: int | None = None) -> list[tuple[int, str, str]]:
    """Server watchdog (franzen_overlays/watchdog.py): after the READY wait (cell 15) a daemon thread
    relaunches the model server if it exits, and requests to the dead endpoint retry in place for 30 min,
    so a mid-run crash (two CUDA OOMs after READY on 2026-10-11) costs a relaunch instead of the rest of
    the run. kill_at rehearses it in a bench build by killing the server once that many seconds after
    READY; it is forced off in a scored run. Cell 23 stops the watch once bm.run returns."""
    code = (OVERLAYS / "watchdog.py").read_text()
    kill = "None" if kill_at is None else f"(None if TRUE_SUBMISSION else {int(kill_at)})"
    return [(SERVE, SERVE_END, SERVE_END + "\n# arm watchdog: relaunch the server if it dies mid-run; requests retry in place\n"
             + "_watchdog = {}\nexec(compile(" + repr(code) + ", 'franzen_overlays/watchdog.py', 'exec'), _watchdog)\n"
             + f"_watchdog_record = _watchdog['start'](globals(), args, env, LOG, url, kill_at={kill}, "
             + "precache=globals().get('precache'), precache_paths=(MODEL_DIR, DRAFT_MODEL_DIR))"),
            (RUN, RUN_ANCHOR, RUN_ANCHOR + "_watchdog_record['stop'].set()  # arm watchdog: the run is over, never relaunch now\n"
             + "print('watchdog:', _watchdog_record['restarts'], 'relaunches', _watchdog_record['events'], flush=True)\n")]


DRAFT_ANCHOR = "    DRAFT_VIEW = prepare_draft_view(MODEL_DIR, DRAFT_MODEL_DIR, PREFIX)\n"
# Inputs an overlay reads, by mount path: build() adds the dataset to kernel-metadata when an edit names the path.
OUR_DATASETS = {"/kaggle/input/datasets/vineethsaivs/flashnext-mtp-d3/": "vineethsaivs/flashnext-mtp-d3"}


def d3draft() -> list[tuple[int, str, str]]:
    """ursuciprian's D3 refit of the drafter's 29 dense mtp.* tensors (franzen_overlays/d3draft_serve.py), spliced
    into a local copy of the draft view's mtp-dense.safetensors right after prepare_draft_view (cell 15). Lossless:
    the target verifies every token; only acceptance can change (+8.1% accepted drafts in ursuciprian's screen)."""
    code = (OVERLAYS / "d3draft_serve.py").read_text()
    return [(SERVE, DRAFT_ANCHOR, DRAFT_ANCHOR + "    # arm d3draft: D3 refit of the 29 dense MTP drafter tensors\n"
             + "    _d3draft = {}\n    exec(compile(" + repr(code) + ", 'franzen_overlays/d3draft_serve.py', 'exec'), _d3draft)\n"
             + "    _d3draft['apply'](DRAFT_VIEW)\n")]


HOTMAP = GEN / "arc-hot-map-2026-10-12.json"  # build_arc_hot_map.py over the y8-family transcripts
FRSPEC_MAP = ('        tok = find_unique(WHEELHOUSE_DIR, "hot_tokens_64k.pt", required=False)\n'
              '        if tok is None: tok = find_unique(WHEELHOUSE_DIR, "flash-next-64k.pt")\n'
              '        assert sha256(tok) == TOKEN_MAP_SHA, "Wrong FR-Spec map; use the map from the pinned Pennyroyal bundle"\n')


def hotmap(size: int) -> list[tuple[int, str, str]]:
    """The ARC traffic draft vocabulary (franzen_overlays/hotmap_serve.py) in place of Pennyroyal's 64K FR-Spec map,
    embedded as a bitmap (no new dataset). Held out by game it misses 0.11% (32K) or 0.04% (64K) of generated tokens
    against the served map's 1.42%. Lossless; the tokenizer assert stays."""
    m = json.loads(HOTMAP.read_text())["maps"][str(size)]
    code = (OVERLAYS / "hotmap_serve.py").read_text()
    return [(SERVE, FRSPEC_MAP, f"        # arm hotmap: ARC traffic draft vocabulary, {size} ids\n        _hotmap = {{}}\n"
             "        exec(compile(" + repr(code) + ", 'franzen_overlays/hotmap_serve.py', 'exec'), _hotmap)\n"
             f"        tok = _hotmap['install']({m['bitmap_zlib_b64']!r}, {m['bitmap_sha256']!r}, {size}, PYTHON, run, "
             "install_env)\n")]


def leantrim(keep: int = 3) -> list[tuple[int, str, str]]:
    """At a trim, reasoning and images older than the last `keep` turns become markers (franzen_overlays/leantrim.py;
    the trim track's prototype, docs/research/2026-10-12-next/trim-economy.md). Changes what the model sees: run only
    after the trimtest decision test passes (gate-predeclared-2026-10-12-trimtest.json)."""
    return [(ENV, "    'ARC3_CONTEXT_DRAIN_TOKENS': 58*1024,",
             f"    'ARC3_CONTEXT_DRAIN_TOKENS': 58*1024,\n    'ARC3_LEAN_KEEP_TURNS': '{keep}',  # arm leantrim")] \
        + overlay("leantrim")


TRIMTEST_PAIRS = "/kaggle/input/datasets/vineethsaivs/arc3-trimtest-pairs/"
OUR_DATASETS[TRIMTEST_PAIRS] = "vineethsaivs/arc3-trimtest-pairs"


def trimtest(pairs_file: str = "pairs.jsonl", samples: int = 2) -> list[tuple[int, str, str]]:
    """Dev kernel only: in place of the benchmark, sample the next reply at recorded post-trim states with and
    without slimming (franzen_overlays/trimtest_replay.py, pairs from solution/analysis/trim_pairs.py). Refuses to
    run on a scored path."""
    code = (OVERLAYS / "trimtest_replay.py").read_text()
    return [(RUN, RUN_ANCHOR, "if TRUE_SUBMISSION:\n    raise RuntimeError('trimtest is a dev kernel, never a submission')\n"
             "# dev trimtest: the leaner-trim decision test replaces the benchmark\n_trimtest = {}\n"
             "exec(compile(" + repr(code) + ", 'franzen_overlays/trimtest_replay.py', 'exec'), _trimtest)\n"
             f"_trimtest['run']({TRIMTEST_PAIRS + pairs_file!r}, f'http://127.0.0.1:{{SERVED_MODEL_PORT}}', "
             f"SERVED_MODEL_NAME, WORKING_DIR, samples={samples})\n")]


LORA = HERE.parents[1] / "solution/lora"
REARC_DIR = "/kaggle/input/datasets/vineethsaivs/rearc127/environment_files"


LORA_ADAPTER = "/kaggle/input/datasets/vineethsaivs/arc3-lora-r16/"  # adapter_model.safetensors + adapter_config.json
OUR_DATASETS[LORA_ADAPTER] = "vineethsaivs/arc3-lora-r16"
GRPO_ADAPTER = "/kaggle/input/datasets/vineethsaivs/arc3-lora-grpo-r1/"  # GRPO round 1 (solution/lora/grpo/GATES.md)
OUR_DATASETS[GRPO_ADAPTER] = "vineethsaivs/arc3-lora-grpo-r1"


def lora(adapter: str = LORA_ADAPTER, scale: float | None = None, sr_seed: int | None = None,
         set_scale: bool = False) -> list[tuple[int, str, str]]:
    """Arm lora: fold the dense r16 LoRA into the BF16 weights while the server loads (solution/lora/lora_merge_serve.py;
    same load_weights generator hook as p448, so kernels, MTP and the D3 drafter run unchanged). The hook goes in right
    before the launch (after p448), SGLANG_LORA_DIR goes into the server env (the watchdog relaunches with it), and the
    arm fails closed: a missing adapter file, a hook that does not install, or a server log without the merge line
    ('lora: merged N tensors') raises, so a run that silently served the base model never scores. With scale, every
    merge line must also name that scale (lora_alpha / r from the adapter's config), so a mis-scaled adapter raises.
    With sr_seed (opt-in; None leaves every edit byte-identical), the merge rounds stochastically with that seed
    (solution/lora/lora_merge_sr.py patches the merge source before install) and the server log must say so.
    With set_scale (opt-in, round C WE3), the adapter is served at `scale`: the merge reads a /tmp copy of the config
    with lora_alpha = scale * r (the weights are symlinked), so the dataset itself stays unchanged."""
    code = (LORA / "lora_merge_serve.py").read_text()
    pre = ("# arm lora: fold the LoRA adapter into the BF16 weights at load (after p448, before launch); fail closed\n"
           f"_lora_dir = {adapter!r}\n"
           "for _f in ('adapter_model.safetensors', 'adapter_config.json'):\n"
           "    if not os.path.isfile(os.path.join(_lora_dir, _f)):\n"
           "        raise RuntimeError(f'arm lora: missing {_lora_dir}{_f}')\n"
           "_lora = {}\nexec(compile(" + repr(code) + ", 'solution/lora/lora_merge_serve.py', 'exec'), _lora)\n")
    if set_scale:
        assert scale is not None, "set_scale needs a scale"
        pre += ("_lora_cfg = __import__('json').load(open(os.path.join(_lora_dir, 'adapter_config.json')))\n"
                "if _lora_cfg.get('use_rslora'):\n"
                "    raise RuntimeError('arm lora: set_scale does not handle rslora')\n"
                f"_lora_cfg['lora_alpha'] = {float(scale)!r} * float(_lora_cfg['r'])\n"
                "_lora_src, _lora_dir = _lora_dir, '/tmp/arc3_lora_scaled/'\n"
                "os.makedirs(_lora_dir, exist_ok=True)\n"
                "open(os.path.join(_lora_dir, 'adapter_config.json'), 'w').write(__import__('json').dumps(_lora_cfg))\n"
                "_lora_w = os.path.join(_lora_dir, 'adapter_model.safetensors')\n"
                "if os.path.lexists(_lora_w):\n"
                "    os.remove(_lora_w)\n"
                "os.symlink(os.path.join(_lora_src, 'adapter_model.safetensors'), _lora_w)\n")
    if sr_seed is not None:
        sr_code = (LORA / "lora_merge_sr.py").read_text()
        pre += ("_lora_sr = {}\nexec(compile(" + repr(sr_code) + ", 'solution/lora/lora_merge_sr.py', 'exec'), _lora_sr)\n"
                f"_lora['REMAP_SRC'] = _lora_sr['patch_src'](_lora['REMAP_SRC'], {int(sr_seed)})\n")
    pre += ("_lora_py = _lora['patch_sglang'](VENV)\n"
            "if _lora['LINE'] not in _lora_py.read_text():\n"
            "    raise RuntimeError('arm lora: the merge hook is not in the installed SGLang')\n")
    if sr_seed is not None:
        pre += ("if 'arm lora sr' not in _lora_py.read_text():\n"
                "    raise RuntimeError('arm lora: stochastic rounding is not in the installed SGLang')\n")
    pre += ("env['SGLANG_LORA_DIR'] = _lora_dir\n"
            "print('LORA PATCHED', _lora_py, flush=True)\n")
    # The merge line is written at the end of load_weights, before READY. A slow node can still be loading at the
    # 722 s release (two WB probes, 10-21: shards at 10 s each), so the check waits for READY in a thread there and
    # kills the kernel (the run errors, no score banked) if the server answers without the merge line or at a wrong scale.
    scales = "None" if scale is None else f"{{{float(scale)!r}}}"
    post = ("\ndef _lora_check(_text):  # arm lora: the merge must have run, at the expected scale\n"
            "    if 'lora: merged' not in _text:\n"
            "        return 'arm lora: server READY but no merge line in serve.log; refusing to play the base model'\n"
            "    _s = {float(_v) for _v in __import__('re').findall(r'lora: merged \\d+ tensors, scale ([0-9.]+)', _text)}\n"
            f"    if {scales} is not None and _s != {scales}:\n"
            "        return f'arm lora: merge scale {sorted(_s)}; refusing to play a mis-scaled adapter'\n")
    if sr_seed is not None:
        post += (f"    if 'lora: stochastic rounding, seed {int(sr_seed)}' not in _text:\n"
                 "        return 'arm lora: no stochastic rounding line in serve.log; refusing to play a nearest-rounded merge'\n")
    post += ("def _lora_ready():\n"
            "    try:\n"
            "        return __import__('urllib.request').request.urlopen('http://127.0.0.1:8001/health', timeout=5).status == 200\n"
            "    except Exception:\n"
            "        return False\n"
            "if 'lora: merged' in Path(LOG).read_text(errors='replace') or _lora_ready():\n"
            "    _lora_err = _lora_check(Path(LOG).read_text(errors='replace'))\n"
            "    if _lora_err:\n"
            "        raise RuntimeError(_lora_err)\n"
            "else:\n"
            "    def _lora_guard():\n"
            "        while 'lora: merged' not in Path(LOG).read_text(errors='replace') and not _lora_ready():\n"
            "            time.sleep(5)\n"
            "        _err = _lora_check(Path(LOG).read_text(errors='replace'))\n"
            "        if _err:\n"
            "            print(_err + '; killing the kernel so the run fails closed', flush=True)\n"
            "            os.kill(os.getpid(), 9)\n"
            "            return\n"
            "        print('arm lora: merge confirmed after release', flush=True)\n"
            "    __import__('threading').Thread(target=_lora_guard, daemon=True).start()\n"
            "    print('arm lora: server still loading at release; merge check deferred to READY', flush=True)")
    return [(SERVE, LAUNCH_ANCHOR, pre + LAUNCH_ANCHOR), (SERVE, SERVE_END, SERVE_END + post)]


def without(edits: list[tuple[int, str, str]], drop: list[tuple[int, str, str]]) -> list[tuple[int, str, str]]:
    """edits minus each edit in drop, each of which must appear exactly once (a twin arm differs in those edits only)."""
    for e in drop:
        if edits.count(e) != 1:
            raise SystemExit(f"without(): edit appears {edits.count(e)} times, expected 1: {e[1][:60]!r}")
    return [e for e in edits if e not in drop]


def reqlog() -> list[tuple[int, str, str]]:
    """DEV ONLY (LoRA data generation, solution/lora/reqlog.py): log every request's exact message list. Raises at the
    top of the launcher cell on a scored path, before any server starts. Output dir comes from ARC3_REQLOG_DIR."""
    return [(ENV, "NOTEBOOK_START_EPOCH = time.time()\n",
             "NOTEBOOK_START_EPOCH = time.time()\nif TRUE_SUBMISSION:\n"
             "    raise RuntimeError('reqlog/rearc is a dev data-generation arm, never a submission')\n")] \
        + overlay("reqlog", LORA)


def rearc(env_dir: str = REARC_DIR) -> list[tuple[int, str, str]]:
    """DEV ONLY: play RE-ARC-3 environments (the file named by env ARC3_REARC_GAMES, a JSON list of game ids) in place
    of the public 25. One thread per game; the shared window is 288 s per game (4,032 slot-s per game: 14 streams x
    7,200 s / 25 games, the replica's). arc_agi 0.9.8 execs a game's source into a module it never registers, which
    breaks module-level @dataclass in many RE-ARC-3 games (reset returns None), so the loader is wrapped to register
    it first (same fix as solution/sim/test_offline_envs.py). Notebook budget raised to 11 h. env_dir: another
    environment_files tree (external games) laid out as RE-ARC-3's."""
    patch = (
        "    # datagen: RE-ARC-3 games in place of the public 25\n"
        "    import importlib.util as _iu\n"
        "    from arc_agi.local_wrapper import LocalEnvironmentWrapper as _LEW\n"
        "    _orig_load = _LEW._load_game_class\n"
        "    def _load_registered(self, seed=0):\n"
        "        _name = f'arc_agi_3.{self.environment_info.game_id}'\n"
        "        if _name not in sys.modules:\n"
        "            sys.modules[_name] = _iu.module_from_spec(_iu.spec_from_loader(_name, loader=None))\n"
        "        _orig_load(self, seed)\n"
        "    _LEW._load_game_class = _load_registered\n"
        f"    bm.games = [g for g in _offline_games({env_dir!r}) if g.env_name in set(_REARC_ALL)]\n"
        "    assert len(bm.games) == _REARC_N, f'opened {len(bm.games)} of {_REARC_N} RE-ARC-3 games'\n")
    return [
        (CUSTOM, "bm.solver.analyzer_timeout = 900.0\n",
         "bm.solver.analyzer_timeout = 900.0\n"
         "import json as _rj\n_REARC_ALL = _rj.load(open(os.environ['ARC3_REARC_GAMES']))\n_REARC_N = len(_REARC_ALL)\n"),
        (CUSTOM, "bm.solver.concurrency = (25 - len(demo_excluded_games)) * bm.n_passes",
         "bm.solver.concurrency = _REARC_N * bm.n_passes  # datagen: one thread per RE-ARC-3 game"),
        (CUSTOM, "bm.solver.max_runtime_s_per_game = 7200  # replica",
         "bm.solver.max_runtime_s_per_game = 288 * _REARC_N  # datagen, 4,032 slot-s per game; was replica"),
        (RUN, '    competition_env_files = str(Path(FORK_WHEELHOUSE).parent / "environment_files")\n'
              '    bm.games = _offline_games(competition_env_files)\n', patch),
        (RUN, '    budget = float(getattr(target, "max_runtime_s", 0.0) or 0.0)\n',
         '    target.max_runtime_s = 39600.0  # datagen: 11 h notebook budget\n'
         '    budget = float(getattr(target, "max_runtime_s", 0.0) or 0.0)\n'),
    ]


PATHS_BLOCK = ("WHEELHOUSE_DIR    = '/kaggle/input/datasets/dfranzen/pennyroyal-v253'\n"
               "MODEL_DIR         = '/kaggle/input/models/dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/transformers/default/1'\n"
               "DRAFT_MODEL_DIR   = '/kaggle/input/models/dfranzen/albucino-qwen3-8-flash-next-drafter/transformers/default/1'\n"
               "ORIG_BUNDLE_DIR   = '/kaggle/input/datasets/dfranzen/taaf-kaggle-source-bundle-copy'\n")
INPUT_COPY = "vineethsaivs/arc3-input-copy-a"
MIRROR_MODELS = ["jamalsaeedi/intel-qwen3-8-flash-next-w4a16-autoround/Transformers/default/1",
                 "jamalsaeedi/intel-qwen3-8-flash-next-w4a16-autoround/Transformers/ple/1"]


def fallback_inputs() -> list[tuple[int, str, str]]:
    """Run on our own inputs instead of Franzen's (rival-input-fallbacks-2026-10-10.json): the wheelhouse, drafter and
    TAAF bundle from our CPU kernel arc3-input-copy-a v1, the model from the public mirror's two instances joined by a
    symlink view. Every model file must match the original's name and size (fallback-model-files-2026-10-12.json)
    before anything loads; off the scored path every copied file is also checked against its recorded sha256
    (input-copy-manifest-2026-10-11.json). build() swaps the kernel's dataset, model and kernel sources."""
    sizes = json.loads((GEN / "fallback-model-files-2026-10-12.json").read_text())["files"]
    manifest = {name: {f: v["sha256"] for f, v in block["files"].items()}
                for name, block in json.loads((GEN / "input-copy-manifest-2026-10-11.json").read_text()).items()}
    code = (
        "# arm fallback: our own copies of Franzen's inputs and the public model mirror\n"
        "def _fallback_dir(*candidates):\n"
        "    for candidate in candidates:\n"
        "        if Path(candidate).is_dir():\n"
        "            return Path(candidate)\n"
        "    raise RuntimeError(f'fallback inputs: none of {candidates} is mounted')\n"
        f"_COPY = _fallback_dir('/kaggle/input/notebooks/{INPUT_COPY}', '/kaggle/input/{INPUT_COPY.split('/')[1]}')\n"
        "WHEELHOUSE_DIR    = str(_COPY / 'pennyroyal-v253')\n"
        "DRAFT_MODEL_DIR   = str(_COPY / 'albucino-drafter')\n"
        "ORIG_BUNDLE_DIR   = str(_COPY / 'taaf-kaggle-source-bundle-copy')\n"
        "_MIRROR = Path('/kaggle/input/models/jamalsaeedi/intel-qwen3-8-flash-next-w4a16-autoround/transformers')\n"
        "MODEL_DIR = '/tmp/fallback-model'\n"
        "Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)\n"
        "for _instance in ('default/1', 'ple/1'):\n"
        "    for _file in sorted((_MIRROR / _instance).iterdir()):\n"
        "        if not (Path(MODEL_DIR) / _file.name).exists():\n"
        "            (Path(MODEL_DIR) / _file.name).symlink_to(_file)\n"
        f"_expected_sizes = {sizes!r}\n"
        "_got_sizes = {p.name: p.stat().st_size for p in Path(MODEL_DIR).iterdir()}\n"
        "if _got_sizes != _expected_sizes:\n"
        "    raise RuntimeError('fallback inputs: the model view differs from the original by name or size: '\n"
        "                       + str(sorted(set(_got_sizes.items()) ^ set(_expected_sizes.items()))[:6]))\n"
        "if os.environ.get('KAGGLE_IS_COMPETITION_RERUN', '').strip().lower() not in {'1', 'true'}:\n"
        "    import hashlib\n"
        f"    _manifest = {manifest!r}\n"
        "    _bad = []\n"
        "    for _name, _files in _manifest.items():\n"
        "        for _rel, _want in _files.items():\n"
        "            _h = hashlib.sha256()\n"
        "            with open(_COPY / _name / _rel, 'rb') as _f:\n"
        "                for _block in iter(lambda: _f.read(1 << 24), b''):\n"
        "                    _h.update(_block)\n"
        "            if _h.hexdigest() != _want:\n"
        "                _bad.append(_name + '/' + _rel)\n"
        "    if _bad:\n"
        "        raise RuntimeError(f'fallback inputs: sha256 mismatch in {_bad[:6]}')\n"
        "    print('fallback inputs: sha256 verified for', sum(len(v) for v in _manifest.values()), 'copied files', flush=True)\n"
        "print('fallback inputs:', _COPY, '| model view', MODEL_DIR, len(_got_sizes), 'files match the original by name and size', flush=True)\n")
    return [(ENV, PATHS_BLOCK, code)]


def fallback_metadata(meta: dict) -> dict:
    """Kernel sources for fallback_inputs(): drop Franzen's datasets and models, mount our copy and the mirror."""
    meta["dataset_sources"] = [d for d in meta["dataset_sources"] if not d.startswith("dfranzen/")]
    meta["model_sources"] = list(MIRROR_MODELS)
    meta["kernel_sources"] = [INPUT_COPY]
    return meta


def serve_overlay(name: str, label: str) -> list[tuple[int, str, str]]:
    """Patch the installed Pennyroyal SGLang in cell 15 (after the install and target validation,
    before the server launch) with franzen_overlays/<name>.py, whose apply(VENV) edits it."""
    code = (OVERLAYS / f"{name}.py").read_text()
    return [(SERVE, TARGET_ANCHOR, TARGET_ANCHOR + f"\n# arm {label}\n_{label} = {{}}\n"
             + "exec(compile(" + repr(code) + f", 'franzen_overlays/{name}.py', 'exec'), _{label})\n"
             + f"_{label}['apply'](VENV)")]


def qsaring() -> list[tuple[int, str, str]]:
    """QSA pending-ring stride 2 * ratio (franzen_overlays/qsaring_serve.py): with the shipped stride a
    4-token MTP verify window that starts mid-group overwrites the boundary group's pending index keys
    before they are compressed (192 of 256 groups in solution/sim/test_franzen_qsaring.py's replay)."""
    return serve_overlay("qsaring_serve", "qsaring")


def asyncmrope() -> list[tuple[int, str, str]]:
    """Speculative M-RoPE deltas uploaded pinned and non_blocking (franzen_overlays/asyncmrope_serve.py)."""
    return serve_overlay("asyncmrope_serve", "asyncmrope")


def qsaparity_probe() -> list[tuple[int, str, str]]:
    """Probe only, never a scored rerun (franzen_overlays/qsaparity_serve.py): after cell 15's READY wait,
    measure on the shipped SGLang how often next-token predictions after a long MTP-decoded reply differ
    between the decode-time cache and a fresh prefill, then apply the ring fix, relaunch and measure again.
    Uses 16 of the recorded parity prompts. The fixed server keeps running for the probe's short game run."""
    ring = (OVERLAYS / "qsaring_serve.py").read_text()
    code = (OVERLAYS / "qsaparity_serve.py").read_text()
    prompts = (NVFP4KV / "parity_prompts.json").read_text()
    return [(SERVE, SERVE_END, SERVE_END + "\nif not TRUE_SUBMISSION:  # probe qsaparity: shipped vs ring-fixed QSA\n"
             "    _qsaring = {}\n    exec(compile(" + repr(ring) + ", 'franzen_overlays/qsaring_serve.py', 'exec'), _qsaring)\n"
             "    _qsaparity = {}\n    exec(compile(" + repr(code) + ", 'franzen_overlays/qsaparity_serve.py', 'exec'), _qsaparity)\n"
             "    proc, log_offset = _qsaparity['probe'](proc, args, env, LOG, url, json.loads(" + repr(prompts) + "), VENV, _qsaring)")]


# The MTP draft's KV in BF16 while the target stays FP8 (vLLM #60023 on Flash-Next: accept length
# 3.215 to 3.473, decode +12%, for about 7% of the KV pool). Gate on serve.log: accept length,
# tokens/s, and KV p90/max at 14 streams.
BF16DRAFT = [(SERVE, '"--speculative-draft-kv-cache-dtype", CFG["KVDTYPE"],',
              '"--speculative-draft-kv-cache-dtype", "bf16",  # arm bf16draft: target KV stays FP8')]


# BF16 draft KV with the target pool capped so total KV bytes stay under today's (serving.md 2.2): SGLang sizes the pool
# as if the draft shared the target's FP8 dtype, so BF16DRAFT alone overshot by 1.42 GB and OOMed (l1a1y8sv3). Per token
# 11,445 B target + 949 B FP8 draft (r3 boot log) becomes 11,445 + 1,898; 1,495,872 x 12,394 / 13,343 less 1%, to a
# multiple of 64 = 1,375,552 tokens (-8%). Lossless; acts on the drafter's attention at depth 2 and beyond.
BF16DRAFTCAP = [(SERVE, '"--speculative-draft-kv-cache-dtype", CFG["KVDTYPE"],',
                 '"--speculative-draft-kv-cache-dtype", "bf16", "--max-total-tokens", "1375552",  # arm bf16c')]


SERVER_DIED = "    if proc.poll() is not None:\n        show_log_tail()\n        break"  # cell 15's READY wait


def failclosed_server() -> list[tuple[int, str, str]]:
    """Fail closed at boot: in a scored run, a model server that exits before READY raises at once. The
    unmodified cell shows the log tail and plays the whole run against the dead server: each game waits out
    its 900 s first-request grace (ARC3_HTTP_RETRY_INITIAL_SECONDS) one slot-batch at a time, then gives up
    after 10 failed turns, so the run "completes" at 0.00 within one to three hours, the shape of the 10-03
    and 10-10 zeros. Raising makes the submission error in minutes, which does not spend the slot, and the
    daily job retries. A server still loading at the 12-minute deadline is released as before (the grace
    covers it). Not combinable with nvfp4kv(), which edits the same lines."""
    return [(SERVE, SERVER_DIED,
             "    if proc.poll() is not None:\n        show_log_tail()\n"
             "        if TRUE_SUBMISSION:  # arm failclosed: never play a scored run against a dead server\n"
             "            raise RuntimeError('failclosed: the model server exited before READY; raising so this '\n"
             "                               'submission errors (no slot spent) and the daily job retries')\n"
             "        break")]


def s20() -> list[tuple[int, str, str]]:
    """B3 serving component: NVFP4 KV with a 2.0M-token pool, 20 admitted streams, 100 Mamba slots (SGLang
    runs max_mamba_cache_size // 5 requests). Replaces streams()/mamba() in whatever base it joins."""
    return nvfp4kv(2_000_000) + streams(20) + mamba(100)


ARMS = {
    "rep": ("Replica baseline: unchanged fork on the 25-game, 10-stream, 7,200 s schedule", lambda: []),
    "a12": ("A1+A2: 12 admitted streams with decode graphs 11-12, and startup trims (FAILED 10-01: cache evictions)", lambda: streams(12) + STARTUP),
    "start": ("New control: replica plus the startup trims (READY 65 s sooner on 10-01)", lambda: STARTUP),
    "kvlite": ("A3a: startup trims plus ~1 GB more KV (memory fraction 0.97)", lambda: STARTUP + KVLITE),
    "hic12": ("A3b: startup trims, 32 GB host-RAM KV tier, 12 admitted streams", lambda: STARTUP + hicache(32) + streams(12)),
    "probe-kvlite": ("PROBE: kvlite boots and holds under load (300 s per game)", lambda: STARTUP + KVLITE + PROBE),
    "probe-hic12": ("PROBE: hic12 boots, attaches the host tier and holds under load (300 s per game)", lambda: STARTUP + hicache(32) + streams(12) + PROBE),
    "hist71": ("A4: drain 46K instead of 58K, keeping ~71K of history after a trim", lambda: STARTUP + HISTORY),
    "levelmem": ("B1: startup trims plus solved-level memory", lambda: STARTUP + overlay("levelmem")),
    "audit": ("B2: startup trims plus one strategy-audit question per stalled level", lambda: STARTUP + overlay("audit")),
    "handover": ("B3: startup trims plus early handover of stuck games after 25K tokens on a level", lambda: STARTUP + overlay("handover")),
    "sandbox": ("B4: startup trims plus the parts of rellik13's Wang/Ludvig port Franzen lacks: turn openers without their "
                "static lines (the act/batch cue stays), the ignore-the-bar lines deleted", lambda: STARTUP + overlay("sandbox")),
    "p448": ("C1: startup trims, REAP keep-448 experts (~1.6M KV tokens), 12 admitted streams", lambda: STARTUP + p448() + streams(12)),
    "p448s10": ("C1 control: startup trims and REAP keep-448 experts at 10 streams", lambda: STARTUP + p448()),
    # 10-02: p448 passed (122 levels vs 92), so the agent levers now stack on it.
    "p448lm": ("p448 plus B1 solved-level memory", lambda: STARTUP + p448() + streams(12) + overlay("levelmem")),
    "p448au": ("p448 plus B2 strategy audit", lambda: STARTUP + p448() + streams(12) + overlay("audit")),
    "p448ho": ("p448 plus B3 early handover", lambda: STARTUP + p448() + streams(12) + overlay("handover")),
    "p448sb": ("p448 plus B4 opener port", lambda: STARTUP + p448() + streams(12) + overlay("sandbox")),
    # 10-03 replicates under fresh slugs (a new version of a pinned kernel would break its pin).
    "p448rep": ("Replicate of p448", lambda: STARTUP + p448() + streams(12)),
    "p448s14rep": ("Replicate of p448s14", lambda: STARTUP + p448() + streams(14) + mamba(70)),
    "p448s14ra": ("p448s14 plus relaxed MTP acceptance at 0.5", lambda: STARTUP + p448() + streams(14) + mamba(70) + relaxed_accept(0.5)),
    "p448h16": ("keep-448 experts, 16 streams, 80 Mamba slots, 32 GB host-RAM KV tier", lambda: STARTUP + p448() + hicache(32) + streams(16) + mamba(80)),
    "p448s14h71": ("p448s14 plus a 46K drain (keeps ~71K of history after a trim), using the KV headroom pruning freed", lambda: STARTUP + p448() + streams(14) + mamba(70) + HISTORY),
    "p448s14": ("C2: startup trims, keep-448 experts, 14 admitted streams, 70 Mamba slots", lambda: STARTUP + p448() + streams(14) + mamba(70)),
    "p448s14dp": ("p448s14 plus the D-prime slot priority (scored run too); its commit run is a 1,800 s SMOKE, "
                  "the evidence is scheduler_sim.py", lambda: STARTUP + p448() + streams(14) + mamba(70) + SMOKE + dprime()),
    "p448s14wf": ("p448s14 plus transcript-waste fixes (sandbox deadline excludes host action time, timeouts keep "
                  "retained functions, stdlib pre-bound, API text corrections)",
                  lambda: STARTUP + p448() + streams(14) + mamba(70) + overlay("wastefix")),
    "p448s14dpwfp": ("PROBE of the next ship candidate: the pinned p448s14dp plus the transcript-waste fixes; "
                     "answers only whether the stack installs and plays", lambda: STARTUP + p448() + streams(14) + mamba(70)
                     + PROBE + dprime() + overlay("wastefix")),
    "p448h14": ("C3: C2 plus the 32 GB host-RAM KV tier (the maximum-throughput serving stack)", lambda: STARTUP + p448() + hicache(32) + streams(14) + mamba(70)),
    "fastdec": ("A6 at 10 streams, NOT QUEUED: online MXFP8 projections, tuned; expected flat tok/s (prefill slower, "
                "decode faster). Attribution control for fastdec12 only", lambda: STARTUP + fastdec()),
    "fastdec12": ("A6: startup trims, online MXFP8 projections (tuned; HC mix and lm_head BF16), +2.7 GiB KV spent on "
                  "12 admitted streams. Judge against start", lambda: STARTUP + fastdec() + streams(12)),
    "p448s14dpwfrc": ("L1: the pinned p448s14dp plus the transcript-waste fixes and the B1 reply-cache round trip "
                      "(history reasoning verbatim, scored run too); bench runs log one serve.log line per finished "
                      "request", lambda: STARTUP + p448() + streams(14) + mamba(70) + dprime() + overlay("wastefix")
                      + overlay("replycache") + replycache_diag()),
    "p448s14dpwfa1": ("A1: the pinned p448s14dp plus the transcript-waste fixes and the agent fixes (a RESET at a "
                      "level start is skipped, the stale guard's catch contract, level_starts/attempt_starts); judge "
                      "on agentfix_counters.py", lambda: STARTUP + p448() + streams(14) + mamba(70) + dprime()
                      + overlay("wastefix") + agentfix()),
    "p448s14dpwftb": ("B5: p448s14dp plus transcript-waste fixes plus a strict thinking budget of 10,240 tokens "
                      "(--enable-strict-thinking, think exclusions emptied); gate on thinking_counters.py",
                      lambda: STARTUP + p448() + streams(14) + mamba(70) + dprime() + overlay("wastefix") + thinkbudget(10240)),
    "l1p": ("PROBE build of L1 (p448s14dpwfrc) used only to create a submittable Kaggle version once the GCP bench "
            "passes L1; the scored path is identical to L1", lambda: STARTUP + p448() + streams(14) + mamba(70) + PROBE
            + dprime() + overlay("wastefix") + overlay("replycache") + replycache_diag()),
    "h16dp": ("pinned p448s14dp served at 16 streams with 80 Mamba slots and the 32 GB host-RAM KV tier (the p448h16 "
              "serving, +7% tok/s), the capacity fallback after NVFP4 failed parity", lambda: STARTUP + p448() + hicache(32)
              + streams(16) + mamba(80) + dprime()),
    "l1a1p": ("PROBE build of l1a1 (L1 plus the A1 agent fixes) used only to create a submittable Kaggle version; the "
              "scored path is identical to l1a1", lambda: STARTUP + p448() + streams(14) + mamba(70) + PROBE + dprime()
              + overlay("wastefix") + overlay("replycache") + replycache_diag() + agentfix()),
    "dpctl": ("R2: full-replica control of the pinned p448s14dp (its commit was only a 1,800 s smoke run)",
              lambda: STARTUP + p448() + streams(14) + mamba(70) + dprime()),
    # Stacks on L1 (plan docs/research/2026-10-05-top5/PLAN.md section 4): each adds one lever to L1.
    "l1a1": ("R4: L1 plus the A1 agent fixes; judge on agentfix_counters.py", lambda: STARTUP + p448() + streams(14)
             + mamba(70) + dprime() + overlay("wastefix") + overlay("replycache") + replycache_diag() + agentfix()),
    "l1tb": ("R6: L1 plus the strict thinking budget of 10,240 tokens; judge on thinking_counters.py", lambda: STARTUP
             + p448() + streams(14) + mamba(70) + dprime() + overlay("wastefix") + overlay("replycache")
             + replycache_diag() + thinkbudget(10240)),
    "l1s20probe": ("P1: PROBE of S20 on L1: NVFP4 KV (2.0M-token pool), 20 streams, 100 Mamba slots; boots FP8 first and "
                   "stops unless FP8 and NVFP4 logprobs agree (300 s per game)", lambda: STARTUP + p448() + s20() + PROBE
                   + dprime() + overlay("wastefix") + overlay("replycache") + nvfp4kv_parity()),
    "l1s20": ("R3: L1 plus S20 (NVFP4 KV, 2.0M-token pool, 20 streams, 100 Mamba slots); gate on serve.log", lambda: STARTUP
              + p448() + s20() + dprime() + overlay("wastefix") + overlay("replycache") + replycache_diag()),
}


ARMS.update({
    # 10-10: l1a1p's second scored draw completed early at 0.00 (as the plain fork's did in the 10-03 outage).
    "l1a1fcp": ("PROBE build of l1a1 plus fail-closed: the promoted l1a1 config, and a scored run that ends with "
                "under 20 levels raises so Kaggle records an error (no slot spent) instead of 0.00; the scored path "
                "is l1a1 plus that check", lambda: STARTUP + p448() + streams(14) + mamba(70) + PROBE + dprime()
                + overlay("wastefix") + overlay("replycache") + replycache_diag() + agentfix() + failclosed()),
    "l1a1fc2p": ("PROBE build of l1a1 plus both fail-closed checks: a scored run whose model server exits before "
                 "READY raises at once, and one that ends under 20 levels raises at the end; supersedes l1a1fcp",
                 lambda: STARTUP + p448() + streams(14) + mamba(70) + PROBE + dprime() + overlay("wastefix")
                 + overlay("replycache") + replycache_diag() + agentfix() + failclosed_server() + failclosed()),
    "l1a1tbfc": ("l1a1 plus the strict thinking budget of 10,240 tokens (l1tb removed every reply cut at the 12,288 "
                 "output cap, 6.5% of tokens, and tied on levels) plus both fail-closed checks; judge on "
                 "thinking_counters.py and agentfix_counters.py, reject only at 104 levels or fewer",
                 lambda: STARTUP + p448() + streams(14) + mamba(70) + dprime() + overlay("wastefix")
                 + overlay("replycache") + replycache_diag() + agentfix() + thinkbudget(10240)
                 + failclosed_server() + failclosed()),
    "l1a1y8": ("l1a1 plus both fail-closed checks, yielding after 8,192 generated tokens instead of 2,048. 34% of "
               "turns end by yield and each re-sends an opener and a board image with no action in between "
               "(docs/research/2026-10-05-top5/top-movers.md proposal 4; Lord Han Solo yields by time only). Judge on "
               "trims and opener tokens per game, levels at matched tokens; reject at 104 levels or fewer",
               lambda: STARTUP + p448() + streams(14) + mamba(70) + dprime() + overlay("wastefix")
               + overlay("replycache") + replycache_diag() + agentfix() + YIELD8K + failclosed_server()
               + failclosed()),
    "l1a1qr": ("l1a1 plus the QSA pending-ring stride fix plus both fail-closed checks. A correctness fix to the "
               "sparse-attention index keys written during MTP verify; gate on boot, serve.log unchanged (tok/s, "
               "accept length, cache hit) and levels not a clear loser (reject at 104 or fewer)",
               lambda: STARTUP + p448() + streams(14) + mamba(70) + dprime() + overlay("wastefix")
               + overlay("replycache") + replycache_diag() + agentfix() + qsaring() + failclosed_server()
               + failclosed()),
    "l1a1qrp": ("PROBE build of l1a1qr used only to create a submittable Kaggle version once its bench passes",
                lambda: STARTUP + p448() + streams(14) + mamba(70) + PROBE + dprime() + overlay("wastefix")
                + overlay("replycache") + replycache_diag() + agentfix() + qsaring() + failclosed_server()
                + failclosed()),
    "l1a1sv2": ("serving v2 on l1a1, the memory-neutral part of v3: the QSA ring fix and the async M-RoPE upload, plus "
                "both fail-closed checks. Gate on boot, serve.log (tok/s not down, accept length within 0.05, cache "
                "hit within 2 points) and levels not a clear loser (reject at 104 or fewer)",
                lambda: STARTUP + p448() + streams(14) + mamba(70) + dprime() + overlay("wastefix")
                + overlay("replycache") + replycache_diag() + agentfix() + qsaring() + asyncmrope()
                + failclosed_server() + failclosed()),
    "l1a1sv2p": ("PROBE build of l1a1sv2 used only to create a submittable Kaggle version once its bench passes",
                 lambda: STARTUP + p448() + streams(14) + mamba(70) + PROBE + dprime() + overlay("wastefix")
                 + overlay("replycache") + replycache_diag() + agentfix() + qsaring() + asyncmrope()
                 + failclosed_server() + failclosed()),
    "l1a1sv3": ("serving v3 on l1a1: the QSA ring fix, async M-RoPE upload and BF16 MTP-draft KV, plus both "
                "fail-closed checks. Gate on serve.log: tok/s up >= 5%, accept length up, cache hit not down more "
                "than 2 points, KV max below 0.98 with no retractions; levels not a clear loser",
                lambda: STARTUP + p448() + streams(14) + mamba(70) + dprime() + overlay("wastefix")
                + overlay("replycache") + replycache_diag() + agentfix() + qsaring() + asyncmrope() + BF16DRAFT
                + failclosed_server() + failclosed()),
    "qsaparity": ("PROBE only, never promoted: on the l1a1 stack, decode-cached vs fresh-prefill next-token agreement "
                  "after long MTP-decoded replies, on the shipped SGLang and again after the QSA ring fix; then a "
                  "300 s-per-game run on the fixed server", lambda: STARTUP + p448() + streams(14) + mamba(70) + PROBE
                  + dprime() + overlay("wastefix") + overlay("replycache") + replycache_diag() + agentfix()
                  + qsaparity_probe() + failclosed_server() + failclosed()),
    "l1a1y8r": ("Replicate of l1a1y8 under a fresh slug (its predeclared gate pools two runs)",
                lambda: ARMS["l1a1y8"][1]()),
    "l1a1y8sv2": ("l1a1 plus yields after 8,192 tokens, the QSA ring fix and the async M-RoPE upload, plus both "
                  "fail-closed checks: the next-pin candidate if l1a1y8 passes",
                  lambda: STARTUP + p448() + streams(14) + mamba(70) + dprime() + overlay("wastefix")
                  + overlay("replycache") + replycache_diag() + agentfix() + YIELD8K + qsaring() + asyncmrope()
                  + failclosed_server() + failclosed()),
    "l1a1y8sv2p": ("PROBE build of l1a1y8sv2 used only to create a submittable Kaggle version",
                   lambda: STARTUP + p448() + streams(14) + mamba(70) + PROBE + dprime() + overlay("wastefix")
                   + overlay("replycache") + replycache_diag() + agentfix() + YIELD8K + qsaring() + asyncmrope()
                   + failclosed_server() + failclosed()),
    "l1a1y0sv2": ("l1a1y8sv2 with no token yield (y8 still yields on 15% of turns, each re-sending a full opener and "
                  "image); judge on openers, actions per opener, replies without a call and turn length; reject at "
                  "104 levels or fewer",
                  lambda: STARTUP + p448() + streams(14) + mamba(70) + dprime() + overlay("wastefix")
                  + overlay("replycache") + replycache_diag() + agentfix() + YIELD0 + qsaring() + asyncmrope()
                  + failclosed_server() + failclosed()),
    "l1a1y8sv2o16": ("l1a1y8sv2 with the output cap raised 12K -> 16K and the window +4K (trim budget unchanged). "
                     "Judge on cut replies (finish_reason length without a call), serve.log running mean and KV, "
                     "tok/s; reject at 104 levels or fewer",
                     lambda: STARTUP + p448() + streams(14) + mamba(70) + dprime() + overlay("wastefix")
                     + overlay("replycache") + replycache_diag() + agentfix() + YIELD8K + OUTCAP16 + qsaring()
                     + asyncmrope() + failclosed_server() + failclosed()),
    "l1a1y8sv2o16p": ("PROBE build of l1a1y8sv2o16 used only to create a submittable Kaggle version",
                      lambda: STARTUP + p448() + streams(14) + mamba(70) + PROBE + dprime() + overlay("wastefix")
                      + overlay("replycache") + replycache_diag() + agentfix() + YIELD8K + OUTCAP16 + qsaring()
                      + asyncmrope() + failclosed_server() + failclosed()),
    "l1a1y8sv2wdt": ("TEST build, never promoted: l1a1y8sv2 plus the server watchdog, with the server killed once "
                     "300 s after READY and 1,800 s per game, to rehearse the relaunch on the card: the server must "
                     "come back, games must keep playing after it, and none may give up",
                     lambda: STARTUP + p448() + streams(14) + mamba(70) + SMOKE + dprime() + overlay("wastefix")
                     + overlay("replycache") + replycache_diag() + agentfix() + YIELD8K + qsaring() + asyncmrope()
                     + watchdog(kill_at=300) + failclosed_server() + failclosed()),
    "l1a1y8sv2wdt2": ("TEST build, never promoted: the watchdog rehearsal again with the relaunch precache, the server "
                      "killed 60 s after READY and 2,700 s per game, so games outlive the relaunch and must resume",
                      lambda: STARTUP + p448() + streams(14) + mamba(70) + REHEARSE + dprime() + overlay("wastefix")
                      + overlay("replycache") + replycache_diag() + agentfix() + YIELD8K + qsaring() + asyncmrope()
                      + watchdog(kill_at=60) + failclosed_server() + failclosed()),
    "l1a1y8sv2wdp": ("PROBE build of l1a1y8sv2 plus the server watchdog, used only to create a submittable version",
                     lambda: STARTUP + p448() + streams(14) + mamba(70) + PROBE + dprime() + overlay("wastefix")
                     + overlay("replycache") + replycache_diag() + agentfix() + YIELD8K + qsaring() + asyncmrope()
                     + watchdog() + failclosed_server() + failclosed()),
    "l1a1y8sv2r1": ("l1a1y8sv2 plus the robustness and throughput riders, each judged on its own counter: the server "
                    "watchdog ('watching' line, 0 relaunches), MEMFRAC 0.955 (pool size, free device memory floor, "
                    "no OOM), the sandbox memory cap (no new tool errors), and image hashes off the scheduler "
                    "(stall per 10K cached tokens, decode share), and toolfast (in-batch gaps, late running requests); reject at 104 levels or fewer",
                    lambda: STARTUP + p448() + streams(14) + mamba(70) + MEMFRAC955 + dprime() + overlay("wastefix")
                    + overlay("replycache") + replycache_diag() + agentfix() + overlay("memcap") + overlay("toolfast")
                    + YIELD8K + qsaring() + asyncmrope() + MMHASH + watchdog() + failclosed_server() + failclosed()),
    "l1a1y8sv2r1p": ("PROBE build of l1a1y8sv2r1 used only to create a submittable Kaggle version",
                     lambda: STARTUP + p448() + streams(14) + mamba(70) + MEMFRAC955 + PROBE + dprime()
                     + overlay("wastefix") + overlay("replycache") + replycache_diag() + agentfix() + overlay("memcap")
                     + overlay("toolfast") + YIELD8K + qsaring() + asyncmrope() + MMHASH + watchdog()
                     + failclosed_server() + failclosed()),
    "l1a1y8sv2r2": ("l1a1y8sv2r1 plus the D3 drafter refit, keepdata and promptobs lite: the stacked candidate, every rider "
                    "judged on its own counters (D3 on accept length, keepdata on NameErrors and blocked imports, "
                    "promptobs on its three error classes); reject at 104 levels or fewer",
                    lambda: STARTUP + p448() + streams(14) + mamba(70) + MEMFRAC955 + dprime() + overlay("wastefix")
                    + overlay("replycache") + replycache_diag() + agentfix() + overlay("memcap") + overlay("toolfast")
                    + overlay("keepdata") + overlay("promptobs") + YIELD8K + qsaring() + asyncmrope() + MMHASH
                    + d3draft() + watchdog() + failclosed_server() + failclosed()),
    "l1a1y8sv2r2p": ("PROBE build of l1a1y8sv2r2 used only to create a submittable Kaggle version",
                     lambda: STARTUP + p448() + streams(14) + mamba(70) + MEMFRAC955 + PROBE + dprime()
                     + overlay("wastefix") + overlay("replycache") + replycache_diag() + agentfix() + overlay("memcap")
                     + overlay("toolfast") + overlay("keepdata") + overlay("promptobs") + YIELD8K + qsaring()
                     + asyncmrope() + MMHASH + d3draft() + watchdog() + failclosed_server() + failclosed()),
    "l1a1y8sv2r2fbp": ("PROBE build of l1a1y8sv2r2 on our own inputs (fallback_inputs): the copied wheelhouse, drafter and "
                       "bundle and the public model mirror; kept unpinned, ready if a Franzen input goes private",
                       lambda: ARMS["l1a1y8sv2r2p"][1]() + fallback_inputs()),
    "l1a1y8sv2r3": ("l1a1y8sv2r1 + D3 + promptobs lite (r2 without keepdata), the shipped set: freeze-audit replicate of "
                    "the pin, judged pooled with r2 and on every rider's own counter",
                    lambda: STARTUP + p448() + streams(14) + mamba(70) + MEMFRAC955 + dprime() + overlay("wastefix")
                    + overlay("replycache") + replycache_diag() + agentfix() + overlay("memcap") + overlay("toolfast")
                    + overlay("promptobs") + YIELD8K + qsaring() + asyncmrope() + MMHASH + d3draft() + watchdog()
                    + failclosed_server() + failclosed()),
    "l1a1y8sv2r3nr": ("l1a1y8sv2r3 with RESET hidden again (no EXPOSE_RESET line; the agentfix overlay stays): run only "
                      "if the board trigger in gate-predeclared-2026-10-12-franzen-l1a1y8sv2r3nr.json fires",
                      lambda: STARTUP + p448() + streams(14) + mamba(70) + MEMFRAC955 + dprime() + overlay("wastefix")
                      + overlay("replycache") + replycache_diag() + overlay("agentfix") + overlay("memcap")
                      + overlay("toolfast") + overlay("promptobs") + YIELD8K + qsaring() + asyncmrope() + MMHASH
                      + d3draft() + watchdog() + failclosed_server() + failclosed()),
    "l1a1y8sv2r3nrp": ("PROBE build of l1a1y8sv2r3nr used only to create a submittable version",
                       lambda: STARTUP + p448() + streams(14) + mamba(70) + MEMFRAC955 + PROBE + dprime()
                       + overlay("wastefix") + overlay("replycache") + replycache_diag() + overlay("agentfix")
                       + overlay("memcap") + overlay("toolfast") + overlay("promptobs") + YIELD8K + qsaring()
                       + asyncmrope() + MMHASH + d3draft() + watchdog() + failclosed_server() + failclosed()),
    "l1a1y8sv2r3p": ("PROBE build of l1a1y8sv2r2 without keepdata (it missed two of its own counters in r2; the gate's "
                     "partial rule drops it): l1a1y8sv2r1 + D3 + promptobs lite, used only to create a submittable version",
                     lambda: STARTUP + p448() + streams(14) + mamba(70) + MEMFRAC955 + PROBE + dprime()
                     + overlay("wastefix") + overlay("replycache") + replycache_diag() + agentfix() + overlay("memcap")
                     + overlay("toolfast") + overlay("promptobs") + YIELD8K + qsaring() + asyncmrope() + MMHASH
                     + d3draft() + watchdog() + failclosed_server() + failclosed()),
    "l1a1y8sv2r3fbp": ("PROBE build of l1a1y8sv2r3p on our own inputs (fallback_inputs); kept unpinned, ready if a "
                       "Franzen input goes private",
                       lambda: ARMS["l1a1y8sv2r3p"][1]() + fallback_inputs()),
    "l1a1y8sv2r3s4": ("the pin's replica (l1a1y8sv2r3) with 4 MTP draft steps instead of 3: judge on decode throughput "
                      "at 14 running, accept length by context and health; reject at 104 or fewer levels",
                      lambda: ARMS["l1a1y8sv2r3"][1]() + SPEC4),
    "l1a1y8sv2r3s4p": ("PROBE build of l1a1y8sv2r3s4 used only to create a submittable version",
                       lambda: ARMS["l1a1y8sv2r3p"][1]() + SPEC4),
    "l1a1y8sv2r3hm32": ("replica A: the pin's replica (l1a1y8sv2r3) with the 32K ARC traffic draft vocabulary and specstats "
                        "(transcript only); judge on gen throughput and clean step at 14 running, accept, health",
                        lambda: ARMS["l1a1y8sv2r3"][1]() + hotmap(32768) + overlay("specstats")),
    "l1a1y8sv2r3hm32p": ("PROBE build of l1a1y8sv2r3hm32 without specstats, used only to create a submittable version",
                         lambda: ARMS["l1a1y8sv2r3p"][1]() + hotmap(32768)),
    "l1a1y8sv2r3hm32bf": ("replica A + BF16 draft KV with the target pool capped at 1,375,552 tokens (BF16DRAFTCAP): run "
                          "only if the bfp probe lifts accept by 0.05", lambda: ARMS["l1a1y8sv2r3hm32"][1]() + BF16DRAFTCAP),
    "l1a1y8sv2r3hm32bfp": ("PROBE of l1a1y8sv2r3hm32p + BF16 draft KV with the cap: installs, fits in memory, and how much "
                           "accept moves on probe contexts", lambda: ARMS["l1a1y8sv2r3hm32p"][1]() + BF16DRAFTCAP),
    "l1a1y8sv2r3lt": ("the pin's replica with leantrim (keep 3 turns): run only if trimtest passes; gate in "
                      "gate-predeclared-2026-10-12-trimtest.json ('if_pass')",
                      lambda: ARMS["l1a1y8sv2r3"][1]() + leantrim(3)),
    "l1a1y8sv2r3ltp": ("PROBE build of l1a1y8sv2r3lt used only to create a submittable version",
                       lambda: ARMS["l1a1y8sv2r3p"][1]() + leantrim(3)),
    # 10-20 (docs/research/2026-10-19-top/PLAN.md 2.5): a scored run with no move for 15 minutes ends with its
    # score so far, and the pin sends one warmup RESET at release, so a server not serving by about 27 minutes
    # banks 0.00. On every path, so the probe runs it too.
    "l1a1y8sv2r3k": ("the pin's replica (l1a1y8sv2r3) plus the warmup-only keepalive (franzen_overlays/keepalive.py): "
                     "the warmup game re-sends RESET every 9 min while /health is not 200, at most 5 times, and stops "
                     "at READY or its first analyzer reply; gate: solution/sim/test_keepalive.py and one probe",
                     lambda: ARMS["l1a1y8sv2r3"][1]() + overlay("keepalive")),
    "l1a1y8sv2r3kp": ("PROBE build of l1a1y8sv2r3k (the pin r3p plus the warmup-only keepalive), used only to create "
                      "a submittable version; it plays as r3p once the server is READY",
                      lambda: ARMS["l1a1y8sv2r3p"][1]() + overlay("keepalive")),
    "trimtest": ("DEV ONLY, never submitted: the pin's server (l1a1y8sv2r3p) samples the next reply at recorded "
                 "post-trim states with and without leaner retention, 2 samples each (gate in "
                 "gate-predeclared-2026-10-12-trimtest.json)", lambda: ARMS["l1a1y8sv2r3p"][1]() + trimtest()),
    "l1a1y8sv2r3hm32fbp": ("PROBE build of l1a1y8sv2r3hm32p on our own inputs (fallback_inputs); the map is embedded, so "
                           "it needs no new input; kept unpinned, ready if a Franzen input goes private",
                           lambda: ARMS["l1a1y8sv2r3hm32p"][1]() + fallback_inputs()),
    "l1a1y8sv2r3hm32rl": ("DEV ONLY, never submitted: the pin (l1a1y8sv2r3hm32p's scored path, without PROBE) playing the clean "
                          "RE-ARC-3 games from ARC3_REARC_GAMES with the reqlog overlay, to generate LoRA training data "
                          "(solution/lora/DATAGEN_STATUS.md)",
                          lambda: ARMS["l1a1y8sv2r3"][1]() + hotmap(32768) + reqlog() + rearc()),
    "l1a1y8sv2r3hm32lora": ("DEV REPLICA, never submitted: the pin's replica (l1a1y8sv2r3hm32, with specstats) serving the r16 LoRA folded into the BF16 weights at load (lora(), adapter from a mounted dataset, fail closed); gate in the LoRA task: both replicas >= 130 levels, tokens per action within 15% of the pin, accept >= 2.70",
                          lambda: ARMS["l1a1y8sv2r3hm32"][1]() + lora()),
    "l1a1y8sv2r3hm64": ("l1a1y8sv2r3hm32 with the 64K traffic map: run only if the 32K replica fails an accept item with "
                        "health green", lambda: ARMS["l1a1y8sv2r3"][1]() + hotmap(65536) + overlay("specstats")),
    "l1a1y8sv2r3s4hm32": ("replica A on the s4 stack (l1a1y8sv2r3s4) if s4 is pinned: + the 32K traffic map + specstats",
                          lambda: ARMS["l1a1y8sv2r3s4"][1]() + hotmap(32768) + overlay("specstats")),
    "l1a1y8sv2r3s4hm32p": ("PROBE build of l1a1y8sv2r3s4hm32 without specstats, used only to create a submittable version",
                           lambda: ARMS["l1a1y8sv2r3s4p"][1]() + hotmap(32768)),
    "l1a1y8sv2d3": ("l1a1y8sv2 plus ursuciprian's D3 refit of the drafter's 29 dense tensors, no other rider; judge on "
                    "accept length (overall and at 60-100K context), clean step time, cache hit; reject at 104 or fewer",
                    lambda: STARTUP + p448() + streams(14) + mamba(70) + dprime() + overlay("wastefix")
                    + overlay("replycache") + replycache_diag() + agentfix() + YIELD8K + qsaring() + asyncmrope()
                    + d3draft() + failclosed_server() + failclosed()),
    "l1a1y8sv2d3p": ("PROBE build of l1a1y8sv2d3: the D3 splice installs and lifts accept length before a replica",
                     lambda: STARTUP + p448() + streams(14) + mamba(70) + PROBE + dprime() + overlay("wastefix")
                     + overlay("replycache") + replycache_diag() + agentfix() + YIELD8K + qsaring() + asyncmrope()
                     + d3draft() + failclosed_server() + failclosed()),
    "l1a1y8sv3": ("l1a1y8sv2 plus BF16 MTP-draft KV (target KV stays FP8): the serving-v3 step on the y8 stack. Gate on "
                  "serve.log: accept length up, tok/s up, KV max below 0.98 with no retractions; levels not a clear loser",
                  lambda: STARTUP + p448() + streams(14) + mamba(70) + dprime() + overlay("wastefix")
                  + overlay("replycache") + replycache_diag() + agentfix() + YIELD8K + qsaring() + asyncmrope()
                  + BF16DRAFT + failclosed_server() + failclosed()),
    "l1a1y8sv3p": ("PROBE build of l1a1y8sv3 used only to create a submittable Kaggle version",
                   lambda: STARTUP + p448() + streams(14) + mamba(70) + PROBE + dprime() + overlay("wastefix")
                   + overlay("replycache") + replycache_diag() + agentfix() + YIELD8K + qsaring() + asyncmrope()
                   + BF16DRAFT + failclosed_server() + failclosed()),
    "l1a1y0sv2p": ("PROBE build of l1a1y0sv2 used only to create a submittable Kaggle version",
                   lambda: STARTUP + p448() + streams(14) + mamba(70) + PROBE + dprime() + overlay("wastefix")
                   + overlay("replycache") + replycache_diag() + agentfix() + YIELD0 + qsaring() + asyncmrope()
                   + failclosed_server() + failclosed()),
    "l1a1y8p": ("PROBE build of l1a1y8 used only to create a submittable Kaggle version",
                lambda: STARTUP + p448() + streams(14) + mamba(70) + PROBE + dprime() + overlay("wastefix")
                + overlay("replycache") + replycache_diag() + agentfix() + YIELD8K + failclosed_server()
                + failclosed()),
    "l1a1tbfcp": ("PROBE build of l1a1tbfc used only to create a submittable Kaggle version once its bench passes; the "
                  "scored path is identical to l1a1tbfc", lambda: STARTUP + p448() + streams(14) + mamba(70) + PROBE
                  + dprime() + overlay("wastefix") + overlay("replycache") + replycache_diag() + agentfix()
                  + thinkbudget(10240) + failclosed_server() + failclosed()),
    "s20probe": ("PROBE of B3 on the p448s14dpwfp stack: NVFP4 KV (2.0M-token pool), 20 streams, 100 Mamba slots; "
                 "boots FP8 first and stops unless FP8 and NVFP4 logprobs agree on 32 recorded prompts (300 s per game)",
                 lambda: STARTUP + p448() + s20() + PROBE + dprime() + overlay("wastefix") + nvfp4kv_parity()),
    "s20": ("B3 replica: p448s14dpwf with NVFP4 KV (2.0M-token pool), 20 admitted streams, 100 Mamba slots; add the B1 "
            "reply-cache overlay when it lands to make L1 + S20",
            lambda: STARTUP + p448() + s20() + dprime() + overlay("wastefix")),
    "l1a1y8sv2r3hob": ("DEV ONLY, never submitted: GRPO gate (b) control, the pin's replica (l1a1y8sv2r3) playing the RE-ARC-3 "
                       "games of ARC3_REARC_GAMES (the 25 of solution/lora/grpo/games_heldout.json, never trained on) with "
                       "reqlog", lambda: ARMS["l1a1y8sv2r3"][1]() + reqlog() + rearc()),
    "l1a1y8sv2r3hobg1": ("DEV ONLY, never submitted: GRPO gate (b) adapter arm, l1a1y8sv2r3hob serving the GRPO round-1 LoRA "
                         "folded into the BF16 weights at load (lora(), fail closed)",
                         lambda: ARMS["l1a1y8sv2r3hob"][1]() + lora(GRPO_ADAPTER)),
    "l1a1y8sv2r3g1": ("DEV REPLICA, never submitted: GRPO gate (c), the pin's replica (l1a1y8sv2r3) serving the GRPO round-1 "
                      "LoRA (lora(), fail closed); gate in solution/lora/grpo/GATES.md",
                      lambda: ARMS["l1a1y8sv2r3"][1]() + lora(GRPO_ADAPTER)),
    "l1a1y8sv2r3g1p": ("PROBE build of l1a1y8sv2r3g1 (GRPO gate d): the pin's probe plus the round-1 LoRA from the private "
                       "dataset vineethsaivs/arc3-lora-grpo-r1", lambda: ARMS["l1a1y8sv2r3p"][1]() + lora(GRPO_ADAPTER)),
    "l1a1y8sv2r3rq": ("DEV REPLICA, never submitted: the slot config's replica (l1a1y8sv2r3) with the reqlog overlay (exact "
                      "request contexts under ARC3_REQLOG_DIR): round A's completion contrast on the public train half, "
                      "the states for gate (a') and same-day pin controls (docs/research/2026-10-20-ft/DECISION.md)",
                      lambda: ARMS["l1a1y8sv2r3"][1]() + reqlog()),
    "l1a1y8sv2r3rqy2": ("DEV REPLICA, never submitted: l1a1y8sv2r3rq without the YIELD8K edit (the harness yields after "
                        "Franzen's 2,048 tokens), every other edit identical: the 2,048 arm of the interleaved yield wave "
                        "(gate-predeclared-2026-10-21-yield-interleaved.json)",
                        lambda: without(ARMS["l1a1y8sv2r3rq"][1](), YIELD8K)),
})


def adapter_arms(dataset: str, sr_seed: int | None = None, scale: float = 1.0) -> None:
    """Wave, held-out and (d) arms for the adapter dataset vineethsaivs/<dataset>, all lora() at scale 1.0, fail closed
    (solution/lora/grpo/GATES.md, round A family amendment 2026-10-20 19:49Z; PARALLEL.md "Do next" 4). Names end in the
    dataset's last '-' part, so arc3-lora-grpo-wa gives:
    - l1a1y8sv2r3wa: wave W = l1a1y8sv2r3rq (the slot config's replica plus reqlog) plus lora(): wave runs log requests;
    - l1a1y8sv2r3wap: gate (d) = l1a1y8sv2r3kp (the slot config's probe, the pin) plus lora() from the private dataset;
    - l1a1y8sv2r3hobwa: wave 2's held-out W = l1a1y8sv2r3hob (the 25 held-out RE-ARC-3 games, reqlog) plus lora().
    With sr_seed the three arms end in "sr" (l1a1y8sv2r3wasr, wasrp, hobwasr) and merge with stochastic rounding
    (lora(sr_seed=...), solution/lora/lora_merge_sr.py); not gated, not pushed (ROUNDB_DRAFT.md, branch ii)."""
    path = f"/kaggle/input/datasets/vineethsaivs/{dataset}/"
    OUR_DATASETS[path] = f"vineethsaivs/{dataset}"
    tag = dataset.rsplit("-", 1)[1] + ("" if scale == 1.0 else f"{scale:g}") + ("" if sr_seed is None else "sr")
    lo = (lambda: lora(path, 1.0, sr_seed)) if scale == 1.0 else (lambda: lora(path, scale, sr_seed, set_scale=True))
    arm = "l1a1y8sv2r3" + tag
    sr = "" if sr_seed is None else f", stochastic rounding seed {sr_seed}"
    ARMS[arm] = (f"DEV REPLICA, never submitted: wave arm W, the slot config's replica with reqlog (l1a1y8sv2r3rq) serving "
                 f"the LoRA from {path} (lora(), scale {scale}{sr}, fail closed)",
                 lambda: ARMS["l1a1y8sv2r3rq"][1]() + lo())
    ARMS[arm + "p"] = (f"PROBE build of {arm} (gate d): the slot config's probe (l1a1y8sv2r3kp) plus the LoRA from the private "
                       f"dataset vineethsaivs/{dataset}{sr}", lambda: ARMS["l1a1y8sv2r3kp"][1]() + lo())
    ARMS["l1a1y8sv2r3hob" + tag] = (f"DEV ONLY, never submitted: wave 2's held-out arm W, l1a1y8sv2r3hob serving the LoRA "
                                    f"from {path} (lora(), scale {scale}{sr}, fail closed)",
                                    lambda: ARMS["l1a1y8sv2r3hob"][1]() + lo())


# one entry per adapter dataset: round A's members wa (the VM), wb and wc (Vertex), the spare wd, and round B's we
for _dataset in ("arc3-lora-grpo-wa", "arc3-lora-grpo-wb", "arc3-lora-grpo-wc", "arc3-lora-grpo-wd", "arc3-lora-grpo-we"):
    adapter_arms(_dataset)
# SR merge (opt-in, lora_merge_sr.py): WA at full strength and round B's WE for branch (ii) of ROUNDB_DRAFT.md; not gated
adapter_arms("arc3-lora-grpo-wa", sr_seed=0)
adapter_arms("arc3-lora-grpo-we", sr_seed=0)
# round C (WE3, GATES.md): WE served at scale 3 (lora_alpha rewritten to 3r in a /tmp config copy), SR seed 0
adapter_arms("arc3-lora-grpo-we", sr_seed=0, scale=3.0)


def keepalive_cap(n: int) -> list[tuple[int, str, str]]:
    """Raise the warmup keepalive's RESET cap (franzen_overlays/keepalive.py, 5) to n; goes after overlay("keepalive").
    10-23 field scan (docs/research/2026-10-23-push/field.md lever 1): since about 10-20 Kaggle model loads take 35 to 63
    minutes or more, and at 5 RESETs a server READY later than about 70 minutes banks a complete 0.00. At 12 the last
    RESET is about 2 h after start. Play is identical once the server is READY (no RESET fires after it)."""
    line = "KEEPALIVE_MAX_RESETS = 5  # to about 57 minutes after notebook start"
    return [(ENV, line, f"KEEPALIVE_MAX_RESETS = {n}  # arm keepalive_cap: to about {9 * n + 12} minutes after notebook start")]


# the slot configs with the keepalive cap at 12: the r3kp pin, and the WE and WE3 probes (pins if they promote)
for _base in ("l1a1y8sv2r3kp", "l1a1y8sv2r3wesrp", "l1a1y8sv2r3we3srp"):
    ARMS[_base + "12"] = (f"PROBE build of {_base} with the warmup keepalive capped at 12 RESETs (about 2 h) instead of 5 "
                          "(keepalive_cap; field.md lever 1); it plays as its base once the server is READY",
                          lambda _b=_base: ARMS[_b][1]() + keepalive_cap(12))
# 10-27 (docs/research/2026-10-27-harness/PLAN.md): harness.md C1 + C2, both overlays run before the pin's own
# (overlay() inserts at the marker, so the last added runs first). Wave W on the public 25 and the held-out 25, and the
# probe that would become the pin.
LF = lambda: overlay("levelfirst") + overlay("imports")  # noqa: E731
ARMS["l1a1y8sv2r3lf"] = ("DEV REPLICA, never submitted: wave W, l1a1y8sv2r3rq (the slot config's replica plus reqlog) with "
                         "levelfirst (at a trim, earlier levels drain to a 12K tail first) and imports (7 more stdlib "
                         "modules, time.sleep a no-op)", lambda: ARMS["l1a1y8sv2r3rq"][1]() + LF())
ARMS["l1a1y8sv2r3hoblf"] = ("DEV ONLY, never submitted: held-out wave W, l1a1y8sv2r3hob (the 25 held-out RE-ARC-3 games, "
                            "reqlog) with levelfirst and imports", lambda: ARMS["l1a1y8sv2r3hob"][1]() + LF())
ARMS["l1a1y8sv2r3kp12lf"] = ("PROBE build of l1a1y8sv2r3lf: the pin (l1a1y8sv2r3kp12) with levelfirst and imports",
                             lambda: ARMS["l1a1y8sv2r3kp12"][1]() + LF())
# 10-20: neither lora() nor a RE-ARC-3 arm had run on Vertex. This proves the held-out W stack (hob plus lora()) with an
# adapter both buckets already hold (arc3-lora-r16: r16, alpha 32, so scale 2.0), in us-east4 before wave 1.
ARMS["l1a1y8sv2r3hobr16"] = ("DEV PROBE ONLY, never waved or submitted: l1a1y8sv2r3hob serving the r16 LoRA of "
                             "arc3-lora-r16 (lora(), scale 2.0, fail closed), to prove the held-out W stack on Vertex",
                             lambda: ARMS["l1a1y8sv2r3hob"][1]() + lora(LORA_ADAPTER, 2.0))

# 10-28: ThinkingCap (bottlecapai's shorter-thinking post-train of Flash-Next, -38.5% thinking tokens at -1.0 accuracy
# on its card), repackaged in Intel's W4A16 AutoRound layout by lamdang: same tensor names, tokenizer and FR-Spec map.
TC_MODEL = "lamdang/thinkingcap-qwen3-8-flash-next-w4a16-autoround/Transformers/default/1"


def thinkingcap() -> list[tuple[int, str, str]]:
    """Serve ThinkingCap in place of Intel's Flash-Next; build() swaps the kernel's model source to TC_MODEL."""
    old = "MODEL_DIR         = '/kaggle/input/models/dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/transformers/default/1'\n"
    return [(ENV, old, "MODEL_DIR         = '/kaggle/input/models/lamdang/thinkingcap-qwen3-8-flash-next-w4a16-autoround/"
                       "transformers/default/1'  # arm thinkingcap\n")]


ARMS["l1a1y8sv2r3lfk"] = ("DEV REPLICA, never submitted: the new pin's agent (l1a1y8sv2r3 plus levelfirst and imports) on the 25 "
                          "public games, Kaggle control for the ThinkingCap bench", lambda: ARMS["l1a1y8sv2r3"][1]() + LF())
ARMS["l1a1y8sv2r3lftc"] = ("DEV REPLICA, never submitted: l1a1y8sv2r3lfk serving ThinkingCap (TC_MODEL) in place of Intel's "
                           "Flash-Next", lambda: ARMS["l1a1y8sv2r3lfk"][1]() + thinkingcap())
ARMS["l1a1y8sv2r3kp12lftc"] = ("PROBE build of l1a1y8sv2r3lftc: the pin (l1a1y8sv2r3kp12lf) serving ThinkingCap",
                               lambda: ARMS["l1a1y8sv2r3kp12lf"][1]() + thinkingcap())

# 10-28: one more admitted game. At 14 streams the server runs 12.7 requests on average (the rest are executing their
# python), so a 15th admitted game fills idle decode slots: about +2.5% tokens per second by the 10-26 scaling
# (docs/research/2026-10-23-push/clock.md s4). Mamba slots stay at 5 per stream.
S15 = lambda edits: without(edits, streams(14) + mamba(70)) + streams(15) + mamba(75)  # noqa: E731
ARMS["l1a1y8sv2r3lf15"] = ("DEV REPLICA, never submitted: l1a1y8sv2r3lfk with 15 admitted streams and 75 Mamba slots",
                           lambda: S15(ARMS["l1a1y8sv2r3lfk"][1]()))
ARMS["l1a1y8sv2r3kp12lf15"] = ("PROBE build of l1a1y8sv2r3lf15: the pin (l1a1y8sv2r3kp12lf) with 15 streams and 75 Mamba slots",
                               lambda: S15(ARMS["l1a1y8sv2r3kp12lf"][1]()))

# 10-28: keepvars (franzen_overlays/keepvars.py): plain-data variables survive between python calls on a level
ARMS["l1a1y8sv2r3lf15kv"] = ("DEV REPLICA, never submitted: l1a1y8sv2r3lf15 with the keepvars overlay",
                             lambda: ARMS["l1a1y8sv2r3lf15"][1]() + overlay("keepvars"))
ARMS["l1a1y8sv2r3kp12lf15kv"] = ("PROBE build of l1a1y8sv2r3lf15kv: the pin (l1a1y8sv2r3kp12lf15) with keepvars",
                                 lambda: ARMS["l1a1y8sv2r3kp12lf15"][1]() + overlay("keepvars"))

# 10-29: keepvars2 (franzen_overlays/keepvars2.py): v1 plus functions that read kept variables are retained (v1 restored
# the data after the retention check fixed its available names, so those functions were refused), and frozensets
ARMS["l1a1y8sv2r3lf15kv2"] = ("DEV REPLICA, never submitted: l1a1y8sv2r3lf15 with the keepvars2 overlay",
                              lambda: ARMS["l1a1y8sv2r3lf15"][1]() + overlay("keepvars2"))
ARMS["l1a1y8sv2r3kp12lf15kv2"] = ("PROBE build of l1a1y8sv2r3lf15kv2: the pin (l1a1y8sv2r3kp12lf15) with keepvars2",
                                  lambda: ARMS["l1a1y8sv2r3kp12lf15"][1]() + overlay("keepvars2"))

# 10-30: unstick (franzen_overlays/unstick.py): a stall check after 3 analyzer turns without an action, the object types
# no click on the level has touched, and Tufa's step-verification rules (mechanics first, replay before a long plan)
ARMS["l1a1y8sv2r3lf15unst"] = ("DEV REPLICA, never submitted: l1a1y8sv2r3lf15 with the unstick overlay",
                               lambda: ARMS["l1a1y8sv2r3lf15"][1]() + overlay("unstick"))
ARMS["l1a1y8sv2r3kp12lf15unst"] = ("PROBE build of l1a1y8sv2r3lf15unst: the pin (l1a1y8sv2r3kp12lf15) with unstick",
                                   lambda: ARMS["l1a1y8sv2r3kp12lf15"][1]() + overlay("unstick"))
# unstick2 (franzen_overlays/unstick2.py): unstick plus, on the first turn of each level, the object types no earlier
# level's board had
ARMS["l1a1y8sv2r3lf15uns2"] = ("DEV REPLICA, never submitted: l1a1y8sv2r3lf15 with the unstick2 overlay",
                               lambda: ARMS["l1a1y8sv2r3lf15"][1]() + overlay("unstick2"))
ARMS["l1a1y8sv2r3kp12lf15uns2"] = ("PROBE build of l1a1y8sv2r3lf15uns2: the pin (l1a1y8sv2r3kp12lf15) with unstick2",
                                   lambda: ARMS["l1a1y8sv2r3kp12lf15"][1]() + overlay("unstick2"))
# seg8 (franzen_overlays/seg8.py): segmentation nodes carry their 8-connected group; on unstick2 (us8) or alone (s8)
ARMS["l1a1y8sv2r3lf15us8"] = ("DEV REPLICA, never submitted: l1a1y8sv2r3lf15uns2 with the seg8 overlay",
                              lambda: ARMS["l1a1y8sv2r3lf15uns2"][1]() + overlay("seg8"))
ARMS["l1a1y8sv2r3kp12lf15us8"] = ("PROBE build of l1a1y8sv2r3lf15us8: the pin with unstick2 and seg8",
                                  lambda: ARMS["l1a1y8sv2r3kp12lf15uns2"][1]() + overlay("seg8"))
ARMS["l1a1y8sv2r3lf15s8"] = ("DEV REPLICA, never submitted: l1a1y8sv2r3lf15 with the seg8 overlay",
                             lambda: ARMS["l1a1y8sv2r3lf15"][1]() + overlay("seg8"))
ARMS["l1a1y8sv2r3kp12lf15s8"] = ("PROBE build of l1a1y8sv2r3lf15s8: the pin with seg8",
                                 lambda: ARMS["l1a1y8sv2r3kp12lf15"][1]() + overlay("seg8"))
# scroll (franzen_overlays/scroll.py): frame_diff undoes a camera scroll and reports it; the opener says when the
# view scrolled; on us8 (unstick2 + seg8)
ARMS["l1a1y8sv2r3lf15usc"] = ("DEV REPLICA, never submitted: l1a1y8sv2r3lf15us8 with the scroll overlay",
                              lambda: ARMS["l1a1y8sv2r3lf15us8"][1]() + overlay("scroll"))
ARMS["l1a1y8sv2r3kp12lf15usc"] = ("PROBE build of l1a1y8sv2r3lf15usc: the pin with unstick2, seg8 and scroll",
                                  lambda: ARMS["l1a1y8sv2r3kp12lf15us8"][1]() + overlay("scroll"))
# tool2 (franzen_overlays/tool2.py): a timed-out snippet keeps its printed output; inspect.getsource reads retained
# functions; on us8 (unstick2 + seg8), without scroll (its gate failed)
ARMS["l1a1y8sv2r3lf15ust"] = ("DEV REPLICA, never submitted: l1a1y8sv2r3lf15us8 with the tool2 overlay",
                              lambda: ARMS["l1a1y8sv2r3lf15us8"][1]() + overlay("tool2"))
ARMS["l1a1y8sv2r3kp12lf15ust"] = ("PROBE build of l1a1y8sv2r3lf15ust: the pin with unstick2, seg8 and tool2",
                                  lambda: ARMS["l1a1y8sv2r3kp12lf15us8"][1]() + overlay("tool2"))
# verify (franzen_overlays/verify.py): check_model and check_goal replay this level's log through the model's own rule
# and goal test; on ust (unstick2 + seg8 + tool2) or, if tool2 is killed, on us8
ARMS["l1a1y8sv2r3lf15ustv"] = ("DEV REPLICA, never submitted: l1a1y8sv2r3lf15ust with the verify overlay",
                               lambda: ARMS["l1a1y8sv2r3lf15ust"][1]() + overlay_last("verify"))
ARMS["l1a1y8sv2r3kp12lf15ustv"] = ("PROBE build of l1a1y8sv2r3lf15ustv: the pin with unstick2, seg8, tool2 and verify",
                                   lambda: ARMS["l1a1y8sv2r3kp12lf15ust"][1]() + overlay_last("verify"))
ARMS["l1a1y8sv2r3lf15usv"] = ("DEV REPLICA, never submitted: l1a1y8sv2r3lf15us8 with the verify overlay",
                              lambda: ARMS["l1a1y8sv2r3lf15us8"][1]() + overlay_last("verify"))
ARMS["l1a1y8sv2r3kp12lf15usv"] = ("PROBE build of l1a1y8sv2r3lf15usv: the pin with unstick2, seg8 and verify",
                                  lambda: ARMS["l1a1y8sv2r3kp12lf15us8"][1]() + overlay_last("verify"))
for _arm in ("l1a1y8sv2r3lf15us8", "l1a1y8sv2r3lf15s8", "l1a1y8sv2r3lf15usc", "l1a1y8sv2r3lf15ust", "l1a1y8sv2r3lf15ustv", "l1a1y8sv2r3lf15usv"):   # replicate copies, own kernel ids
    for _suffix in "bc":
        ARMS[_arm + _suffix] = (ARMS[_arm][0] + f" (copy {_suffix})", ARMS[_arm][1])


def cell_text(cell: dict) -> str:
    src = cell["source"]
    return "".join(src) if isinstance(src, list) else src


def apply_edits(nb: dict, edits: list[tuple[int, str, str]]) -> None:
    for index, old, new in edits:
        text = cell_text(nb["cells"][index])
        count = text.count(old)
        if count != 1:
            raise SystemExit(f"edit for cell {index} matches {count} times, expected 1: {old[:80]!r}")
        nb["cells"][index]["source"] = text.replace(old, new)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(arm: str) -> Path:
    description, edits_fn = ARMS[arm]
    out = GEN / f"candidates-franzen-{arm}"
    if out.exists():
        raise SystemExit(f"{out} exists; delete it first (a pushed candidate must not change silently)")
    nb = json.loads((BASE / "submission.ipynb").read_text())
    edits = REPLICA + edits_fn()
    apply_edits(nb, edits)
    out.mkdir()
    (out / "submission.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    meta = json.loads((BASE / "kernel-metadata.json").read_text())
    meta["id"] = f"vineethsaivs/arc3-franzen-{arm}-a"
    meta["title"] = f"arc3-franzen-{arm}-a"
    for path, ref in OUR_DATASETS.items():
        if any(path in new for _, _, new in edits) and ref not in meta["dataset_sources"]:
            meta["dataset_sources"].append(ref)
    if any("arm fallback: our own copies" in new for _, _, new in edits):
        meta = fallback_metadata(meta)
    if any("# arm thinkingcap" in new for _, _, new in edits):
        meta["model_sources"] = [m for m in meta["model_sources"] if "intel-qwen3.8-flash-next" not in m] + [TC_MODEL]
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    base_manifest = json.loads((BASE / "experiment-manifest.json").read_text())
    manifest = {
        "schema_version": 1,
        "candidate": out.name,
        "mode": "ordinary-commit-replica-arm",
        "arm": arm,
        "description": description,
        "kernel_id": meta["id"],
        "submission_notebook_sha256": sha(out / "submission.ipynb"),
        "kernel_metadata_sha256": sha(out / "kernel-metadata.json"),
        "source_dataset": base_manifest["source_dataset"],
        "scored_submission_authorized": False,
        "base_candidate": BASE.name,
        "base_notebook_sha256": sha(BASE / "submission.ipynb"),
        "upstream": base_manifest["upstream"],
        "edits": [{"cell": i, "old": o, "new": n} for i, o, n in edits],
        "submission_path_changed": any(i != CUSTOM for i, _, _ in edits),
        "plan": "docs/PLAN_2026-10-01.md",
    }
    if INPUT_COPY in meta.get("kernel_sources", []):  # fallback_inputs(): the wheelhouse comes from our kernel copy
        manifest["source_dataset"] = None
        manifest["source_kernel"] = INPUT_COPY
    (out / "experiment-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    subprocess.run([sys.executable, str(HERE / "runtime_preflight.py"), str(out), "--seal"],
                   check=True, cwd=HERE, stdout=subprocess.DEVNULL,
                   env={"PYTHONDONTWRITEBYTECODE": "1", "PATH": "/usr/bin:/bin"})
    return out


def self_check() -> None:
    base = json.loads((BASE / "submission.ipynb").read_text())
    for arm, (_, edits_fn) in ARMS.items():
        try:
            edits = REPLICA + edits_fn()
        except FileNotFoundError as exc:
            print(f"skip {arm}: overlay not written yet ({exc.filename})")
            continue
        nb = json.loads(json.dumps(base))
        apply_edits(nb, edits)
        for index in {i for i, _, _ in edits}:
            text = cell_text(nb["cells"][index])
            if text.lstrip().startswith("%%"):
                continue
            compile("\n".join("pass" if l.lstrip().startswith(("!", "%")) else l
                              for l in text.split("\n")), f"{arm}:cell{index}", "exec",
                    flags=__import__("ast").PyCF_ALLOW_TOP_LEVEL_AWAIT)
        custom = cell_text(nb["cells"][CUSTOM])
        assert "bm.solver.concurrency = 120" in custom and "532*60" in custom, arm  # scored path intact
        print(f"ok   {arm}: {len(edits)} edits")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("arms", nargs="*")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.list:
        for name, (desc, _) in ARMS.items():
            print(f"{name:10} {desc}")
    if args.self_check:
        self_check()
    for arm in args.arms:
        print(build(arm))
