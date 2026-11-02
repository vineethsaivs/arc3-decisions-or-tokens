"""Arm nvfp4kv (lever B3): NVFP4 KV for Flash-Next's 12 QSA layers, so 20 admitted streams fit.

Runs inside the launcher cell (cell 15), exec'd into a fresh namespace:
  CFG.update(apply(VENV, env, max_total_tokens, patches)) after the install and target validation,
  before the launch arguments are built; patches are the two vendored patch texts by file name.
  relaunch_fp8(...) in the cell's READY wait, scored runs only: an NVFP4 server that exits before READY
  is relaunched once on FP8 KV, so the run is not played against a dead server.
  reference_args(args) and parity(...) only in the probe arm, around the cell's launch and READY wait.

Patches: Lesilva/qwen38-flashnext-nvfp4-kv @ 540722120a (GitHub, Apache-2.0), vendored unchanged in
franzen_overlays/nvfp4kv/ with PROVENANCE.json.
  0001 stores the QSA layers' K and V as NVFP4 (packed E2M1 plus one E4M3 scale per 16 values) and
       dequantizes to FP8 inside QSA's own top-k and prefix gathers, so the attention kernels and
       their CUDA graphs stay as they are and no full-pool dequant workspace is allocated.
  0002 adds a Mamba-slot admission check before batching (the scheduler crash that more streams make
       likelier) and a COW-source unlock that stays inert here (it needs SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK).
Every hunk applies as an exact-text edit: its old text must occur exactly once in Franzen's wheel
(sglang 0.5.19+gd00d88efc8d6), all eight files are checked before any is written, and a rerun of the
cell is a no-op. 0002's radix-cache hunk lands 42 lines lower than in stock Pennyroyal 2.5.3, under
Franzen's Mamba sparse-prefill code; its text is unchanged. With --kv-cache-dtype fp8_e4m3 every 0001
path is inert (0002's admission check stays on), so the probe's FP8 reference boots the same patched
install with stock FP8 numerics. So parity() does not test 0002: its admission check is live in both
probe boots and in the scored run, and a scheduler regression in s20 can come from 0002 as well as NVFP4.

Capacity, from the wheel's pool math (2 KV heads x 256 dims x 12 QSA layers, the BF16 QSA index at
768 B, one FP8 draft layer at 1,024 B): FP8 costs 14,080 B per token today, NVFP4 8,704 B. The planner
still prices NVFP4 at 9,429 B (a 1,024 B dequant workspace 0001 never allocates, and the draft at
1/12 of the target), so it under-fills: about 2.1M tokens at 100 Mamba slots. --max-total-tokens
can only lower that (kv_cache_configurator._apply_token_constraints takes the min).

Scales: the Intel W4A16 checkpoint carries no k/v scales (0 of 224,280 tensors), and SGLang's
auto-round loader gives attention no scale parameters anyway, so every layer's NVFP4 global scale
stays 1.0 (fp4_kv_cache_quant_method.py:346, 357) and a block's E4M3 scale is its amax/6: full
precision for blocks whose largest magnitude is 0.094 to 2,688, coarser below. Lesilva measured
+0.004 to +0.013 nats tail NLL in that same mode on another checkpoint; parity() measures ours.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import socket
import statistics
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

SWITCH = "SGLANG_QSA_NVFP4_KV"
KV4_LINE = "KV4: QSA NVFP4 KV enabled via SGLANG_QSA_NVFP4_KV=1"
PATCH_SHA256 = {
    "0001-qsa-nvfp4-kv.patch": "a31b03c136d8c6cfb37d0e708a7ca856c2212fa634b77e00d31383c9119f93cb",
    "0002-optional-mamba-pingpong-fix.patch": "12a15462ebe5ad4ae7b6989a05bd0b03e3c4f878b21a94f1c971ee32f54d22bb",
}
# Target KV in NVFP4; the MTP draft stays FP8 because --speculative-draft-kv-cache-dtype rejects nvfp4.
SERVER_CFG = {"KVDTYPE": "nvfp4", "DRAFT_KVDTYPE": "fp8_e4m3"}
POLL_S = 3  # seconds between READY and shutdown checks
# The model's chat template for one system and one user turn, thinking on, reasoning effort xhigh.
XHIGH = ("Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, "
         "consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.")


def hunks(text: str) -> list[tuple[str, str, str]]:
    """(path under site-packages, old text, new text) for every hunk of a `diff -ruN` patch, in order."""
    out, path, lines, i = [], None, text.splitlines(keepends=True), 0
    while i < len(lines):
        line = lines[i]
        i += 1
        if line.startswith("+++ "):
            path = line[4:].split("\t")[0].strip()
            if not path.startswith("b/python/sglang/"):
                raise RuntimeError(f"nvfp4kv: unexpected patch path {path}")
            path = path[len("b/python/"):]
        elif line.startswith("@@ "):
            m = re.match(r"@@ -\d+(?:,(\d+))? \+\d+(?:,(\d+))? @@", line)
            if not m:
                raise RuntimeError(f"nvfp4kv: bad hunk header in {path}: {line!r}")
            want_old, want_new = int(m[1] or 1), int(m[2] or 1)
            old, new = [], []
            while i < len(lines) and (len(old) < want_old or len(new) < want_new):
                tag, body = lines[i][:1], lines[i][1:]
                if tag not in " -+":
                    raise RuntimeError(f"nvfp4kv: unexpected hunk line in {path}: {lines[i]!r}")
                if tag != "+":
                    old.append(body)
                if tag != "-":
                    new.append(body)
                i += 1
            if (len(old), len(new)) != (want_old, want_new):
                raise RuntimeError(f"nvfp4kv: truncated hunk in {path}")
            out.append((path, "".join(old), "".join(new)))
    return out


def patch_sglang(venv, patches: dict) -> list[Path]:
    """Apply both patches to the installed SGLang exactly once; a rerun of the cell is a no-op."""
    roots = sorted(Path(venv).glob("lib/python*/site-packages/sglang"))
    if len(roots) != 1:
        raise RuntimeError(f"nvfp4kv: expected one installed sglang under {venv}, found {roots}")
    site = roots[0].parent
    edits: dict[str, list] = {}
    for name, sha in PATCH_SHA256.items():
        if hashlib.sha256(patches[name].encode()).hexdigest() != sha:
            raise RuntimeError(f"nvfp4kv: {name} is not the vendored Lesilva patch")
        for rel, old, new in hunks(patches[name]):
            edits.setdefault(rel, []).append((old, new))
    pending = {}
    for rel, file_edits in edits.items():
        path = site / rel
        text = path.read_text(encoding="utf-8") if path.is_file() else None
        states = set()
        for old, new in file_edits:
            if old == "":  # a new file (0001's qsa/nvfp4_kv.py)
                if text is None and path.parent.is_dir():
                    text, state = new, "todo"
                elif text == new:
                    state = "done"
                else:
                    raise RuntimeError(f"nvfp4kv: {rel} exists with other content, or its package is missing")
            elif text is not None and text.count(new) == 1 and text.count(old) == new.count(old):
                state = "done"
            elif text is not None and text.count(old) == 1 and text.count(new) == 0:
                text, state = text.replace(old, new), "todo"
            else:
                raise RuntimeError(f"nvfp4kv: {rel} does not match the Pennyroyal v2.5.3 source the patch was "
                                   f"written for (a hunk's text is missing or not unique)")
            states.add(state)
        if states == {"todo", "done"}:
            raise RuntimeError(f"nvfp4kv: {rel} is half patched")
        if "todo" in states:
            pending[path] = text
    for path, text in pending.items():  # nothing is written until every file has checked out
        path.write_text(text, encoding="utf-8")
    return [site / rel for rel in edits]


def apply(venv, env, max_total_tokens: int, patches: dict) -> dict:
    """Patch SGLang, turn the QSA path on, and return the CFG entries cell 15 turns into server flags."""
    patched = patch_sglang(venv, patches)  # raises before the switch is set, so a failed patch never launches nvfp4
    env[SWITCH] = "1"
    print(f"nvfp4kv: {SWITCH}=1, target KV nvfp4, draft KV fp8_e4m3, pool cap {max_total_tokens:,} tokens; "
          f"patched {', '.join(p.name for p in patched)}", flush=True)
    return dict(SERVER_CFG, MAX_TOTAL_TOKENS=int(max_total_tokens))


def reference_args(args: list) -> list:
    """The same launch with the FP8 target KV the pinned config ships: the parity reference."""
    at = [i for i, a in enumerate(args) if a == "--kv-cache-dtype"]
    if len(at) != 1 or args[at[0] + 1] != SERVER_CFG["KVDTYPE"]:
        raise RuntimeError("nvfp4kv: expected exactly one --kv-cache-dtype nvfp4 in the launch arguments")
    out = list(args)
    out[at[0] + 1] = "fp8_e4m3"
    return out


def _post(url: str, payload: dict, timeout: float = 600) -> dict:
    request = urllib.request.Request(url, json.dumps(payload).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"nvfp4kv: {url} answered {exc.code}: {exc.read()[:500]!r}") from exc


def _render(system: str, user: str) -> str:
    return (f"<|im_start|>system\n{XHIGH}\n\n{system.strip()}<|im_end|>\n<|im_start|>user\n{user.strip()}"
            "<|im_end|>\n<|im_start|>assistant\n<think>\n")


def measure(base: str, prompts: dict, ids: list | None = None, decode_tokens: int = 8) -> list[dict]:
    """Per prompt: warm the cache with the rendered request A, then send A plus the recorded reply B and
    take B's teacher-forced logprobs. A prefill with no cached prefix attends over fresh BF16 K/V and never
    reads the pool, so only the second request, whose prefix comes from the cache, sees the KV dtype.
    Greedy decoding of a few tokens after B exercises the decode path. SGLang returns no logprob and no
    top row for the first scored position (logprob_result_processor.py prepends None), so B's first token
    is dropped; logprob_start_len stays len(A) because it also caps the cached prefix."""
    out = []
    for n, p in enumerate(prompts["prompts"]):
        if ids is None:  # tokenize once, on the reference server; the NVFP4 run reuses the same ids
            a = _post(base + "/tokenize", {"prompt": _render(prompts["system"], p["user"]), "add_special_tokens": False})["tokens"]
            b = _post(base + "/tokenize", {"prompt": p["continuation"], "add_special_tokens": False})["tokens"]
        else:
            a, b = ids[n]
        _post(base + "/generate", {"input_ids": a, "sampling_params": {"max_new_tokens": 1, "temperature": 0}})
        meta = _post(base + "/generate", {
            "input_ids": a + b, "return_logprob": True, "logprob_start_len": len(a), "top_logprobs_num": 1,
            "sampling_params": {"max_new_tokens": decode_tokens, "temperature": 0}})["meta_info"]
        out.append({"ids": (a, b), "cached": meta["cached_tokens"],
                    "lp": [x[0] for x in meta["input_token_logprobs"]][1:],
                    "top": [x[0][1] if x else None for x in meta["input_top_logprobs"]][1:],
                    "gen": [x[1] for x in meta["output_token_logprobs"]]})
    return out


def compare(ref: list, test: list, max_nll: float, min_agree: float) -> str:
    """Teacher-forced FP8 vs NVFP4 logprobs on the same tokens; raises unless the gates and the sanity
    checks hold, else returns the report line. The NLL gate is the signed change Lesilva reported; mean
    |dlogp| is always at least as large and has no measured null, so it is reported, not gated."""
    diffs, nlls, agree, common, cached = [], [], 0, [], []
    for n, (r, t) in enumerate(zip(ref, test, strict=True)):
        size = len(r["ids"][1]) - 1  # B minus its first token, which measure() drops
        for run, label in ((r, "fp8"), (t, "nvfp4")):
            if len(run["lp"]) != size or len(run["top"]) != size or None in run["lp"] or None in run["top"]:
                raise RuntimeError(f"nvfp4kv parity: prompt {n} ({label}) returned {len(run['lp'])} logprobs "
                                   f"for {size} scored tokens")
            if run["cached"] <= 0:
                raise RuntimeError(f"nvfp4kv parity: prompt {n} ({label}) reused no cached prefix, so its logprobs "
                                   "never read the KV pool; the comparison would be vacuous")
        diffs += [abs(x - y) for x, y in zip(r["lp"], t["lp"])]
        nlls += [x - y for x, y in zip(r["lp"], t["lp"])]  # NVFP4 NLL minus FP8 NLL per scored token
        agree += sum(x == y for x, y in zip(r["top"], t["top"]))
        common.append(next((k for k, (x, y) in enumerate(zip(r["gen"], t["gen"])) if x != y),
                           min(len(r["gen"]), len(t["gen"]))))
        cached.append(t["cached"] / len(t["ids"][0]))
    ranked = sorted(diffs)
    nll = sum(nlls) / len(nlls)
    mean_abs, agreement, decode = sum(diffs) / len(diffs), agree / len(diffs), statistics.median(common)
    report = (f"{len(diffs):,} scored tokens over {len(ref)} prompts; NLL change (nvfp4 minus fp8) {nll:+.4f} "
              f"nats/token (gate {max_nll}); top-1 agreement {agreement:.2%} (gate {min_agree:.0%}); greedy decode "
              f"agrees for a median of {decode:g} tokens (gate 2); not gated: mean |dlogp| {mean_abs:.4f}, p50/p90/p99 "
              f"{ranked[len(ranked) // 2]:.4f}/{ranked[int(len(ranked) * 0.9)]:.4f}/{ranked[int(len(ranked) * 0.99)]:.4f}; "
              f"prefix read from the pool for {min(cached):.0%} to {max(cached):.0%} of A")
    if nll > max_nll or agreement < min_agree or decode < 2:
        raise RuntimeError(f"nvfp4kv parity FAILED: {report}")
    return report


def _port_open(port: int) -> bool:
    with socket.socket() as probe:
        return probe.connect_ex(("127.0.0.1", port)) == 0


def _gpu_used_mib() -> int:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    return int(out.split()[0])


def _wait_ready(health_url: str, proc, timeout: float) -> None:
    t0, last = time.time(), 0
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"nvfp4kv: the server exited with code {proc.returncode} before READY; see serve.log")
        try:
            with urllib.request.urlopen(health_url, timeout=5) as r:
                if r.status == 200:
                    print(f"nvfp4kv: READY after {int(time.time() - t0)}s", flush=True)
                    return
        except Exception:
            pass
        if time.time() - t0 - last >= 60:
            last = int(time.time() - t0)
            print(f"  nvfp4kv: waiting for READY, {last}s ...", flush=True)
        time.sleep(POLL_S)
    raise RuntimeError(f"nvfp4kv: the server was not READY within {timeout:.0f}s")


def _stop(proc, port: int, timeout: float = 180) -> None:
    """End the server's process group (cell 15 starts it in its own session) and wait until the port is closed
    and the GPU has given its memory back, escalating SIGTERM to SIGKILL."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            pass
        deadline = time.time() + timeout / 2
        while time.time() < deadline:
            if proc.poll() is not None and not _port_open(port) and _gpu_used_mib() < 2048:
                return
            time.sleep(POLL_S)
    raise RuntimeError("nvfp4kv: the previous server did not release its port and GPU memory")


def relaunch_fp8(proc, args: list, env: dict, log, port: int) -> tuple:
    """Scored run only, from cell 15's READY wait: the NVFP4 server exited before READY. Clears what is left
    of it and starts the same launch on FP8 KV, the probe's reference boot (every 0001 path keys on the
    nvfp4 dtype, so the switch is inert). The cell then waits for it like a first boot, and each game's first
    request already retries for 900 s while a server comes up (ARC3_HTTP_RETRY_INITIAL_SECONDS), which covers
    a reload. FP8 at 20 streams is worse than the pinned config, but not zero. Returns (proc, args)."""
    try:
        _stop(proc, port)
    except Exception as exc:  # a slow GPU release is no reason to skip the attempt; the boot logs its own failure
        print(f"nvfp4kv: {exc}", flush=True)
    args = reference_args(args)
    with open(log, "ab", buffering=0) as logf:
        proc = subprocess.Popen(args, env=env, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
    print(f"nvfp4kv: the NVFP4 server exited before READY; relaunched once on FP8 KV, pid {proc.pid} -> {log}",
          flush=True)
    return proc, args


def parity(proc, args: list, env: dict, log, health_url: str, prompts: dict, max_nll: float = 0.02,
           min_agree: float = 0.95, timeout: float = 1800) -> tuple:
    """Probe arm only. `proc` is the FP8 reference server cell 15 just launched. Scores the prompts on it,
    restarts the server with `args` (NVFP4), scores the same token ids again and compares. Raises unless the
    NVFP4 NLL of the scored tokens is at most max_nll nats per token above FP8 and top-1 agreement is at
    least min_agree (teacher-forced argmax flips at near-ties even on a healthy pool, hence 95%), and also
    when the NVFP4 boot never logged the QSA switch or a scored request read nothing from the pool.
    Returns the NVFP4 server process and its serve.log offset for cell 15 to keep: 0, because the FP8 boot's
    log moves to serve-fp8-reference.log first."""
    base = health_url.rsplit("/", 1)[0]
    _wait_ready(health_url, proc, timeout)
    ref = measure(base, prompts)
    _stop(proc, urllib.parse.urlsplit(health_url).port)
    # serve.log keeps one boot, as in every other run, so the readout parses the NVFP4 server; the FP8 one moves aside.
    reference_log = Path(log).with_name(Path(log).stem + "-fp8-reference.log")
    os.replace(log, reference_log)
    with open(log, "ab", buffering=0) as logf:
        proc = subprocess.Popen(args, env=env, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
    print(f"nvfp4kv: FP8 reference scored ({reference_log.name}); NVFP4 server pid {proc.pid} -> {log}", flush=True)
    _wait_ready(health_url, proc, timeout)
    boot = Path(log).read_bytes().decode(errors="replace")
    if KV4_LINE not in boot:
        raise RuntimeError(f"nvfp4kv: the NVFP4 boot never logged '{KV4_LINE}', so QSA does not read NVFP4")
    for line in boot.splitlines():
        if "KV Cache is allocated" in line or "max_total_num_tokens=" in line:
            print("nvfp4kv:", line.strip(), flush=True)
    report = compare(ref, measure(base, prompts, [r["ids"] for r in ref]), max_nll, min_agree)
    print(f"nvfp4kv parity PASS: {report}", flush=True)
    return proc, 0
