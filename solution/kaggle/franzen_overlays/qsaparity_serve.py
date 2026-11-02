"""Probe qsaparity: does the QSA ring defect change what the model predicts, on the real card?

Probe runs only, never a scored rerun. Cell 15 boots the shipped SGLang as usual; this then measures,
applies the ring fix (franzen_overlays/qsaring_serve.py, passed in as `qsaring`), relaunches the same
server command and measures again.

What is measured, per recorded prompt A (our system prompt plus one recorded turn, from
nvfp4kv/parity_prompts.json):
  1. flush the radix cache, then greedy-generate G tokens D from A with MTP on: every verify forward
     writes D's QSA index keys through the pending ring, the path the defect corrupts;
  2. greedy-generate K tokens after A + D with the prefix served from the cache (D's decode-time keys);
  3. flush again and greedy-generate K tokens after A + D from a fresh prefill (D's keys written by the
     extend path, which the defect does not touch);
  4. flush and prefill once more, as the noise floor of step 3 against itself.
The defect predicts that, on the shipped server, step 2 disagrees with step 3 more often than it does
once the fix is applied. Decode and prefill also differ in ordinary numerics, so the reading is the
CHANGE in agreement between the two boots, never the level on one boot. Nothing is gated; the report
goes to stdout and /kaggle/working/qsaparity.json.
"""
from __future__ import annotations

import json
import os
import signal
import socket
import statistics
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

POLL_S = 3
XHIGH = ("Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, "
         "consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.")


def _post(url: str, payload: dict | None = None, timeout: float = 600) -> dict | str:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url, data, {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"qsaparity: {url} answered {exc.code}: {exc.read()[:400]!r}") from exc
    try:
        return json.loads(body)
    except ValueError:
        return body.decode(errors="replace")


def _render(system: str, user: str) -> str:
    return (f"<|im_start|>system\n{XHIGH}\n\n{system.strip()}<|im_end|>\n<|im_start|>user\n{user.strip()}"
            "<|im_end|>\n<|im_start|>assistant\n<think>\n")


def _greedy(base: str, ids: list, n: int) -> dict:
    meta = _post(base + "/generate", {
        "input_ids": ids, "return_logprob": True, "top_logprobs_num": 1,
        "sampling_params": {"max_new_tokens": n, "temperature": 0}})["meta_info"]
    out = meta["output_token_logprobs"]
    return {"ids": [x[1] for x in out], "lp": [x[0] for x in out], "cached": int(meta.get("cached_tokens") or 0)}


def _flush(base: str) -> None:
    for _ in range(20):
        reply = _post(base + "/flush_cache")
        if "Cache flushed" in str(reply):
            return
        time.sleep(1)
    raise RuntimeError(f"qsaparity: /flush_cache did not flush: {reply!r}")


def measure(base: str, prompts: dict, n_prompts: int, gen: int, tail: int) -> list[dict]:
    rows = []
    for p in prompts["prompts"][:n_prompts]:
        a = _post(base + "/tokenize", {"prompt": _render(prompts["system"], p["user"]), "add_special_tokens": False})["tokens"]
        _flush(base)
        d = _greedy(base, a, gen)["ids"]
        cached = _greedy(base, a + d, tail)
        _flush(base)
        fresh = _greedy(base, a + d, tail)
        _flush(base)
        again = _greedy(base, a + d, tail)
        rows.append({"a": len(a), "d": len(d), "cached_tokens": cached["cached"], "fresh_cached_tokens": fresh["cached"],
                     "cached": cached, "fresh": fresh, "again": again})
    return rows


def _agree(x: dict, y: dict) -> tuple[int, int, float]:
    """(first token equal, tokens equal before the first divergence, |first logprob difference|)."""
    same = next((k for k, (i, j) in enumerate(zip(x["ids"], y["ids"])) if i != j), min(len(x["ids"]), len(y["ids"])))
    return int(x["ids"][:1] == y["ids"][:1]), same, abs(x["lp"][0] - y["lp"][0]) if x["lp"] and y["lp"] else 0.0


def summarize(rows: list[dict]) -> dict:
    valid = [r for r in rows if r["cached_tokens"] >= r["a"] + r["d"] - 8]
    def stats(key_x: str, key_y: str) -> dict:
        got = [_agree(r[key_x], r[key_y]) for r in valid]
        return {"first_token_agreement": sum(g[0] for g in got) / max(1, len(got)),
                "median_tokens_before_divergence": statistics.median(g[1] for g in got) if got else None,
                "mean_abs_first_logprob_diff": sum(g[2] for g in got) / max(1, len(got))}
    return {"prompts": len(rows), "prompts_reusing_decode_keys": len(valid),
            "cached_vs_fresh": stats("cached", "fresh"), "fresh_vs_fresh": stats("fresh", "again")}


def _port_open(port: int) -> bool:
    with socket.socket() as probe:
        return probe.connect_ex(("127.0.0.1", port)) == 0


def _gpu_used_mib() -> int:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    return int(out.split()[0])


def _wait_ready(health_url: str, proc, timeout: float) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"qsaparity: the server exited with code {proc.returncode} before READY; see serve.log")
        try:
            with urllib.request.urlopen(health_url, timeout=5) as response:
                if response.status == 200:
                    print(f"qsaparity: READY after {int(time.time() - t0)}s", flush=True)
                    return
        except Exception:
            pass
        time.sleep(POLL_S)
    raise RuntimeError(f"qsaparity: the server was not READY within {timeout:.0f}s")


def _stop(proc, port: int, timeout: float = 180) -> None:
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
    raise RuntimeError("qsaparity: the server did not release its port and GPU memory")


def probe(proc, args: list, env: dict, log, health_url: str, prompts: dict, venv, qsaring: dict,
          n_prompts: int = 16, gen: int = 512, tail: int = 16, timeout: float = 1800) -> tuple:
    """Measure on the shipped server, apply the ring fix, relaunch, measure again; returns (proc, 0) with the
    fixed server running so the probe's short game run exercises it too."""
    base = health_url.rsplit("/", 1)[0]
    _wait_ready(health_url, proc, timeout)
    shipped = measure(base, prompts, n_prompts, gen, tail)
    _stop(proc, urllib.parse.urlsplit(health_url).port)
    os.replace(log, Path(log).with_name(Path(log).stem + "-shipped.log"))
    qsaring["apply"](venv)
    with open(log, "ab", buffering=0) as logf:
        proc = subprocess.Popen(args, env=env, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
    print(f"qsaparity: shipped server measured; fixed server pid {proc.pid} -> {log}", flush=True)
    _wait_ready(health_url, proc, timeout)
    fixed = measure(base, prompts, n_prompts, gen, tail)
    report = {"shipped": summarize(shipped), "fixed": summarize(fixed),
              "settings": {"prompts": n_prompts, "generated_tokens": gen, "compared_tokens": tail}}
    Path("/kaggle/working/qsaparity.json").write_text(json.dumps({"report": report, "shipped_rows": shipped,
                                                                  "fixed_rows": fixed}, indent=1))
    print("qsaparity REPORT:", json.dumps(report), flush=True)
    return proc, 0
