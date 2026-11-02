"""Dev kernel only: the leaner-trim decision test (docs/research/2026-10-12-next/PLAN.md 5.1).

What. solution/analysis/trim_pairs.py turns a replica's request logs into post-trim states, each as the logged
messages ("full") and the same messages with older reasoning and images slimmed ("lean"). This samples the next
reply SAMPLES times per variant from the notebook's own server, with the live sampling (cell 7: T 0.7, top_p 0.95,
top_k 20) and no seed, so full-vs-full disagreement is the noise floor that full-vs-lean is judged against
(solution/analysis/trim_compare.py). Nothing here plays a game; build_franzen_arms refuses it on a scored path.

How. run() posts every (state, variant, sample) to /v1/chat/completions on a thread pool the size of the admitted
streams, state by state so the second sample of a variant reuses the first one's prefix, and appends each reply
(reasoning, tool calls, finish reason, usage) to OUT_DIR/trimtest_replies.jsonl as it lands. Stdlib only.
"""
from __future__ import annotations

import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SAMPLING = {"temperature": 0.7, "top_p": 0.95, "top_k": 20}


def _post(url: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def logprob_tail(pairs: list, url: str, model: str, out_dir, n: int, workers: int, timeout: float) -> None:
    """D4's K0 input at no extra session: one full-history reply for the first n states with the top-20 logprobs of
    every generated token (post-temperature, the sampler's default), to OUT_DIR/trimtest_logprobs.jsonl. Errors
    are recorded, never raised: logprobs under MTP are untested on this wheel."""
    out = Path(out_dir) / "trimtest_logprobs.jsonl"

    def call(i):
        p = pairs[i]
        body = {"model": model, "messages": p["full"], "max_tokens": 2048, "stream": False, "logprobs": True,
                "top_logprobs": 20, **SAMPLING}
        for key in ("tools", "tool_choice", "chat_template_kwargs"):
            if p.get(key):
                body[key] = p[key]
        try:
            r = _post(url, body, timeout)
            content = ((r["choices"][0].get("logprobs") or {}).get("content")) or []
            return {"state": i, "error": None, "tokens": [
                [t.get("logprob"), [x.get("logprob") for x in t.get("top_logprobs") or []]] for t in content]}
        except Exception as exc:
            return {"state": i, "error": repr(exc)[:500]}

    with out.open("w") as f, ThreadPoolExecutor(workers) as pool:
        rows = list(pool.map(call, range(min(n, len(pairs)))))
        for row in rows:
            f.write(json.dumps(row) + "\n")
    bad = sum(r["error"] is not None for r in rows)
    print(f"trimtest: logprob tail {len(rows) - bad} ok, {bad} errors -> {out}", flush=True)


def run(pairs_path, base_url: str, model: str, out_dir, samples: int = 2, max_tokens: int = 8192,
        workers: int = 14, timeout: float = 1800.0, logprob_states: int = 30) -> Path:
    pairs = [json.loads(line) for line in Path(pairs_path).open()]
    jobs = [(i, v, s) for i in range(len(pairs)) for v in ("full", "lean") for s in range(samples)]
    out = Path(out_dir) / "trimtest_replies.jsonl"
    url = base_url.rstrip("/") + "/v1/chat/completions"
    t0 = time.time()

    def call(job):
        i, variant, sample = job
        p = pairs[i]
        body = {"model": model, "messages": p[variant], "max_tokens": max_tokens, "stream": False, **SAMPLING}
        for key in ("tools", "tool_choice", "chat_template_kwargs"):
            if p.get(key):
                body[key] = p[key]
        start = time.time()
        try:
            r = _post(url, body, timeout)
            choice = r["choices"][0]
            msg = choice.get("message") or {}
            row = {"reasoning": msg.get("reasoning_content") or "", "content": msg.get("content") or "",
                   "tool_calls": msg.get("tool_calls") or [], "finish_reason": choice.get("finish_reason"),
                   "usage": r.get("usage"), "error": None}
        except Exception as exc:  # recorded, never fatal: one bad state must not sink the batch
            row = {"error": repr(exc)[:500]}
        return {"state": i, "game": p.get("game"), "request_index": p.get("request_index"), "variant": variant,
                "sample": sample, "seconds": round(time.time() - start, 1), **row}

    done = errors = 0
    with out.open("w") as f, ThreadPoolExecutor(workers) as pool:
        for row in pool.map(call, jobs):
            f.write(json.dumps(row) + "\n")
            f.flush()
            done += 1
            errors += row["error"] is not None
            if done % 20 == 0 or done == len(jobs):
                print(f"trimtest: {done}/{len(jobs)} replies, {errors} errors, {time.time() - t0:.0f} s", flush=True)
    print(f"trimtest: wrote {out} ({len(pairs)} states x 2 variants x {samples} samples)", flush=True)
    if logprob_states:
        logprob_tail(pairs, url, model, out_dir, logprob_states, workers, timeout)
    return out
