"""Arm thinkbudget (B5): a strict thinking budget that holds under NEXTN. Research:
docs/research/2026-10-05-top5/throughput.md section 2.1.

Runs inside the launcher cell (cell 15): args = apply(VENV, env, args, budget) after the argv is
assembled and before the launch; check(url, proc, CFG["SERVED_NAME"], strict=not TRUE_SUBMISSION)
after the cell's READY wait.

Why not a custom logit processor: NEXTN resolves to EAGLE, whose verify samples through
eagle_sample (speculative/eagle_utils.py). That applies penalties, logit_bias and the grammar mask,
never custom logit processors, so a processor would touch only the first token after a prefill.

What the flag does in Pennyroyal v2.5.3: --enable-strict-thinking gives every thinking request a
ReasonerGrammarObject (constrained/reasoner_grammar_backend.py). Its budget is
SGLANG_MAX_THINK_TOKENS (default -1, read once at startup) or custom_params["thinking_budget"] per
request; once the budget is reached the mask allows only </think>. EAGLE verify builds that mask
over the draft chain on the host while the target forward runs (spec_utils.build_grammar_vocab_mask)
and applies it before sampling; supports_grammar_overlap() is true for EAGLE, so overlap stays on.
A reply ended by the budget reports reasoning_tokens = budget + 1 (the forced </think> counts).

Why the one-line exclusion patch: strict mode also blocks Qwen3Detector.think_excluded_tokens
(<tool_call>, </tool_call>, <|im_end|>, <|endoftext|>) on every thinking step under the budget, a
filter call per thinking row per verify step, and the list alone switches the filter on. Emptying
it leaves the budget as the only change. The filter then turns on only because the budget is >= 0
(ReasonerGrammarBackend.enable_token_filter), so a self-test whose thinking is cut at 64 tokens
also proves the env reached the server (one that stops thinking sooner proves neither).
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path

MARK = "arm thinkbudget"
FLAG = "--enable-strict-thinking"
ENV = "SGLANG_MAX_THINK_TOKENS"
TEST_BUDGET, TEST_MAX_REASONING = 64, 80  # a bound reply reports 65

EXCLUDED_OLD = ('        think_excluded_tokens = [\n'
                '            "<tool_call>",\n'
                '            "</tool_call>",\n'
                '            "<|im_end|>",\n'
                '            "<|endoftext|>",\n'
                '        ]\n')
EXCLUDED_NEW = EXCLUDED_OLD + f"        think_excluded_tokens = []  # {MARK}: strict thinking enforces only the budget\n"

PROMPT = ("Use the python tool to find the smallest positive integer n such that n! ends in exactly 1000 "
          "zeros, and print it. Before you call the tool, reason carefully about how trailing zeros of n! "
          "are counted, which counts are impossible, and how to search efficiently.")
TOOL = {"type": "function", "function": {
    "name": "python", "description": "Run Python code and return what it prints.",
    "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}}


def patch_sglang(venv) -> Path:
    """Empty Qwen3Detector's exclusion list in the installed SGLang; a rerun of the cell is a no-op."""
    roots = sorted(Path(venv).glob("lib/python*/site-packages/sglang"))
    if len(roots) != 1:
        raise RuntimeError(f"thinkbudget: expected one installed sglang under {venv}, found {roots}")
    path = roots[0] / "srt/parser/reasoning_parser.py"
    text = path.read_text()
    if EXCLUDED_NEW in text:
        return path
    if MARK in text or text.count(EXCLUDED_OLD) != 1:
        raise RuntimeError(f"thinkbudget: {path.name} does not match the Pennyroyal v2.5.3 source; "
                           "this SGLang build is not the one the arm was written against")
    path.write_text(text.replace(EXCLUDED_OLD, EXCLUDED_NEW))
    return path


def apply(venv, env, args: list, budget: int) -> list:
    if not isinstance(budget, int) or budget < 1:
        raise RuntimeError(f"thinkbudget: budget must be a positive int, got {budget!r}")
    parser = args[args.index("--reasoning-parser") + 1] if "--reasoning-parser" in args[:-1] else None
    if parser != "qwen3":
        raise RuntimeError("thinkbudget: the exclusion patch is for Qwen3Detector; launch with --reasoning-parser qwen3")
    path = patch_sglang(venv)  # raises before the flag and env are set, so a failed patch never ships them
    env[ENV] = str(budget)
    print(f"thinkbudget: {FLAG} with {ENV}={budget}; Qwen3Detector think exclusions emptied in {path.name}",
          flush=True)
    return args if FLAG in args else args + [FLAG]


def verdict(reply: dict) -> tuple[str | None, str]:
    """(problem or None, summary) for the self-test reply. A reply cut at max_tokens can fail only on
    the reasoning bound and the split: its answer is unfinished, so the parse checks do not apply."""
    choice = (reply.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    content, calls = msg.get("content") or "", msg.get("tool_calls") or []
    n = (reply.get("usage") or {}).get("reasoning_tokens")
    summary = (f"{n} reasoning tokens at a {TEST_BUDGET}-token budget, {len(calls)} tool call(s), "
               f"finish {choice.get('finish_reason')}")
    if not isinstance(n, int) or n < 1:
        return "no reasoning tokens reported; the request did not run in thinking mode", summary
    if n > TEST_MAX_REASONING:
        return f"the budget did not stop thinking ({n} reasoning tokens, limit {TEST_MAX_REASONING})", summary
    if not reasoning.strip() or any(tag in reasoning + content for tag in ("<think>", "</think>")):
        return "reasoning and answer did not split cleanly at the forced </think>", summary
    if choice.get("finish_reason") == "length":
        summary += " (answer cut at max_tokens, parse not exercised)"
    else:
        if "<tool_call>" in content:
            return "tool-call markup left in the answer text: the call did not parse", summary
        for call in calls:
            try:
                code = json.loads(call["function"]["arguments"])["code"]
            except (KeyError, TypeError, ValueError):
                code = None
            if not isinstance(code, str):
                return "a tool call after the forced </think> did not parse", summary
        if not calls and not content.strip():
            return "nothing after the forced </think>: no tool call and no answer", summary
    if n < TEST_BUDGET:
        summary += " (stopped thinking before the budget, so the cut itself was not exercised)"
    return None, summary


def check(health_url: str, proc, model: str, strict: bool) -> None:
    """After cell 15's wait, bench runs only (strict): one request with a 64-token budget must stop
    thinking by 80 reasoning tokens and still parse, or the bench stops; so does a server that exited.
    A server still loading at the deadline skips the request instead of holding the release: game
    clocks start at release, so waiting would give this arm the boot time its control loses (about
    2, 10 and 14 minutes in the three slow boots so far); thinking_counters.py reads the run instead.
    The scored run skips it too: its log is never readable, so the request would only delay play."""
    if not strict:
        print("thinkbudget: scored run, self-test skipped", flush=True)
        return
    try:
        with urllib.request.urlopen(health_url, timeout=5) as r:
            up = r.status == 200
    except Exception:
        up = False
    if not up:
        if proc.poll() is not None:
            raise RuntimeError(f"thinkbudget: the server exited (code {proc.returncode}) before READY")
        print("thinkbudget: server still loading at the startup deadline, self-test skipped so the games' "
              "clocks start as in the base fork; read the budget off the run with thinking_counters.py", flush=True)
        return
    payload = {"model": model, "messages": [{"role": "user", "content": PROMPT}], "tools": [TOOL],
               "tool_choice": "auto", "max_tokens": 4096, "temperature": 0.7, "top_p": 0.95, "top_k": 20,
               "chat_template_kwargs": {"enable_thinking": True},
               "custom_params": {"thinking_budget": TEST_BUDGET}}
    url = health_url.rsplit("/health", 1)[0] + "/v1/chat/completions"
    request = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=180) as r:
            body = r.read()
    except urllib.error.HTTPError as exc:
        body = b""
        try:
            body = exc.read()
        except Exception:
            pass
        problem, summary = f"self-test request rejected, HTTP {exc.code}", body[:400].decode(errors="replace")
    except Exception as exc:
        problem, summary = f"self-test request failed ({exc!r})", "the server answered /health but not this request"
    else:
        try:
            problem, summary = verdict(json.loads(body))
        except Exception as exc:
            problem, summary = f"self-test reply unreadable ({exc!r})", body[:400].decode(errors="replace")
    if problem:
        raise RuntimeError(f"thinkbudget: {problem}. {summary}")
    print(f"thinkbudget: self-test passed, {summary}", flush=True)
