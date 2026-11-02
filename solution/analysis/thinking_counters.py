"""Thinking-budget counters for Franzen-fork runs (lever B5, docs/research/2026-10-05-top5/PLAN.md).

    python3 solution/analysis/thinking_counters.py RUN_DIR [RUN_DIR ...] [--budget 10240]
    python3 solution/analysis/thinking_counters.py --self-check

RUN_DIR as for transcript_waste.py: transcripts-dl/*_p0.txt, plus usage/*_usage.jsonl from
`transcript_waste.py --fetch-usage RUN_DIR` (the request logs cut down to their usage fields).
Replies pair with usage rows by order and every pair's finish_reason must agree. Without usage
only the transcript counters are filled.

  cut, no tool call   finish_reason=length and no tool call: cut at the output cap while still
                      thinking. Control p448s14: 67 in two runs. The B5 gate wants 0.
  past the budget     reasoning_tokens > budget. Under strict thinking at this budget each one
                      was ended by it and reports budget + 1 (the forced </think> counts); in a
                      control they are the replies the budget would have ended.
    with a call       ... and the reply still carried a tool call (the harness drops malformed
                      calls before counting, so a call cut off mid-arguments does not count)
    call ran          ... and its code ran: a tool result that is neither a timeout nor a
                      SyntaxError (runtime errors count as ran, as for any reply)
  max reasoning       the B5 gate wants at most budget + 1
  implicit close      a tool call with reasoning_tokens >= completion_tokens: <tool_call> opened
                      without </think>, so the think count ran through the call. Under a budget,
                      such a reply that passes it gets </think> forced inside its call.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from transcript_waste import EXC, classify, generations  # noqa: E402

BUDGET = 10240
SYNTAX = {"SyntaxError", "IndentationError", "TabError"}


def replies(run: Path) -> list[dict]:
    """Every transcript reply with its usage dict (None if the run has no usage row for it)."""
    out = []
    for path in sorted((run / "transcripts-dl").glob("*_p0.txt")):
        up = run / "usage" / path.name.replace(".txt", "_usage.jsonl")
        rows = [r for r in map(json.loads, up.open()) if r["event"] == "response"] if up.exists() else []
        for i, g in enumerate(generations(path.read_text(errors="replace"))):
            row = rows[i] if i < len(rows) else None  # the final in-flight reply has no row
            if row is not None and (row["finish_reason"] or "") != g["finish"]:
                raise SystemExit(f"{path.name} reply {i}: transcript says finish {g['finish']!r}, usage says "
                                 f"{row['finish_reason']!r}; replies and usage rows are misaligned")
            out.append(dict(g, usage=None if row is None else (row["usage"] or {})))
    return out


def ran(g: dict) -> bool:
    r = g["result"]
    return g["calls"] > 0 and bool(r.strip()) and "Tool timed out after" not in r and not SYNTAX & set(EXC.findall(r))


def counters(run: Path, budget: int = BUDGET) -> dict:
    c, classes, max_rt = Counter(), Counter(), 0
    for g in replies(run):
        c["replies"] += 1
        cut = g["finish"] == "length" and g["calls"] == 0
        c["cut_no_call"] += cut
        if g["usage"] is None:
            continue
        rt, ct = g["usage"].get("reasoning_tokens") or 0, g["usage"].get("completion_tokens") or 0
        c["with_usage"] += 1
        c["tokens"] += ct
        c["cut_no_call_tokens"] += ct * cut
        c["implicit_close"] += g["calls"] > 0 and rt >= ct > 0
        max_rt = max(max_rt, rt)
        if rt > budget:
            c["past_budget"] += 1
            c["past_budget_with_call"] += g["calls"] > 0
            c["past_budget_call_ran"] += ran(g)
            classes[classify(g)] += 1
    return {"budget": budget, **c, "max_reasoning_tokens": max_rt if c["with_usage"] else None,
            "past_budget_classes": dict(classes.most_common())}


def report(name: str, c: dict) -> str:
    if not c.get("with_usage"):
        return f"{name}: {c.get('replies', 0)} replies, {c.get('cut_no_call', 0)} cut at the cap with no tool call; no usage rows"
    share = 100 * c["cut_no_call_tokens"] / c["tokens"] if c["tokens"] else 0.0
    classes = ", ".join(f"{k} {v}" for k, v in c["past_budget_classes"].items()) or "none"
    return "\n".join([
        f"{name}: {c['replies']} replies ({c['with_usage']} with usage), budget {c['budget']}",
        f"  cut at the cap, no tool call  {c['cut_no_call']} ({share:.1f}% of {c['tokens']} generated tokens)",
        f"  past the budget               {c['past_budget']}: with a call {c['past_budget_with_call']}, "
        f"call ran {c['past_budget_call_ran']} ({classes})",
        f"  max reasoning_tokens          {c['max_reasoning_tokens']} (strict thinking caps it at {c['budget'] + 1})",
        f"  implicit close                {c['implicit_close']}"])


SAMPLE = """--- analysis_step=1 | action=1 | 00:00:01 | tool-agent ---
[MODEL RESPONSE META]
finish_reason: length
tool_call_count: 0

[MODEL RESPONSE META]
finish_reason: tool_calls
tool_call_count: 1

[TOOL RESULT: python]
[action] UP -> moved

[MODEL RESPONSE META]
finish_reason: tool_calls
tool_call_count: 1

[TOOL RESULT: python]
Traceback (most recent call last):
  File "<python_tool>", line 3, in <module>
SyntaxError: '(' was never closed

--- analysis_step=2 | action=2 | 00:00:09 | tool-agent ---
[MODEL RESPONSE META]
finish_reason: tool_calls
tool_call_count: 1

[TOOL RESULT: python]
42

[MODEL RESPONSE META]
finish_reason: stop
tool_call_count: 0
"""
USAGE = [("length", 12291, 12291), ("tool_calls", 10241, 10900), ("tool_calls", 10241, 10350), ("tool_calls", 700, 700)]


def self_check() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp)
        (run / "transcripts-dl").mkdir()
        (run / "usage").mkdir()
        (run / "transcripts-dl/g-1_p0.txt").write_text(SAMPLE)
        rows = [{"event": e, "finish_reason": f if e == "response" else None,
                 "usage": {"reasoning_tokens": rt, "completion_tokens": ct} if e == "response" else None}
                for f, rt, ct in USAGE for e in ("request", "response")]
        (run / "usage/g-1_p0_usage.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        c = counters(run)
        want = {"replies": 5, "with_usage": 4, "cut_no_call": 1, "past_budget": 3, "past_budget_with_call": 2,
                "past_budget_call_ran": 1, "implicit_close": 1, "max_reasoning_tokens": 12291}
        assert {k: c[k] for k in want} == want, c
        assert c["past_budget_classes"] == {"no_tool_call_length": 1, "ok": 1, "exception_no_output": 1}, c
        assert counters(run, budget=10241)["past_budget"] == 1  # a forced reply sits exactly at budget + 1
        rows[1]["finish_reason"] = "stop"  # a shifted pairing must stop the report, not skew it
        (run / "usage/g-1_p0_usage.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        try:
            counters(run)
            raise AssertionError("misaligned usage rows were accepted")
        except SystemExit as exc:
            assert "misaligned" in str(exc)
    print("self-check ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", type=Path)
    ap.add_argument("--budget", type=int, default=BUDGET)
    ap.add_argument("--self-check", action="store_true")
    a = ap.parse_args()
    if a.self_check:
        self_check()
    total = Counter()
    for run in a.runs:
        c = counters(run, a.budget)
        print(report(run.name, c))
        total.update({k: c.get(k, 0) for k in ("cut_no_call", "past_budget", "past_budget_with_call",
                                               "past_budget_call_ran", "implicit_close")})
    if len(a.runs) > 1:
        print(f"all {len(a.runs)} runs: cut at the cap with no tool call {total['cut_no_call']}, past the budget "
              f"{total['past_budget']} (with a call {total['past_budget_with_call']}, call ran "
              f"{total['past_budget_call_ran']}), implicit close {total['implicit_close']}")
