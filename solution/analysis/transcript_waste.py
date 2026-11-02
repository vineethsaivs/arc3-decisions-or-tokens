"""Where a TAAF/Franzen run spends generated tokens: per-generation waste classes from transcripts.

    python3 solution/analysis/transcript_waste.py RUN_DIR [RUN_DIR ...]
    python3 solution/analysis/transcript_waste.py --fetch-usage RUN_DIR   # once, while signed URLs live
    python3 solution/analysis/transcript_waste.py --self-check

RUN_DIR holds transcripts-dl/*_p0.txt (kapi.sh output <slug> RUN_DIR/transcripts-dl "transcripts/").
Exact token counts come from RUN_DIR/usage/*_usage.jsonl: --fetch-usage streams each
*_p0_requests.jsonl listed in transcripts-dl/_raw.json, keeps only the usage fields, and deletes
the multi-MB file. Without usage the report still counts generations, just not tokens.
"""
from __future__ import annotations

import json
import re
import sys
import urllib.request
from collections import Counter
from pathlib import Path

HEADER = re.compile(r"^--- analysis_step=(\d+) \| action=(\d+) \| \S+ \| tool-agent ---$", re.M)
LABELS = ("SYSTEM PROMPT", "USER PROMPT", "ANALYZER STATUS", "MODEL RESPONSE META", "THINKING",
          "ASSISTANT", "TOOL CALL: python", "TOOL RESULT: python")
SECTION = re.compile(r"^\[(" + "|".join(map(re.escape, LABELS)) + r")\]$", re.M)
EXC = re.compile(r"^([A-Za-z_][\w.]*(?:Error|Exception))(?::|$)", re.M)
ECHO = re.compile(r"^\[action\] (.*)$", re.M)
STRICT = ("no_tool_call_length", "no_tool_call_stop", "exception_no_output", "tool_timeout")


def generations(text: str) -> list[dict]:
    """One dict per model reply, in transcript order (= request-log order)."""
    out, cur = [], None
    marks = list(SECTION.finditer(text))
    for i, m in enumerate(marks):
        body = text[m.end(): marks[i + 1].start() if i + 1 < len(marks) else len(text)]
        body = HEADER.split(body)[0].strip("\n")  # a turn header can end the last section
        label = m.group(1)
        if label == "MODEL RESPONSE META":
            f = dict(line.split(": ", 1) for line in body.split("raw_tool_calls")[0].splitlines() if ": " in line)
            cur = {"finish": f.get("finish_reason", ""), "calls": int(f.get("tool_call_count", "0") or 0),
                   "result": ""}
            out.append(cur)
        elif label == "TOOL RESULT: python" and cur is not None:
            cur["result"] += body
    return out


def verdicts(result: str) -> list[str]:
    """Per executed action: 'noop' or 'moved', from the harness's [action] echo lines.

    A batch echo names only its stop reason; ARC3_BATCH_NOOP_BLOCK stops a batch at its
    first no-op, so only the last executed action of a no_op_action batch is a no-op."""
    out = []
    for line in ECHO.findall(result):
        m = re.match(r"batch of \d+ actions, (\d+) executed.*?(?:\| stop_reason: (\S+))?$", line)
        if m:
            k = int(m.group(1))
            out += ["moved"] * (k - 1) + (["noop" if m.group(2) == "no_op_action" else "moved"] if k else [])
        elif "-> not executed" not in line:
            out.append("noop" if "-> NO-OP" in line else "moved")
    return out


def classify(g: dict) -> str:
    r = g["result"]
    v = verdicts(r)
    acted = bool(v)
    if g["calls"] == 0:
        return "no_tool_call_" + (g["finish"] or "none")
    if "Tool timed out after" in r:
        return "tool_timeout"
    if "Traceback (most recent call last)" in r:
        if acted:
            return "exception_after_actions"
        before = re.sub(r"^\s*error:\s*$", "", r.split("Traceback (most recent call last)")[0], flags=re.M)
        return "exception_partial_output" if before.strip() else "exception_no_output"
    if acted and set(v) == {"noop"}:
        return "noop_only_actions"
    return "ok"


def fetch_usage(run: Path) -> None:
    usage = run / "usage"
    usage.mkdir(exist_ok=True)
    for e in json.loads((run / "transcripts-dl/_raw.json").read_text())["files"]:
        name = e["fileName"]
        if "/" in name or not name.endswith("_p0_requests.jsonl"):
            continue
        part = usage / (name + ".part")
        urllib.request.urlretrieve(e["url"], part)
        keep = ("event", "analysis_step", "action", "request_index_within_turn", "finish_reason", "usage")
        rows = [{k: json.loads(line).get(k) for k in keep} for line in part.open()]
        (usage / name.replace("_requests.jsonl", "_usage.jsonl")).write_text(
            "".join(json.dumps(x) + "\n" for x in rows))
        part.unlink()
        print(name, len(rows), flush=True)


def report(runs: list[Path]) -> Counter:
    n, tok, exc = Counter(), Counter(), Counter()
    for run in runs:
        for path in sorted((run / "transcripts-dl").glob("*_p0.txt")):
            gens = generations(path.read_text())
            up = run / "usage" / path.name.replace(".txt", "_usage.jsonl")
            resp = [json.loads(x) for x in up.open()] if up.exists() else []
            resp = [x for x in resp if x["event"] == "response"]
            for i, g in enumerate(gens):  # the final reply of a run can lack its usage row
                c = classify(g)
                t = (resp[i]["usage"] or {}).get("completion_tokens", 0) if i < len(resp) else 0
                n[c] += 1
                tok[c] += t
                tok["_all"] += t
                if "Traceback" in g["result"]:
                    found = EXC.findall(g["result"])
                    exc[found[-1] if found else "?"] += 1
    total = tok["_all"] or 1
    print(f"{'class':28s} {'gens':>6s} {'tokens':>10s} {'share':>7s}")
    for c in sorted(n, key=lambda k: -tok[k]):
        print(f"{c:28s} {n[c]:6d} {tok[c]:10d} {100 * tok[c] / total:6.2f}%")
    strict = sum(tok[c] for c in STRICT)
    print(f"strict waste (no code output at all): {strict} tokens, {100 * strict / total:.1f}% of {tok['_all']}")
    print("top exceptions:", exc.most_common(8))
    return n


SAMPLE = """--- analysis_step=1 | action=1 | 00:00:01 | tool-agent ---
[MODEL RESPONSE META]
finish_reason: tool_calls
tool_call_count: 1

[TOOL RESULT: python]
Traceback (most recent call last):
  File "<python_tool>", line 1, in <module>
NameError: name 'frames' is not defined

[MODEL RESPONSE META]
finish_reason: length
tool_call_count: 0

--- analysis_step=2 | action=2 | 00:00:09 | tool-agent ---
[MODEL RESPONSE META]
finish_reason: tool_calls
tool_call_count: 1

[TOOL RESULT: python]
[action] UP -> NO-OP (executed, no change inside the board area)
"""

if __name__ == "__main__":
    if sys.argv[1:2] == ["--self-check"]:
        got = [classify(g) for g in generations(SAMPLE)]
        assert got == ["exception_no_output", "no_tool_call_length", "noop_only_actions"], got
        print("self-check ok")
    elif sys.argv[1:2] == ["--fetch-usage"]:
        for arg in sys.argv[2:]:
            fetch_usage(Path(arg))
    else:
        report([Path(a) for a in sys.argv[1:]])
