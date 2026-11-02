"""Post-trim request pairs for the leaner-trim decision test (docs/research/2026-10-12-next/PLAN.md 5.1).

    python3 -I solution/analysis/trim_pairs.py OUT.jsonl REQUESTS.jsonl [REQUESTS.jsonl ...] [--keep 3]

A replica's <game>_p0_requests.jsonl holds every request the harness sent, images included. A trim shows as the
message count falling between consecutive requests of one game; the first request after it is a post-trim state.
For each one this writes the logged messages (variant "full") and the same messages slimmed as the leantrim
prototype would (variant "lean"): before the last KEEP turn starts, assistant reasoning becomes a marker and user
images are dropped. Tool calls, tool results and opener text stay. A turn starts at a fresh opener ("The code
executed ..." or the first opener); resume openers and nudges carry a control tag that is stripped before logging,
so they are recognised by their text.

Each output row: game, request index, analysis step, tools, chat_template_kwargs, and both message lists.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

MARKER = "[reasoning from this older turn omitted]"
PLACEHOLDER = "[grid image omitted from history]"  # tool_agent.py:889
# turn openers (tool_agent.py:370-385, 5049); a resume opener is a verbatim copy of its turn's opener (:6694)
FRESH = ("The code executed ", "No previous sequence has been executed yet.",
         "Your last python snippet executed NO actions", "Your previous exchange this turn produced no tool call")


def text_of(message: dict) -> str:
    c = message.get("content")
    if isinstance(c, str):
        return c
    return " ".join(p.get("text", "") for p in c or [] if isinstance(p, dict) and p.get("type") == "text")


def turn_starts(messages: list[dict]) -> list[int]:
    seen, out = set(), []
    for i, m in enumerate(messages):
        if m.get("role") == "user":
            t = text_of(m)
            if t.lstrip().startswith(FRESH) and t not in seen:
                out.append(i)
            seen.add(t)
    return out


def lean(messages: list[dict], keep: int) -> list[dict]:
    """The leantrim prototype's rule on a logged request (system message first)."""
    starts = turn_starts(messages)
    if len(starts) <= keep:
        return messages
    cut = starts[-keep]
    out = []
    for i, m in enumerate(messages):
        if 0 < i < cut:
            if m.get("role") == "assistant" and m.get("reasoning_content"):
                m = {**m, "reasoning_content": MARKER}
            elif m.get("role") == "user" and isinstance(m.get("content"), list):  # _strip_message_images (:927)
                parts = [{"type": "text", "text": PLACEHOLDER} if isinstance(p, dict) and p.get("type") == "image_url"
                         else p for p in m["content"]]
                m = {**m, "content": parts}
        out.append(m)
    return out


def post_trim_requests(path: Path):
    prev = None
    rows = (json.loads(line) for line in path.open())
    for index, row in enumerate(r for r in rows if r.get("event") == "request"):
        n = len(row["messages"])
        if prev is not None and n < prev:
            yield index, row
        prev = n


def size(messages: list[dict]) -> tuple[int, int]:
    """(characters outside images, images)."""
    chars = images = 0
    for m in messages:
        chars += len(text_of(m)) + len(m.get("reasoning_content") or "") + len(json.dumps(m.get("tool_calls") or ""))
        if isinstance(m.get("content"), list):
            images += sum(isinstance(p, dict) and p.get("type") == "image_url" for p in m["content"])
    return chars, images


def main(argv: list[str]) -> None:
    keep = int(argv[argv.index("--keep") + 1]) if "--keep" in argv else 3
    paths = [a for a in argv[1:] if a.endswith(".jsonl") and a != argv[0]]
    pairs = full_c = lean_c = full_i = lean_i = 0
    with open(argv[0], "w") as out:
        for path in map(Path, paths):
            game = path.name.split("-")[0]
            for index, row in post_trim_requests(path):
                slim = lean(row["messages"], keep)
                (fc, fi), (lc, li) = size(row["messages"]), size(slim)
                pairs += 1
                full_c, lean_c, full_i, lean_i = full_c + fc, lean_c + lc, full_i + fi, lean_i + li
                out.write(json.dumps({"game": game, "request_index": index, "analysis_step": row.get("analysis_step"),
                                      "tools": row.get("tools"), "tool_choice": row.get("tool_choice"),
                                      "chat_template_kwargs": row.get("chat_template_kwargs"),
                                      "full": row["messages"], "lean": slim}) + "\n")
    print(f"{pairs} post-trim states; text chars {full_c} -> {lean_c} ({100 * (lean_c - full_c) / max(full_c, 1):+.1f}%), "
          f"images {full_i} -> {lean_i}")


if __name__ == "__main__":
    main(sys.argv[1:])
