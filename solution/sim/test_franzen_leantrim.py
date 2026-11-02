"""Offline check for leantrim (franzen_overlays/leantrim.py), adapted from the trim track's prototype test.

    python3 solution/sim/test_franzen_leantrim.py

On a copy of Franzen's patched bundle with the pin's harness overlays applied in cell 7's order, then leantrim:
1. both anchors match once, it compiles, a second application refuses;
2. the real ToolAgent._trim_messages_for_context at the pin's budget drops exactly the blocks the pristine trimmer
   drops, then slims what it keeps: reasoning before the 3rd-last fresh opener becomes the marker, images there the
   history placeholder, the last 3 turns and every tool call, tool result and opener text are unchanged, the input
   is not mutated and the eviction flag is set;
3. a history under the budget comes back as the same objects;
4. with ARC3_LEAN_KEEP_TURNS unset the trimmer is the pristine one.
solution/analysis/trim_pairs.py applies the same rule to logged requests; check 5 asserts they agree.
No GPU, no network.
"""
from __future__ import annotations

import copy
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

from test_franzen_sandbox import FORK_ENV, TREE, apply, import_tool_agent

ORDER = ("promptobs", "toolfast", "memcap", "agentfix", "replycache", "wastefix", "leantrim")
IMG = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}


def history(turns: int) -> list[dict]:
    msgs = [{"role": "system", "content": "s" * 15000}]
    for t in range(turns):
        msgs.append({"role": "user", "content": [{"type": "text", "text": f"The code executed {t} actions " + "o" * 2500},
                                                 IMG, {"type": "text", "text": "Diff image"}, IMG]})
        for r in range(2):
            msgs.append({"role": "assistant", "reasoning_content": f"think {t}.{r} " + "r" * 6000,
                         "tool_calls": [{"id": f"c{t}{r}", "type": "function", "function": {
                             "name": "python", "arguments": json.dumps({"code": f"x={t}{r}\n" + "c" * 800})}}]})
            msgs.append({"role": "tool", "tool_call_id": f"c{t}{r}", "content": f"result {t}.{r} " + "t" * 1500})
        if t % 5 == 4:  # a resume opener (control-tagged copy of the turn's opener) after a yield
            msgs.append({**msgs[-5], "_arc3_control": "resume"})
            msgs.append({"role": "assistant", "reasoning_content": f"think {t}.resume " + "r" * 3000, "content": None})
    return msgs


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="leantrim-") as tmp:
        bundle = Path(tmp) / "bundle"
        shutil.copytree(TREE, bundle, ignore=shutil.ignore_patterns("__pycache__"))
        for name in ORDER:
            apply(name, bundle)
        try:
            apply("leantrim", bundle)
        except RuntimeError as exc:
            assert "already applied" in str(exc), exc
        else:
            raise AssertionError("a second application did not refuse")
        print("ok 1: anchors matched once after the pin's overlays; compiles; a second application refuses")

        os.environ.update(FORK_ENV)
        os.environ.update({"LOCAL_ANALYZER_CONTEXT_WINDOW": str(128 * 1024), "LOCAL_ANALYZER_MAX_OUTPUT": str(12 * 1024),
                           "ARC3_CONTEXT_DRAIN_TOKENS": str(58 * 1024), "ARC3_IMAGE_TOKENS_FLAT": "400"})
        ta = import_tool_agent(bundle)

        def agent():
            a = object.__new__(ta.ToolAgent)
            a._context_budget_tokens = max(1024, 128 * 1024 - 12 * 1024 - 512)
            a._text_chars_per_token = 3.05
            a._has_evicted = False
            a._context_was_trimmed = False
            return a

        msgs = history(26)
        frozen = copy.deepcopy(msgs)
        os.environ.pop("ARC3_LEAN_KEEP_TURNS", None)
        plain = agent()._trim_messages_for_context(list(msgs))
        assert msgs == frozen and len(plain) < len(msgs)
        print(f"ok 4: knob unset = pristine trimmer; kept {len(plain) - 1} of {len(msgs) - 1}")

        os.environ["ARC3_LEAN_KEEP_TURNS"] = "3"
        a = agent()
        lean = a._trim_messages_for_context(list(msgs))
        assert msgs == frozen, "input mutated"
        assert a._context_was_trimmed and a._has_evicted and len(lean) == len(plain)
        fresh = [i for i, m in enumerate(lean) if i and m["role"] == "user" and not m.get("_arc3_control")]
        cut = fresh[-3]
        for i, (p, m) in enumerate(zip(plain, lean)):
            if i == 0 or i >= cut or m["role"] == "tool":
                assert m is p, i
            elif m["role"] == "assistant":
                assert m["reasoning_content"] == ta._LEAN_REASONING_MARKER and m.get("tool_calls") == p.get("tool_calls"), i
            elif m["role"] == "user":
                assert not any(x.get("type") == "image_url" for x in m["content"]), i
        print(f"ok 2: same blocks dropped; slimmed before the 3rd-last fresh opener (message {cut} of {len(lean)})")

        small = history(3)
        out = agent()._trim_messages_for_context(list(small))
        assert all(x is y for x, y in zip(out, small)) and len(out) == len(small)
        print("ok 3: under the budget nothing changes")

        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
        import trim_pairs
        wire = [{k: v for k, v in m.items() if k != "_arc3_control"} for m in plain]  # as logged on the wire
        offline = trim_pairs.lean(wire, 3)
        online = [{k: v for k, v in m.items() if k != "_arc3_control"} for m in lean]
        assert offline == online, "trim_pairs.lean differs from the overlay on the same retained history"
        print("ok 5: trim_pairs.lean on the logged form equals the overlay's slimming")


if __name__ == "__main__":
    main()
