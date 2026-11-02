"""Offline check for specstats (franzen_overlays/specstats.py).

    python3 solution/sim/test_franzen_specstats.py

Applies specstats first and then last among the pin's harness overlays (wastefix, replycache, agentfix, memcap,
toolfast, promptobs) to copies of Franzen's patched bundle. On the real code: a request asks for
return_spec_tokens_details and keeps everything else; the result carries the response's sglext; the spec line reads
the dict and list forms, and says (none) when the server sent nothing; the transcript block appends it. A second
application refuses. No GPU, no network.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

from test_franzen_sandbox import FORK_ENV, TREE, apply, import_tool_agent

PIN = ("wastefix", "replycache", "agentfix", "memcap", "toolfast", "promptobs")
SPEC = {"spec_accept_length": 2.8312, "spec_verify_ct": 812, "spec_num_correct_drafts": 1487,
        "spec_num_proposed_drafts": 2436, "spec_correct_drafts_histogram": [120, 233, 181, 278]}
LINE = "\nspec: accept_length=2.831 verify_ct=812 correct_drafts=1487 proposed_drafts=2436 hist=120/233/181/278"


class _Response:
    status_code = 200
    text = ""

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return {"choices": [{"message": {"role": "assistant", "content": "", "reasoning_content": "Hmm."},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2299}, "sglext": {"spec_tokens_details": SPEC}}


def check(order: tuple[str, ...]) -> None:
    with tempfile.TemporaryDirectory(prefix="specstats-") as tmp:
        bundle = Path(tmp) / "bundle"
        shutil.copytree(TREE, bundle, ignore=shutil.ignore_patterns("__pycache__"))
        for name in order:
            apply(name, bundle)
        try:
            apply("specstats", bundle)
        except RuntimeError as exc:
            assert "already applied" in str(exc), exc
        else:
            raise AssertionError("a second application did not refuse")
        text = (bundle / "src/ARC3-Inference/inference/agent/tool_agent.py").read_text()
        assert "response_meta + _spec_line(result.sglext)" in text
        os.environ.update(FORK_ENV)
        for mod in [m for m in sys.modules if m == "inference" or m.startswith("inference.")]:
            del sys.modules[mod]
        tool_agent = import_tool_agent(bundle)
        sys.path.remove(str(bundle / "src" / "ARC3-Inference"))

        assert tool_agent._spec_line({"spec_tokens_details": SPEC}) == LINE
        assert tool_agent._spec_line({"spec_tokens_details": [SPEC]}) == LINE
        for empty in (None, {}, {"spec_tokens_details": None}, {"spec_tokens_details": {"spec_verify_ct": 0}}):
            assert tool_agent._spec_line(empty) == "\nspec: (none)", empty

        sent = []
        agent = tool_agent.ToolAgent(model="stub")
        tool_agent.requests.post = lambda url, **kw: sent.append(kw["json"]) or _Response()
        result = agent._chat_completion([{"role": "user", "content": "hi"}], tools=None)
        assert sent and sent[0]["return_spec_tokens_details"] is True, sent
        assert sent[0]["messages"] and "temperature" in sent[0], sent[0]
        assert result.sglext == {"spec_tokens_details": SPEC} and result.usage["completion_tokens"] == 2299
        print(f"ok   order {'specstats first' if order[0] == 'specstats' else 'specstats last'}: request flag, "
              "result field, spec line, transcript hook")


def main() -> None:
    check(("specstats",) + PIN)
    check(PIN + ("specstats",))


if __name__ == "__main__":
    main()
