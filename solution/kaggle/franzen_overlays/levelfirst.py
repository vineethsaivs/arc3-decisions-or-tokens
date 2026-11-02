"""levelfirst: at a trim, drain earlier levels' history down to a short tail before the usual drain.

A trim drops history oldest first until the request is a drain (58K tokens) under the budget, and the next request
re-prefills everything it kept. Early in a level most of that is the previous level's history, which the 10-23
diagnosis measured as worth little (more older-level context at a level start: +2.3, se 1.1 bench; -1.5, se 2.5
held out), while a trim deep into the current level is the one harmful context event (-3.6 points of 30K-token
clear chance, se 1.3). This pass runs only when the trimmer is about to drop history anyway, so the prefix is
invalid either way: it drops whole blocks of earlier levels, oldest first, until at most ARC3_LEVEL_DRAIN_TAIL
tokens (12,288) of them remain before the current level's opener, then the usual drain continues to its target.
It never drops a message the base trimmer keeps: the base drops a prefix of the history, this drops a longer
prefix only while that prefix is still earlier levels. Replayed on 33 pin runs (docs/research/2026-10-23-push/
harness.md C1): trim re-prefill -19%, trims -9%, current-level tokens evicted -3%.

The current level's opener is the last user message without the harness's control tag whose text announces a
level-up (LEVEL_START_USER_PROMPT, or the plain line when ARC3_LEVEL_TRANSFER_GUIDANCE is off). Resume openers
repeat that text but carry the tag. On level 1, or once a drain has already removed the opener, nothing changes.

Run by build_franzen_arms.py right after Franzen's harness patch, as exec(code, {'BUNDLE_DIR': BUNDLE_DIR}).
Stdlib only. Two exact edits asserted to match once; nothing is written unless the file still compiles.
"""
from pathlib import Path

_AGENT = Path(BUNDLE_DIR) / "src" / "ARC3-Inference" / "inference" / "agent" / "tool_agent.py"  # noqa: F821
_MARK = "overlay levelfirst"
_EDITS = [
    # the pass, as a method next to the trimmer's other eviction helper
    ("    def _force_reduce_messages(\n",
     "    def _drain_older_levels(self, history: list[dict[str, Any]], *, preserve_recent: int) -> bool:\n"
     "        \"\"\"overlay levelfirst: drop whole blocks of earlier levels, oldest first, until at most\n"
     "        ARC3_LEVEL_DRAIN_TAIL tokens of them remain before the current level's opener. Only called when the\n"
     "        trimmer is about to drop history anyway. Returns whether anything was dropped.\"\"\"\n"
     "        tail = _get_env_int(\"ARC3_LEVEL_DRAIN_TAIL\", 12288)\n"
     "        if tail < 0:\n"
     "            return False\n"
     "        marks = (\"You have completed the previous level.\", \"You have progressed to a new level!\")\n"
     "        idx = 0\n"
     "        for i in range(len(history) - 1, 0, -1):\n"
     "            m = history[i]\n"
     "            if (str(m.get(\"role\", \"\")).strip() == \"user\" and not m.get(_CONTROL_MESSAGE_KEY)\n"
     "                    and any(k in _normalize_message_content(m.get(\"content\", \"\")) for k in marks)):\n"
     "                idx = i\n"
     "                break\n"
     "        envelope = self._estimate_request_input_tokens([])\n"
     "        dropped = False\n"
     "        while idx > 0 and self._estimate_request_input_tokens(history[:idx]) - envelope > tail:\n"
     "            before = len(history)\n"
     "            if not self._drop_oldest_history_block(\n"
     "                history, preserve_recent=max(preserve_recent, len(history) - idx)\n"
     "            ):\n"
     "                break\n"
     "            dropped = True\n"
     "            idx -= before - len(history)\n"
     "        return dropped\n"
     "\n"
     "    def _force_reduce_messages(\n"),
    # the trimmer: before its drop loop, only when it is about to drop (over budget, or a forced drain)
    ("        draining = bool(force_drain and drain_tokens)\n",
     "        draining = bool(force_drain and drain_tokens)\n"
     "        if drain_tokens and (draining or self._estimate_request_input_tokens(\n"
     "                [system_message, *history], tools=tools) > budget_tokens):\n"
     "            # overlay levelfirst: then drain on to the usual target, as the base trimmer would\n"
     "            draining = self._drain_older_levels(history, preserve_recent=preserve_recent) or draining\n"),
]

_text = _AGENT.read_text()
if _MARK in _text:
    raise RuntimeError("levelfirst overlay: already applied to " + str(_AGENT))
for _old, _new in _EDITS:
    _count = _text.count(_old)
    if _count != 1:
        raise RuntimeError(f"levelfirst overlay: anchor matches {_count} times in {_AGENT.name}, expected 1: "
                           f"{_old.strip()[:70]!r}")
    _text = _text.replace(_old, _new)
compile(_text, str(_AGENT), "exec")
_AGENT.write_text(_text)
print("arm overlay levelfirst: at a trim, earlier levels drain to a 12K tail before the usual drain")
