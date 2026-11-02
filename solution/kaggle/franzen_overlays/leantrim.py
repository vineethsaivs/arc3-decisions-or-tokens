"""leantrim: when the token trimmer has just dropped history, slim what it keeps.

A trim drops the oldest blocks, which shifts every surviving token, so the next request re-prefills the
whole retained history (about 54K tokens, 4 to 5 times per game, stop-the-world: chunks run back to back
with no decode). That one moment is free for any other change to the retained history. This overlay uses
it: assistant messages older than the last ARC3_LEAN_KEEP_TURNS turns (a turn starts at a user message
without the harness's control tag, i.e. a fresh opener, not a resume opener or nudge) get their reasoning
replaced by a short marker, and user messages before that point get their images replaced by the
harness's own placeholder (_strip_message_images). Tool calls, tool results and opener text stay. Nothing
changes between trims, so the prefix cache is untouched.

Measured offline on franzen-l1a1y8sv2r1/r2/r3 (docs/research/2026-10-12-next/trim-economy.md): with 3 turns
kept, the retained context after a trim falls from 54.4K to about 39.4K tokens, trims per game from 4.69 to
3.85, re-prefilled tokens by 41%.

Run by build_franzen_arms.py right after Franzen's harness patch, as exec(code, {'BUNDLE_DIR': BUNDLE_DIR}).
Stdlib only. Two exact edits asserted to match once; nothing is written unless the file still compiles.
"""
from pathlib import Path

_AGENT = Path(BUNDLE_DIR) / "src" / "ARC3-Inference" / "inference" / "agent" / "tool_agent.py"  # noqa: F821
_MARK = "overlay leantrim"
_EDITS = [
    # the helper, next to the harness's own image-window helper
    ("def _apply_history_image_window(\n",
     "_LEAN_REASONING_MARKER = \"[reasoning from this older turn omitted]\"\n"
     "\n"
     "\n"
     "def _lean_retained_history(history: list[dict[str, Any]]) -> list[dict[str, Any]]:\n"
     "    \"\"\"overlay leantrim: called only right after the trimmer dropped history, when the prefix is\n"
     "    invalid anyway. Before the last ARC3_LEAN_KEEP_TURNS fresh openers: reasoning becomes a marker,\n"
     "    images become the history placeholder. Originals are never mutated (a failed turn rolls back to\n"
     "    them).\"\"\"\n"
     "    keep = _get_env_int(\"ARC3_LEAN_KEEP_TURNS\", 0)\n"
     "    if keep <= 0:\n"
     "        return history\n"
     "    starts = [\n"
     "        i for i, m in enumerate(history)\n"
     "        if str(m.get(\"role\", \"\")).strip() == \"user\" and not m.get(_CONTROL_MESSAGE_KEY)\n"
     "    ]\n"
     "    if len(starts) <= keep:\n"
     "        return history\n"
     "    cut = starts[-keep]\n"
     "    keys = _reasoning_history_keys()\n"
     "    out = []\n"
     "    for i, m in enumerate(history):\n"
     "        if i < cut:\n"
     "            role = str(m.get(\"role\", \"\")).strip()\n"
     "            if role == \"assistant\" and any(m.get(k) for k in keys):\n"
     "                m = {**m, **{k: _LEAN_REASONING_MARKER for k in keys if m.get(k)}}\n"
     "            elif role == \"user\" and _message_has_image(m):\n"
     "                m = _strip_message_images(m)\n"
     "        out.append(m)\n"
     "    return out\n"
     "\n"
     "\n"
     "def _apply_history_image_window(\n"),
    # the trimmer's own eviction point (_trim_messages_for_context)
    ("        if len(history) != len(messages) - 1:\n"
     "            self._note_history_evicted()\n"
     "        return [system_message, *history]\n",
     "        if len(history) != len(messages) - 1:\n"
     "            history = _lean_retained_history(history)  # overlay leantrim\n"
     "            self._note_history_evicted()\n"
     "        return [system_message, *history]\n"),
]

_text = _AGENT.read_text()
if _MARK in _text:
    raise RuntimeError("leantrim overlay: already applied to " + str(_AGENT))
for _old, _new in _EDITS:
    _count = _text.count(_old)
    if _count != 1:
        raise RuntimeError(f"leantrim overlay: anchor matches {_count} times in {_AGENT.name}, expected 1: "
                           f"{_old.strip()[:70]!r}")
    _text = _text.replace(_old, _new)
compile(_text, str(_AGENT), "exec")
_AGENT.write_text(_text)
print("arm overlay leantrim: at a trim, reasoning and images older than the last "
      "ARC3_LEAN_KEEP_TURNS turns are replaced by markers (tool calls, results and opener text kept)")
