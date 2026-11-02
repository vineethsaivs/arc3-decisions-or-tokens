"""B1 reply-cache round trip: re-send the model's reasoning as the server returned it.

docs/research/2026-10-05-top5/top-movers.md section 4 measured that the next request
gets the previous reply from the prefix cache for only 36 to 38% of reply tokens, all
or nothing per reply. The harness stores each reply's reasoning for history through
_extract_reasoning_text, which runs it through _normalize_message_content: think tags
stripped and every blank line deleted. The chat template then renders that text back
into the next prompt, so a reply with a paragraph break no longer matches the text
the model generated, and the radix cache falls back to the previous prompt's end (the
hybrid model can only resume where a recurrent state was saved).

The usage logs point here. In franzen-p448s14-v1 and -rep-v1, replies of 128+ tokens
whose reasoning was under 32 tokens (rarely more than one paragraph) were fully reused
99 times in 100; replies with longer reasoning about 40%. For those, reuse climbs from
1% under 16K tokens of context to about 50% past 96K, though early replies are not
longer: most likely the model copies the blank-line-free layout of its own history as
history fills, so more replies happen to round-trip. The server side donates
correctly; franzen_overlays/replycache_serve.py has the code reading and
solution/analysis/reply_reuse.py the numbers.

The fix keeps the server's string, minus the think tags. Its two ends are stripped as
before: the Flash-Next template applies |trim to reasoning_content and writes
'<think>\\n' + reasoning + '\\n</think>\\n\\n' itself, so the rendered text is the same
either way and every truthiness check on the reasoning behaves as before (lengths grow
by the blank lines). The reply's content and tool calls are not touched. This
function is the only source of the reasoning the harness stores for history (the
analyze loop's raw_reasoning, under ARC3_REASONING_HISTORY_KEY); the prompt log and
the rolling-summary logs read it too and now show the verbatim text.

Limits. This makes the text match; whether the server's retokenized history equals the
generated tokens is the L1 reuse gate's to show. A reply that a thinking budget ends
(thinkbudget_serve.py) rarely round-trips with or without this fix: its forced </think>
follows whatever token hit the budget, where the template writes '\\n</think>'
(solution/analysis/reply_reuse.py --think-budget counts those replies apart).

Run by build_franzen_arms.py right after Franzen's harness patch, as
exec(code, {'BUNDLE_DIR': BUNDLE_DIR}). Stdlib only. One exact replacement asserted to
match once; nothing is written unless the edited file still compiles.
Offline check: solution/sim/test_franzen_replycache.py.
"""
from pathlib import Path

_AGENT = Path(BUNDLE_DIR) / "src" / "ARC3-Inference" / "inference" / "agent" / "tool_agent.py"
_MARK = "overlay replycache"
_OLD = (
    "def _extract_reasoning_text(message: dict[str, Any]) -> str:\n"
    '    reasoning = message.get("reasoning")\n'
    '    if reasoning in (None, ""):\n'
    '        reasoning = message.get("reasoning_content", "")\n'
    "    return _normalize_message_content(reasoning)\n"
)
_NEW = (
    "def _extract_reasoning_text(message: dict[str, Any]) -> str:\n"
    '    reasoning = message.get("reasoning")\n'
    '    if reasoning in (None, ""):\n'
    '        reasoning = message.get("reasoning_content", "")\n'
    "    if isinstance(reasoning, str):\n"
    "        # overlay replycache: the server's text minus the think tags, blank lines\n"
    "        # kept, so the history turn renders the text the model generated and the\n"
    "        # prefix cache can serve the reply (the template trims the two ends itself)\n"
    '        return _THINK_TAG_RE.sub("", reasoning).strip()\n'
    "    return _normalize_message_content(reasoning)\n"
)

_text = _AGENT.read_text()
if _MARK in _text:
    raise RuntimeError("replycache overlay: already applied to " + str(_AGENT))
_count = _text.count(_OLD)
if _count != 1:
    raise RuntimeError(f"replycache overlay: anchor matches {_count} times in {_AGENT.name}, expected 1: "
                       "def _extract_reasoning_text(...)")
_text = _text.replace(_OLD, _NEW)
compile(_text, str(_AGENT), "exec")
_AGENT.write_text(_text)
print("arm overlay replycache: history reasoning is the server's text minus the think tags (blank lines kept)")
