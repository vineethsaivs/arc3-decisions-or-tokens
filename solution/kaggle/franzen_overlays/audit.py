"""B2 strategy audit: one host-injected question per level that has stalled.

Lord Han Solo (Milestone 2 winner, 23.84) adds a strategy-audit block to the
turn once a level has used 25% of the game's time, and repeats it on that
interval. Franzen's games have no fixed per-game time, so this keys on the
per-level counts his priority snapshot already keeps (generated tokens and
actions since the last level-up): the audit fires once a level has cost
ARC3_AUDIT_TOKENS tokens (default 30000) and ARC3_AUDIT_MIN_ACTIONS actions
(default 20). It fires at most once per level and is a short question, not an
advice block, because rellik13 measured advice blocks hurting at a capped
context. The text rides at the end of the turn's opening user message, after
the cached prefix, so it costs no prefill.

Run by build_franzen_arms.py right after Franzen's harness patch, with
BUNDLE_DIR set to the patched bundle.
"""
from pathlib import Path

TOOL_AGENT = Path(BUNDLE_DIR) / "src" / "ARC3-Inference" / "inference" / "agent" / "tool_agent.py"

PROMPT = '''_STRATEGY_AUDIT_PROMPT = (
    "Strategy check: this level has gone a long time without a level-up. "
    "Before your next action, answer in a few lines: what have you tried on "
    "this level, what has that ruled out, and what single cheapest experiment "
    "would most change your plan? If recent observations already match your "
    "predictions and a concrete plan is advancing, name that evidence and "
    "continue the plan. Do not reset or give up progress because of this question."
)


'''

METHOD = '''    def _strategy_audit_context(self) -> str:
        """The strategy-audit question, at most once per level.

        Due once the level has cost ARC3_AUDIT_TOKENS generated tokens and
        ARC3_AUDIT_MIN_ACTIONS actions: the same per-level counts the priority
        snapshot reads, which restart at every level-up. Read at call time so
        the thresholds tune without a rebuild. Never raises, because an
        optional prompt line must not end a game.
        """
        try:
            summary = self._last_step_summary or {}
            level = int(summary.get("level") or 1)
            # getattr: an agent unpickled from before this field existed
            audited = getattr(self, "_audited_levels", None)
            if audited is None:
                audited = self._audited_levels = set()
            if level in audited:
                return ""
            tokens = self._session_generated_tokens - getattr(self, "_tokens_at_level_start", 0)
            actions = _priority_action_count(summary) - self._actions_at_level_start
            if (
                tokens < _get_env_int("ARC3_AUDIT_TOKENS", 30000)
                or actions < _get_env_int("ARC3_AUDIT_MIN_ACTIONS", 20)
            ):
                return ""
            audited.add(level)
            log.warning(
                "%s strategy audit on level %d after %d tokens and %d actions",
                self._diag_name, level, tokens, actions,
            )
            return "\\n\\n" + _STRATEGY_AUDIT_PROMPT
        except Exception as exc:
            log.warning("strategy audit skipped: %s", exc)
            return ""

'''

# Pure insertions: every anchor survives verbatim, so another overlay that
# inserts at the same spot still finds it, in either order.
EDITS = [
    ("\nclass ToolAgent:\n", "\n" + PROMPT + "class ToolAgent:\n"),
    ("    def _retained_function_context(self) -> str:\n",
     METHOD + "    def _retained_function_context(self) -> str:\n"),
    # _build_user_message is the one place every turn opener goes through
    # (full, short and state_only resumptions alike)
    ("        user_prompt += self._retained_function_context()\n",
     "        user_prompt += self._retained_function_context()\n"
     "        user_prompt += self._strategy_audit_context()\n"),
]

text = TOOL_AGENT.read_text()
if "_strategy_audit_context" in text:
    raise RuntimeError("audit overlay: already applied to " + str(TOOL_AGENT))
for old, new in EDITS:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"audit overlay: anchor {old.strip()!r} matches {count} times, expected 1")
    text = text.replace(old, new)
TOOL_AGENT.write_text(text)
print("audit overlay applied: one strategy-audit question per level after "
      "ARC3_AUDIT_TOKENS (30000) tokens and ARC3_AUDIT_MIN_ACTIONS (20) actions")
