"""B3 overlay: early handover of stuck games.

Run by build_franzen_arms.py right after Franzen's harness patch, as
exec(code, {'BUNDLE_DIR': BUNDLE_DIR}). Stdlib only; every edit is an exact
string replacement asserted to match once.

Franzen's gate yields a slot only at a context trim, and every unstarted game
outranks every trimmed one, so the opening phase is a round robin in which each
game holds a slot until its first trim even when stuck on level 1. On the fork's
10 games that first trim lands 12 to 19 min in, at 45K to 73K generated tokens.
This also hands the slot over, through the same _maybe_handover path (same
priority, same snapshot), once the current level has generated
ARC3_EARLY_HANDOVER_TOKENS (default 25000, about half the first-trim count, so
the game yields a half-full context instead of a nearly full one a few K tokens
before a free trim) without a level-up, and again ARC3_EARLY_HANDOVER_STEP
(default 40000) after the last handover of either kind, but only while another
game is waiting at the gate. Later trims come every 25K to 38K tokens, so the
step only matters when an inter-trim stint outlasts it. Set the tokens to 0 to
disable it; set the step to 0 or below to stop early handovers on a level after
its first handover of either kind.
Readout: one "early handover:" warning per fire, naming the game and its token
count; kernel.log carries every warning twice, so count unique lines.
Plan: docs/PLAN_2026-10-01.md, track B, step B3.
"""
from pathlib import Path

_path = Path(BUNDLE_DIR) / "src" / "ARC3-Inference" / "inference" / "agent" / "tool_agent.py"
_text = _path.read_text()

_EDITS = [
    # 1. the check, as a new method just above _maybe_handover
    (
        "    def _maybe_handover(self) -> None:\n",
        '''    def _early_handover_due(self) -> bool:
        """Whether a stuck game should yield its slot although nothing was trimmed.

        Due once this level has generated ARC3_EARLY_HANDOVER_TOKENS without a
        level-up, then again ARC3_EARLY_HANDOVER_STEP after the last handover of
        either kind (never again on this level when the step is 0). Only while another game waits: with nobody to take the
        slot, waiting would just expose the cached prefix to eviction, so the
        threshold stays armed until someone queues. Never raises: a failure
        here keeps the slot, which is the behaviour without this check.
        """
        try:
            first = _get_env_int("ARC3_EARLY_HANDOVER_TOKENS", 25000)
            gate = _priority_gate()
            if gate is None or first <= 0:
                return False
            step = _get_env_int("ARC3_EARLY_HANDOVER_STEP", 40000)
            level_start = getattr(self, "_tokens_at_level_start", 0)
            tokens = self._session_generated_tokens - level_start
            if getattr(self, "_early_handover_level_start", None) != level_start:
                # a level-up restarts the token count, so the threshold too
                self._early_handover_level_start = level_start
                self._early_handover_at = first
            nxt = tokens + step if step > 0 else float("inf")
            if self._context_was_trimmed:
                # the trim hands over anyway; the next early one is a step on
                self._early_handover_at = max(self._early_handover_at, nxt)
                return False
            # unlocked read: if that waiter is admitted elsewhere first, the
            # handover simply wins the slot straight back
            if tokens < self._early_handover_at or not gate._waiting:
                return False
            self._early_handover_at = nxt
            log.warning(
                "early handover: %s yields after %d tokens on level %s, %d waiting",
                getattr(self, "_diag_name", "?"), tokens,
                (self._last_step_summary or {}).get("level"), len(gate._waiting),
            )
            return True
        except Exception as exc:
            log.warning("early handover check failed, keeping the slot: %s", exc)
            return False

    def _maybe_handover(self) -> None:
''',
    ),
    # 2. let it through the trim-only early return; the rest of the path is unchanged
    (
        "        if not self._context_was_trimmed:\n"
        "            return\n"
        "        self._context_was_trimmed = False\n",
        "        early = self._early_handover_due()  # overlay B3\n"
        "        if not self._context_was_trimmed and not early:\n"
        "            return\n"
        "        self._context_was_trimmed = False\n",
    ),
]

for _old, _new in _EDITS:
    _count = _text.count(_old)
    if _count != 1:
        raise RuntimeError(
            f"handover overlay: anchor matches {_count} times, expected 1: {_old.splitlines()[0].strip()!r}"
        )
    _text = _text.replace(_old, _new)
_path.write_text(_text)
print("arm overlay handover: stuck games yield their slot to a waiting game after "
      "ARC3_EARLY_HANDOVER_TOKENS (default 25000) on a level, then every ARC3_EARLY_HANDOVER_STEP (40000)")
