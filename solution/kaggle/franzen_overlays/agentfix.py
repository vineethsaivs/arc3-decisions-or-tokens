"""Agent fixes A1: RESET behind a level-start guard, the stale-guard catch contract, and
level/attempt index globals. docs/research/2026-10-05-top5/PLAN.md section 3 item 5;
the measured failures are in agent-quality.md proposals 1, 4 and 6.

(a) RESET. The arm sets EXPOSE_RESET=on in cell 7, which turns on Franzen's own RESET
   support (RESET in valid_actions, RESET_INFO_ADDENDUM, RESET first in a snippet).
   step_env now skips a RESET heading the call, stop_reason "reset_at_level_start",
   when the last history entry is the seeded start, any RESET (manual, the warmup
   one, or the automatic one after a game over) or an action that completed a
   level: nothing has happened on that level yet. The RESET does not run and no
   action is spent; a lone RESET returns at once and the snippet goes on, and in a
   batch the other actions run. With ARC3_ACTION_ECHO (on in the fork) the skip
   prints its own [action] line. One sentence is added to RESET_INFO_ADDENDUM.
   What such a RESET does, checked offline through this harness on arcengine
   0.9.3 and arc_agi 0.9.8 (solution/sim/test_franzen_agentfix.py): arcengine's
   handle_reset does a level reset whenever ONLY_RESET_LEVELS is "true", and the
   fork sets it twice (cell 7, and taaf's GameAPI after arcade.make). So on the
   replica a RESET on an untouched level executes as a no-op that costs one
   action, and a mid-level RESET restarts the level with levels_completed kept.
   With the variable unset, the engine turns a RESET at engine action count 0
   into a FULL reset: level 1, levels_completed 0 (what agent-quality.md
   replayed on the bare engine). The scored gateway is not readable; arc_agi
   0.9.8's competition server answers such a RESET with the current frame and
   still records it on the scorecard. So the skip saves an action on both paths
   and closes the full-reset case if a path runs without the variable.
   PLAN.md says "refused", raising like reset_not_first. Skipping is the same
   guard without the cost: the board already is the board a RESET would give, so
   the snippet's plan still holds, and a raise would throw away the rest of the
   snippet and a model turn to save one action.
(b) Stale-state catch contract. The refusal says StaleStateActionError can be
   caught to keep acting through no-ops, but the latch stayed armed, so every
   later call in the snippet was refused too (36 of 72 catching snippets were
   refused two or more times in a row). Now the first caught refusal in a snippet
   clears the latch and waives it for the next 6 no-op actions; the 7th re-arms
   it, and a refusal after that is not waived, so a careless retry loop spends at
   most 7 more no-ops. The refusal message says so. Each refusal also prints its
   [action] echo line before raising, so a caught one still shows in stdout and
   solution/analysis/agentfix_counters.py can count it. Known wording mismatch,
   left alone (tool_agent.py is not touched): the next opener still says "the
   sequence was stopped before X" for a caught refusal the snippet acted past.
(c) level_starts and attempt_starts are provided globals, refreshed with history:
   history indices where each level began and where the current level began or
   was last reset. One prompt bullet defines them and the refreshed-globals line
   names them.

Run by build_franzen_arms.py right after Franzen's harness patch, as
exec(code, {'BUNDLE_DIR': BUNDLE_DIR}). Stdlib only. Every edit is an exact
replacement asserted to match once, and nothing is written until all three files,
and the sandbox's child script inside one of them, still compile.
"""
import textwrap
from pathlib import Path

_inf = Path(BUNDLE_DIR) / "src" / "ARC3-Inference" / "inference"
_SOLVER = _inf / "framework" / "solver.py"
_SANDBOX = _inf / "agent" / "python_tool_sandbox.py"
_PROMPTS = _inf / "agent" / "prompts.py"
_MARK = "overlay agentfix"

_EDITS = {
    _SOLVER: [
        # (a) before the loop, so a skipped RESET never reaches the stale and repeat guards
        # and the batch bookkeeping (batch_size, displays, trace) sees only what runs.
        # ponytail: a snippet looping on RESET alone at a level start spins to its deadline
        # (skips are not executed actions, so reset_not_first never fires); count skips per
        # snippet if the counters ever show one.
        ('        if self.should_stop() or _is_engine_game_over(self.game):\n'
         '            return self._terminal_payload(requested_actions)\n',
         '        if self.should_stop() or _is_engine_game_over(self.game):\n'
         '            return self._terminal_payload(requested_actions)\n'
         '        # overlay agentfix: nothing has happened on this level since it began or was\n'
         '        # last reset, so a RESET heading the call restarts nothing. It would cost one\n'
         '        # action, and an engine run without ONLY_RESET_LEVELS makes it a full reset\n'
         '        # to level 1. Skip it and run the rest of the call.\n'
         '        _last = self.history_entries[-1] if self.history_entries else None\n'
         '        reset_skipped = (\n'
         '            requested_actions[0].id.name == "RESET"\n'
         '            and reset_exposed()\n'
         '            and (\n'
         '                _last is None\n'
         '                or str(_last.action).strip().upper() in ("", "RESET")\n'
         '                or (_last.result or {}).get("level_completed")\n'
         '            )\n'
         '        )\n'
         '        if reset_skipped:\n'
         '            requested_actions = requested_actions[1:]\n'
         '        if reset_skipped and not requested_actions:\n'
         '            return {\n'
         '                "executed": False,\n'
         '                "action_num": self.action_count,\n'
         '                "level": _level_number(self.game),\n'
         '                "score": int(self.game.current_state.levels_completed),\n'
         '                "valid_actions": to_model_actions(_engine_action_names(self.game)),\n'
         '                "stop_reason": "reset_at_level_start",\n'
         '                "stop_detail": (\n'
         '                    "RESET was skipped and no action was spent: nothing has happened on "\n'
         '                    "this level since it began or was last reset, so it is already at its "\n'
         '                    "start."\n'
         '                ),\n'
         '                **({"action_echo": _build_action_echo(["RESET"], [], "reset_at_level_start")}\n'
         '                   if ACTION_ECHO else {}),\n'
         '                **self.timing_payload(),\n'
         '            }\n'),
        ('            echo = _build_action_echo(requested_displays, executed_payloads, stop_reason)\n',
         '            echo = _build_action_echo(requested_displays, executed_payloads, stop_reason)\n'
         '            if reset_skipped:  # overlay agentfix: the skipped RESET\'s line comes first\n'
         '                echo = _build_action_echo(["RESET"], [], "reset_at_level_start") + "\\n" + echo\n'),
        # (b) the refusal's own echo line, printed by the sandbox before it raises
        ('                    "blocked_action": requested_displays[0] if requested_displays else None,\n'
         '                }\n'
         '            return self._error_payload("No action was executed.")\n',
         '                    "blocked_action": requested_displays[0] if requested_displays else None,\n'
         '                    # overlay agentfix: printed before the raise, so a caught\n'
         '                    # refusal still leaves its line in the snippet\'s stdout\n'
         '                    **({"action_echo": _build_action_echo(requested_displays, [], blocked_kind)}\n'
         '                       if ACTION_ECHO and blocked_kind == "stale_state" else {}),\n'
         '                }\n'
         '            return self._error_payload("No action was executed.")\n'),
        ('        "(StaleStateActionError can be caught if a snippet has good reason to "\n'
         '        "keep acting through no-ops.)"\n',
         '        "(StaleStateActionError can be caught if a snippet has good reason to "\n'
         '        # overlay agentfix: the catch contract the sandbox now keeps\n'
         '        "keep acting through no-ops. The first time a snippet catches this, the "\n'
         '        "check is waived for its next 6 actions that change nothing; the 7th "\n'
         '        "re-arms it, and later refusals in that snippet are not waived.)"\n'),
    ],
    _SANDBOX: [
        # (c), inside the child script: refreshed with history after every action
        ('            runtime_globals["history"] = history\n',
         '            runtime_globals["history"] = history\n'
         '            # overlay agentfix: history indices where each level, and each attempt\n'
         '            # at the current level, began\n'
         '            _level_starts, _attempt_starts = [0], [0]\n'
         '            for _index, _entry in enumerate(history):\n'
         '                if _entry.result.get("level_completed"):\n'
         '                    _level_starts.append(_index)\n'
         '                    _attempt_starts = [_index]\n'
         '                elif _index and _entry.action.strip().upper() == "RESET":\n'
         '                    _attempt_starts.append(_index)\n'
         '            runtime_globals["level_starts"] = _level_starts\n'
         '            runtime_globals["attempt_starts"] = _attempt_starts\n'),
        # (b)
        ('        _stale_state: list[str] = []\n',
         '        _stale_state: list[str] = []\n'
         '        # overlay agentfix: empty until the snippet catches its first stale\n'
         '        # refusal, then how many no-op actions are still waived\n'
         '        _stale_waiver: list[int] = []\n'),
        # a skipped RESET (stop_reason reset_at_level_start) needs nothing here: it raises
        # nothing, and action() prints its echo line like any executed action's
        ('            if action_result.get("stop_reason") == "stale_state":\n'
         '                raise StaleStateActionError(\n',
         '            if action_result.get("stop_reason") == "stale_state":\n'
         '                # overlay agentfix: a caught refusal still leaves its echo line\n'
         '                _refusal_echo = action_result.pop("action_echo", None)\n'
         '                if _refusal_echo:\n'
         '                    print(_refusal_echo)\n'
         '                if not _stale_waiver:\n'
         '                    # overlay agentfix: the catch contract. Once per snippet the\n'
         '                    # latch clears, so a snippet that catches this can go on\n'
         '                    del _stale_state[:]\n'
         '                    _stale_waiver.append(6)\n'
         '                raise StaleStateActionError(\n'),
        ('                if not _stale_state:\n'
         '                    _stale_state.append(str(_name))\n',
         '                if _stale_waiver and _stale_waiver[0] > 0:\n'
         '                    _stale_waiver[0] -= 1  # overlay agentfix: a waived no-op\n'
         '                elif not _stale_state:\n'
         '                    _stale_state.append(str(_name))\n'),
    ],
    _PROMPTS: [
        # (a), the end of RESET_INFO_ADDENDUM
        ('    "or computation may precede it.\\n"\n)',
         '    "or computation may precede it. RESET restarts only the current level, so when that "\n'
         '    "level is already at its start (right after a level-up, a game over or another "\n'
         '    "RESET) it is skipped, spends no action, and the rest of the snippet runs.\\n"\n)'),
        # (c)
        ('do not compare `current_frame` to `history[-1].frame`.\\n"\n',
         'do not compare `current_frame` to `history[-1].frame`.\\n"\n'
         '    "- `level_starts` and `attempt_starts` are lists of `history` indices. '
         '`history[level_starts[k]].frame` is the first board of level k+1 (`level_starts[0]` is 0), so for a '
         'finished level `history[level_starts[k+1]]` is the action that completed it and '
         '`history[level_starts[k+1] - 1].frame` its last board. `attempt_starts` holds where the current level '
         'began and every RESET since: `history[attempt_starts[-1]].frame` is the board the current attempt '
         'started from.\\n"\n'),
        ('`current_frame`, `previous_frame`, `history`, `transitions`, `valid_actions`, '
         'and `last_action_call_result` are refreshed.',
         '`current_frame`, `previous_frame`, `history`, `transitions`, `level_starts`, `attempt_starts`, '
         '`valid_actions`, and `last_action_call_result` are refreshed.'),
    ],
}

_texts = {path: path.read_text() for path in _EDITS}
for _path, _text in _texts.items():
    if _MARK in _text:
        raise RuntimeError("agentfix overlay: already applied to " + str(_path))
for _path, _edits in _EDITS.items():
    for _old, _new in _edits:
        _count = _texts[_path].count(_old)
        if _count != 1:
            raise RuntimeError(f"agentfix overlay: anchor matches {_count} times in {_path.name}, "
                               f"expected 1: {_old.strip()[:70]!r}")
        _texts[_path] = _texts[_path].replace(_old, _new)
for _path, _text in _texts.items():
    compile(_text, str(_path), "exec")
# the child script is a string, so the line above cannot see a mistake inside it
_child = _texts[_SANDBOX].split('textwrap.dedent(\n    r"""', 1)[1].split('"""', 1)[0]
compile(textwrap.dedent(_child), "python_tool_sandbox bootstrap", "exec")
for _path, _text in _texts.items():
    _path.write_text(_text)
print("arm overlay agentfix: RESET skipped at a level start (reset_at_level_start), the stale guard's catch "
      "contract kept (6 waived no-ops once per snippet), level_starts and attempt_starts provided")
