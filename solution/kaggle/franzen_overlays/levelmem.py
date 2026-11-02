"""B1 overlay: solved-level memory.

Run by build_franzen_arms.py right after Franzen's harness patch, as
exec(code, {'BUNDLE_DIR': BUNDLE_DIR}). Stdlib only. Writes one new module,
inference/agent/levelmem.py, and adds four call sites to tool_agent.py. Every
edit is a pure insertion asserted to match once, so its anchor survives
verbatim and another overlay inserting at the same spot still matches, in any
order. Thresholds are read at call time (ARC3_LEVELMEM_*). Plan:
docs/PLAN_2026-10-01.md, track B, step B1.
"""
from pathlib import Path

_agent_dir = Path(BUNDLE_DIR) / "src" / "ARC3-Inference" / "inference" / "agent"

_MODULE = r'''"""Solved-level memory: each cleared level's rule and winning actions, kept all game.

Ported from rellik13's M85 (Milestone 2 winner): pinning the rule the agent wrote
for a cleared level plus its last winning actions was worth +2.8 on the board,
"the only text addition that helped", because it returns evidence a trim would
otherwise take away rather than adding advice.

What "the rule the agent wrote" is here. With ARC3_MEMORY_SECTIONS=off this
harness elicits no written summary, and rellik measured an unasked notes field
filled in 0 of 989 prompts. So, as in M85, the level-up opener asks once (at most
ARC3_LEVELMEM_ASK_TURNS openers) for a short paragraph starting
"Rule for level N:", read back from the reply's text or its reasoning. The tail
of the reasoning in the turn that cleared the level is kept beside it, so a level
whose rule was never stated still carries the model's own words.

Prefix caching. A new record rides on the level-up opener, which is appended, so
the cached prefix survives. The whole block is folded into the end of the system
prompt only where a history drain already drops the head of history: the record
lives in history until a drain and in the system prompt after it. Every entry
point catches its own exceptions; a failure costs the memory, never the game.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any

log = logging.getLogger(__name__)

HEADER = "=== SOLVED LEVELS (facts from this game plus your own rules; kept for the whole game) ==="
FOOTER = "=== END SOLVED LEVELS ==="
REUSE = "Reuse these where a new level has the same controls; verify with one probe before relying on them."
REASONING_CHARS = 900
RULE_CHARS = 600
# text before a label on its line that marks the model restating the ask
_ECHO = re.compile(r"\b(start\w*|begin\w*|write|asks?|wants?|says?|with)\W*$", re.I)
# the ask's own wording, which a real rule does not repeat
_ASK_WORDS = ("states the goal", "what each control did", "the plan that won")


def _env_int(name: str, default: int) -> int:
    try:
        return max(0, int(os.environ.get(name, "").strip() or default))
    except ValueError:
        return default


def _state(agent: Any) -> dict[str, Any]:
    # lazy, so an agent built or unpickled without it still works
    state = getattr(agent, "_levelmem", None)
    if state is None:
        state = agent._levelmem = {"records": [], "folded": ""}
    return state


def _text(message: dict[str, Any]) -> str:
    """Reasoning, then visible text, in the order written: with thinking on,
    most replies carry no text, and the last 'Rule for level N:' must win."""
    parts: list[str] = []
    for key in ("reasoning_content", "reasoning", "content"):
        value = message.get(key)
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, list):
            parts += [p["text"] for p in value if isinstance(p, dict) and isinstance(p.get("text"), str)]
    return "\n".join(p for p in parts if p)


def level_attempts(entries: list[Any], level: int) -> list[list[str]]:
    """Actions taken on `level`, split into attempts at each RESET.

    history[i].frame is the board AFTER history[i].action, so an action belongs
    to the level of the frame before it; the seeded first entry has no action.
    A RESET with no action before it (the warmup) opens no attempt.
    """
    attempts: list[list[str]] = []
    current: list[str] = []
    prev = None
    for entry in entries or []:
        action = str(getattr(entry, "action", "") or "").strip()
        if action and prev == level:
            if action == "RESET":
                if current:
                    attempts.append(current)
                current = []
            else:
                current.append(action)
        prev = getattr(getattr(entry, "frame", None), "level", prev)
    if current:
        attempts.append(current)
    return attempts


def encode_actions(actions: list[str]) -> str:
    """Run-length: 'UP x3, MOUSE(row=4, col=7), SPACE'."""
    runs: list[list[Any]] = []
    for action in actions:
        if runs and runs[-1][0] == action:
            runs[-1][1] += 1
        else:
            runs.append([action, 1])
    return ", ".join(a if n == 1 else f"{a} x{n}" for a, n in runs)


def find_rule(text: str, level: int) -> str | None:
    """The model's own 'Rule for level N:' paragraph, or None.

    Its own line when that line carries 40+ chars, else up to five following
    non-empty lines. The LAST match wins. The model restating the ask is not an
    answer, so a label followed by a quote, one after 'starting with' or
    'write', and text repeating the ask's wording are all skipped.
    """
    rule = None
    pattern = r"Rule for level\s*" + str(int(level)) + r"[*_]*\s*:(?![\"'`])[ \t]*"
    for match in re.finditer(pattern, text or "", re.I):
        if _ECHO.search(text[text.rfind("\n", 0, match.start()) + 1:match.start()]):
            continue
        lines = text[match.end():].split("\n")
        first = lines[0].strip()
        if len(first) >= 40:
            found = first
        else:
            kept = [first] if first else []
            for line in lines[1:6]:
                if not line.strip():
                    if kept:
                        break
                    continue
                kept.append(line.strip())
            found = " ".join(kept)
        found = " ".join(found.split()).lstrip("*_ ")[:RULE_CHARS]
        if len(found) >= 20 and not any(words in found.lower() for words in _ASK_WORDS):
            rule = found
    return rule


def _capture(state: dict[str, Any], messages: list[Any] | None, *, count: bool) -> None:
    """Read rules back from assistant messages, newest first, while an ask is
    unanswered: a later statement in the same window replaces an earlier one."""
    pending = [r for r in state["records"] if r["asks"] > r["checks"]]
    if not pending:
        return
    texts = [_text(m) for m in messages or [] if isinstance(m, dict) and m.get("role") == "assistant"]
    for record in pending:
        if count:
            record["checks"] += 1
        for text in reversed(texts):
            rule = find_rule(text, record["level"])
            if rule:
                record["rule"] = rule
                break


def _new_record(level: int, entries: list[Any], history: list[Any]) -> dict[str, Any]:
    attempts = level_attempts(entries, level)
    winning = attempts[-1] if attempts else []
    reasoning = ""
    for message in reversed(history):
        if isinstance(message, dict) and message.get("role") == "assistant":
            reasoning = " ".join(_text(message).split())[-REASONING_CHARS:]
            if reasoning:
                break
    keep = _env_int("ARC3_LEVELMEM_ACTIONS", 30)
    return {
        "level": level,
        "n_actions": sum(len(a) for a in attempts),
        "attempts": max(1, len(attempts)),
        "n_winning": len(winning),
        "winning": winning[-keep:] if keep else [],
        "reasoning": reasoning,
        "rule": None,
        "asks": 0,
        "checks": 0,
    }


def _count(n: int, noun: str) -> str:
    return f"{n} {noun}{'s' if n != 1 else ''}"


def _facts(record: dict[str, Any]) -> str:
    return (f"Level {record['level']}: cleared after {_count(record['n_actions'], 'action')} in "
            f"{_count(record['attempts'], 'attempt')}.")


def _winning_line(record: dict[str, Any]) -> str:
    kept = record["winning"]
    if not kept:
        return "Winning attempt: no actions recorded"
    label = (f"last {len(kept)} of {record['n_winning']} actions"
             if record["n_winning"] > len(kept) else _count(len(kept), "action"))
    return f"Winning attempt, {label}: {encode_actions(kept)}"


def _ask(level: int) -> str:
    return (f"Level {level} is solved. In your reply this turn, before acting, write one short "
            f"paragraph that starts exactly with 'Rule for level {level}:' and states the goal, "
            "what each control did, and the plan that won (at most 5 lines). It is kept for every "
            "later level of this game.")


def block(agent: Any, state: dict[str, Any]) -> str:
    """The pinned block, at most ARC3_LEVELMEM_MAX_TOKENS by the trimmer's own measure.

    The newest ARC3_LEVELMEM_FULL_LEVELS levels keep actions and reasoning, older
    ones the rule only; on overflow only the most recent levels that fit are kept.
    """
    chars_per_token = getattr(agent, "_text_chars_per_token", None) or 3.0
    room = int(_env_int("ARC3_LEVELMEM_MAX_TOKENS", 3000) * chars_per_token)
    room -= len(HEADER) + len(REUSE) + len(FOOTER) + 3
    full = _env_int("ARC3_LEVELMEM_FULL_LEVELS", 3)
    kept: list[str] = []
    for index, record in enumerate(reversed(state["records"])):
        rule = f"Your rule: {record['rule'] or 'not stated'}"
        short = text = f"{_facts(record)} {rule}"
        if index < full:
            lines = [_facts(record), rule, _winning_line(record)]
            if record["reasoning"]:
                lines.append(f"Your reasoning in the turn that cleared it: {record['reasoning']}")
            text = "\n  ".join(lines)
        if len(text) + 1 > room:
            text = short
        if len(text) + 1 > room:
            break
        kept.append(text)
        room -= len(text) + 1
    if not kept:
        return ""
    return "\n".join([HEADER, *reversed(kept), REUSE, FOOTER])


def on_opener(agent: Any, prompt: str, frame: Any, entries: list[Any] | None) -> str:
    """Turn start: read back rules, record newly solved levels, and append the
    new records and the rule request to this turn's opener (appended, so free).

    A turn rolled back on a request error leaves history exactly as this hook
    last saw it; that turn's records and ask counts are then undone, so the
    retried opener carries them again. A resumption after a yield restates the
    turn, often without this prompt, so it neither asks nor counts a check.
    """
    try:
        state = _state(agent)
        history = list(getattr(agent, "_history_messages", None) or [])
        resuming = bool(getattr(agent, "_resume_after_yield", False))
        seen = (len(history), history[-1] if history else None)
        undo = state.pop("undo", None)
        if undo and undo[0] == seen[0] and undo[1] is seen[1]:
            del state["records"][undo[2]:]
            for record, asks, checks in undo[3]:
                record["asks"], record["checks"] = asks, checks
        state["undo"] = (*seen, len(state["records"]),
                         [(r, r["asks"], r["checks"]) for r in state["records"]])
        _capture(state, history, count=not resuming)
        level = getattr(frame, "level", None)
        summary = getattr(agent, "_last_step_summary", None) or {}
        added: list[dict[str, Any]] = []
        if isinstance(level, int) and not summary.get("run_complete"):
            done = {r["level"] for r in state["records"]}
            for solved in range(1, level):
                if solved not in done:
                    # only the clearing turn's reasoning is in history now
                    own = solved == level - 1 and summary.get("level_transition")
                    added.append(_new_record(solved, entries or [], history if own else []))
            state["records"].extend(added)
        lines: list[str] = []
        if added:
            lines.append("Solved-level memory, kept for the rest of this game:")
            lines += [f"{_facts(r)} {_winning_line(r)}." for r in added]
        last = state["records"][-1] if state["records"] else None
        if (last is not None and not resuming and not last["rule"]
                and last["asks"] < _env_int("ARC3_LEVELMEM_ASK_TURNS", 2)):
            last["asks"] += 1
            lines.append(_ask(last["level"]))
        if lines:
            prompt += "\n\n" + "\n".join(lines)
    except Exception as exc:
        log.warning("levelmem: opener hook failed, prompt unchanged: %s: %s", type(exc).__name__, exc)
    return prompt


def _splice(text: str, old: str, new: str) -> str:
    if old and old in text:
        return text.replace(old, new)
    return text.rstrip("\n") + "\n\n" + new


def fold(agent: Any, system_message: dict[str, Any] | None, history: list[Any] | None) -> dict[str, Any] | None:
    """Fold the block into the end of the system prompt and return the system
    message to send. Call ONLY where the head of history is being dropped, so
    the prefix is already lost; a block that has not changed is never re-sent."""
    try:
        state = getattr(agent, "_levelmem", None)
        if not state or not state["records"]:
            return system_message
        # the drain may be about to drop the reply that stated a pending rule
        _capture(state, history, count=False)
        new = block(agent, state)
        old = state["folded"]
        if not new or new == old:
            return system_message
        agent._system_prompt = _splice(agent._system_prompt, old, new)
        state["folded"] = new
        log.info("levelmem: %d solved level(s) folded into the system prompt (%d chars)",
                 len(state["records"]), len(new))
        if system_message is None:
            return None
        return {**system_message, "content": _splice(str(system_message.get("content") or ""), old, new)}
    except Exception as exc:
        log.warning("levelmem: fold failed, system prompt unchanged: %s: %s", type(exc).__name__, exc)
        return system_message
'''

_EDITS = [
    ("from inference.agent.runtime_state import Frame, HistoryEntry, RUNTIME_STATE_FILENAME, load_runtime_state\n",
     "from inference.agent import levelmem as _levelmem  # overlay B1\n"),
    # analyze(): the opener is the newest message, so appending to it is free
    ("        display_action_num = _display_action_number(action_num)\n",
     "        # overlay B1: record newly solved levels and ask for their rule\n"
     "        user_prompt = _levelmem.on_opener(self, user_prompt, current_frame, history_entries)\n"),
    # the token drain: the head of history goes, so the prefix is lost anyway;
    # folding before the next estimate makes the drain leave room for the block.
    # Only when a drop will happen: otherwise the fold costs the prefix for
    # nothing and the overflow retry mistakes the larger request for a recovery
    ("            draining = True\n",
     "            if self._drop_oldest_history_block(list(history), preserve_recent=preserve_recent):  # overlay B1\n"
     "                system_message = _levelmem.fold(self, system_message, history)\n"),
    # the turn-cap drain drops the head without the token trimmer
    ("        if history != original_history:\n"
     "            self._note_history_evicted()\n",
     "        if history[:1] != original_history[:1]:\n"
     "            _levelmem.fold(self, None, original_history)  # overlay B1\n"),
]

_module_path = _agent_dir / "levelmem.py"
_tool_agent = _agent_dir / "tool_agent.py"
if _module_path.exists():
    raise RuntimeError(f"levelmem overlay: {_module_path} already exists (applied twice?)")
_text = _tool_agent.read_text()
for _anchor, _insert in _EDITS:
    _count = _text.count(_anchor)
    if _count != 1:
        raise RuntimeError(f"levelmem overlay: anchor matches {_count} times, expected 1: {_anchor.strip()!r}")
    _text = _text.replace(_anchor, _anchor + _insert)
_module_path.write_text(_MODULE)
_tool_agent.write_text(_text)
print("arm overlay levelmem: solved levels' rule and winning actions ride on the level-up opener, "
      "folded into the system prompt at the next drain (ARC3_LEVELMEM_MAX_TOKENS default 3000)")
