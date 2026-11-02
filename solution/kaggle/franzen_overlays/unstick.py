"""unstick: a stall check, an untested-object list and the step-verification rules, aimed at levels the agent never clears.

Read on four replicas of the r3 agent (10-12; 562 level plays, 63 never cleared) and in eight paired plays of the same
level, cleared in one run and stuck in another (docs/research/2026-10-30-unstick/FINDINGS.md):
- Stuck runs stop acting. 27% of never-cleared levels have 3 or more analyzer turns in a row with no executed action
  (each is at least one 8,192-token yield), against 8% of cleared ones; 40% of their clock goes to such turns (29%).
  Cases: s5i5 L3 r1 (10 turns without acting after a trim garbled its own clicks), sk48 L2 r1 and r3s4 (40 minutes of
  search and simulation, no action), s5i5 L2 r3 ("52 actions left! ... think REALLY carefully", 51 minutes, 2 actions).
- Stuck runs write off an object type untested. tn36 L2 r2 never clicked the 12 vertical bars it called "connectors"
  (the clearing runs found they were separate toggles); sp80 L2 r2 clicked MOUSE once, on the wrong object, wrote
  "MOUSE is not a mechanic here" and never clicked the red bars the clearing runs used.
- Clearing runs coded a simulator, checked it against recorded moves, stopped a batch on the first mismatch, and read
  "impossible under my model" as a missing mechanic; stuck runs changed the goal instead (sp80 r2). Tufa's
  STEP_VERIFICATION_ADDENDUM says most of this and is off in Franzen's settings.

What changes (tool_agent.py and prompts.py; nothing in the sandbox):
1. Stall check. The agent counts analyzer turns on the level since the last executed action. At 3, 5, 7, ... the opener
   says so and asks for at least one action this turn: a short, low-risk probe with its effect predicted first, or the
   verified first steps of the plan.
2. Untested list. With every stall check, and on the first opener after a game over or a model RESET on the level, it
   lists the object types (same color and shape) on the current board that no click on this level has touched (a click
   counts for the object under it, objects of its color and bounding-box size, and anything within one cell),
   when MOUSE is valid, and the valid actions not yet used on this level. Large regions (a tenth of the board, or three
   quarters of its height or width) and thin border strips (timer bars) are left out; at most six types.
3. Prompt. STEP_VERIFICATION_ADDENDUM is on by default, its "no solution" rule says to suspect the mechanics first (an
   untried action or object type, an object taken for decoration, a misread animation) and the goal last, and one line
   asks the model to replay this level's recorded transitions through a simulator before running a long plan from it.
Only text in the opener and the system prompt changes; no action is ever taken or refused by this overlay. Any error in
the bookkeeping returns no lines.

Run by build_franzen_arms.py right after Franzen's harness patch, as exec(code, {'BUNDLE_DIR': BUNDLE_DIR}).
Stdlib only. Every edit is asserted to match once; nothing is written unless both files still compile.
"""
from pathlib import Path

_AGENT_DIR = Path(BUNDLE_DIR) / "src" / "ARC3-Inference" / "inference" / "agent"  # noqa: F821
_MARK = "overlay unstick"

_HELPERS = '''
# overlay unstick: stall check and untested-object list for the turn opener
_UNSTICK_STALL_TURNS = 3
_UNSTICK_MAX_TYPES = 6
_UNSTICK_MOUSE = re.compile(r"MOUSE\\(row=(\\d+), col=(\\d+)\\)")


def _unstick_component(grid, r, c):
    """Cells of the 4-connected same-color component holding (r, c)."""
    if not (0 <= r < len(grid) and 0 <= c < len(grid[r])):
        return set()
    color = grid[r][c]
    seen = {(r, c)}
    stack = [(r, c)]
    while stack:
        y, x = stack.pop()
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if ((ny, nx) not in seen and 0 <= ny < len(grid) and 0 <= nx < len(grid[ny])
                    and grid[ny][nx] == color):
                seen.add((ny, nx))
                stack.append((ny, nx))
    return seen


def _unstick_box(cells):
    """(height, width) of a component's bounding box: with its color, the object type."""
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return max(rows) - min(rows) + 1, max(cols) - min(cols) + 1


def _unstick_coverage(current_frame, history_entries, valid_actions):
    """Untested-object and unused-action lines for the current level."""
    from inference.utils.grid_utils import ARC_COLOR_CHARS
    level = current_frame.level
    clicked, clicked_keys, used = set(), set(), set()
    previous = None
    for entry in history_entries:
        frame = entry.frame
        name = str(entry.action or "").strip()
        if (previous is not None and frame is not None and previous.level == level and name
                and not (entry.result or {}).get("automatic")):
            match = _UNSTICK_MOUSE.match(name)
            if match:
                r, c = int(match.group(1)), int(match.group(2))
                clicked.add((r, c))
                cells = _unstick_component(previous.grid, r, c)
                if cells:
                    clicked_keys.add((previous.grid[r][c], _unstick_box(cells)))
                used.add("MOUSE")
            else:
                used.add(name.split("(")[0].strip().upper())
        previous = frame
    lines = []
    valid = [str(a).strip().upper() for a in (valid_actions or [])]
    if "MOUSE" in valid:
        grid = current_frame.grid
        rows = len(grid)
        cols = max((len(row) for row in grid), default=0)
        near = {(r + dr, c + dc) for r, c in clicked for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
        types, seen = {}, set()
        for r in range(rows):
            for c in range(len(grid[r])):
                if (r, c) in seen:
                    continue
                cells = _unstick_component(grid, r, c)
                seen |= cells
                r0 = min(y for y, _ in cells)
                r1 = max(y for y, _ in cells)
                c0 = min(x for _, x in cells)
                c1 = max(x for _, x in cells)
                if len(cells) * 10 >= rows * cols or (r1 - r0 + 1) * 4 >= rows * 3 or (c1 - c0 + 1) * 4 >= cols * 3:
                    continue
                if (r0 <= 1 or c0 <= 1 or r1 >= rows - 2 or c1 >= cols - 2) and min(r1 - r0, c1 - c0) <= 1:
                    continue
                key = (grid[r][c], (r1 - r0 + 1, c1 - c0 + 1))
                kind = types.setdefault(key, {"touched": False, "count": 0, "box": key[1], "at": None,
                                              "pixels": len(cells)})
                kind["count"] += 1
                if key in clicked_keys or not cells.isdisjoint(near):
                    kind["touched"] = True
                if kind["at"] is None:
                    cy, cx = (r0 + r1) / 2, (c0 + c1) / 2
                    kind["at"] = min(cells, key=lambda p: (p[0] - cy) ** 2 + (p[1] - cx) ** 2)
        untested = sorted(((k, v) for k, v in types.items() if not v["touched"]),
                          key=lambda kv: (-kv[1]["count"], -kv[1]["pixels"]))
        if untested:
            shown = []
            for (color, _), v in untested[:_UNSTICK_MAX_TYPES]:
                char = ARC_COLOR_CHARS[color] if 0 <= color < len(ARC_COLOR_CHARS) else str(color)
                h, w = v["box"]
                count = f" x{v['count']}" if v["count"] > 1 else ""
                shown.append(f"{char} {h}x{w}{count} (e.g. row {v['at'][0]}, col {v['at'][1]})")
            more = len(untested) - len(shown)
            lines.append("Untested on this level: no click on this level has touched an object of these types: "
                         + "; ".join(shown) + (f"; and {more} more type(s)" if more > 0 else "")
                         + ". One no-op on one object does not show that a whole type does nothing.")
    unused = [a for a in valid if a not in used and a not in ("RESET", "MOUSE")]
    if unused:
        lines.append("Valid actions not yet used on this level: " + ", ".join(unused) + ".")
    return lines


def _unstick_lines(agent, action_num, current_frame, history_entries, valid_actions, summary):
    """Stall check and untested list; [] on the first turn of a level and on any error."""
    try:
        if current_frame is None:
            return []
        level = current_frame.level
        if getattr(agent, "_unstick_level", None) != level:
            agent._unstick_level, agent._unstick_idle, agent._unstick_last = level, 0, action_num
            return []
        acted = action_num != agent._unstick_last
        agent._unstick_last = action_num
        agent._unstick_idle = 0 if acted else agent._unstick_idle + 1
        idle = agent._unstick_idle
        summary = summary or {}
        failed = acted and (bool(summary.get("game_over")) or any(
            str(a).strip().upper() == "RESET" for a in (summary.get("executed_actions") or [])))
        stall = idle >= _UNSTICK_STALL_TURNS and (idle - _UNSTICK_STALL_TURNS) % 2 == 0
        lines = []
        if stall:
            lines.append(
                f"Stall check: your last {idle} turns on this level executed no action. Reasoning and code spend "
                "no actions but do spend this game's limited time, and a level still unsolved when time runs out "
                "scores nothing. In this turn, execute at least one action: a short, low-risk probe whose effect "
                "you state first, or the verified first steps of your plan; then compare the result with your "
                "prediction.")
        if stall or failed:
            try:
                lines.extend(_unstick_coverage(current_frame, history_entries, valid_actions))
            except Exception:
                pass
        return lines
    except Exception:
        return []


'''

_AGENT_EDITS = [
    ("def _stale_summary_lines(summary: dict[str, Any]) -> list[str]:\n",
     _HELPERS.lstrip("\n") + "def _stale_summary_lines(summary: dict[str, Any]) -> list[str]:\n"),
    ("        lines.extend(self._prepare_auto_diff(current_frame, previous_step_summary))\n",
     "        lines.extend(self._prepare_auto_diff(current_frame, previous_step_summary))\n"
     "        lines.extend(_unstick_lines(self, action_num, current_frame, history_entries, valid_actions,\n"
     "                                    previous_step_summary))  # overlay unstick\n"),
    ("    if _get_env_bool(\"ARC3_STEP_VERIFICATION_HINT\", False):\n",
     "    if _get_env_bool(\"ARC3_STEP_VERIFICATION_HINT\", True):  # overlay unstick: on by default\n"),
]

_PROMPT_EDITS = [
    ("    \"that the game is solvable. Reconsider your mechanics, goal, search implementation, \"\n"
     "    \"or search limits, including interactions with new elements. Take a targeted action \"\n",
     "    \"that the game is solvable, so your model is missing something. Suspect the mechanics \"\n"
     "    \"first: an action or object type you have not tried, an object you took for decoration, \"\n"
     "    \"or a misread animation; then your search implementation or limits; change the goal \"\n"
     "    \"last. Take a targeted action \"\n"),
    ("    \"the result.\\n\"\n)\n",
     "    \"the result.\\n\"\n"
     "    \"- Before running a long plan from a simulator or rule you wrote, replay this level's \"\n"
     "    \"recorded `transitions` through it in code and fix any mismatch first.\\n\"  # overlay unstick\n)\n"),
]

_plan = []
for _name, _edits in (("tool_agent.py", _AGENT_EDITS), ("prompts.py", _PROMPT_EDITS)):
    _path = _AGENT_DIR / _name
    _text = _path.read_text()
    if _MARK in _text:
        raise RuntimeError("unstick overlay: already applied to " + str(_path))
    for _old, _new in _edits:
        _count = _text.count(_old)
        if _count != 1:
            raise RuntimeError(f"unstick overlay: anchor matches {_count} times in {_name}, expected 1: "
                               f"{_old.strip()[:70]!r}")
        _text = _text.replace(_old, _new)
    compile(_text, str(_path), "exec")
    _plan.append((_path, _text))
for _path, _text in _plan:
    _path.write_text(_text)
print("arm overlay unstick: stall check and untested-object list in the opener; step verification on")
