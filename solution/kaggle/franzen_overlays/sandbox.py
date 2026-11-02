"""B4 overlay: the parts of rellik13's Wang/Ludvig harness port (M90) that Franzen lacks.

rellik13 (Milestone 2, 22.53) went 9.96 -> 13.40 on porting Wang's harness and
called it "cheaper, more precise turns". Diffed line by line against Franzen's
patched tree, most of it is already here: the same -I -S sandbox and module
whitelist (Franzen's is a superset), 4-connected segmentation (Franzen's adds
rotation-aware hashes), the same code-first Python guidance verbatim, an
object-level frame_diff, UNDO, animation frames and a detailed game-over
message. Two parts are absent, and neither adds advice text:

1. The turn opener drops the static lines it repeats every turn ("Only tool:
   python...", the tool-call format rule, ...). Each restates a line of the
   system prompt. This is Wang's five-line turn message: every dynamic line
   Franzen added (outcomes, guard stops, game-over and level-up text, auto diff,
   state, valid actions) stays. About 1,480 characters per opener, so more of
   each game's own turns fit before a trim. The "When ready, call
   action(actions) ... prefer batching it in one call" line STAYS: it is the
   opener's only act/batch cue, and Franzen's _opener_commit_hint docstring
   measured that kind of wording moving batch size (6.7 to 6.9 actions against
   3.9 to 5.4), so deleting it would confound shorter openers with smaller
   batches.
2. The system prompt loses the line telling the model to ignore the edge bar
   and the clause about countdown bars (rellik's v18.1gs). Franzen's own flag
   semantics line and BAR RULE say the bar is the action budget to plan around.

Not ported, with reasons: Wang's node bbox and KeyError field hints (Franzen's
system prompt lists every node field, and 2,226 baseline tool calls in
franzen-rep-v1 show 0 KeyErrors on a node field, so there is nothing to fix),
Wang's priors and turn-protocol prose (added advice; the protocol asks for a
world model every turn, which Franzen measured worse), the cell-level diff
(Franzen's object-level one covers it), per-transition animation frames
(Franzen exposes the last animation, his strongest perception gain), "Level
restarted" and UNDO (already better here).

Run by build_franzen_arms.py right after Franzen's harness patch, as
exec(code, {'BUNDLE_DIR': BUNDLE_DIR}). Stdlib only. Every edit is an exact
replacement asserted to match once; both files are checked before either is
written. Plan: docs/PLAN_2026-10-01.md, track B.
"""
from pathlib import Path

_inf = Path(BUNDLE_DIR) / "src" / "ARC3-Inference" / "inference"
_PROMPTS = _inf / "agent" / "prompts.py"
_AGENT = _inf / "agent" / "tool_agent.py"

_STATIC_LINES = (
    "Only tool: `python`. It receives `current_frame`, `previous_frame`, `history`, `transitions`, `last_transition`, `valid_actions`, `last_action_call_result`, `frame_diff(before, after)`, and `action(actions)`.",
    "Only letter-coded board views and lightweight metadata are exposed; raw numeric color IDs are not available.",
    "Keep tool output compact: use `current_frame.segmentation` as the primary view, and `current_frame.ascii` only for a small specific region; never print full boards.",
    "For the most recent change, compare `previous_frame` to `current_frame`, or `last_transition.before_frame` to `last_transition.after_frame`; `history[-1].frame` is the current frame, not the previous one.",
    "Use Python to inspect the evidence from the newest history, and search or score candidate actions or short sequences against the current goal as you currently understand it.",
    "Ground yourself in `current_frame` before acting, but start with a compact structural summary rather than restating the full frame.",
    "Focus on what changed most recently in `history`, update the target environment change if needed, and separate gameplay-object changes from HUD-only changes.",
    "You may call `action(actions)` more than once in one Python snippet if your search or control loop needs it.",
    "If you use MOUSE, include integer row and col arguments.",
)

_FILTER_DEF = (
    "# overlay B4: the turn opener's static lines, each a restatement of the system\n"
    "# prompt, dropped as in Wang's five-line turn message\n"
    "_WANG_STATIC_OPENER_LINES = frozenset((\n"
    + "".join(f"    {line!r},\n" for line in _STATIC_LINES)
    + "    TOOL_CALL_FORMAT_GUIDANCE,\n"
    "))\n\n\n"
)

_EDITS = {
    _PROMPTS: [
        (" If you identify such a bar, do not get distracted by it or treat it as core gameplay state"
         " unless there is concrete evidence that it interacts with the puzzle mechanics.", ""),
        (" and whether the last action affected gameplay state or only HUD elements such as countdown bars", ""),
    ],
    _AGENT: [
        # pure insertion, same anchor audit.py inserts at: it survives either order
        ("\nclass ToolAgent:\n", "\n" + _FILTER_DEF + "class ToolAgent:\n"),
        ("        return \"\\n\".join(lines)\n\n    def _tools(self, state_path: Path)",
         "        return \"\\n\".join(line for line in lines if line not in _WANG_STATIC_OPENER_LINES)\n\n"
         "    def _tools(self, state_path: Path)"),
    ],
}

_texts = {path: path.read_text() for path in _EDITS}
if "_WANG_STATIC_OPENER_LINES" in _texts[_AGENT]:
    raise RuntimeError("sandbox overlay: already applied to " + str(_AGENT))
# a reworded static line would silently stay in every opener: require each verbatim
for _line in _STATIC_LINES:
    if _texts[_AGENT].count('"' + _line + '"') != 1:
        raise RuntimeError(f"sandbox overlay: static opener line not found once: {_line[:60]!r}")
if _texts[_AGENT].count("                TOOL_CALL_FORMAT_GUIDANCE,\n") != 1:
    raise RuntimeError("sandbox overlay: TOOL_CALL_FORMAT_GUIDANCE opener line not found once")
for _path, _edits in _EDITS.items():
    for _old, _new in _edits:
        _count = _texts[_path].count(_old)
        if _count != 1:
            raise RuntimeError(f"sandbox overlay: anchor matches {_count} times in {_path.name}, expected 1: {_old.strip()[:70]!r}")
        _texts[_path] = _texts[_path].replace(_old, _new)
for _path, _text in _texts.items():
    _path.write_text(_text)
print("arm overlay sandbox: turn openers drop their static lines (the act/batch cue stays); "
      "the ignore-the-bar lines are gone")
