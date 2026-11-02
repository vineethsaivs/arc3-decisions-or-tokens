"""promptobs (lite): two hard-error fixes in what the model is told and what its code may call. Facts, no advice.

Measured in the y8-family transcripts (docs/research/2026-10-11-improve/prompts-observation.md and its verifier):
PO3 the animation fields are typed in the prompt: `frames`, `transient_pixels` and the timeline's `changed` are
    ints and its `bbox` is text. All 22 "object of type 'int' has no len()" tool errors hit these fields.
PO5 forgiveness for two API forms the model guesses: single-letter U/D/L/R action names (every "Unknown action"
    rejection was one of them, and each discarded the whole batch), and step, level, ascii, shape and
    segmentation read through history entries and transitions (all 8 view AttributeErrors).
The verifier killed the rest of the original proposal: the image-geometry line (no downstream gain), row_runs
(2.66x a raw crop's tokens) and the teal diff background (lower contrast for 75% of changed cells). Expected
effect about 0 levels; it rides along, gated on its counters. +74 system-prompt tokens, nothing per opener.

Run by build_franzen_arms.py with the other harness overlays, as exec(code, {'BUNDLE_DIR': BUNDLE_DIR}). Stdlib
only. Every edit is an exact replacement asserted to match once; nothing is written unless every edited file and
the sandbox's child script still compile. solution/sim/test_franzen_promptobs.py checks it offline.
"""
import textwrap
from pathlib import Path

_inf = Path(BUNDLE_DIR) / "src" / "ARC3-Inference" / "inference"
_PROMPTS = _inf / "agent" / "prompts.py"
_AGENT = _inf / "agent" / "tool_agent.py"
_SANDBOX = _inf / "agent" / "python_tool_sandbox.py"
_NAMES = _inf / "agent" / "action_names.py"
_MARK = "overlay promptobs"

_EDITS = {
    _PROMPTS: [('    "reports `frames`, `transient_pixels` and `transient_bbox`. Transient cells "\n', '    # overlay promptobs: field types (22 len() errors on these ints)\n    "reports `frames` (an int: how many frames the action passed through), `transient_pixels` "\n    "(an int count) and `transient_bbox` (a list `[top, left, bottom, right]`). Transient cells "\n'), ('    "shows which cells changed at each step. It is a dict with `action`, `frames` and "\n    "`steps`, one entry per step that changed anything. Each entry has `step`, "\n    "`changed`, `bbox`, and "\n', '    # overlay promptobs: field types\n    "shows which cells changed at each step. It is a dict with `action`, `frames` (an int count; "\n    "the frames themselves are `last_animation_frames`) and `steps`, one entry per step that changed "\n    "anything. Each entry has `step`, `changed` (an int: how many cells changed), `bbox` (a text "\n    "string such as \'rows 3-7, cols 10-12\'), and "\n')],
    _SANDBOX: [('        def __str__(self):\n            return f"AsciiHistoryEntryView(action={self.action!r}, frame={self.frame})"\n', '        def __getattr__(self, name):\n            # overlay promptobs: h.step, h.level, h.ascii read the entry\'s frame\n            frame = self.__dict__.get("frame")\n            if name in ("step", "level", "ascii", "shape", "segmentation") and frame is not None:\n                return getattr(frame, name)\n            raise AttributeError(f"HistoryEntryView has no attribute {name!r}; its board is .frame")\n\n        def __str__(self):\n            return f"AsciiHistoryEntryView(action={self.action!r}, frame={self.frame})"\n'), ('        def __str__(self):\n            return (\n                "ActionTransitionView("\n', '        def __getattr__(self, name):\n            # overlay promptobs: t.step, t.level, t.ascii read the after-frame\n            if name == "action_display":\n                return self.__dict__.get("action")\n            frame = self.__dict__.get("after_frame")\n            if name in ("step", "level", "ascii", "shape", "segmentation") and frame is not None:\n                return getattr(frame, name)\n            raise AttributeError(f"TransitionView has no attribute {name!r}; use .before_frame or .after_frame")\n\n        def __str__(self):\n            return (\n                "ActionTransitionView("\n')],
    _NAMES: [('MODEL_TO_ENGINE_ACTION = {value: key for key, value in ENGINE_TO_MODEL_ACTION.items()}\n', 'MODEL_TO_ENGINE_ACTION = {value: key for key, value in ENGINE_TO_MODEL_ACTION.items()}\n# overlay promptobs: U/D/L/R from the model\'s own path search (11 refused calls in three runs)\nMODEL_TO_ENGINE_ACTION.update({"U": "ACTION1", "D": "ACTION2", "L": "ACTION3", "R": "ACTION4"})\n')],
}

_texts = {path: path.read_text() for path in _EDITS}
for _path, _text in _texts.items():
    if _MARK in _text:
        raise RuntimeError("promptobs overlay: already applied to " + str(_path))
for _path, _edits in _EDITS.items():
    for _old, _new in _edits:
        _count = _texts[_path].count(_old)
        if _count != 1:
            raise RuntimeError(f"promptobs overlay: anchor matches {_count} times in {_path.name}, "
                               f"expected 1: {_old.strip()[:70]!r}")
        _texts[_path] = _texts[_path].replace(_old, _new)
for _path, _text in _texts.items():
    compile(_text, str(_path), "exec")
_child = _texts[_SANDBOX].split('textwrap.dedent(\n    r"""', 1)[1].split('"""', 1)[0]
compile(textwrap.dedent(_child), "python_tool_sandbox bootstrap", "exec")
for _path, _text in _texts.items():
    _path.write_text(_text)
print("arm overlay promptobs-lite: animation field types, U/D/L/R aliases, frame fields on history entries and transitions")
