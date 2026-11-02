"""Transcript-waste fixes: corrections and lossless repairs, no new advice text.

docs/research/2026-10-04-climb/transcript-waste.md measured where the pinned config
(p448s14, two 25-game runs) wastes replies. This carries only its fixes 4, 2(a) and 5:

4. Tool timeouts. The 30 s snippet deadline was wall clock including the host's work
   for every action and animation request (stepping the game, reloading and
   reserializing the whole history: about 2 s per action past step 500), so late
   batches died with all their output. A killed snippet also reports no
   keepable_functions, which read as "every retained function dropped": 24 timeouts
   cleared 629 functions. Host handler time now extends the deadline, and a result
   without keepable_functions leaves the retained set alone. The sandbox's own work,
   including its parse of each state payload, still counts against the deadline and
   RLIMIT_CPU (31 CPU-s). The incremental-history rewrite is not here.
2(a). The sandbox's own SAFE_MODULES plus deque/Counter/defaultdict are pre-bound in
   the snippet globals: 19 tracebacks used an import an earlier call made.
   _safe_import is unchanged, so blocked modules stay blocked.
5. The API matches its text. last_animation_timeline gains items() and values() and
   keeps the documented action/frames/steps keys when the last action did not
   animate. The host then answers {"error": ...}, which is truthy, so a plain {}
   default never applied; the reply is merged over the default instead. Nodes carry
   bbox and the node line names it; the last_action_call_result line names its
   fields; the stop rule names level_completed again (the terminal latch still
   enforces it); the tool schema stops calling retained functions ephemeral; the
   diff-image sentence says when the image comes; one prompt line regains its
   newline and grammar.

Run by build_franzen_arms.py right after Franzen's harness patch, as
exec(code, {'BUNDLE_DIR': BUNDLE_DIR}). Stdlib only. Every edit is an exact
replacement asserted to match once, and nothing is written until all four files,
and the sandbox's child script inside one of them, still compile.
"""
import textwrap
from pathlib import Path

_inf = Path(BUNDLE_DIR) / "src" / "ARC3-Inference" / "inference"
_SANDBOX = _inf / "agent" / "python_tool_sandbox.py"
_AGENT = _inf / "agent" / "tool_agent.py"
_PROMPTS = _inf / "agent" / "prompts.py"
_SEGMENTATION = _inf / "utils" / "segmentation.py"
_MARK = "overlay wastefix"

_EDITS = {
    _SANDBOX: [
        # fix 2(a), inside the child script: before _available_names is taken, so a
        # retained function may rely on these names like any other provided global
        ('        runtime_globals["__builtins__"]["__import__"] = _safe_import\n',
         '        runtime_globals["__builtins__"]["__import__"] = _safe_import\n'
         '        # overlay wastefix: the allowed modules and the usual collections names\n'
         '        # come pre-bound, so a snippet relying on an earlier call\'s import runs\n'
         '        for _prebound in sorted(SAFE_MODULES):\n'
         '            runtime_globals[_prebound] = __import__(_prebound)\n'
         '        for _prebound in ("deque", "Counter", "defaultdict"):\n'
         '            runtime_globals[_prebound] = getattr(runtime_globals["collections"], _prebound)\n'),
        # fix 5, _LazyAnimation
        ('                        self._value = _animation_fetch(self._kind) or self._empty\n',
         '                        self._value = _animation_fetch(self._kind) or self._empty\n'
         '                        if isinstance(self._empty, dict) and isinstance(self._value, dict):\n'
         '                            # overlay wastefix: a non-animating action gets {"error": ...}\n'
         '                            # back; keep that note under the documented keys\n'
         '                            self._value = {**self._empty, **self._value}\n'),
        ('                def keys(self):\n'
         '                    return self._get().keys()\n',
         '                def keys(self):\n'
         '                    return self._get().keys()\n'
         '\n'
         '                def items(self):\n'
         '                    return self._get().items()\n'
         '\n'
         '                def values(self):\n'
         '                    return self._get().values()\n'),
        ('_LazyAnimation("timeline", {})\n',
         '_LazyAnimation("timeline", {"action": None, "frames": 0, "steps": []})\n'),
        # fix 4, host side: the clock stops while the host serves the child
        ('        deadline = time.monotonic() + max(1, int(timeout_seconds))\n'
         '        while True:\n'
         '            remaining = deadline - time.monotonic()\n',
         '        deadline = time.monotonic() + max(1, int(timeout_seconds))\n'
         '        hard_stop = deadline + 120  # overlay wastefix: host time is excused for 2 minutes at most\n'
         '        host_since = None  # overlay wastefix\n'
         '        while True:\n'
         '            if host_since is not None:\n'
         '                # the host\'s work on an action or animation request, up to its\n'
         '                # reply, is not snippet time: it grows with game length\n'
         '                deadline = min(hard_stop, deadline + time.monotonic() - host_since)\n'
         '                host_since = None\n'
         '            remaining = deadline - time.monotonic()\n'),
        ('            msg_type = str(message.get("type", "")).strip()\n',
         '            msg_type = str(message.get("type", "")).strip()\n'
         '            if msg_type in ("animation", "action"):\n'
         '                host_since = time.monotonic()  # overlay wastefix\n'),
    ],
    _AGENT: [
        # fix 4: a killed or crashed snippet returns no keepable_functions key at all
        ('    def _record_retained_functions(self, sandbox_result, payload) -> None:\n',
         '    def _record_retained_functions(self, sandbox_result, payload) -> None:\n'
         '        if "keepable_functions" not in sandbox_result:\n'
         '            # overlay wastefix: a timeout or crash says nothing about the\n'
         '            # functions, so keep them instead of reading it as all dropped\n'
         '            return\n'),
        # fix 5: the schema contradicted the retained functions when they are on
        ('                                    "Python code to run. The snippet is ephemeral and is not saved across tool calls."\n',
         '                                    # overlay wastefix: true in both persistence modes\n'
         '                                    "Python code to run. Variables do not persist between calls; eligible functions do."\n'
         '                                    if _persistent_functions() else\n'
         '                                    "Python code to run. The snippet is ephemeral and is not saved across tool calls."\n'),
        # fix 5: resumes and unchanged boards carry no diff image (44% of openers)
        ('            " Alongside the current grid image you receive a diff image: cells that "\n'
         '            "changed since your previous analyzer turn are shown in their new true "\n',
         '            " When the board changed since your previous analyzer turn, a diff image "\n'
         '            "follows the current grid image: changed cells are shown in their new true "\n'),
    ],
    _PROMPTS: [
        # fix 5: the level-transfer patch dropped level_completed from the stop rule
        ('        .replace(\n'
         '            "- If an action result reports `game_over`, `run_complete`, `level_completed`, or `done`, '
         'stop acting immediately and re-ground on the next turn.\\n",\n'
         '            "- If an action result reports `game_over`, `run_complete`, or `done`, '
         'stop acting immediately and re-ground on the next turn.\\n",\n'
         '        )\n', ''),
        ('    "- Game entities are usually be rendered as connected multi-tile shapes such as 2\u00d72, 2\u00d73, 3\u00d73, '
         'or longer patterned structures. Sometime they might also be 1x1 tokens."\n',
         '    "- Game entities are usually rendered as connected multi-tile shapes such as 2\u00d72, 2\u00d73, 3\u00d73, '
         'or longer patterned structures. Sometimes they might also be 1x1 tokens.\\n"\n'),
        ("`pixels` (cell count), `boundary` (",
         "`pixels` (cell count), `bbox` (`[min_row, min_col, max_row, max_col]`, inclusive), `boundary` ("),
        ("- `last_action_call_result` contains the result of the latest model-issued `action(...)` call, "
         "including batch totals, executed/skipped actions, and stop reasons.",
         "- `last_action_call_result` is a dict with the result of the latest model-issued `action(...)` call: "
         "`executed` (a bool), `executed_actions` (the list of actions that ran), `stop_reason` (what stopped "
         "the call, if anything: a game event such as `level_completed`, or a guard), "
         "`gameplay_changed_per_action` (one bool per executed action, for batches), plus batch totals and "
         "skipped actions."),
    ],
    _SEGMENTATION: [
        # fix 5: spliced into the sandbox too, so every node view carries it
        ('                "pixels": len(comp["cells"]),\n'
         '                "boundary": [[r, c] for r, c in boundary],\n',
         '                "pixels": len(comp["cells"]),\n'
         '                # overlay wastefix: inclusive bounding box; boundary is a corner list\n'
         '                "bbox": [min(r for r, _ in comp["cells"]), min(c for _, c in comp["cells"]),\n'
         '                         max(r for r, _ in comp["cells"]), max(c for _, c in comp["cells"])],\n'
         '                "boundary": [[r, c] for r, c in boundary],\n'),
    ],
}

_texts = {path: path.read_text() for path in _EDITS}
for _path, _text in _texts.items():
    if _MARK in _text:
        raise RuntimeError("wastefix overlay: already applied to " + str(_path))
for _path, _edits in _EDITS.items():
    for _old, _new in _edits:
        _count = _texts[_path].count(_old)
        if _count != 1:
            raise RuntimeError(f"wastefix overlay: anchor matches {_count} times in {_path.name}, "
                               f"expected 1: {_old.strip()[:70]!r}")
        _texts[_path] = _texts[_path].replace(_old, _new)
for _path, _text in _texts.items():
    compile(_text, str(_path), "exec")
# the child script is a string, so the line above cannot see a mistake inside it
_child = _texts[_SANDBOX].split('textwrap.dedent(\n    r"""', 1)[1].split('"""', 1)[0]
compile(textwrap.dedent(_child), "python_tool_sandbox bootstrap", "exec")
for _path, _text in _texts.items():
    _path.write_text(_text)
print("arm overlay wastefix: host action time is outside the 30 s snippet deadline, timeouts keep retained "
      "functions, allowed stdlib pre-bound, API text corrected (timeline, bbox, call result, stop rule, schema)")
