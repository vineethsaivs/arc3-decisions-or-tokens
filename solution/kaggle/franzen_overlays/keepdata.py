"""keepdata: plain data a snippet bound survives into the next python call, until an action executes.

Measured (docs/research/2026-10-11-improve/transcript-forensics.md and its verifier): every run has 47 to 64
NameErrors on a name an earlier call of the same level bound, and 9 to 12 blocked `time` imports; 88% of the
same-level data NameErrors have no executed action between the binding and the use. The sandbox restarts per call,
so only retained functions survive today.

Sandbox (python_tool_sandbox.py): top-level names a snippet assigned whose values are plain literals (bool, int,
float, str, None, list, tuple, dict, set; repr round-trips through ast.literal_eval; at most 20,000 chars each and
60,000 in all; not private, not a provided name) come back as keepable_data with the retention payload and are
restored, with ast.literal_eval, at the start of the next call. Function retention is unchanged, so a function
that reads such data is still not retained. `time` and `hashlib` join the importable modules.
Host (tool_agent.py): the agent keeps the last keepable_data, passes it to the next call, and drops it whenever an
action executes (and at a new session), so restored data always describes the current board. No prompt text
invites relying on it; only the module list in prompts.py gains time and hashlib.

Run by build_franzen_arms.py with the other harness overlays, as exec(code, {'BUNDLE_DIR': BUNDLE_DIR}). Stdlib
only. Every edit is an exact replacement asserted to match once, and nothing is written unless every file and the
sandbox's child script still compile. solution/sim/test_franzen_keepdata.py runs the real sandbox and host.
"""
import textwrap
from pathlib import Path

_inf = Path(BUNDLE_DIR) / "src" / "ARC3-Inference" / "inference"
_SANDBOX = _inf / "agent" / "python_tool_sandbox.py"
_AGENT = _inf / "agent" / "tool_agent.py"
_PROMPTS = _inf / "agent" / "prompts.py"
_MARK = "overlay keepdata"

_SANDBOX_EDITS = [
    ('        "string",\n    }\n',
     '        "string",\n        "time",  # overlay keepdata\n        "hashlib",  # overlay keepdata\n    }\n'),
    ('        _available_names = set(runtime_globals) | set(runtime_globals["__builtins__"])\n',
     '        _available_names = set(runtime_globals) | set(runtime_globals["__builtins__"])\n'
     '        # overlay keepdata: plain data from this turn\'s earlier calls; the host\n'
     '        # drops it whenever an action executes, so it always matches the board\n'
     '        _provided = set(_available_names)\n'
     '        _kept_data_in = dict(initial.get("kept_data") or {})\n'
     '        for _kd_name, _kd_text in _kept_data_in.items():\n'
     '            if _kd_name not in _provided:\n'
     '                try:\n'
     '                    runtime_globals[_kd_name] = ast.literal_eval(_kd_text)\n'
     '                except Exception:\n'
     '                    pass\n'
     '\n'
     '        def _keepable_data():\n'
     '            names = list(_kept_data_in)\n'
     '            try:\n'
     '                body = ast.parse(_code_text).body\n'
     '            except SyntaxError:\n'
     '                body = []\n'
     '            for node in body:\n'
     '                if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):\n'
     '                    for target in (node.targets if isinstance(node, ast.Assign) else [node.target]):\n'
     '                        names += [n.id for n in ast.walk(target) if isinstance(n, ast.Name)]\n'
     '            kept, total = {}, 0\n'
     '            for name in dict.fromkeys(names):\n'
     '                if name.startswith("_") or name in _provided or name not in runtime_globals:\n'
     '                    continue\n'
     '                value = runtime_globals[name]\n'
     '                if type(value) not in (\n'
     '                        bool, int, float, str, type(None), list, tuple, dict, set):\n'
     '                    continue\n'
     '                try:\n'
     '                    text = repr(value)\n'
     '                    if len(text) <= 20000 and total + len(text) <= 60000 and ast.literal_eval(text) == value:\n'
     '                        kept[name] = text\n'
     '                        total += len(text)\n'
     '                except Exception:\n'
     '                    pass\n'
     '            return kept\n'),
    ('                return {\n'
     '                    "keepable_functions": [{"name": n, "source": src} for n, src in kept.items()],\n',
     '                return {\n'
     '                    "keepable_data": _keepable_data(),  # overlay keepdata\n'
     '                    "keepable_functions": [{"name": n, "source": src} for n, src in kept.items()],\n'),
    # host side of the pipe
    ('    repair_hints: bool = False,\n) -> dict[str, Any]:\n',
     '    repair_hints: bool = False,\n    kept_data: dict[str, str] | None = None,  # overlay keepdata\n) -> dict[str, Any]:\n'),
    ('                "kept_functions": list(kept_functions or []),\n',
     '                "kept_functions": list(kept_functions or []),\n'
     '                "kept_data": dict(kept_data or {}),  # overlay keepdata\n'),
    ('                    "keepable_functions": list(message.get("keepable_functions") or []),\n',
     '                    "keepable_functions": list(message.get("keepable_functions") or []),\n'
     '                    **({"keepable_data": message["keepable_data"]}  # overlay keepdata\n'
     '                       if isinstance(message.get("keepable_data"), dict) else {}),\n'),
]
_AGENT_EDITS = [
    ("        self._kept_functions: dict[str, str] = {}\n        self._last_animation_chain: list[Any] = []\n",
     "        self._kept_functions: dict[str, str] = {}\n"
     "        self._kept_data: dict[str, str] = {}  # overlay keepdata\n"
     "        self._last_animation_chain: list[Any] = []\n"),
    ("            if _persistent_functions():\n"
     "                self._kept_functions = {}\n"
     "                self._retained_clear_notice = \"\"\n"
     "            self._history_messages = []\n",
     "            if _persistent_functions():\n"
     "                self._kept_functions = {}\n"
     "                self._retained_clear_notice = \"\"\n"
     "            self._kept_data = {}  # overlay keepdata\n"
     "            self._history_messages = []\n"),
    ("            kept_functions=self._retained_sources(),\n"
     "            retain_imports=_get_env_bool(\"ARC3_PERSISTENT_FUNCTIONS_IMPORTS\", False),\n",
     "            kept_functions=self._retained_sources(),\n"
     "            kept_data=getattr(self, \"_kept_data\", None) if _persistent_functions() else None,  # overlay keepdata\n"
     "            retain_imports=_get_env_bool(\"ARC3_PERSISTENT_FUNCTIONS_IMPORTS\", False),\n"),
    ("        self._record_retained_functions(sandbox_result, payload)\n"
     "        rendered_stdout = str(sandbox_result.get(\"stdout\", \"\") or \"\")\n",
     "        self._record_retained_functions(sandbox_result, payload)\n"
     "        if isinstance(sandbox_result.get(\"keepable_data\"), dict):  # overlay keepdata\n"
     "            self._kept_data = dict(sandbox_result[\"keepable_data\"])\n"
     "        rendered_stdout = str(sandbox_result.get(\"stdout\", \"\") or \"\")\n"),
    ("        if step_executed:\n"
     "            self._last_step_summary = self._summarize_step_sequence(action_results)\n",
     "        if step_executed:\n"
     "            self._kept_data = {}  # overlay keepdata: data bound before an action may not match the board\n"
     "            self._last_step_summary = self._summarize_step_sequence(action_results)\n"),
]
_PROMPT_EDITS = [
    ("json, math, operator, random, re, statistics, string.\\n\"\n",
     "json, math, operator, random, re, statistics, string, time, hashlib.\\n\"  # overlay keepdata\n"),
]

_FILES = {_SANDBOX: _SANDBOX_EDITS, _AGENT: _AGENT_EDITS, _PROMPTS: _PROMPT_EDITS}
_texts = {path: path.read_text() for path in _FILES}
for _path, _text in _texts.items():
    if _MARK in _text:
        raise RuntimeError("keepdata overlay: already applied to " + str(_path))
for _path, _edits in _FILES.items():
    for _old, _new in _edits:
        _count = _texts[_path].count(_old)
        if _count != 1:
            raise RuntimeError(f"keepdata overlay: anchor matches {_count} times in {_path.name}, expected 1: "
                               f"{_old.strip()[:70]!r}")
        _texts[_path] = _texts[_path].replace(_old, _new)
for _path, _text in _texts.items():
    compile(_text, str(_path), "exec")
_child = _texts[_SANDBOX].split('textwrap.dedent(\n    r"""', 1)[1].split('"""', 1)[0]
compile(textwrap.dedent(_child), "python_tool_sandbox bootstrap", "exec")
for _path, _text in _texts.items():
    _path.write_text(_text)
print("arm overlay keepdata: plain snippet data kept until the next executed action; time and hashlib importable")
