"""keepvars: plain-data variables survive between python calls on a level; globals, locals and SystemExit exist.

Each python call runs in a fresh sandbox process and only retained functions carry over, so a snippet that reuses its
own earlier variable dies at a NameError. On a 25-game Kaggle replica of the pin's agent (lfk, 10-28) 226 of 2,866
calls ended in a traceback, 91 of them NameErrors, nearly all on the model's own names (G, plan, MAZE, guards5, sols,
R4) and a few on missing builtins (globals, time, SystemExit); the model even writes `globals()['x'] = ...` to keep
state. Each failed call costs about one reply (about 2.2K tokens): roughly 3% of the run's tokens.

What changes:
1. Sandbox (python_tool_sandbox.py, inside the bootstrap): after a call, top-level names the snippet assigned whose
   values are plain data (int, finite float, str, bool, None, and lists, tuples, sets and dicts of those, nested, at
   most 20,000 elements) are returned as repr text, up to 100K characters each and 300K in all, on both the success and
   the error path. Before the next call they are restored with ast.literal_eval, never over a provided global or a
   retained function. Objects, functions, modules, numpy arrays and names starting with "_" are never kept. The
   builtins globals, locals and SystemExit are added.
2. Agent (tool_agent.py): the kept text travels with the call; it is cleared at every level transition and with each
   new session, so nothing crosses a level.
3. Prompt (prompts.py): "Python variables reset between tool calls." becomes a sentence saying plain-data variables
   are kept on the level and cleared when it changes, and that anything derived from an earlier frame may be stale.

Run by build_franzen_arms.py right after Franzen's harness patch, as exec(code, {'BUNDLE_DIR': BUNDLE_DIR}).
Stdlib only. Every edit is asserted to match once; nothing is written unless all three files still compile.
"""
from pathlib import Path

_AGENT_DIR = Path(BUNDLE_DIR) / "src" / "ARC3-Inference" / "inference" / "agent"  # noqa: F821
_MARK = "overlay keepvars"

_SANDBOX_EDITS = [
    # builtins: the line itself stays (wastefix and imports anchor on it and insert after it)
    ("        runtime_globals[\"__builtins__\"][\"__import__\"] = _safe_import\n",
     "        runtime_globals[\"__builtins__\"][\"__import__\"] = _safe_import\n"
     "        # overlay keepvars: builtins the model reaches for\n"
     "        for _kv_b in (\"globals\", \"locals\", \"SystemExit\"):\n"
     "            runtime_globals[\"__builtins__\"].setdefault(_kv_b, getattr(builtins, _kv_b))\n"),
    # restore before the snippet runs, and add the kept text to both payloads
    ("        try:\n"
     "            _code_text = str(initial.get(\"code\", \"\"))\n",
     "        # overlay keepvars: plain-data variables kept between calls on a level\n"
     "        _kv_base = set(runtime_globals)\n"
     "\n"
     "        def _kv_plain(value, depth=0, budget=None):\n"
     "            budget = budget if budget is not None else [20000]\n"
     "            budget[0] -= 1\n"
     "            if budget[0] < 0 or depth > 12:\n"
     "                return False\n"
     "            kind = type(value)\n"
     "            if kind in (int, str, bool) or value is None:\n"
     "                return True\n"
     "            if kind is float:\n"
     "                return value == value and value not in (float(\"inf\"), float(\"-inf\"))\n"
     "            if kind in (list, tuple, set):\n"
     "                return all(_kv_plain(item, depth + 1, budget) for item in value)\n"
     "            if kind is dict:\n"
     "                return all(_kv_plain(k, depth + 1, budget) and _kv_plain(v, depth + 1, budget)\n"
     "                           for k, v in value.items())\n"
     "            return False\n"
     "\n"
     "        def _kv_collect():\n"
     "            kept, total = {}, 0\n"
     "            for name, value in list(runtime_globals.items()):\n"
     "                if name.startswith(\"_\") or name in _kv_base or name == \"result\":\n"
     "                    continue\n"
     "                try:\n"
     "                    if not _kv_plain(value):\n"
     "                        continue\n"
     "                    text = repr(value)\n"
     "                except Exception:\n"
     "                    continue\n"
     "                if len(text) > 100000 or total + len(text) > 300000:\n"
     "                    continue\n"
     "                kept[name] = text\n"
     "                total += len(text)\n"
     "            return kept\n"
     "\n"
     "        for _kv_name, _kv_text in dict(initial.get(\"kept_vars\") or {}).items():\n"
     "            if not str(_kv_name).isidentifier() or _kv_name in runtime_globals:\n"
     "                continue\n"
     "            try:\n"
     "                runtime_globals[_kv_name] = ast.literal_eval(str(_kv_text))\n"
     "            except Exception:\n"
     "                pass\n"
     "\n"
     "        _kv_collect_functions = _collect_retention_payload\n"
     "\n"
     "        def _collect_retention_payload():\n"
     "            payload = dict(_kv_collect_functions() or {})\n"
     "            try:\n"
     "                payload[\"kept_vars\"] = _kv_collect()\n"
     "            except Exception:\n"
     "                payload[\"kept_vars\"] = {}\n"
     "            return payload\n"
     "\n"
     "        try:\n"
     "            _code_text = str(initial.get(\"code\", \"\"))\n"),
    # host: accept, send and return the kept text
    ("    kept_functions: list[str] | None = None,\n",
     "    kept_functions: list[str] | None = None,\n"
     "    kept_vars: dict[str, str] | None = None,  # overlay keepvars\n"),
    ("                \"repair_hints\": repair_hints,\n",
     "                \"repair_hints\": repair_hints,\n"
     "                \"kept_vars\": dict(kept_vars or {}),  # overlay keepvars\n"),
    ("                    \"action_results\": list(message.get(\"action_results\") or host_action_results),\n",
     "                    \"action_results\": list(message.get(\"action_results\") or host_action_results),\n"
     "                    \"kept_vars\": dict(message.get(\"kept_vars\") or {}),  # overlay keepvars\n"),
]

_AGENT_EDITS = [
    ("            animation_handler=_handle_animation if _animation_enabled() else None,\n",
     "            animation_handler=_handle_animation if _animation_enabled() else None,\n"
     "            kept_vars=dict(getattr(self, \"_kept_vars\", None) or {}),  # overlay keepvars\n"),
    ("        payload: dict[str, Any] = {\"tool\": \"python\"}\n",
     "        payload: dict[str, Any] = {\"tool\": \"python\"}\n"
     "        if \"kept_vars\" in sandbox_result:  # overlay keepvars\n"
     "            self._kept_vars = dict(sandbox_result.get(\"kept_vars\") or {})\n"),
    ("        if summary.get(\"level_transition\") and _persistent_functions_scope() == \"levelup\":\n",
     "        if summary.get(\"level_transition\"):\n"
     "            self._kept_vars = {}  # overlay keepvars: variables never cross a level\n"
     "        if summary.get(\"level_transition\") and _persistent_functions_scope() == \"levelup\":\n"),
    ("            self._history_messages = []\n"
     "            self._session_total_tokens = 0\n",
     "            self._history_messages = []\n"
     "            self._kept_vars = {}  # overlay keepvars\n"
     "            self._session_total_tokens = 0\n"),
]

_PROMPT_EDITS = [
    ("    \"- Python variables reset between tool calls. Eligible functions you define are \"\n",
     "    \"- Plain-data variables you assign at top level (numbers, strings, booleans, lists, tuples, sets, \"\n"
     "    \"dicts) are kept for later calls on this level and cleared when the level changes; anything you \"\n"
     "    \"derived from an earlier frame may be stale, so recompute it from `current_frame` when the board \"\n"
     "    \"may have changed. Eligible functions you define are \"\n"),
]

_plan = []
for _name, _edits in (("python_tool_sandbox.py", _SANDBOX_EDITS), ("tool_agent.py", _AGENT_EDITS),
                      ("prompts.py", _PROMPT_EDITS)):
    _path = _AGENT_DIR / _name
    _text = _path.read_text()
    if _MARK in _text:
        raise RuntimeError("keepvars overlay: already applied to " + str(_path))
    for _old, _new in _edits:
        _count = _text.count(_old)
        if _count != 1:
            raise RuntimeError(f"keepvars overlay: anchor matches {_count} times in {_name}, expected 1: "
                               f"{_old.strip()[:70]!r}")
        _text = _text.replace(_old, _new)
    compile(_text, str(_path), "exec")
    _plan.append((_path, _text))
for _path, _text in _plan:
    _path.write_text(_text)
print("arm overlay keepvars: plain-data variables kept between python calls on a level (cleared at level change); "
      "globals, locals and SystemExit available")
