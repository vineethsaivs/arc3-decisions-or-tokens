"""imports: more stdlib modules and builtins in the python tool, with time.sleep a no-op.

A blocked import or a missing builtin raises and the whole snippet is lost; the reply after an error needs about 1.5K
more tokens to reach an action (7.6K against 6.1K). Two gaps:
1. Modules. The 10-23 census (docs/research/2026-10-23-push/harness.md C2) counted, per bench run, time 9.6, inspect
   4.3, traceback 0.6 and types 0.3 blocked imports (held out: time 11.3, inspect 4.9). This allows time, inspect,
   traceback, types, hashlib, pprint, timeit, dataclasses, typing and enum. sys, builtins and pickle stay blocked
   (escape and persistence routes). `import time` returns a copy of the module whose sleep returns at once, so a
   snippet cannot idle its 30 s budget away.
2. Builtins. SAFE_BUILTINS has no __build_class__, so `class State: pass` raised "NameError: __build_class__ not
   found", and object, setattr, vars, super, id and ImportError were missing too (checked 10-27 on the pin's tree):
   any search code built on a class died at its first line. Those are added, and runtime globals get __name__.
The new modules are not pre-bound (wastefix pre-binds SAFE_MODULES only) and the prompt's module list is unchanged, so
behaviour changes only where a snippet used to raise, as when difflib was added.

Run by build_franzen_arms.py right after Franzen's harness patch, as exec(code, {'BUNDLE_DIR': BUNDLE_DIR}).
Stdlib only. Two exact edits asserted to match once; nothing is written unless the file still compiles.
"""
from pathlib import Path

_SANDBOX = Path(BUNDLE_DIR) / "src" / "ARC3-Inference" / "inference" / "agent" / "python_tool_sandbox.py"  # noqa: F821
_MARK = "overlay imports"
_OLD = (
    "    def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):\n"
    "        root = str(name or \"\").split(\".\", 1)[0]\n"
    "        if root not in SAFE_MODULES:\n"
)
_NEW = (
    "    # overlay imports: allowed but not pre-bound; `import time` gets a copy whose sleep is a no-op\n"
    "    EXTRA_SAFE_MODULES = {\"time\", \"inspect\", \"traceback\", \"types\", \"hashlib\", \"pprint\", \"timeit\",\n"
    "                          \"dataclasses\", \"typing\", \"enum\"}\n"
    "    _TIME_SHIM = None\n"
    "\n"
    "\n"
    "    def _time_shim():\n"
    "        global _TIME_SHIM\n"
    "        if _TIME_SHIM is None:\n"
    "            import time as _real_time\n"
    "            import types as _types\n"
    "            _TIME_SHIM = _types.ModuleType(\"time\")\n"
    "            for _key in dir(_real_time):\n"
    "                if not _key.startswith(\"__\"):\n"
    "                    setattr(_TIME_SHIM, _key, getattr(_real_time, _key))\n"
    "            _TIME_SHIM.sleep = lambda *args, **kwargs: None\n"
    "        return _TIME_SHIM\n"
    "\n"
    "\n"
    "    def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):\n"
    "        root = str(name or \"\").split(\".\", 1)[0]\n"
    "        if name == \"time\":\n"
    "            return _time_shim()\n"
    "        if root not in SAFE_MODULES and root not in EXTRA_SAFE_MODULES:\n"
)

# the model's builtins: wastefix anchors on this one line and inserts after it, so the line itself stays as it is
_OLD_BUILTINS = "        runtime_globals[\"__builtins__\"][\"__import__\"] = _safe_import\n"
_NEW_BUILTINS = (
    _OLD_BUILTINS
    + "        # overlay imports: class definitions and the builtins they need\n"
    "        for _name in (\"__build_class__\", \"object\", \"setattr\", \"delattr\", \"id\", \"super\", \"property\",\n"
    "                      \"staticmethod\", \"classmethod\", \"vars\", \"ImportError\", \"ModuleNotFoundError\",\n"
    "                      \"BaseException\", \"TimeoutError\", \"MemoryError\"):\n"
    "            runtime_globals[\"__builtins__\"].setdefault(_name, getattr(builtins, _name))\n"
    "        runtime_globals.setdefault(\"__name__\", \"__main__\")\n"
)

_text = _SANDBOX.read_text()
if _MARK in _text:
    raise RuntimeError("imports overlay: already applied to " + str(_SANDBOX))
for _old, _new in ((_OLD, _NEW), (_OLD_BUILTINS, _NEW_BUILTINS)):
    _count = _text.count(_old)
    if _count != 1:
        raise RuntimeError(f"imports overlay: anchor matches {_count} times in {_SANDBOX.name}, expected 1: "
                           f"{_old.strip()[:70]!r}")
    _text = _text.replace(_old, _new)
compile(_text, str(_SANDBOX), "exec")
_SANDBOX.write_text(_text)
print("arm overlay imports: 10 more stdlib modules (time.sleep a no-op); class definitions, object, setattr, vars, "
      "super, id and ImportError available")
