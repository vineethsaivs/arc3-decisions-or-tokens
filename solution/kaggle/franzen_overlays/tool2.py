"""tool2: a timed-out snippet keeps its printed output, and retained functions can be read back.

Two defects found by reading the newest replica transcripts (kaggle-lf15us8-1031-1, kaggle-lf15uns2-1030-3):

1. A snippet that used its 30 s was killed from outside and the model got `{"error": "Tool timed out after 30s",
   "stdout": ""}`: the progress prints, the line where the search stalled and any functions the snippet defined were all
   lost (16 timeouts in 12 game runs, two on a final level). Now the sandbox child sets a timer for the limit minus 3 s,
   counted as its own compute (the time it waits for the host to answer an action or an animation request is not
   counted, as the host does not count it). The timer raises `SnippetTimeout`, a BaseException so a model's `except
   Exception` cannot swallow it, and a timer that fires while the child waits for the host is deferred so a reply is
   never cut. The ordinary error path then returns the printed output, the traceback line and the retention payload. A
   snippet that swallows it with a bare `except` is stopped again a second later. The host's own hard deadline is
   unchanged and still applies if the child cannot stop.
2. Retained functions are compiled from a source string, so `inspect.getsource(fn)` raised `OSError: could not get
   source code` (15 calls in 12 game runs): the model rewrote the function from memory or called code it no longer
   understood. Now `inspect.getsource` returns the stored source of a retained function and the source of a function
   defined in the same snippet; any other object falls back to the original behaviour.
3. The system prompt says both.

Run by build_franzen_arms.py after Franzen's harness patch, as exec(code, {'BUNDLE_DIR': BUNDLE_DIR}). Stdlib only.
Every edit is asserted to match once; nothing is written unless both files still compile.
"""
from pathlib import Path

_inf = Path(BUNDLE_DIR) / "src" / "ARC3-Inference" / "inference"  # noqa: F821
_MARK = "overlay tool2"

_BLOCK = '''    # overlay tool2: a snippet past its compute limit stops with its printed output; getsource reads retained code
    import signal as _t2_signal
    import time as _t2_time

    class SnippetTimeout(BaseException):
        pass

    _T2 = {"wait": 0.0, "recv_at": None, "start": 0.0, "limit": 27, "retained": {}, "code": ""}
    _t2_recv_plain = _recv

    def _recv():
        _T2["recv_at"] = _t2_time.monotonic()
        try:
            return _t2_recv_plain()
        finally:
            _T2["wait"] += _t2_time.monotonic() - _T2["recv_at"]
            _T2["recv_at"] = None

    def _t2_used():
        now = _t2_time.monotonic()
        waited = _T2["wait"] + (now - _T2["recv_at"] if _T2["recv_at"] is not None else 0.0)
        return now - _T2["start"] - waited

    def _t2_handler(signum, frame):
        if _T2["recv_at"] is not None:  # blocked on the host: never cut a reply, look again soon
            _t2_signal.setitimer(_t2_signal.ITIMER_REAL, 0.25)
            return
        left = _T2["limit"] - _t2_used()
        if left > 0.05:
            _t2_signal.setitimer(_t2_signal.ITIMER_REAL, left)
            return
        _t2_signal.setitimer(_t2_signal.ITIMER_REAL, 1.0)  # a snippet that swallows this is stopped again
        raise SnippetTimeout(f"the snippet used its {int(_T2['limit'])} s of compute and was stopped; the output it "
                             "printed before that is returned. Search in smaller steps, or call action() earlier.")

    @contextlib.contextmanager
    def _t2_guard(limit):
        _T2.update(start=_t2_time.monotonic(), wait=0.0, recv_at=None, limit=limit)
        old = _t2_signal.signal(_t2_signal.SIGALRM, _t2_handler)
        _t2_signal.setitimer(_t2_signal.ITIMER_REAL, limit)
        try:
            yield
        finally:
            _t2_signal.setitimer(_t2_signal.ITIMER_REAL, 0)
            _t2_signal.signal(_t2_signal.SIGALRM, old)

    def _t2_install_getsource():
        import inspect as _t2_inspect
        original = _t2_inspect.getsource

        def getsource(obj):
            code = getattr(obj, "__code__", None)
            try:
                if code is not None and code.co_filename == "<retained>" and obj.__name__ in _T2["retained"]:
                    return _T2["retained"][obj.__name__]
                if code is not None and code.co_filename == "<python_tool>":
                    for node in ast.walk(ast.parse(_T2["code"])):
                        if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == obj.__name__
                                and min([node.lineno] + [d.lineno for d in node.decorator_list]) == code.co_firstlineno):
                            return "".join(_T2["code"].splitlines(True)[code.co_firstlineno - 1:node.end_lineno])
            except Exception:
                pass
            return original(obj)

        _t2_inspect.getsource = getsource


'''

_EDITS = {
    _inf / "agent" / "python_tool_sandbox.py": [
        ("    def main():\n        initial = _recv()\n", _BLOCK + "    def main():\n        initial = _recv()\n"),
        ("            with contextlib.redirect_stdout(stdout):\n",
         "            _T2[\"retained\"] = dict(_kept_pool)  # overlay tool2\n"
         "            _T2[\"code\"] = _code_text\n"
         "            _t2_install_getsource()\n"
         "            with _t2_guard(max(1, timeout_seconds - 3)), contextlib.redirect_stdout(stdout):\n"),
        ("            TerminalStateActionError,\n            Exception,\n        ) as exc:",
         "            TerminalStateActionError,\n            SnippetTimeout,  # overlay tool2\n            Exception,\n        ) as exc:"),
    ],
    _inf / "agent" / "prompts.py": [
        ('    "Tool results report which functions were retained and explain any rejection.\\n"\n',
         '    "Tool results report which functions were retained and explain any rejection. '
         '`inspect.getsource(fn)` returns the source of a retained function or of one defined in the same call.\\n"\n'),
        ('    "- Each `python` tool call has a hard time limit of 30 seconds.\\n"\n',
         '    "- Each `python` tool call has a hard time limit of 30 seconds. A call that uses its time is stopped with '
         'a `SnippetTimeout` message and the output it printed so far, so print progress while you search.\\n"\n'),
    ],
}

_plan = []
for _path, _edits in _EDITS.items():
    _text = _path.read_text()
    if _MARK in _text or "SnippetTimeout" in _text:
        raise RuntimeError("tool2 overlay: already applied to " + str(_path))
    for _old, _new in _edits:
        _count = _text.count(_old)
        if _count != 1:
            raise RuntimeError(f"tool2 overlay: anchor matches {_count} times in {_path.name}, expected 1: "
                               f"{_old.strip()[:70]!r}")
        _text = _text.replace(_old, _new)
    compile(_text, str(_path), "exec")
    _plan.append((_path, _text))
for _path, _text in _plan:
    _path.write_text(_text)
print("arm overlay tool2: a timed-out snippet keeps its printed output; inspect.getsource reads retained functions")
