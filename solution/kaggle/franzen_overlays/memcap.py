"""Sandbox memory cap: a runaway snippet dies with MemoryError instead of taking the model server with it.

The python tool's child process limits CPU time, file size and open files, but not memory, and host RAM sits at
115 to 120 of 177 GB in every run. The SGLang scheduler holds the Flash-Next PLE table as page-locked anonymous
memory (models/qwen4_exp.py), about 105 GB of RSS, so a host OOM kill would very likely pick the server, not the
snippet that caused it: a scored run would lose its server. Son Pham's team added the same cap on 10-05 as a crash
guard. Now the child sets RLIMIT_AS to 8 GiB (an allocation past it raises MemoryError inside the snippet) and
its own oom_score_adj to 1000, so if the host still runs out the kernel kills a sandbox child first. No recorded
legitimate snippet comes near 8 GiB (0 MemoryErrors in 17,508 calls), so normal play is unchanged.

Run by build_franzen_arms.py after the other harness overlays, as exec(code, {'BUNDLE_DIR': BUNDLE_DIR}). Stdlib
only. The edit is an exact replacement asserted to match once, and nothing is written unless the file and the
sandbox's child script inside it still compile. solution/sim/test_franzen_memcap.py checks the edit offline
(macOS ignores RLIMIT_AS; the behaviour itself is checked on Linux).
"""
import textwrap
from pathlib import Path

_SANDBOX = Path(BUNDLE_DIR) / "src" / "ARC3-Inference" / "inference" / "agent" / "python_tool_sandbox.py"
_MARK = "overlay memcap"
_OLD = ('        cpu_limit = max(1, int(timeout_seconds)) + 1\n'
        '        for limit, value in (\n'
        '            (getattr(resource, "RLIMIT_CPU", None), cpu_limit),\n')
_NEW = ('        cpu_limit = max(1, int(timeout_seconds)) + 1\n'
        '        try:  # overlay memcap: a host OOM kills this child, not the SGLang scheduler\n'
        '            with open("/proc/self/oom_score_adj", "w") as _oom:\n'
        '                _oom.write("1000")\n'
        '        except OSError:\n'
        '            pass\n'
        '        for limit, value in (\n'
        '            (getattr(resource, "RLIMIT_AS", None), 8 << 30),  # overlay memcap: MemoryError at 8 GiB\n'
        '            (getattr(resource, "RLIMIT_CPU", None), cpu_limit),\n')

_text = _SANDBOX.read_text()
if _MARK in _text:
    raise RuntimeError("memcap overlay: already applied to " + str(_SANDBOX))
if _text.count(_OLD) != 1:
    raise RuntimeError(f"memcap overlay: anchor matches {_text.count(_OLD)} times in {_SANDBOX.name}, expected 1")
_text = _text.replace(_OLD, _NEW)
compile(_text, str(_SANDBOX), "exec")
_child = _text.split('textwrap.dedent(\n    r"""', 1)[1].split('"""', 1)[0]
compile(textwrap.dedent(_child), "python_tool_sandbox bootstrap", "exec")
_SANDBOX.write_text(_text)
print("arm overlay memcap: sandbox child RLIMIT_AS 8 GiB and oom_score_adj 1000")
