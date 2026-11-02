"""Offline check for B1, the reply-cache round trip: franzen_overlays/replycache.py (bundle
overlay) and franzen_overlays/replycache_serve.py (cell 15).

    python3 solution/sim/test_franzen_replycache.py

Harness side, on the real code with a fake server (no GPU, no network):
  - two recorded replies, real THINKING blocks from franzen-p448s14-v1 (the transcript keeps
    the history form, so each is rebuilt with its paragraph breaks; when the transcripts are
    on disk the recorded text is compared against them);
  - one real analyze() turn per reply: the fake server returns it, the harness runs its
    python call in the real sandbox and builds the next request;
  - both requests rendered through the chat template the way SGLang renders them (jinja2 from
    this interpreter, else from the ARC-AGI-3-Agents venv); there the hand port of the
    template's assistant branch is checked against the render, and without jinja2 it stands in;
  - unpatched: the re-sent reasoning is the transcript text with the blank lines gone, and the
    next prompt does not extend the previous prompt plus the generated text;
  - patched (pristine tree, and with wastefix in either order): the re-sent reasoning is the
    server's minus the think tags, and the next prompt extends the previous prompt plus the
    generated text, character for character, which is what the prefix cache needs (text, not
    tokens: the server retokenizes the history, so token identity is the L1 reuse gate's);
  - exactly tool_agent.py changes, it compiles, a second application refuses, an anchor that
    matches 0 or 2 times raises without writing, and it composes with every other overlay.
Serving side, on the wheel's own unified_radix_cache.py: the diagnostic patches only with
ARC3_REPLYCACHE_DIAG=1, once, nothing else moves, a rerun is a no-op, drift and
ARC3_REPLYCACHE_DONATE=1 raise, it composes with nvfp4kv_serve.py (whose 0002 edits the same
file) in either order, its template note tells the checked template from any other, and the
patched cache_finished_req (with the wheel's MambaComponent, RadixKey and InsertParams) runs
and logs the documented line. The notebook and wheel facts the serve docstring rests on are
asserted as well.
"""
from __future__ import annotations

import ast
import hashlib
import json
import logging
import os
import py_compile
import re
import shutil
import sys
import tempfile
import types
import zipfile
from pathlib import Path
from types import SimpleNamespace

sys.dont_write_bytecode = True
try:
    import requests  # noqa: F401
except ImportError:  # the harness catches requests.RequestException, which a MagicMock stub cannot be
    _requests = types.ModuleType("requests")
    _requests.RequestException = type("RequestException", (Exception,), {})
    _requests.HTTPError = type("HTTPError", (_requests.RequestException,), {})
    sys.modules["requests"] = _requests

import overlay_orders  # noqa: E402
from test_franzen_sandbox import FORK_ENV, INF, OVERLAYS, TREE, apply  # noqa: E402
from test_franzen_wastefix import changed, load  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "solution" / "kaggle"))
import build_franzen_arms as builder  # noqa: E402

AGENT = INF / "agent/tool_agent.py"
SERVE = OVERLAYS / "replycache_serve.py"
WHEEL = Path(os.environ.get("REPLYCACHE_SGLANG_WHEEL") or __import__("franzen_tree").pennyroyal_wheel())
TRANSCRIPTS = REPO / "solution/kaggle/generated/kernel-outputs/franzen-p448s14-v1/transcripts-dl"
# Byte-identical to the chat_template.jinja of Intel/Qwen3.8-Flash-Next-W4A16-AutoRound, the
# checkpoint Franzen serves (Hugging Face tree oid, a git blob id, checked 2026-10-05).
TEMPLATE = REPO / "solution/sim/fixtures/q38_chat_template.jinja"
TEMPLATE_BLOB = "c0c686f9c38d70d179fb7b5f5aa7530bc913dda3"
TEMPLATE_LINES = (
    "{%- set reasoning_content = reasoning_content|trim %}",
    "{%- set content = render_content(message.content, true)|trim %}",
    "{%- if preserve_thinking is undefined or preserve_thinking is true or loop.index0 > ns.last_query_index %}",
    "{{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content + '\\n</think>\\n\\n' + content }}",
    "{{- '<tool_call>\\n<function=' + tool_call.name + '>\\n' }}",
    "{{- '<parameter=' + args_name + '>\\n' }}",
    "{%- set args_value = args_value | string if args_value is string else args_value | tojson | safe %}",
    "{{- '\\n</parameter>\\n' }}",
    "{{- '</function>\\n</tool_call>' }}",
    "        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}",
    "{{- '<|im_start|>assistant\\n' }}",
    "{{- '<think>\\n' }}",
)
PROMPT = "<|im_start|>assistant\n<think>\n"  # the generation prompt every reply continues
CODE = "nodes = current_frame.segmentation['nodes']\n\nprint(len(nodes), current_frame.shape)"

# (transcript, index of its [THINKING] block, the reply's paragraphs as the server returned them).
# The transcript records "\n".join(paragraphs); the server sent "\n\n".join(paragraphs) + "\n".
REPLIES = (
    ("ar25-0c556536_p0.txt", 28, (
        "Both objects are now on their targets:\n"
        "- B48 (T object) at rows 42-47, cols 9-20 = target A' \u2713\n"
        "- B56 (L object) at rows 42-53, cols 33-44 = target B' \u2713",
        "The timer bar is now orange ('O' 40) \u2014 a warning that time is running low (40 left).",
        "Now stage 3: SPACE (\u2192 line control), 5 UP \u2192 line center from 43 to 28. Then the reflections "
        "should land on A and B.",
        "Remaining: 40 actions, need 6 \u2713.",
        "Let's execute.",
    )),
    ("m0r0-492f87ba_p0.txt", 9, (
        "The maze map:\n```\n1 ...######...\n2 ..###.####..\n3 .###...###..\n4 .##.....##..\n5 ..#......#..\n"
        "6 ..#......#..\n7 .###....##..\n8 .####.####..\n9 .####.####..\n10 .####.###...\n```",
        "Note j=5 column is '.' except i=1. So left and right halves connect only via the top bar (i=1).",
        "Now BFS joint states. Let me implement and test goal candidates.",
        "Note: the maze in block coordinates: left half cells are j<=4, right half j>=6, connected at i=1 (j=3..8).",
        "Goal candidates:\nA) both at i==1\nB) left at (1,3), right at (1,8)  [outer ends]\n"
        "C) left at (1,8), right at (1,3)  [swap - impossible probably]\n"
        "D) left in right chamber & right in left chamber (swap)\nE) squares adjacent (merged)",
        "Let me BFS from current state (7,3),(8,7) and compute min distance to each candidate set. Also compute "
        "the reachable state space size.",
    )),
)
SECTION = re.compile(r"^\[[A-Z][A-Z ]*(?::[^\]]*)?\]$")

# wheel members the serve patch edits or its docstring cites
URC = "sglang/srt/mem_cache/unified_radix_cache.py"
WHEEL_FACTS = {
    "sglang/srt/runtime_context.py": [
        '        and get_exec().mamba.mamba_radix_cache_strategy == "extra_buffer_lazy"\n'],
    "sglang/srt/managers/scheduler_components/batch_result_processor.py": [
        "                is_insert = (\n                    req.mamba_lazy_is_insert\n"
        "                    if mamba_extra_buffer_lazy_enabled()\n                    else True\n                )\n"
        "                release_kv_cache(req, self.tree_cache, is_insert=is_insert)\n",
        "        result.num_correct_drafts_per_req_cpu = [x - 1 for x in accept_lens]\n",
        "        elif result.num_correct_drafts_per_req_cpu is not None:\n"],
    "sglang/srt/mem_cache/unified_cache/components/mamba_component.py": [
        "            cache_len = req.mamba_last_track_seqlen\n",
        "                    req, mamba_ping_pong_track_buffer_to_keep=keep_idx\n"],
    "sglang/srt/mem_cache/memory_pool.py": [
        "        self.mamba_ping_pong_track_buffer_size = 2 if enable_overlap_schedule else 1\n",
        '        """Return the ping-pong index holding the most recent tracked state."""\n'
        "        return req.mamba_last_track_idx\n"],
    "sglang/srt/mem_cache/unified_cache/unified_tree_core.py": [
        "        self.is_eagle = params.is_eagle and ComponentType.MAMBA not in components\n"],
    "sglang/srt/mem_cache/registry.py": ["    cache = UnifiedRadixCache(params)\n"],
    # the server's reasoning is the text before </think>, verbatim; a tool parameter loses one LF each side
    "sglang/srt/parser/reasoning_parser.py": ["                reasoning_text=processed_text[:end_idx],\n"],
    "sglang/srt/function_call/qwen3_coder_detector.py": [
        '                        if p_val.startswith("\\n"):\n                            p_val = p_val[1:]\n'
        '                        if p_val.endswith("\\n"):\n                            p_val = p_val[:-1]\n'],
    "sglang/srt/mem_cache/radix_cache.py": [],
    "sglang/srt/mem_cache/base_prefix_cache.py": [],
    URC: [],
}
NOTEBOOK_FACTS = ('    MAMBA_RADIX="extra_buffer",\n',
                  '"--mamba-radix-cache-strategy", CFG["MAMBA_RADIX"], "--mamba-track-interval", "64",',
                  '"--page-size", "64"', '"--default-chat-template-kwargs", \'{"preserve_thinking":true}\'',
                  '"--reasoning-parser", "qwen3", "--tool-call-parser", "qwen3_coder"')


class Captured(BaseException):
    """Ends analyze() once the next request is in hand (BaseException passes every handler)."""


def server_form(paragraphs) -> str:
    return "\n\n".join(paragraphs) + "\n"


def generated(paragraphs) -> str:
    """Everything the model wrote after PROMPT: the reasoning, then the python call."""
    return (server_form(paragraphs) + "</think>\n\n<tool_call>\n<function=python>\n<parameter=code>\n" + CODE
            + "\n</parameter>\n</function>\n</tool_call><|im_end|>")


def history_turn(message: dict) -> str:
    """A hand port of the template's assistant branch for one call with preserve_thinking on
    (TEMPLATE_LINES); check_round_trip holds it to the real render whenever jinja2 imports."""
    (call,) = message["tool_calls"]
    content = (message.get("content") or "").strip()
    assert not content, message
    text = PROMPT + message.get("reasoning_content", "").strip() + "\n</think>\n\n" + content
    text += "<tool_call>\n<function=" + call["function"]["name"] + ">\n"
    for name, value in json.loads(call["function"]["arguments"]).items():
        assert isinstance(value, str), value
        text += "<parameter=" + name + ">\n" + value + "\n</parameter>\n"
    return text + "</function>\n</tool_call><|im_end|>\n"


def chat_renderer():
    """render(messages, tools) -> the prompt SGLang builds for a request, or None without jinja2.
    As the wheel does it: template_manager._load_jinja_template reads the file (strip, then its \\n
    escapes made literal); serving_chat._apply_jinja_template parses tool-call arguments, turns None
    content into "" and drops None fields; transformers renders in a sandbox with trim_blocks and
    lstrip_blocks. enable_thinking is the harness's, preserve_thinking cell 15's default. jinja2 is
    pure Python, so the ARC-AGI-3-Agents venv's copy imports here when this interpreter has none."""
    saved = list(sys.path)
    sys.path += [str(p) for p in sorted((REPO / "ARC-AGI-3-Agents/.venv/lib").glob("python3*/site-packages"))]
    try:
        import jinja2.ext
        import jinja2.sandbox
    except ImportError:
        return None
    finally:
        sys.path[:] = saved

    def raise_exception(message):
        raise jinja2.TemplateError(message)

    env = jinja2.sandbox.ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                                       extensions=[jinja2.ext.loopcontrols])
    env.filters["tojson"] = lambda value: json.dumps(value, ensure_ascii=False)
    env.globals["raise_exception"] = raise_exception
    template = env.from_string(TEMPLATE.read_text().strip("\n").replace("\\n", "\n"))

    def render(messages: list, tools: list) -> str:
        dicts = []
        for message in json.loads(json.dumps(messages)):
            if message.get("content") is None:
                message["content"] = ""
            for call in message.get("tool_calls") or []:
                call["function"]["arguments"] = json.loads(call["function"]["arguments"])
            dicts.append({k: v for k, v in message.items() if v is not None})
        return template.render(messages=dicts, tools=tools, add_generation_prompt=True, enable_thinking=True,
                               preserve_thinking=True)

    return render


def two_requests(tool_agent, tmp: Path, reasoning: str) -> tuple[list, list, list]:
    """One real analyze() turn whose first reply is this reasoning plus a python call; returns the
    messages of that request and of the one that follows the tool result, and the tools both carry."""
    sent, tools = [], []

    def fake_server(messages, **kwargs):
        sent.append(json.loads(json.dumps(messages)))
        tools.append(kwargs.get("tools"))
        if len(sent) > 1:
            raise Captured
        message = {"role": "assistant", "content": "\n\n", "reasoning_content": reasoning,
                   "tool_calls": [{"id": "call_0", "type": "function",
                                   "function": {"name": "python", "arguments": json.dumps({"code": CODE})}}]}
        return tool_agent._ChatCompletionResult(
            message=message, finish_reason="tool_calls",
            usage={"prompt_tokens": 6000, "completion_tokens": 900, "total_tokens": 6900})

    agent = tool_agent.ToolAgent(model="stub")
    agent._chat_completion = fake_server
    state = Path(tempfile.mkdtemp(dir=tmp)) / "game_p0_tool_runtime_state.json"
    grid = [[0] * 5 for _ in range(5)]
    grid[2][2] = 3
    state.write_text(json.dumps({"current_frame": {"grid": grid, "step": 0, "level": 1}, "history": []}))
    try:
        agent.analyze(state, 0, valid_actions=["ACTION1"], step_env=lambda payload: {})
        raise AssertionError("analyze() ended before sending the next request")
    except Captured:
        pass
    (assistant,) = [m for m in sent[1] if m.get("role") == "assistant"]
    assert sent[1][-1].get("role") == "tool" and assistant["tool_calls"][0]["function"]["name"] == "python", sent[1][-2:]
    assert tools[0] and tools[0] == tools[1], tools
    return sent[0], sent[1], tools[0]


def check_round_trip(bundle: Path, tmp: Path, render, *, patched: bool) -> None:
    tool_agent, _ = load(bundle)
    for _, _, paragraphs in REPLIES:
        server = server_form(paragraphs)
        first, second, tools = two_requests(tool_agent, tmp, server)
        (message,) = [m for m in second if m.get("role") == "assistant"]
        reasoning = message["reasoning_content"]
        assert "reasoning" not in message  # the fork sends history reasoning under reasoning_content only
        verbatim = reasoning == server.strip()
        turn = history_turn(message)
        if render is None:
            served = turn == PROMPT + generated(paragraphs) + "\n"
        else:  # the cache can serve the reply only if the next prompt extends this prompt plus the reply
            p1, p2 = render(first, tools), render(second, tools)
            at = len(p1) - len(PROMPT)
            assert p1.endswith(PROMPT) and p2.startswith(p1[:at]), "the next request changed text before the reply"
            assert p2[at:at + len(turn)] == turn, "history_turn() no longer renders what the template renders"
            served = p2.startswith(p1 + generated(paragraphs))
        if patched:
            assert verbatim and served, (reasoning, server)
            tagged = tool_agent._extract_reasoning_text({"reasoning": "<think>\n" + server + "</think>\n"})
            assert tagged == server.strip(), tagged
        else:
            assert reasoning == "\n".join(paragraphs), reasoning  # what the transcript recorded
            assert not verbatim and not served, "the unpatched harness must fail the round trip"


def check_recorded() -> None:
    """The recorded text is the transcript's, when the transcripts are on disk."""
    if not TRANSCRIPTS.is_dir():
        print(f"SKIP recorded-text compare ({TRANSCRIPTS} not present)")
        return
    for name, index, paragraphs in REPLIES:
        blocks, current = [], None
        for line in (TRANSCRIPTS / name).read_text(errors="replace").split("\n"):
            if SECTION.match(line) or line.startswith("--- analysis_step="):
                if current is not None:
                    blocks.append("\n".join(current).strip())
                current = [] if line == "[THINKING]" else None
            elif current is not None:
                current.append(line)
        assert blocks[index] == "\n".join(paragraphs), (name, index, blocks[index][:200])
    print("ok   recorded replies match their franzen-p448s14-v1 transcripts")


def git_blob(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def def_source(text: str, name: str, cls: str | None = None) -> str:
    """One def or class from a module's text, dedented."""
    tree = ast.parse(text)
    scope = tree if cls is None else next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == cls)
    node = next(n for n in ast.walk(scope) if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)
    start = min([node.lineno] + [d.lineno for d in node.decorator_list])
    lines = text.splitlines()[start - 1:node.end_lineno]
    indent = len(lines[0]) - len(lines[0].lstrip())
    return "\n".join(line[indent:] for line in lines)


def check_harness(tmp: Path) -> None:
    text = (OVERLAYS / "replycache.py").read_text()
    assert "\u2014" not in text and "\u2013" not in text, "dash in overlay"
    os.environ.update(FORK_ENV)  # read at import: set before any harness loads
    check_recorded()
    render = chat_renderer()
    if render is None:
        print("SKIP template render (no jinja2 here or in ARC-AGI-3-Agents/.venv): the hand port history_turn() "
              "stands in for it")

    pristine = tmp / "pristine"
    shutil.copytree(TREE, pristine)
    check_round_trip(pristine, tmp, render, patched=False)
    print("ok   unpatched: the re-sent reasoning is the transcript text, blank lines gone; the next prompt does not "
          "extend the previous prompt plus the generated text")

    bundle = tmp / "bundle"
    shutil.copytree(TREE, bundle)
    apply("replycache", bundle)
    assert changed(bundle) == [AGENT], changed(bundle)
    py_compile.compile(str(bundle / AGENT), doraise=True)
    try:
        apply("replycache", bundle)
        raise AssertionError("second application should refuse")
    except RuntimeError as exc:
        assert "already applied" in str(exc), exc
    check_round_trip(bundle, tmp, render, patched=True)
    print("ok   patched: the re-sent reasoning is the server's text minus the think tags, and the next prompt "
          "extends the previous prompt plus the generated text" + ("" if render is None else " (template render)"))

    drift = tmp / "drift"
    shutil.copytree(TREE, drift)
    source = (drift / AGENT).read_text()
    func = def_source(source, "_extract_reasoning_text") + "\n"
    assert source.count(func) == 1, func
    for text, count in ((source.replace(func, func.replace(" -> str:", " ->  str:")), 0), (source + "\n\n" + func, 2)):
        (drift / AGENT).write_text(text)
        try:
            apply("replycache", drift)
            raise AssertionError(f"an anchor that matches {count} times must refuse")
        except RuntimeError as exc:
            assert f"anchor matches {count} times" in str(exc), exc
        assert (drift / AGENT).read_text() == text, "a refused overlay wrote tool_agent.py"
    shutil.rmtree(drift)
    print("ok   an anchor that matches 0 or 2 times raises and leaves tool_agent.py as it was")

    for order in (("wastefix", "replycache"), ("replycache", "wastefix")):
        combo = tmp / ("ship-" + "-".join(order))
        shutil.copytree(TREE, combo)
        for name in order:
            apply(name, combo)
        check_round_trip(combo, tmp, render, patched=True)
        print("ok   round trip holds with", " -> ".join(order))

    names = sorted(p.stem for p in OVERLAYS.glob("*.py") if not p.stem.endswith("_serve"))
    for order in overlay_orders.orders(names, "replycache"):
        combo = tmp / ("combo-%08x" % __import__("zlib").crc32("-".join(order).encode()))
        shutil.copytree(TREE, combo)
        for name in order:
            apply(name, combo)
        for rel in changed(combo):
            py_compile.compile(str(combo / rel), doraise=True)
        assert (combo / AGENT).read_text().count("# overlay replycache") == 1
        shutil.rmtree(combo)
        print("composed:", " -> ".join(order))


def wheel_sources() -> dict[str, str]:
    with zipfile.ZipFile(WHEEL) as wheel:
        return {rel: wheel.read(rel).decode() for rel in WHEEL_FACTS}


def check_facts(src: dict[str, str]) -> None:
    """What the serve docstring claims, read from this notebook and this wheel."""
    cell = builder.cell_text(json.loads((builder.BASE / "submission.ipynb").read_text())["cells"][builder.SERVE])
    for fact in NOTEBOOK_FACTS:
        assert cell.count(fact) == 1, fact
    assert "disable-overlap-schedule" not in cell and "extra_buffer_lazy" not in cell
    for rel, facts in WHEEL_FACTS.items():
        for fact in facts:
            assert src[rel].count(fact) == 1, (rel, fact[:80])
    processor = src["sglang/srt/managers/scheduler_components/batch_result_processor.py"]
    assert processor.count("self._mamba_lazy_spec_update(") == 1
    assert "\n    if lazy and not batch.spec_algorithm.is_none():\n" in def_source(processor, "_mamba_prefix_cache_update")
    assert "MambaRadixCache(" not in src["sglang/srt/mem_cache/registry.py"]
    assert git_blob(TEMPLATE) == TEMPLATE_BLOB, "chat template fixture changed"
    template = TEMPLATE.read_text()
    for line in TEMPLATE_LINES:
        assert template.count(line) == 1, line
    print("ok   facts: notebook runs extra_buffer with overlap on, so the lazy-spec skip never runs and every finish "
          "inserts; the donated depth is mamba_last_track_seqlen; the live cache is UnifiedRadixCache")


def finished_line(patched: str, src: dict[str, str], *, is_insert: bool, track: int | None) -> str:
    """Run the patched UnifiedRadixCache.cache_finished_req with the wheel's MambaComponent,
    RadixKey and InsertParams/InsertResult; fakes only for the tree walk, allocator and locks."""
    lines = []
    logger = logging.getLogger("replycache-test")
    logger.propagate, logger.handlers = False, []
    logger.setLevel(logging.INFO)
    handler = logging.Handler()
    handler.emit = lambda record: lines.append(record.getMessage())
    logger.addHandler(handler)
    ns = {"dataclasses": __import__("dataclasses"), "array": __import__("array").array, "logger": logger,
          "torch": SimpleNamespace(int64="int64")}
    prelude = "from __future__ import annotations\nfrom typing import Any, Iterator, Optional, Tuple, Union\n"
    for rel, name, cls in (("sglang/srt/mem_cache/radix_cache.py", "RadixKey", None),
                           ("sglang/srt/mem_cache/base_prefix_cache.py", "InsertParams", None),
                           ("sglang/srt/mem_cache/base_prefix_cache.py", "InsertResult", None),
                           ("sglang/srt/mem_cache/unified_cache/components/mamba_component.py",
                            "prepare_for_caching_req", "MambaComponent"),
                           (URC, "cache_finished_req", "UnifiedRadixCache")):
        text = patched if rel == URC else src[rel]
        exec(compile(prelude + def_source(text, name, cls), rel, "exec"), ns)

    class Rows(list):  # a req_to_token row and its slices; .to() is a copy
        def __getitem__(self, key):
            return Rows(list.__getitem__(self, key[1] if isinstance(key, tuple) else key))

        def to(self, **kwargs):
            return Rows(self)

    class Slot(int):  # a mamba pool slot id
        def unsqueeze(self, dim):
            return self

        def clone(self):
            return self

    in_len, out_len, protected = 6295, 630, 6272
    req = SimpleNamespace(rid="r1", origin_input_ids=list(range(in_len)), output_ids=list(range(out_len)),
                          req_pool_idx=0, cache_protected_len=protected, priority=0, extra_key=None, cache_salt=None,
                          mamba_last_track_seqlen=track, mamba_last_track_idx=1,
                          mamba_ping_pong_track_buffer=[Slot(11), Slot(12)], swa_prefix_lock_released=False)
    component = SimpleNamespace(int8_ckpt_pool=None, cache=SimpleNamespace(
        enable_mamba_extra_buffer=True,
        req_to_token_pool=SimpleNamespace(get_mamba_ping_pong_keep_idx=lambda r: r.mamba_last_track_idx)))
    component.prepare_for_caching_req = types.MethodType(ns["prepare_for_caching_req"], component)
    component.cleanup_after_caching_req = lambda *a, **k: None
    inserted = {}

    def insert(params):  # an empty key keeps no state, as UnifiedTreeCore.begin_insert answers it
        inserted.update(depth=len(params.key), mamba=params.mamba_value)
        return ns["InsertResult"](prefix_len=min(protected, len(params.key)), last_device_node="node",
                                  mamba_exist=len(params.key) == 0)

    cache = SimpleNamespace(
        session=SimpleNamespace(try_cache_finished_req=lambda *a, **k: False), disable=False,
        req_to_token_pool=SimpleNamespace(req_to_token=Rows(range(100_000, 100_000 + in_len + out_len))),
        _components_tuple=(component,), tree_core=SimpleNamespace(is_eagle=False), page_size=64, insert=insert,
        token_to_kv_pool_allocator=SimpleNamespace(free_segments=lambda segments: None,
                                                   free_segment=lambda indices, start_pos: None),
        _dec_req_lock=lambda r, skip_swa: None, enable_session_radix_cache=False)
    ns["cache_finished_req"](cache, req, is_insert=is_insert, kv_len_to_handle=in_len + out_len - 1)
    (line,) = lines
    assert inserted == ({"depth": track or 0, "mamba": 12} if is_insert else {}), inserted  # the keep slot
    return line


def site_bytes(site: Path) -> dict[str, bytes]:
    return {str(p.relative_to(site)): p.read_bytes() for p in sorted(site.rglob("*.py"))}


def check_nvfp4kv(tmp: Path, ov: dict, patched: str) -> None:
    """nvfp4kv_serve.py (B3) vendors 0002, whose ping-pong hunk edits cache_unfinished_req in the same
    unified_radix_cache.py, and the PLAN stacks B3 on L1: either order leaves the same bytes, which
    compile and keep cache_finished_req as this patch alone leaves it, and a rerun of both is a no-op."""
    nv: dict = {}
    exec(compile((OVERLAYS / "nvfp4kv_serve.py").read_text(), "franzen_overlays/nvfp4kv_serve.py", "exec"), nv)
    patches = {name: (OVERLAYS / "nvfp4kv" / name).read_text() for name in nv["PATCH_SHA256"]}
    # every file a hunk edits, and for the file 0001 adds (qsa/nvfp4_kv.py) its package's __init__
    rels = {rel if old else str(Path(rel).parent / "__init__.py")
            for text in patches.values() for rel, old, _ in nv["hunks"](text)}
    assert URC in rels, "0002 no longer edits unified_radix_cache.py"
    with zipfile.ZipFile(WHEEL) as wheel:
        files = {rel: wheel.read(rel) for rel in rels}
    run = {"nvfp4kv": lambda venv: nv["patch_sglang"](venv, patches), "replycache": ov["patch_sglang"]}
    states = []
    for order in (("nvfp4kv", "replycache"), ("replycache", "nvfp4kv")):
        venv = tmp / ("venv-" + "-".join(order))
        site = venv / "lib/python3.12/site-packages"
        for rel, data in files.items():
            (site / rel).parent.mkdir(parents=True, exist_ok=True)
            (site / rel).write_bytes(data)
        for name in order:
            run[name](venv)
        state = site_bytes(site)
        for name in order:
            run[name](venv)
        assert site_bytes(site) == state, ("a rerun moved something", order)
        text = state[URC].decode()
        compile(text, URC, "exec")
        assert text.count("arm replycache") == 1 and text.count("Local fix (ping-pong") == 1, order
        assert (def_source(text, "cache_finished_req", "UnifiedRadixCache")
                == def_source(patched, "cache_finished_req", "UnifiedRadixCache")), order
        states.append(state)
    assert states[0] == states[1], "the two orders leave different files"
    print("ok   composes with nvfp4kv_serve.py, whose 0002 edits cache_unfinished_req in the same file: both orders "
          "leave the same bytes, which compile, and a rerun of both is a no-op")


def check_serve(tmp: Path) -> None:
    text = SERVE.read_text()
    assert "\u2014" not in text and "\u2013" not in text, "dash in serving patch"
    src = wheel_sources()
    check_facts(src)
    ov: dict = {}
    exec(compile(text, "franzen_overlays/replycache_serve.py", "exec"), ov)
    site = tmp / "venv/lib/python3.12/site-packages"
    path = site / URC
    path.parent.mkdir(parents=True)
    original = src[URC]
    path.write_text(original)
    model = tmp / "model"  # MODEL_DIR, no chat template in it yet
    model.mkdir()

    for env in ({}, {"ARC3_REPLYCACHE_DIAG": "0"}, {"ARC3_REPLYCACHE_DIAG": "true"}):
        ov["apply"](tmp / "venv", env, model)
        assert path.read_text() == original, env
    try:
        ov["apply"](tmp / "venv", {"ARC3_REPLYCACHE_DIAG": "1", "ARC3_REPLYCACHE_DONATE": "1"}, model)
        raise AssertionError("ARC3_REPLYCACHE_DONATE=1 must raise: there is no donate fix")
    except RuntimeError as exc:
        assert "no fix behind it" in str(exc), exc
    assert path.read_text() == original
    ov["apply"](tmp / "venv", {"ARC3_REPLYCACHE_DIAG": "1"}, model)
    patched = path.read_text()
    compile(patched, URC, "exec")
    assert patched.count("arm replycache") == 1 and patched.replace(ov["LOG_NEW"], ov["LOG_OLD"]) == original
    ov["apply"](tmp / "venv", {"ARC3_REPLYCACHE_DIAG": "1"}, model)
    assert path.read_text() == patched, "a rerun of the cell must be a no-op"
    print("ok   serve patch: off unless ARC3_REPLYCACHE_DIAG=1; one edit to unified_radix_cache.py, nothing else "
          "moves; rerun is a no-op; ARC3_REPLYCACHE_DONATE=1 raises")

    assert ov["TEMPLATE_BLOB"] == TEMPLATE_BLOB
    notes = [ov["template_note"](model)]
    shutil.copy(TEMPLATE, model / "chat_template.jinja")
    notes.append(ov["template_note"](model))
    (model / "chat_template.jinja").write_bytes(TEMPLATE.read_bytes() + b"\n")
    notes.append(ov["template_note"](model))
    assert ("no chat_template.jinja" in notes[0] and f"{TEMPLATE_BLOB}, the one" in notes[1]
            and f"NOT {TEMPLATE_BLOB}" in notes[2]), notes
    print("ok   template note: names the fixture's template, flags any other and a missing one")

    check_nvfp4kv(tmp, ov, patched)

    assert finished_line(patched, src, is_insert=True, track=6912) == (
        "replycache rid=r1 in=6295 out=630 kv=6924 track=6912 cache_len=6912 protected=6272 donated=True")
    assert finished_line(patched, src, is_insert=True, track=None) == (
        "replycache rid=r1 in=6295 out=630 kv=6924 track=None cache_len=0 protected=6272 donated=False")
    assert finished_line(patched, src, is_insert=False, track=6912) == (
        "replycache rid=r1 in=6295 out=630 kv=6924 track=6912 cache_len=0 protected=6272 donated=None")
    print("ok   patched cache_finished_req runs with the wheel's MambaComponent and RadixKey: a reply donated to its "
          "last tracked boundary logs cache_len > protected; no boundary logs track=None cache_len=0")

    path.write_text(original.replace("            req.last_node = result.last_device_node\n",
                                     "            req.last_node = result.last_device_node  # drift\n", 1))
    try:
        ov["apply"](tmp / "venv", {"ARC3_REPLYCACHE_DIAG": "1"}, model)
        raise AssertionError("a drifted unified_radix_cache.py must raise")
    except RuntimeError as exc:
        assert "does not match the Pennyroyal v2.5.3 source" in str(exc), exc
    print("ok   drifted source raises before anything is written")


def main() -> None:
    assert TREE.is_dir(), f"pristine Franzen tree missing: {TREE} (set FRANZEN_TREE)"
    if not WHEEL.is_file():
        raise SystemExit(f"FAIL: SGLang wheel not found at {WHEEL}; set REPLYCACHE_SGLANG_WHEEL")
    with tempfile.TemporaryDirectory(prefix="franzen-replycache-") as name:
        tmp = Path(name)
        check_harness(tmp)
        check_serve(tmp)
    print("PASS")


if __name__ == "__main__":
    main()
