"""Offline checks for arm thinkbudget (B5), solution/kaggle/franzen_overlays/thinkbudget_serve.py.

    python3 solution/sim/test_franzen_thinkbudget.py

Reads the real SGLang sources from Franzen's Pennyroyal v2.5.3 wheel (franzen_tree.pennyroyal_wheel(),
a sha256-checked zip that is read, never installed; THINKBUDGET_SGLANG_WHEEL overrides the path) and
checks against those files:
  1. --enable-strict-thinking and SGLANG_MAX_THINK_TOKENS exist and parse as the arm uses them, and
     the default grammar backend (xgrammar) has the token filter the budget needs.
  2. Under NEXTN the budget is the grammar mask inside EAGLE verify, never a custom logit processor:
     the wheel's own ReasonerGrammarObject, traverse_tree and update_reasoning_tokens, driven over
     4-token draft chains, end thinking at exactly the budget and report budget + 1 reasoning tokens.
  3. Why the exclusion patch: unpatched, Qwen3Detector's list blocks four tokens on every thinking
     row and switches the filter on by itself. Patched, the filter acts only at the budget and only
     when the env is set, so the 64-token self-test also proves the env reached the server.
  4. The cell-15 code the builder inserts: it patches the real reasoning_parser.py once (rerun is a
     no-op, a drifted source raises before the flag or env is set), sets the env, adds the flag
     once, runs after the argv is assembled and before the launch, and its self-test is the cell's
     last statement. The arm's edits land exactly once beside every other cell-15 patch, in the
     orders overlay_orders gives.
  5. The self-test verdict and its bench/scored behavior, and thinking_counters.py (self-check, plus
     the published 67 cap cuts with no call when the two p448s14 control runs are present).
No torch, no GPU, no network once the wheel is cached.
"""
from __future__ import annotations

import ast
import copy
import functools
import hashlib
import inspect
import io
import json
import logging
import os
import re
import sys
import tempfile
import typing
import urllib.error
import warnings
import zipfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
for sub in ("solution/kaggle", "solution/sim", "solution/analysis"):
    sys.path.insert(0, str(REPO / sub))
import build_franzen_arms as builder  # noqa: E402
import franzen_tree  # noqa: E402
import overlay_orders  # noqa: E402
import thinking_counters  # noqa: E402

OVERLAY = REPO / "solution/kaggle/franzen_overlays/thinkbudget_serve.py"
CONTROL_RUNS = [REPO / "solution/kaggle/generated/kernel-outputs" / n for n in ("franzen-p448s14-v1", "franzen-p448s14rep-v1")]
ARM, BUDGET = "p448s14dpwftb", 10240
FLAG, ENV = "--enable-strict-thinking", "SGLANG_MAX_THINK_TOKENS"
PARSER = "sglang/srt/parser/reasoning_parser.py"

# The builder snippet handed to the integrator, verbatim. Until it is pasted into
# build_franzen_arms.py it runs here in a copy of the builder's namespace; afterwards the test
# checks that the integrated helper and arm produce exactly these edits.
SNIPPET = '''
LAUNCH_ANCHOR = "# ---- launch detached and wait for health ----\\n"  # cell 15, after the argv is assembled


def thinkbudget(tokens: int) -> list[tuple[int, str, str]]:
    """B5: SGLang's strict thinking budget. --enable-strict-thinking plus SGLANG_MAX_THINK_TOKENS forces
    </think> at the budget through the grammar mask that NEXTN verify applies (a custom logit processor
    never runs there), and Qwen3Detector's think exclusions are emptied so only the budget changes.
    After the READY wait a bench run sends one request with a 64-token budget, which must stop thinking
    by 80 reasoning tokens and still parse, or the bench stops. A server still loading at the deadline
    skips it rather than holding the games' clocks; the scored run skips it.
    franzen_overlays/thinkbudget_serve.py; counters: solution/analysis/thinking_counters.py."""
    code = (OVERLAYS / "thinkbudget_serve.py").read_text()
    return [(SERVE, LAUNCH_ANCHOR, f"# arm thinkbudget: strict thinking budget of {tokens} tokens, think exclusions emptied\\n"
             + "_thinkbudget = {}\\nexec(compile(" + repr(code) + ", 'franzen_overlays/thinkbudget_serve.py', 'exec'), _thinkbudget)\\n"
             + f"args = _thinkbudget['apply'](VENV, env, args, {tokens})\\n" + LAUNCH_ANCHOR),
            (SERVE, SERVE_END, SERVE_END + "\\n_thinkbudget['check'](url, proc, CFG['SERVED_NAME'], strict=not TRUE_SUBMISSION)"
             "  # arm thinkbudget: 64-token self-test on a bench run")]
'''
ARMS_ENTRY = '''
    "p448s14dpwftb": ("B5: p448s14dp plus transcript-waste fixes plus a strict thinking budget of 10,240 tokens "
                      "(--enable-strict-thinking, think exclusions emptied); gate on thinking_counters.py",
                      lambda: STARTUP + p448() + streams(14) + mamba(70) + dprime() + overlay("wastefix") + thinkbudget(10240)),
'''

# Stand-in token ids for the five markup strings, one id each as in Qwen3.8's tokenizer.
THOUGHT, ANSWER, THINK_END, TOOL, TOOL_END, IM_END, EOT, V = 5, 6, 9, 10, 11, 12, 13, 32
IDS = {"</think>": [THINK_END], "<tool_call>": [TOOL], "</tool_call>": [TOOL_END], "<|im_end|>": [IM_END],
       "<|endoftext|>": [EOT]}


def load_sources() -> dict[str, str]:
    wheel = Path(os.environ.get("THINKBUDGET_SGLANG_WHEEL") or franzen_tree.pennyroyal_wheel())
    assert hashlib.sha256(wheel.read_bytes()).hexdigest() == franzen_tree.WHEEL_SHA256, "not Franzen's pennyroyal-v253 wheel"
    names = {"parser": "parser/reasoning_parser.py", "reasoner": "constrained/reasoner_grammar_backend.py",
             "base": "constrained/base_grammar_backend.py", "manager": "constrained/grammar_manager.py",
             "xgr": "constrained/xgrammar_backend.py", "filter": "constrained/torch_ops/token_filter_torch_ops.py",
             "environ": "environ.py", "args": "server_args.py", "argutils": "arg_groups/arg_utils.py",
             "hook": "arg_groups/speculative_hook.py", "matcher": "utils/token_sequence_matcher.py",
             "spec": "speculative/spec_utils.py", "eagle": "speculative/eagle_utils.py",
             "worker": "speculative/eagle_worker_common.py", "specinfo": "speculative/spec_info.py",
             "protocol": "entrypoints/openai/protocol.py", "batch": "managers/schedule_batch.py"}
    with zipfile.ZipFile(wheel) as z:
        return {k: z.read("sglang/srt/" + rel).decode() for k, rel in names.items()}


def node(text: str, name: str, cls: str | None = None):
    scope = ast.parse(text) if cls is None else node(text, cls)
    return next(n for n in ast.walk(scope) if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name)


def source(text: str, name: str, cls: str | None = None) -> str:
    """Dedented source of one class or def, decorators dropped."""
    n = node(text, name, cls)
    lines = text.splitlines()[n.lineno - 1:n.end_lineno]
    indent = len(lines[0]) - len(lines[0].lstrip())
    return "\n".join(line[indent:] for line in lines)


def wheel_namespace(src: dict, parser_text: str) -> dict:
    """The wheel's own grammar, matcher, env and detector classes, exec'd without torch."""
    ns = {k: getattr(typing, k) for k in ("Any", "Dict", "List", "NamedTuple", "Optional", "Tuple", "Type", "Union")}
    ns.update(copy=copy, re=re, inspect=inspect, json=json, logging=logging, logger=logging.getLogger("thinkbudget"),
              torch=SimpleNamespace(Tensor=object), ThreadPoolExecutor=ThreadPoolExecutor,
              PreTrainedTokenizer=object, PreTrainedTokenizerFast=object, ReasoningParser=object)
    env_ns = {"os": os, "warnings": warnings, "contextmanager": contextmanager, "Any": typing.Any}
    for name in ("EnvField", "EnvInt"):
        exec(source(src["environ"], name), env_ns)
    think_env = env_ns["EnvInt"](-1)
    think_env.__set_name__(None, ENV)
    ns["envs"] = SimpleNamespace(SGLANG_MAX_THINK_TOKENS=think_env)
    for key, names in (("matcher", ["TokenSequenceMatcher"]),
                       ("base", ["GrammarRow", "BaseGrammarObject", "InvalidGrammarObject", "BaseGrammarBackend"]),
                       ("reasoner", ["ReasonerGrammarObject", "ReasonerGrammarBackend"])):
        for name in names:
            exec(source(src[key], name), ns)
    for name in ("_ReasoningCodeQuotes", "BaseReasoningFormatDetector", "Qwen3Detector"):
        exec(source(parser_text, name), ns)
    manager = SimpleNamespace()  # the two GrammarManager methods, bound to one object as on the scheduler
    for name in ("_get_request_thinking_budget", "_apply_request_reasoning_budget"):
        exec(source(src["manager"], name, "GrammarManager"), ns)
        setattr(manager, name, functools.partial(ns[name], manager))
    ns["manager"] = manager
    for key, name in (("spec", "traverse_tree"), ("batch", "update_reasoning_tokens")):
        exec(source(src[key], name), ns)
    return ns


class Tok:
    def encode(self, text, add_special_tokens=False):
        return list(IDS[text])


class Vec(list):
    @property
    def shape(self):
        return (len(self),)


def allowed(row: list, t: int) -> bool:
    return bool(row[t // 32] & (1 << (t % 32)))


def filter_backend(ns: dict):
    """xgrammar's role: all-allowed int32 bitmask rows, and set_token_filter with the exact reset and
    bit semantics of the wheel's set_token_filter_torch (checked against its source in main)."""
    class FilterBackend(ns["BaseGrammarBackend"]):
        calls = 0

        @property
        def is_support_token_filter(self):
            return True

        @staticmethod
        def allocate_vocab_mask(vocab_size, batch_size, device):
            return [[-1] * ((vocab_size + 31) // 32) for _ in range(batch_size)]

        @staticmethod
        def move_vocab_mask(vocab_mask, device):
            return vocab_mask

        @staticmethod
        def apply_vocab_mask(logits, vocab_mask):
            raise AssertionError("the simulation picks tokens from the mask rows itself")

        def set_token_filter(self, vocab_mask, token_ids, batch_idx, is_allowed=True, reset_vocab_mask=True):
            FilterBackend.calls += 1
            row = vocab_mask[batch_idx]
            if reset_vocab_mask:
                row[:] = [0 if is_allowed else -1] * len(row)
            for t in token_ids:
                row[t // 32] = row[t // 32] | (1 << t % 32) if is_allowed else row[t // 32] & ~(1 << t % 32)
    return FilterBackend


def grammar(ns: dict, env: dict, request_budget=None):
    """One thinking request's grammar, built the way the scheduler builds it under strict thinking."""
    backend = filter_backend(ns)()
    with mock.patch.dict(os.environ, env, clear=False):
        if ENV not in env:
            os.environ.pop(ENV, None)
        reasoner = ns["ReasonerGrammarBackend"](backend, SimpleNamespace(detector=ns["Qwen3Detector"]()), Tok(),
                                                enable_strict_thinking=True)
    g = reasoner.init_strict_reasoning_grammar(True)
    params = {"thinking_budget": request_budget, "__req__": None} if request_budget is not None else None
    ns["manager"]._apply_request_reasoning_budget(SimpleNamespace(sampling_params=SimpleNamespace(custom_params=params),
                                                                  grammar=g))
    return reasoner, g, backend


def decode(ns: dict, g, chain: int, cap: int) -> tuple[list[int], int]:
    """Drive a model that always wants to keep thinking through the NEXTN path: the first token from
    the extend sampler's mask, then verify steps over draft chains of `chain` tokens. Each step fills
    the masks with the wheel's traverse_tree, the target takes its preferred token where its row
    allows it and the one allowed token where not, the accepted run is drafts up to the first
    mismatch plus the bonus, and the grammar advances over it (the scheduler's grammar barrier)."""
    req = SimpleNamespace(_is_reasoning_over=False, _think_end_matcher=None, _think_end_match_len=0, reasoning_tokens=0)

    def pick(row):
        if allowed(row, THOUGHT):
            return THOUGHT
        only = [t for t in range(V) if allowed(row, t)]
        assert len(only) == 1, only
        return only[0]

    mask = g.allocate_vocab_mask(V, 1, "cpu")
    g.fill_vocab_mask(mask, 0)
    out = [pick(mask[0])]
    g.accept_token(out[0])
    ns["update_reasoning_tokens"](req, out[0], [THINK_END])
    while out[-1] != THINK_END and len(out) < cap:
        draft = Vec([out[-1]] + [THOUGHT] * (chain - 1))
        mask = g.allocate_vocab_mask(V, chain, "cpu")
        ns["traverse_tree"](Vec(list(range(1, chain)) + [-1]), Vec([-1] * chain), draft, g, mask, vocab_size=V)
        picks = [pick(row) for row in mask]
        k = next((i for i in range(chain - 1) if picks[i] != draft[i + 1]), chain - 1)
        run = picks[:k + 1]
        for t in run:
            g.accept_token(t)
        ns["update_reasoning_tokens"](req, run, [THINK_END])
        out += run
    return out, req.reasoning_tokens


def base_nb() -> dict:
    return json.loads((builder.BASE / "submission.ipynb").read_text())


def helpers() -> tuple:
    """(thinkbudget, arm edits): the integrated builder's if present, else the snippet's."""
    ns = dict(vars(builder))
    exec(SNIPPET, ns)
    tb = ns["thinkbudget"]
    arm = (builder.STARTUP + builder.p448() + builder.streams(14) + builder.mamba(70) + builder.dprime()
           + builder.overlay("wastefix") + tb(BUDGET))
    if hasattr(builder, "thinkbudget"):
        assert builder.thinkbudget(BUDGET) == tb(BUDGET), "integrated thinkbudget() differs from the tested snippet"
        tb = builder.thinkbudget
    if ARM in builder.ARMS:
        assert builder.ARMS[ARM][1]() == arm, f"integrated ARMS[{ARM!r}] differs from the tested entry"
    return tb, arm


def cell(edits: list) -> str:
    nb = base_nb()
    builder.apply_edits(nb, builder.REPLICA + edits)  # raises unless every anchor matches exactly once
    for index in {i for i, _, _ in builder.REPLICA + edits}:
        text = builder.cell_text(nb["cells"][index])
        compile("\n".join("pass" if l.lstrip().startswith(("!", "%")) else l for l in text.split("\n")),
                f"cell{index}", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    return builder.cell_text(nb["cells"][builder.SERVE])


def make_venv(root: Path, parser_text: str) -> Path:
    path = root / "venv/lib/python3.12/site-packages" / PARSER
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(parser_text)
    return root / "venv"


ALIVE, EXITED = SimpleNamespace(poll=lambda: None, returncode=None), SimpleNamespace(poll=lambda: 1, returncode=1)


class BodyLost(io.BytesIO):
    def read(self, *args):
        raise ConnectionResetError(104, "Connection reset by peer")


def run_check(ov: dict, reply=None, error=None, strict=True, health=True, proc=ALIVE) -> tuple[str, dict]:
    """Call check() with urlopen mocked: /health answers 200 when `health` and refuses otherwise, the
    chat request returns `reply` (bytes as is) or raises `error`. Returns (printed text, what was sent)."""
    sent = {"health": 0}

    class Resp(io.BytesIO):
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def urlopen(request, timeout):
        if isinstance(request, str):
            assert (request, timeout) == ("http://127.0.0.1:8001/health", 5), (request, timeout)
            sent["health"] += 1
            if not health:
                raise urllib.error.URLError(ConnectionRefusedError(111, "Connection refused"))
            return Resp(b"")
        sent.update(url=request.full_url, payload=json.loads(request.data), timeout=timeout)
        if error is not None:
            raise error
        return Resp(reply if isinstance(reply, bytes) else json.dumps(reply).encode())

    out = io.StringIO()
    with mock.patch("urllib.request.urlopen", urlopen), redirect_stdout(out):
        ov["check"]("http://127.0.0.1:8001/health", proc, "flashnext", strict=strict)
    return out.getvalue(), sent


def reply(n: int, *, reasoning="Trailing zeros come from factors of five.", content="", calls=None, finish="tool_calls"):
    calls = [{"type": "function", "function": {"name": "python", "arguments": json.dumps({"code": "print(4005)"})}}] \
        if calls is None else calls
    return {"choices": [{"finish_reason": finish, "message": {"role": "assistant", "reasoning_content": reasoning,
                                                              "content": content, "tool_calls": calls}}],
            "usage": {"reasoning_tokens": n, "completion_tokens": n + 40}}


def main() -> None:
    src = load_sources()
    print(f"ok   real SGLang sources from the Pennyroyal v2.5.3 wheel, sha256 {franzen_tree.WHEEL_SHA256[:12]}")

    # 1. The flag and the env exist and parse as the arm uses them.
    field = next(n for n in node(src["args"], "ServerArgs").body
                 if isinstance(n, ast.AnnAssign) and getattr(n.target, "id", "") == "enable_strict_thinking")
    parts = field.annotation.slice.elts
    assert ast.unparse(parts[0]) == "bool" and isinstance(parts[1], ast.Constant) and field.value.value is False
    cli_ns: dict = {}
    exec(source(src["argutils"], "_field_to_cli_name"), cli_ns)
    assert cli_ns["_field_to_cli_name"]("enable_strict_thinking") == FLAG
    assert "if isinstance(a, str):\n                return inner, Arg(help=a)" in source(src["argutils"], "_unwrap_annotated")
    adder = source(src["argutils"], "add_cli_args_from_dataclass")
    assert "if inner_type is bool:" in adder and 'kwargs = dict(action="store_true"' in adder
    assert 'grammar_backend="xgrammar"' in source(src["args"], "_handle_grammar_backend")
    assert "--grammar-backend" not in builder.cell_text(base_nb()["cells"][builder.SERVE])
    assert ast.unparse(node(src["xgr"], "is_support_token_filter", "XGrammarGrammarBackend").body[0]) == "return True"
    assert ("if get_serving().reasoning_parser and think_end_ids:" in src["base"]
            and "enable_strict_thinking=get_serving().enable_strict_thinking," in src["base"])
    assert "    SGLANG_MAX_THINK_TOKENS = EnvInt(-1)\n" in src["environ"]
    assert "self.max_think_tokens = envs.SGLANG_MAX_THINK_TOKENS.get()" in source(src["reasoner"], "__init__", "ReasonerGrammarBackend")
    pristine = src["parser"]
    ns = wheel_namespace(src, pristine)
    with mock.patch.dict(os.environ, {ENV: str(BUDGET)}):
        assert ns["envs"].SGLANG_MAX_THINK_TOKENS.get() == BUDGET
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop(ENV, None)
        assert ns["envs"].SGLANG_MAX_THINK_TOKENS.get() == -1
    print(f"ok   {FLAG} is a ServerArgs bool flag; {ENV} is EnvInt(-1), read once when the reasoner backend is "
          "built; the default grammar backend (xgrammar, the notebook sets none) has the token filter")

    # 2. Under NEXTN the budget is the grammar mask inside EAGLE verify.
    assert re.search(r'if speculative_algorithm == "NEXTN" or speculative_algorithm == "EAGLE":\n(?:.*\n){1,8}?'
                     r'\s+return "EAGLE"\n', src["hook"])
    assert "return self.is_eagle() or" in source(src["specinfo"], "supports_grammar_overlap")
    assert "SpeculativeAlgorithm.EAGLE," in source(src["specinfo"], "is_eagle")
    assert "return self.has_grammar and not self.spec_algorithm.supports_grammar_overlap()" in src["batch"]
    worker = src["worker"]
    assert worker.index("grammar_mask = build_grammar_vocab_mask(") < worker.index(
        "= eagle_sample(verify_input, batch, logits_output, grammar_mask)")
    sample = node(src["eagle"], "eagle_sample")
    assert "grammar_mask.apply(next_token_logits)" in ast.unparse(sample)
    assert "custom_logit" not in ast.unparse(sample), "eagle_sample now applies custom logit processors"
    assert "allocate_token_bitmask = req.grammar.allocate_vocab_mask(" in source(src["spec"], "generate_token_bitmask")
    assert "vocab_mask[batch_idx].fill_(-1 if (not is_allowed) else 0)" in src["filter"]
    assert "row[element_indices] = torch.bitwise_or(" in src["filter"] and "torch.bitwise_and(" in src["filter"]
    patched_parser = pristine.replace(ns_overlay()["EXCLUDED_OLD"], ns_overlay()["EXCLUDED_NEW"])
    pns = wheel_namespace(src, patched_parser)
    env_on = {ENV: str(BUDGET)}
    chain = int(re.search(r"\n    SPEC_STEPS=(\d+),", builder.cell_text(base_nb()["cells"][builder.SERVE]))[1]) + 1
    assert chain == 4, chain  # --speculative-num-draft-tokens is SPEC_STEPS + 1
    for draft_len in (1, chain):
        for budget in (61, 62, 63, 64):
            _, g, backend = grammar(pns, env_on, request_budget=budget)
            out, counted = decode(pns, g, draft_len, cap=400)
            assert out == [THOUGHT] * budget + [THINK_END] and counted == budget + 1, (draft_len, budget, len(out), counted)
            assert backend.calls == 1, backend.calls  # one filter call, on the row where the budget binds
    assert ns_overlay()["TEST_BUDGET"] + 1 <= ns_overlay()["TEST_MAX_REASONING"]
    _, g, _ = grammar(pns, env_on)
    out, counted = decode(pns, g, chain, cap=BUDGET + 10)
    assert len(out) == BUDGET + 1 and out[-1] == THINK_END and counted == BUDGET + 1
    print(f"ok   NEXTN resolves to EAGLE, whose verify applies the grammar mask (never a custom logit processor) "
          f"with overlap kept; over plain decode and {chain}-token draft chains the wheel's grammar ends thinking at "
          f"exactly the budget wherever it falls in the chain: 64 -> 65 reasoning tokens, env {BUDGET} -> {counted}")

    # 3. Why the exclusion patch is needed, and the interlock it creates.
    q = ns["Qwen3Detector"]()
    assert q.think_excluded_tokens == ["<tool_call>", "</tool_call>", "<|im_end|>", "<|endoftext|>"]
    assert pns["Qwen3Detector"]().think_excluded_tokens == []
    reasoner, g, backend = grammar(ns, {})
    assert reasoner.think_excluded_token_ids == [TOOL, TOOL_END, IM_END, EOT] and reasoner.enable_token_filter
    mask = g.allocate_vocab_mask(V, 1, "cpu")
    g.fill_vocab_mask(mask, 0)
    assert allowed(mask[0], THOUGHT) and not any(allowed(mask[0], t) for t in (TOOL, TOOL_END, IM_END, EOT))
    _, g, backend = grammar(ns, env_on, request_budget=64)
    decode(ns, g, 4, cap=400)
    blocked_calls = backend.calls
    assert blocked_calls >= 64, blocked_calls  # a filter call on every thinking row
    reasoner, g, backend = grammar(pns, env_on)
    assert reasoner.think_excluded_token_ids is None and reasoner.enable_token_filter
    mask = g.allocate_vocab_mask(V, 1, "cpu")
    g.fill_vocab_mask(mask, 0)
    assert mask == [[-1]] and backend.calls == 0
    reasoner, g, _ = grammar(pns, {}, request_budget=64)
    assert not reasoner.enable_token_filter
    out, _ = decode(pns, g, 4, cap=400)
    assert THINK_END not in out and len(out) >= 400  # without the env the request budget does nothing
    _, g, _ = grammar(ns, {}, request_budget=64)
    assert decode(ns, g, 4, cap=400)[0][-1] == THINK_END  # unpatched, the exclusions alone enable it
    _, g, _ = grammar(pns, env_on)
    for params, want in (({"thinking_budget": 33}, 33), ({"thinking_budget": "64"}, 33), (None, 33)):
        pns["manager"]._apply_request_reasoning_budget(
            SimpleNamespace(sampling_params=SimpleNamespace(custom_params=params), grammar=g))
        assert g.max_think_tokens == want, (params, g.max_think_tokens)  # only an int budget is taken
    assert src["manager"].count("self._apply_request_reasoning_budget(req)") == 3  # every grammar path applies it
    to_params = source(src["protocol"], "to_sampling_params", "ChatCompletionRequest")
    assert '"custom_params": self.custom_params,' in to_params
    print(f"ok   unpatched, Qwen3Detector's 4 exclusions are blocked on every thinking row ({blocked_calls} filter "
          "calls for one 64-token think) and switch the filter on alone; patched, rows stay unconstrained until "
          "the budget, and a request budget binds only with the env set; the chat endpoint passes custom_params")

    # 4. The cell-15 code the builder inserts.
    tb, arm = helpers()
    serve = cell(arm)
    start = serve.index("# arm thinkbudget: strict thinking budget of 10240 tokens")
    line = f"args = _thinkbudget['apply'](VENV, env, args, {BUDGET})"
    end = serve.index(line) + len(line)
    assert serve.index('args = [SGLANG, "serve"') < start < end < serve.index("precache_model_thread.start()") \
        < serve.index("subprocess.Popen(args")
    assert '"--reasoning-parser", "qwen3"' in serve and serve.count("_thinkbudget[") == 2
    block = serve[start:end]
    literal = next(n.value for n in ast.walk(ast.parse(block)) if isinstance(n, ast.Constant)
                   and isinstance(n.value, str) and "def apply" in n.value)
    assert literal == OVERLAY.read_text(), "cell carries a stale copy of the overlay"
    body = ast.parse("\n".join("pass" if l.lstrip().startswith(("!", "%")) else l for l in serve.split("\n"))).body
    assert isinstance(body[-2], ast.While) and "NOTEBOOK_START_TIME" in ast.unparse(body[-2].test)
    assert ast.unparse(body[-1]) == "_thinkbudget['check'](url, proc, CFG['SERVED_NAME'], strict=not TRUE_SUBMISSION)"
    assert "proc = subprocess.Popen(args" in serve and 'url, t0 = f"http://127.0.0.1:{SERVED_MODEL_PORT}/health"' in serve
    argv = ["sglang", "serve", "--reasoning-parser", "qwen3", "--tool-call-parser", "qwen3_coder"]
    with tempfile.TemporaryDirectory() as tmp:
        venv = make_venv(Path(tmp), pristine)
        target = venv / "lib/python3.12/site-packages" / PARSER
        env, cellns = {"PATH": "/usr/bin"}, {"VENV": str(venv), "args": list(argv)}
        cellns["env"] = env
        with redirect_stdout(io.StringIO()):
            exec(compile(block, "cell15-thinkbudget", "exec"), cellns)
        patched = target.read_text()
        assert env[ENV] == str(BUDGET) and cellns["args"] == argv + [FLAG]
        assert patched == patched_parser and patched.count("arm thinkbudget") == 1
        compile(patched, PARSER, "exec")
        assert patched.replace(cellns["_thinkbudget"]["EXCLUDED_NEW"], cellns["_thinkbudget"]["EXCLUDED_OLD"]) == pristine
        for args in (list(argv), argv + [FLAG]):  # a rerun of the cell, with a fresh or an already-flagged argv
            cellns["args"] = args
            with redirect_stdout(io.StringIO()):
                exec(compile(block, "cell15-thinkbudget", "exec"), cellns)
            assert target.read_text() == patched and cellns["args"].count(FLAG) == 1
        old = cellns["_thinkbudget"]["EXCLUDED_OLD"]
        drifted = pristine.replace(old, old.replace('"</tool_call>",\n', '"</tool_call>",\n            "<|eot_id|>",\n'))
        assert drifted != pristine
        for text, args, why in ((drifted, list(argv), "does not match the Pennyroyal v2.5.3 source"),
                                (pristine, ["sglang", "serve", "--reasoning-parser", "deepseek-r1"], "--reasoning-parser qwen3"),
                                (pristine, ["sglang", "serve"], "--reasoning-parser qwen3")):
            target.write_text(text)
            env, cellns = {}, {"VENV": str(venv), "args": list(args)}
            cellns["env"] = env
            try:
                exec(compile(block, "cell15-thinkbudget", "exec"), cellns)
                raise AssertionError(f"accepted a launch it must refuse ({why})")
            except RuntimeError as exc:
                assert why in str(exc), exc
            assert env == {} and cellns["args"] == args and target.read_text() == text
    other = {"thinkbudget": tb(BUDGET), "fastdec": builder.fastdec(), "p448": builder.p448(), "hicache": builder.hicache(32)}
    orders = overlay_orders.orders(other, "thinkbudget")
    for order in orders:
        text = cell(builder.STARTUP + [e for name in order for e in other[name]])
        assert text.count("_thinkbudget['apply']") == text.count("_thinkbudget['check']") == ("thinkbudget" in order)
    print(f"ok   {ARM}: cell 15 patches the real reasoning_parser.py once (rerun no-op, drifted source and a non-qwen3 "
          f"parser raise before the flag or env is set), sets {ENV}={BUDGET}, adds {FLAG} once after the argv is built "
          f"and before the launch; the self-test is the cell's last statement; edits land once in {len(orders)} "
          "orders beside fastdec, p448 and the host tier")

    # 5. The self-test verdict, its bench/scored behavior, and the run counters.
    ov = ns_overlay()
    good = reply(65)
    assert ov["verdict"](good)[0] is None
    assert ov["verdict"](reply(65, calls=[], content="n = 4005"))[0] is None
    assert "not exercised" in ov["verdict"](reply(40))[1]
    cut = ov["verdict"](reply(65, content="<tool_call>\n<function=python>", calls=[], finish="length"))
    assert cut[0] is None and cut[1].endswith("(answer cut at max_tokens, parse not exercised)"), cut
    for bad_reply, why in ((reply(300), "did not stop thinking"), (reply(0), "thinking mode"),
                           (reply(65, content="<tool_call>\n<function=python>"), "did not parse"),
                           (reply(65, reasoning=""), "split cleanly"), (reply(65, content="</think> 4005"), "split cleanly"),
                           (reply(65, calls=[{"function": {"name": "python", "arguments": '{"code": "print(1'}}]), "did not parse"),
                           (reply(65, calls=[], content=""), "nothing after"),
                           (reply(300, calls=[], finish="length"), "did not stop thinking"),  # a cut reply still
                           (reply(65, reasoning="", finish="length"), "split cleanly")):      # checks bound and split
        assert why in (ov["verdict"](bad_reply)[0] or ""), (why, ov["verdict"](bad_reply))
    printed, sent = run_check(ov, good)
    assert "self-test passed, 65 reasoning tokens" in printed and sent["health"] == 1
    assert sent["url"] == "http://127.0.0.1:8001/v1/chat/completions" and sent["timeout"] == 180
    p = sent["payload"]
    assert p["custom_params"] == {"thinking_budget": 64} and p["chat_template_kwargs"] == {"enable_thinking": True}
    assert p["model"] == "flashnext" and p["tools"][0]["function"]["name"] == "python" and p["tool_choice"] == "auto"
    assert (p["temperature"], p["top_p"], p["top_k"]) == (0.7, 0.95, 20) and p["max_tokens"] == 4096  # the harness's sampling
    printed, _ = run_check(ov, reply(65, content="<tool_call>\n<function=python>", calls=[], finish="length"))
    assert "self-test passed" in printed and "parse not exercised" in printed
    # A slow boot (3 of the Franzen-fork runs so far) keeps its run and its timing: no request, no wait.
    printed, sent = run_check(ov, reply(900), health=False)
    assert "still loading at the startup deadline" in printed and sent == {"health": 1}, (printed, sent)
    # The scored run sends nothing: its log is never readable.
    printed, sent = run_check(ov, reply(900), strict=False, health=False, proc=EXITED)
    assert printed == "thinkbudget: scored run, self-test skipped\n" and sent == {"health": 0}, (printed, sent)
    refused = urllib.error.URLError(ConnectionRefusedError(111, "Connection refused"))
    rejected = urllib.error.HTTPError("u", 400, "Bad Request", {}, io.BytesIO(b'{"message": "bad custom_params"}'))
    lost = urllib.error.HTTPError("u", 500, "Server Error", {}, BodyLost())
    for kwargs, why in (({"reply": reply(900)}, "did not stop thinking"),
                        ({"health": False, "proc": EXITED}, "the server exited (code 1) before READY"),
                        ({"error": refused}, "self-test request failed"),
                        ({"error": rejected}, 'HTTP 400. {"message": "bad custom_params"}'),
                        ({"error": lost}, "rejected, HTTP 500"),
                        ({"reply": {"choices": [{"message": "4005"}]}}, "self-test reply unreadable (AttributeError"),
                        ({"reply": {"choices": {"0": {}}}}, "self-test reply unreadable (KeyError"),
                        ({"reply": b"<html>busy</html>"}, "self-test reply unreadable (JSONDecodeError")):
        try:
            run_check(ov, **kwargs)
            raise AssertionError(f"a bench run must stop ({why})")
        except RuntimeError as exc:
            assert why in str(exc), (why, exc)
    print("ok   self-test, bench runs only: 64-token budget with the harness's sampling, thinking mode and python tool; "
          "passes at 65 reasoning tokens with a parsed call or answer, or cut at max_tokens after a clean split; stops "
          "the bench on more than 80, no thinking, an unparsed call, a dirty split, a rejected or unreadable reply, or a "
          "server that exited; a server still loading at the deadline and the scored run skip it, sending nothing")
    with redirect_stdout(io.StringIO()):
        thinking_counters.self_check()
    if all((r / "usage").is_dir() for r in CONTROL_RUNS):
        cs = [thinking_counters.counters(r) for r in CONTROL_RUNS]
        total = {k: sum(c[k] for c in cs) for k in ("cut_no_call", "past_budget", "past_budget_with_call", "implicit_close")}
        assert total == {"cut_no_call": 67, "past_budget": 89, "past_budget_with_call": 22, "implicit_close": 0}, total
        print("ok   thinking_counters: self-check; on the two p448s14 control runs 67 replies cut at the cap with no "
              "call (transcript-waste.md's count), 89 past 10,240 of which 22 still called, 0 implicit closes")
    else:
        print("ok   thinking_counters: self-check (control runs not present, published-count check skipped)")
    print("PASS")


_OVERLAY_NS: dict = {}


def ns_overlay() -> dict:
    if not _OVERLAY_NS:
        exec(compile(OVERLAY.read_text(), "franzen_overlays/thinkbudget_serve.py", "exec"), _OVERLAY_NS)
    return _OVERLAY_NS


if __name__ == "__main__":
    main()
