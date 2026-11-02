"""Offline checks for arm fastdec, solution/kaggle/franzen_overlays/fastdec_serve.py.

    python3 solution/sim/test_franzen_fastdec.py

Uses the real SGLang that Franzen's notebook installs: the six files the arm patches or reads,
from the sglang wheel in his pennyroyal-v253 dataset (Pennyroyal v2.5.3 plus his patches). By
default they come from solution/sim/fixtures/pennyroyal-v253/, each pinned by sha256 to that
wheel (PROVENANCE.json). To read them from the wheel itself (25 MB), fetch it once with
  TOKEN="$(cat ARC-AGI-3-Kaggle-Starter/.kaggle/access_token)"
  curl -sSL -H "Authorization: Bearer $TOKEN" -o <path> "https://www.kaggle.com/api/v1/datasets/\
download/dfranzen/pennyroyal-v253/wheels%2Fsglang-0.5.19%2Bgd00d88efc8d6-cp312-cp312-linux_x86_64.whl"
and set FASTDEC_SGLANG_WHEEL=<path>.

Checks the two row caps that make the stock switch a regression at 10 streams; runs the exact
cell-15 code the builder inserts (both arms) against a temp venv of the real files, then checks the
edits, a rerun, the mismatch guard, the autotune gate in this notebook's server config, the
post-READY check, and which modules the wheel's own conversion rule turns into MXFP8.
No torch, no GPU, no network.
"""
from __future__ import annotations

import ast
import contextlib
import hashlib
import json
import os
import re
import sys
import tempfile
import types
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "solution" / "kaggle"))
import build_franzen_arms as builder  # noqa: E402

FIX = REPO / "solution/sim/fixtures/pennyroyal-v253"
WHEEL_SHA = "d0620216732b82b6efd4f8a78cc9b5f55343a4d3837fdfa3bcbb841b7718a225"
HC, Q4 = "sglang/srt/layers/hyperconnection.py", "sglang/srt/models/qwen4_exp.py"
TUNE = "sglang/srt/model_executor/runner/flashinfer_autotune.py"
ENVIRON, HCMIX = "sglang/srt/environ.py", "sglang/srt/layers/hc_mix_triton.py"
OFP8 = "sglang/kernels/ops/gemm/sm120_online_fp8.py"
PATCHED = (HC, Q4, TUNE)
ARMS = ("fastdec", "fastdec12")
ESCAPED_DASH = chr(92) + "u2014"  # how the fixtures write U+2014 (repo rule: no em dashes)


def load_sources() -> tuple[dict[str, str], str]:
    """The six files, from FASTDEC_SGLANG_WHEEL or the fixtures; either way checked against the pins."""
    prov = json.loads((FIX / "PROVENANCE.json").read_text())
    assert prov["wheel_sha256"] == WHEEL_SHA, prov["wheel_sha256"]
    pins = {f["wheel_path"]: f["sha256"] for f in prov["files"].values()}
    wheel = os.environ.get("FASTDEC_SGLANG_WHEEL")
    if wheel:
        sha = hashlib.sha256(Path(wheel).read_bytes()).hexdigest()
        assert sha == WHEEL_SHA, f"wheel sha256 {sha} is not Franzen's pennyroyal-v253 build"
        with zipfile.ZipFile(wheel) as z:
            raw = {rel: z.read(rel) for rel in pins}
        origin = f"wheel {Path(wheel).name} sha256 {sha[:12]}"
    else:  # fixtures store each U+2014 as its 6-character escape text (PROVENANCE.json, stored_form)
        raw = {rel: (FIX / Path(rel).name).read_text(encoding="utf-8").replace(ESCAPED_DASH, chr(0x2014)).encode()
               for rel in pins}
        origin = f"fixtures {FIX.relative_to(REPO)}, pinned to wheel sha256 {WHEEL_SHA[:12]}"
    for rel, data in raw.items():
        assert hashlib.sha256(data).hexdigest() == pins[rel], f"{rel} is not the pinned wheel member"
    return {rel: data.decode() for rel, data in raw.items()}, origin


def find_def(tree: ast.AST, name: str, cls: str | None = None) -> ast.FunctionDef:
    scope = tree if cls is None else next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == cls)
    return next(n for n in ast.walk(scope) if isinstance(n, ast.FunctionDef) and n.name == name)


def def_source(text: str, name: str, cls: str | None = None) -> str:
    node = find_def(ast.parse(text), name, cls)
    lines = text.splitlines()[node.lineno - 1:node.end_lineno]
    indent = len(lines[0]) - len(lines[0].lstrip())
    return "\n".join(line[indent:] for line in lines)


def module_int(text: str, name: str) -> int:
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return node.value.value
    raise AssertionError(f"{name} not found")


def make_venv(root: Path, files: dict[str, str]) -> Path:
    venv = root / "venv"
    for rel, text in files.items():
        dest = venv / "lib/python3.12/site-packages" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text)
    return venv


def serve_cell(arm: str) -> str:
    nb = json.loads((builder.BASE / "submission.ipynb").read_text())
    builder.apply_edits(nb, builder.REPLICA + builder.ARMS[arm][1]())
    return builder.cell_text(nb["cells"][builder.SERVE])


def snippet(arm: str) -> str:
    """The fastdec block the builder inserts into cell 15, exactly as the arm will run it."""
    cell = serve_cell(arm)
    start = cell.index("# arm fastdec")
    end = cell.index("_fastdec['apply'](VENV, env)") + len("_fastdec['apply'](VENV, env)")
    assert cell.index(builder.TARGET_ANCHOR) < start < end < cell.index("precache_model_thread.start()"), \
        "fastdec must run after target validation and before the server launch"
    return cell[start:end]


def autotune_gate(src: str, *, draft_worker: bool, draft_call: bool, switch: bool) -> bool:
    """The wheel's should_run_flashinfer_autotune on this notebook's server config, as franzen-rep-v1's
    server_args record it: quantization auto-round (draft compressed-tensors), MoE runner auto for target
    and draft, a2a none, autotune not disabled, not deterministic, speculative, SM120."""
    ns = {"torch": SimpleNamespace(cuda=SimpleNamespace(get_device_capability=lambda *a: (12, 0))),
          "envs": SimpleNamespace(SGLANG_SM120_ONLINE_MXFP8=SimpleNamespace(get=lambda: switch)),
          "get_exec": lambda: SimpleNamespace(kernel=SimpleNamespace(disable_flashinfer_autotune=False),
                                              deterministic=SimpleNamespace(enable_deterministic_inference=False),
                                              moe=SimpleNamespace(moe_runner_backend="auto", moe_a2a_backend="none")),
          "get_spec": lambda: SimpleNamespace(speculative_moe_runner_backend="auto", speculative_moe_a2a_backend=None)}
    exec(compile("from __future__ import annotations\n" + def_source(src, "should_run_flashinfer_autotune"), TUNE, "exec"), ns)
    fp4, fp8 = types.ModuleType("fp4_utils"), types.ModuleType("fp8_utils")
    fp4.get_fp4_gemm_runner_backend = lambda: SimpleNamespace(is_flashinfer_cutlass=lambda: False,
                                                              is_flashinfer_cutedsl=lambda: False)
    fp8.flashinfer_per_tensor_fp8_supported = lambda: True
    fp8.resolve_mxfp8_dense_gemm_backend = lambda: SimpleNamespace(is_flashinfer=lambda: True)
    mr = SimpleNamespace(device="cuda", is_draft_worker=draft_worker,
                         model_config=SimpleNamespace(quantization="compressed-tensors" if draft_worker else "auto-round"),
                         spec_algorithm=SimpleNamespace(is_speculative=lambda: True))
    with mock.patch.dict(sys.modules, {"sglang.srt.layers.quantization.fp4_utils": fp4,
                                       "sglang.srt.layers.quantization.fp8_utils": fp8}):
        return ns["should_run_flashinfer_autotune"](mr, for_speculative_draft=draft_call)


def converted_params(rule_src: str) -> tuple[set[str], set[str], int]:
    """Run the wheel's convert_eligible_linears_to_mxfp8 on one GDN layer and one attention layer of
    the AutoRound checkpoint (only its routed experts are INT4). Shapes from
    Intel/Qwen3.8-Flash-Next-W4A16-AutoRound config.json: hidden 2560, GDN 16 key / 48 value heads of
    128, attention 24 heads of 256 with output gate, 2 KV heads, QSA indexer 4+1 heads of 128, shared
    expert 640, HC low rank 320 over 4 x 2560. Layout as in the wheel: qkv_proj and o_proj are direct
    children of the attention decoder layer (qwen3_5.py), and so is the QSA indexer (qwen4_exp.py)."""
    class Param:
        def __init__(self, rows, cols):
            self.shape, self.dtype = (rows, cols), "bf16"

        def dim(self):
            return 2

    class Unquant:  # UnquantizedLinearMethod: what AutoRound returns for its 16-bit layers
        pass

    class Mod:
        def __init__(self, **children):
            self.children = children

        def named_children(self):
            return self.children.items()

    class Lin(Mod):
        def __init__(self, rows, cols, quant=True):
            super().__init__()
            self.weight = Param(rows, cols)
            if quant:
                self.quant_method = Unquant()

    class FusedMoE(Mod):
        pass

    class PLE(Mod):
        pass

    ns = {"torch": SimpleNamespace(nn=SimpleNamespace(Parameter=Param, Module=Mod), bfloat16="bf16")}
    exec(compile("from __future__ import annotations\n" + rule_src, "convert_rule", "exec"), ns)

    def hc():  # plain nn.Linear weights, no quant_method
        return Mod(input_mix_weight_down=Lin(320, 10240, quant=False),
                   input_mix_weight_up=Lin(10240, 320, quant=False), block_inject_weight=Lin(4, 10240, quant=False))

    def mlp():
        return Mod(gate=Lin(512, 2560), experts=FusedMoE(),
                   shared_expert=Mod(gate_up_proj=Lin(1280, 2560), down_proj=Lin(2560, 640)),
                   shared_expert_gate=Lin(1, 2560))

    gdn = Mod(linear_attn=Mod(in_proj_qkvz=Lin(16384, 2560), in_proj_ba=Lin(96, 2560), out_proj=Lin(2560, 6144)),
              mlp=mlp(), attn_hyper_connection=hc(), mlp_hyper_connection=hc(), ple=PLE())
    attn = Mod(qkv_proj=Lin(13312, 2560), o_proj=Lin(2560, 6144), mlp=mlp(),
               indexer=Mod(index_qk_proj=Lin(640, 2560)), attn_hyper_connection=hc(), mlp_hyper_connection=hc())
    names, params = {}, 0
    for kind, layer, count in (("gdn", gdn, 36), ("attn", attn, 12)):
        got = ns["convert_eligible_linears_to_mxfp8"](
            layer, enabled=True, method_factory=lambda: "mxfp8", unquantized_method_type=Unquant,
            excluded_module_type=(FusedMoE, PLE))
        names[kind] = set(got)

        def lookup(path, node=layer):
            for part in path.split("."):
                node = node.children[part]
            return node
        params += count * sum(lookup(n).weight.shape[0] * lookup(n).weight.shape[1] for n in got)
    return names["gdn"], names["attn"], params


@contextlib.contextmanager
def _healthy(url, timeout):
    yield SimpleNamespace(status=200)


def main() -> None:
    src, origin = load_sources()
    print(f"ok   real SGLang sources: {origin}")

    # 1. Why the stock switch is a regression here: both row-wise FP8 paths dequantize past their cap.
    hc_cap, head_cap = module_int(src[HCMIX], "_FUSED_MIX_MAX_ROWS"), module_int(src[OFP8], "_MAX_KERNEL_ROWS")
    assert "dequantize_rowwise_weight(" in def_source(src[HC], "mix", "GatedResidual")
    assert "* scale[start:stop, None]" in def_source(src[OFP8], "rowwise_fp8_lm_head_logits")
    for arm in ARMS:
        cfg = serve_cell(arm)
        maxreq = int(re.search(r"\n    MAXREQ=(\d+),", cfg)[1])
        verify_rows = maxreq * (int(re.search(r"\n    SPEC_STEPS=(\d+),", cfg)[1]) + 1)
        assert verify_rows > head_cap > hc_cap, (arm, verify_rows, head_cap, hc_cap)
        print(f"ok   {arm}: verify runs {verify_rows} rows at {maxreq} streams; HC mix FP8 kernel stops at "
              f"{hc_cap}, lm_head FP8 GEMV at {head_cap}, then both dequantize the weight per call")

    # 2. The switch exists and parses "true".
    assert "SGLANG_SM120_ONLINE_MXFP8 = EnvBool(False)" in src[ENVIRON]
    parse_ns: dict = {}
    exec(compile(def_source(src[ENVIRON], "parse", "EnvBool"), "EnvBool.parse", "exec"), parse_ns)
    assert parse_ns["parse"](None, "true") is True

    # 3. Run the inserted cell-15 code against a temp venv of real files.
    code = snippet("fastdec")
    assert snippet("fastdec12") == code
    with tempfile.TemporaryDirectory() as tmp:
        venv = make_venv(Path(tmp), {rel: src[rel] for rel in PATCHED})
        site = venv / "lib/python3.12/site-packages"
        env = {"SGLANG_SM120_ONLINE_MXFP8": "0"}  # what cell 15 sets before the anchor
        cell_ns = {"VENV": str(venv), "env": env}
        exec(compile(code, "cell15-fastdec", "exec"), cell_ns)
        hc, q4, tune = ((site / rel).read_text() for rel in PATCHED)
        assert env["SGLANG_SM120_ONLINE_MXFP8"] == "true" and parse_ns["parse"](None, env["SGLANG_SM120_ONLINE_MXFP8"])
        for rel, text in zip(PATCHED, (hc, q4, tune)):
            compile(text, rel, "exec")
            assert text.count("arm fastdec") == 1, rel

        # HC: the row-wise FP8 mix is never selected, so HC weights load as plain BF16.
        init = find_def(ast.parse(hc), "__init__", "GatedResidual")
        rowwise = [n for n in ast.walk(init) if isinstance(n, ast.Assign)
                   and any(getattr(t, "id", None) == "rowwise_mix" for t in n.targets)]
        assert len(rowwise) == 1 and isinstance(rowwise[0].value, ast.Constant) and rowwise[0].value.value is False

        # lm_head: post_load_weights returns before any row-wise FP8 conversion.
        post = find_def(ast.parse(q4), "post_load_weights")
        calls = {n.func.id for n in ast.walk(post) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        assert "replace_linear_weight_rowwise_fp8" not in calls
        top = [type(s).__name__ for s in post.body]
        assert top.index("Return") < top.index("For"), top  # the HC scale loop and log are now dead code
        guard = post.body[2]
        assert isinstance(guard, ast.If) and "online_fp8_enabled" in ast.unparse(guard.test), \
            "the early return must sit behind the switch guard, so the off path is unchanged"

        # The projections still convert: the conversion code is untouched.
        for name in ("_maybe_convert_linears_to_mxfp8", "_init_qwen4_exp_layer_extensions"):
            assert def_source(q4, name) == def_source(src[Q4], name), name
        ov = cell_ns["_fastdec"]
        for rel, text, old, new in ((Q4, q4, "HEAD_OLD", "HEAD_NEW"), (HC, hc, "HC_OLD", "HC_NEW"),
                                    (TUNE, tune, "TUNE_OLD", "TUNE_NEW")):
            assert text.replace(ov[new], ov[old]) == src[rel], rel  # nothing else in any file moved
        print("ok   inserted snippet: switch on, HC mix and lm_head kept BF16, target autotune on, "
              "projection conversion untouched")

        # Autotune: shut for every runner today; after the edit open for the target only, and only with the switch.
        runners = {"target": (False, False), "draft warmup": (True, False), "draft autotune": (True, True)}
        before = {k: autotune_gate(src[TUNE], draft_worker=w, draft_call=c, switch=True) for k, (w, c) in runners.items()}
        after = {k: autotune_gate(tune, draft_worker=w, draft_call=c, switch=True) for k, (w, c) in runners.items()}
        assert before == dict.fromkeys(runners, False), before
        assert after == {"target": True, "draft warmup": False, "draft autotune": False}, after
        assert autotune_gate(tune, draft_worker=False, draft_call=False, switch=False) is False
        print("ok   autotune gate in this notebook's config: never opens unpatched; patched it opens for the "
              "target only, and only with the switch on (the draft stays untuned, as in Pennyroyal)")

        # Rerun of the cell is a no-op.
        before_bytes = {rel: (site / rel).read_bytes() for rel in PATCHED}
        exec(compile(code, "cell15-fastdec", "exec"), {"VENV": str(venv), "env": env})
        assert before_bytes == {rel: (site / rel).read_bytes() for rel in PATCHED}
        print("ok   rerun leaves all three files unchanged")

        # The post-READY check: placed after the READY wait at top level, and it stops the run unless
        # the server answers and this server's log shows the autotune finished.
        for arm in ARMS:
            body = ast.parse("\n".join("pass" if l.lstrip().startswith(("!", "%")) else l
                                       for l in serve_cell(arm).split("\n"))).body
            assert isinstance(body[-2], ast.While) and "NOTEBOOK_START_TIME" in ast.unparse(body[-2].test), arm
            assert ast.unparse(body[-1]) == "_fastdec['check'](LOG, log_offset, url, strict=not TRUE_SUBMISSION)", arm
        log = Path(tmp) / "serve.log"
        log.write_text(f"[old run] {ov['AUTOTUNE_DONE']}\n")
        offset = log.stat().st_size
        url = "http://127.0.0.1:1/health"
        with mock.patch("urllib.request.urlopen", _healthy):
            for text in ("Load weight end\n", ""):
                log.write_text(f"[old run] {ov['AUTOTUNE_DONE']}\n{text}")
                try:
                    ov["check"](log, offset, url, strict=True)
                    raise AssertionError("no autotune line after the offset must stop a bench run")
                except RuntimeError as exc:
                    assert "untuned fallback tile" in str(exc)
                ov["check"](log, offset, url, strict=False)  # the scored run warns and keeps going
            log.write_text(log.read_text() + f"[2026-10-03 10:00:00] {ov['AUTOTUNE_DONE']}\n")
            ov["check"](log, offset, url, strict=True)
        with mock.patch("urllib.request.urlopen", side_effect=OSError("connection refused")):
            try:
                ov["check"](log, offset, url, strict=True)
                raise AssertionError("a server that is not up must stop a bench run")
            except RuntimeError as exc:
                assert "720 s startup deadline" in str(exc)
            ov["check"](log, offset, url, strict=False)
        print("ok   post-READY check: last statement of cell 15 in both arms; a bench run stops on no server or "
              "no autotune line, the scored run only warns")

    # 4. A build that differs from v2.5.3 stops the cell before the switch is set.
    with tempfile.TemporaryDirectory() as tmp:
        drifted = src[HC].replace("rowwise_mix = online_fp8 and online_fp8_enabled()",
                                  "rowwise_mix = online_fp8 and bool(online_fp8_enabled())")
        venv = make_venv(Path(tmp), {HC: drifted, Q4: src[Q4], TUNE: src[TUNE]})
        env = {"SGLANG_SM120_ONLINE_MXFP8": "0"}
        try:
            exec(compile(code, "cell15-fastdec", "exec"), {"VENV": str(venv), "env": env})
            raise AssertionError("a drifted hyperconnection.py must raise")
        except RuntimeError as exc:
            assert "does not match the Pennyroyal v2.5.3 source" in str(exc)
        assert env["SGLANG_SM120_ONLINE_MXFP8"] == "0"
    print("ok   drifted source raises and the switch stays off")

    # 5. What still turns into MXFP8, by the wheel's own rule on the AutoRound module shapes.
    gdn, attn, params = converted_params(def_source(src[OFP8], "convert_eligible_linears_to_mxfp8"))
    assert gdn == {"linear_attn.in_proj_qkvz", "linear_attn.out_proj",
                   "mlp.shared_expert.gate_up_proj", "mlp.shared_expert.down_proj"}, gdn
    assert attn == {"qkv_proj", "o_proj", "indexer.index_qk_proj",
                    "mlp.shared_expert.gate_up_proj", "mlp.shared_expert.down_proj"}, attn
    bf16_gb, saved_gb = params * 2 / 1e9, params * (1 - 1 / 32) / 1e9  # MXFP8: 1 byte plus a 1-byte scale per 32
    assert 2.8 < saved_gb < 2.9, saved_gb
    print(f"ok   MXFP8 rule converts {params / 1e9:.2f}B target params: {bf16_gb:.2f} GB of BF16 read per verify "
          f"forward becomes {bf16_gb - saved_gb:.2f} GB; {saved_gb / 2**30 * 1e9:.2f} GiB freed for KV. "
          "Router (name 'gate'), in_proj_ba (96 rows), shared_expert_gate (1 row), experts, PLE, HC stay as they are")
    print("PASS")


if __name__ == "__main__":
    main()
