"""Arm lora: fold a dense LoRA into the BF16 checkpoint tensors while SGLang loads them.

Same mechanism as franzen_overlays/p448_serve.py: Qwen4ExpForConditionalGeneration.load_weights first
passes the checkpoint through a generator that rewrites tensors by name. Here the generator adds
(alpha / r) * B @ A to every targeted BF16 tensor, so the server runs unchanged kernels, CUDA graphs,
MTP speculation and the D3 drafter (no SGLang LoRA runtime, which has no qwen4_exp wiring).

Targets must be BF16 in the checkpoint: linear_attn.{in_proj_qkv,in_proj_z,out_proj},
self_attn.{q,k,v,o}_proj, mlp.shared_expert.{gate,up,down}_proj. The int4 routed experts
(mlp.experts.N.*.{qweight,qzeros,scales}) are refused. Env: SGLANG_LORA_DIR = dir holding
adapter_model.safetensors and adapter_config.json (r, lora_alpha). Unset = no-op.

Adapter keys: "<module>.lora_A.weight" / "<module>.lora_B.weight", optionally prefixed
"base_model.model.". <module> + ".weight" must equal a checkpoint name. Every adapter pair must be
consumed exactly once or the load raises (a silently unmerged adapter would score as the base model).

Compose with p448: patch_sglang accepts either an unpatched or a p448-patched load_weights.
Tested on CPU by solution/lora/test_lora_merge.py.
"""
from __future__ import annotations

from pathlib import Path

MARK = "arm lora"
DENSE_OK = ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj",
            "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
            "mlp.shared_expert.gate_proj", "mlp.shared_expert.up_proj", "mlp.shared_expert.down_proj")

REMAP_SRC = r'''

# ---- arm lora (solution/lora/lora_merge_serve.py): fold a dense LoRA into BF16 weights at load ----
def _lora_merge_weights(weights, config):
    import json, os
    d = os.environ.get("SGLANG_LORA_DIR")
    if not d:
        yield from weights
        return
    import torch
    from safetensors.torch import load_file
    cfg = json.load(open(os.path.join(d, "adapter_config.json")))
    scale = float(cfg["lora_alpha"]) / float(cfg["r"])
    if cfg.get("use_rslora"):
        scale = float(cfg["lora_alpha"]) / float(cfg["r"]) ** 0.5
    raw = load_file(os.path.join(d, "adapter_model.safetensors"))
    pairs = {}
    for k, v in raw.items():
        k = k[len("base_model.model."):] if k.startswith("base_model.model.") else k
        mod, _, rest = k.rpartition(".lora_")
        ab = rest.split(".")[0]
        if ab not in ("A", "B"):
            raise RuntimeError(f"lora: unexpected adapter key {k}")
        pairs.setdefault(mod + ".weight", {})[ab] = v
    for name, p in pairs.items():
        if set(p) != {"A", "B"}:
            raise RuntimeError(f"lora: {name} lacks its A or B half")
        if ".mlp.experts." in name:
            raise RuntimeError(f"lora: {name} targets a quantized routed expert")
    done = set()
    for name, tensor in weights:
        p = pairs.get(name)
        if p is not None:
            if tensor.dtype != torch.bfloat16:
                raise RuntimeError(f"lora: {name} is {tensor.dtype}, only BF16 tensors can be merged")
            A, B = p["A"].float(), p["B"].float()
            if B.shape[0] != tensor.shape[0] or A.shape[1] != tensor.shape[1] or A.shape[0] != B.shape[1]:
                raise RuntimeError(f"lora: {name} shape {tuple(tensor.shape)} vs A {tuple(A.shape)} B {tuple(B.shape)}")
            dev = tensor.device
            delta = (B.to(dev) @ A.to(dev)) * scale
            tensor = (tensor.float() + delta).to(torch.bfloat16)
            done.add(name)
        yield name, tensor
    missing = sorted(set(pairs) - done)
    if missing:
        raise RuntimeError(f"lora: {len(missing)} adapter targets not in the checkpoint, e.g. {missing[:3]}")
    print(f"lora: merged {len(done)} tensors, scale {scale}", flush=True)
'''

DEF = "    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):\n"
P448 = "        weights = _p448_prune_weights(weights, self.config)  # arm p448\n"
STACK = "        stacked_params_mapping = [\n"
LINE = f"        weights = _lora_merge_weights(weights, self.config)  # {MARK}\n"


def patch_text(text: str) -> str:
    """Return text with the merge hook installed exactly once (idempotent)."""
    if LINE in text:
        if "def _lora_merge_weights" not in text:
            raise RuntimeError("lora: hook present but merge function missing")
        return text
    olds = [DEF + STACK, DEF + P448 + STACK]
    hits = [o for o in olds if text.count(o) == 1]
    if len(hits) != 1 or MARK in text:
        raise RuntimeError("lora: qwen4_exp.load_weights does not match Pennyroyal v2.5.3 (p448-patched or not)")
    old = hits[0]
    new = old.replace(STACK, LINE + STACK)
    return text.replace(old, new) + REMAP_SRC


def patch_sglang(venv) -> Path:
    roots = sorted(Path(venv).glob("lib/python*/site-packages/sglang"))
    if len(roots) != 1:
        raise RuntimeError(f"lora: expected one installed sglang under {venv}, found {roots}")
    path = roots[0] / "srt/models/qwen4_exp.py"
    path.write_text(patch_text(path.read_text()))
    return path
