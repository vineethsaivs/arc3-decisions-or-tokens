"""Offline checks for arm p448, solution/kaggle/franzen_overlays/p448_serve.py.

    python3 solution/sim/test_franzen_p448.py

Runs the launcher-cell snippet the builder inserts, against a temp venv holding the real
Pennyroyal SGLang sources (read from the sglang wheel) and a fake model directory. Then drives
the exact code the patch put into the installed qwen4_exp.py: a tiny fake safetensors
checkpoint with the Intel AutoRound expert layout and the real REAP lists for two layers goes
through _p448_prune_weights and then through SGLang's own expert-name mapping. Also runs the
patched draft fallback in logical_to_all_physical. No torch, no GPU, no network.
"""
from __future__ import annotations

import ast
import json
import os
import shutil
import struct
import sys
import tempfile
import typing
import zipfile
from pathlib import Path
from types import SimpleNamespace

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "solution" / "kaggle"))
import build_franzen_arms as builder  # noqa: E402

OVERLAY = REPO / "solution" / "kaggle" / "franzen_overlays" / "p448_serve.py"
SCRATCH = Path("<scratch>/"
               "c8309e29-5c7b-46fe-8bbb-77b013bc9554/scratchpad/p448")
WHEEL = Path(os.environ.get("P448_SGLANG_WHEEL") or __import__("franzen_tree").pennyroyal_wheel())  # cached, sha-checked
MATCHED = SCRATCH / "reap_k448_kept.json"  # router-row match output, provenance check only
SOURCES = ["sglang/srt/models/qwen4_exp.py", "sglang/srt/eplb/expert_location.py",
           "sglang/srt/layers/moe/fused_moe_triton/layer.py"]
PROJ, PARTS = ("gate_proj", "up_proj", "down_proj"), ("qweight", "qzeros", "scales")


class FakeTensor:
    """Bytes plus a shape; t[list] gathers rows like torch advanced indexing on dim 0."""

    def __init__(self, shape, data: bytes):
        self.shape, self.data = tuple(shape), data

    def __getitem__(self, rows):
        assert isinstance(rows, list), rows
        width = len(self.data) // self.shape[0]
        return FakeTensor((len(rows),) + self.shape[1:],
                          b"".join(self.data[r * width:(r + 1) * width] for r in rows))


def write_safetensors(path: Path, tensors: dict) -> None:
    header, blobs, offset = {}, [], 0
    for name, (dtype, shape, data) in tensors.items():
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + len(data)]}
        blobs.append(data)
        offset += len(data)
    raw = json.dumps(header).encode()
    raw += b" " * (-len(raw) % 8)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"".join(blobs))


def read_safetensors(path: Path):
    raw = path.read_bytes()
    size = struct.unpack("<Q", raw[:8])[0]
    for name, meta in json.loads(raw[8:8 + size]).items():
        a, b = meta["data_offsets"]
        yield name, FakeTensor(meta["shape"], raw[8 + size + a:8 + size + b])


def function_source(path: Path, name: str) -> str:
    """Source of one def (method or function) in a file, dedented, decorators dropped."""
    text = path.read_text()
    node = next(n for n in ast.walk(ast.parse(text))
                if isinstance(n, ast.FunctionDef) and n.name == name)
    lines = text.splitlines()[node.lineno - 1:node.end_lineno]
    indent = len(lines[0]) - len(lines[0].lstrip())
    return "\n".join(line[indent:] for line in lines)


def fake_venv(root: Path) -> Path:
    site = root / "venv/lib/python3.12/site-packages"
    with zipfile.ZipFile(WHEEL) as wheel:
        for name in SOURCES:
            (site / name).parent.mkdir(parents=True, exist_ok=True)
            (site / name).write_bytes(wheel.read(name))
    return site / "sglang"


def fake_model(root: Path) -> Path:
    model = root / "model"
    model.mkdir()
    cfg = {"architectures": ["Qwen4ExpForConditionalGeneration"],
           "quantization_config": {"quant_method": "auto-round", "bits": 4},
           "text_config": {"num_experts": 512, "num_hidden_layers": 48, "num_experts_per_tok": 10}}
    (model / "config.json").write_text(json.dumps(cfg))
    for name in ("tokenizer.json", "chat_template.jinja", "model.safetensors.index.json",
                 "model-00001-of-00017.safetensors", "preprocessor_config.json"):
        (model / name).write_text(name)
    return model


def main() -> None:
    ns: dict = {}
    exec(compile(OVERLAY.read_text(), "franzen_overlays/p448_serve.py", "exec"), ns)
    kept = ns["kept_experts"]()
    assert len(kept) == 48 and all(len(v) == 448 and v == sorted(set(v)) for v in kept.values())
    if MATCHED.is_file():
        assert kept == json.loads(MATCHED.read_text()), "table differs from the router-row match"
        print("PASS table equals the router-row match (48 layers, 448 kept each)")
    else:
        print(f"SKIP provenance compare ({MATCHED} not present)")
    if not WHEEL.is_file():
        raise SystemExit(f"FAIL: SGLang wheel not found at {WHEEL}; set P448_SGLANG_WHEEL")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        sglang = fake_venv(root)
        model = fake_model(root)
        location = sglang / "srt/eplb/expert_location.py"
        original_lookup = function_source(location, "logical_to_all_physical")

        # 1. The exact snippet the builder inserts into the launcher cell.
        nb = json.loads((builder.BASE / "submission.ipynb").read_text())
        builder.apply_edits(nb, builder.REPLICA + builder.ARMS["p448"][1]())
        cell = builder.cell_text(nb["cells"][builder.SERVE])
        start = cell.index("# arm p448: serve")
        end = cell.index("\n", cell.index("MODEL_DIR = _p448['apply']"))
        snippet = cell[start:end]
        literal = next(n for n in ast.walk(ast.parse(snippet)) if isinstance(n, ast.Constant)
                       and isinstance(n.value, str) and "def apply" in n.value)
        assert literal.value == OVERLAY.read_text(), "cell carries a stale copy of the overlay"
        assert "MAXREQ=12" in cell and "'ARC3_MAX_ACTIVE_STREAMS': 12" in builder.cell_text(nb["cells"][builder.ENV])
        assert cell.index("# arm p448: serve") < cell.index("prepare_draft_view(MODEL_DIR")
        env = {"PATH": "/usr/bin"}
        cellns = {"MODEL_DIR": str(model), "PREFIX": str(root / "sgl-intel"),
                  "VENV": str(root / "venv"), "env": env}
        for _ in range(2):  # a rerun of the cell must be a no-op
            exec(snippet, cellns)
        view = Path(cellns["MODEL_DIR"])
        assert view == root / "sgl-intel" / "p448-view"
        vcfg = json.loads((view / "config.json").read_text())
        assert vcfg["text_config"]["num_experts"] == 448
        assert json.loads((model / "config.json").read_text())["text_config"]["num_experts"] == 512
        assert env["SGLANG_P448_KEPT"] == str(view / "p448_kept_experts.json")
        assert json.loads(Path(env["SGLANG_P448_KEPT"]).read_text()) == kept
        for src in model.iterdir():
            if src.name != "config.json":
                assert (view / src.name).is_symlink() and (view / src.name).resolve() == src.resolve()
        print("PASS cell snippet: view (num_experts 448, links, kept list), env, idempotent rerun")

        qwen = sglang / "srt/models/qwen4_exp.py"
        text = qwen.read_text()
        assert text.count("weights = _p448_prune_weights(weights, self.config)") == 1
        assert text.count("def _p448_prune_weights") == 1
        assert location.read_text().count("# arm p448") == 1
        for path in (qwen, location):
            compile(path.read_text(), str(path), "exec")
        loader = function_source(qwen, "load_weights")
        assert loader.splitlines()[1].strip().startswith("weights = _p448_prune_weights"), loader[:200]
        print("PASS sglang patch: both edits land once, files compile, call is load_weights' first line")

        patched = location.read_text()
        broken = patched.replace("# arm p448", "# arm x")
        location.write_text(broken.replace("        cpu_map = self.logical_to_all_physical_map_cpu\n", "", 1))
        try:
            ns["patch_sglang"](root / "venv")
            raise AssertionError("patch_sglang accepted a source it was not written against")
        except RuntimeError as exc:
            assert "Pennyroyal" in str(exc)
        location.write_text(patched)
        print("PASS sglang patch refuses an unknown source")

        # 2. The remap exactly as installed, on a fake checkpoint with the real layout.
        rns: dict = {}
        exec(compile(function_source(qwen, "_p448_prune_weights"), "qwen4_exp.py", "exec"), rns)
        prune = rns["_p448_prune_weights"]
        layers = (0, 47)
        tensors = {"model.language_model.embed_tokens.weight": ("BF16", (2, 2), b"emb!" * 2)}
        for layer in (0, 1, 47):
            tensors[f"model.language_model.layers.{layer}.mlp.gate.weight"] = (
                "BF16", (512, 2), b"".join(struct.pack("<HH", layer, e) for e in range(512)))
            for e in range(512):
                for p, proj in enumerate(PROJ):
                    for q, part in enumerate(PARTS):
                        tensors[f"model.language_model.layers.{layer}.mlp.experts.{e}.{proj}.{part}"] = (
                            "U8", (1, 5), struct.pack("<HHB", layer, e, 3 * p + q))
            tensors[f"model.language_model.layers.{layer}.mlp.shared_expert.gate_proj.weight"] = ("BF16", (1, 2), b"shar")
        tensors["mtp.layers.0.mlp.gate.weight"] = ("BF16", (512, 2), bytes(2048))
        tensors["mtp.layers.0.mlp.experts.500.gate_proj.weight"] = ("BF16", (1, 2), b"mtp!")
        ckpt = root / "fake.safetensors"
        write_safetensors(ckpt, tensors)
        kept_path = root / "kept.json"
        kept_path.write_text(json.dumps({str(layer): kept[str(layer)] for layer in layers}))
        cfg = SimpleNamespace(num_experts=448)

        os.environ.pop("SGLANG_P448_KEPT", None)
        assert [n for n, _ in prune(read_safetensors(ckpt), cfg)] == list(tensors)
        os.environ["SGLANG_P448_KEPT"] = str(kept_path)
        out = list(prune(read_safetensors(ckpt), cfg))
        names = [n for n, _ in out]
        assert len(names) == len(set(names)) == len(tensors) - 2 * 64 * 9

        typing_ns = {k: getattr(typing, k) for k in ("List", "Tuple")}
        exec(function_source(sglang / "srt/layers/moe/fused_moe_triton/layer.py",
                             "make_expert_params_mapping"), typing_ns)
        mapping = typing_ns["make_expert_params_mapping"](None, "gate_proj", "down_proj", "up_proj", 448)
        slots = {layer: {} for layer in layers}
        for name, t in out:
            if ".mlp.experts." not in name or not name.startswith("model.language_model.layers."):
                if name not in {f"model.language_model.layers.{layer}.mlp.gate.weight" for layer in layers}:
                    assert t.data == tensors[name][2], name  # everything else untouched
                continue
            layer, old, code = struct.unpack("<HHB", t.data)
            if layer not in layers:
                assert name == f"model.language_model.layers.{layer}.mlp.experts.{old}.{PROJ[code // 3]}.{PARTS[code % 3]}"
                continue
            new = int(name.split(".mlp.experts.")[1].split(".")[0])
            assert kept[str(layer)][new] == old, (name, old)
            sglang_name = name.replace("model.language_model.", "model.")  # as load_weights does
            hits = [m for m in mapping if m[1] in sglang_name]
            assert len(hits) == 1 and hits[0][2] == new, (sglang_name, hits)
            slots[layer].setdefault(new, set()).add(code)
        for layer in layers:
            assert sorted(slots[layer]) == list(range(448))
            assert all(codes == set(range(9)) for codes in slots[layer].values())
        routers = dict(out)
        for layer in layers:
            r = routers[f"model.language_model.layers.{layer}.mlp.gate.weight"]
            assert r.shape == (448, 2)
            assert [struct.unpack("<HH", r.data[4 * j:4 * j + 4])[1] for j in range(448)] == kept[str(layer)]
        assert routers["model.language_model.layers.1.mlp.gate.weight"].shape == (512, 2)
        assert routers["mtp.layers.0.mlp.gate.weight"].shape == (512, 2)
        print("PASS remap: pruned experts dropped, kept renumbered 0..447 into the right SGLang slots, "
              "router rows sliced, layers outside the list, MTP and dense tensors untouched")

        for bad_cfg, cut, why in ((SimpleNamespace(num_experts=512), None, "config says"),
                                  (cfg, f"layers.47.mlp.experts.{kept['47'][5]}.", "missing"),
                                  (cfg, "model.language_model.layers.0.mlp.gate.weight", "not sliced")):
            stream = ((n, t) for n, t in read_safetensors(ckpt) if not (cut and cut in n))
            try:
                list(prune(stream, bad_cfg))
                raise AssertionError(f"accepted a bad load ({why})")
            except RuntimeError as exc:
                assert why in str(exc), exc
        os.environ.pop("SGLANG_P448_KEPT")
        print("PASS remap fails loudly on a 512-expert config, a missing kept expert, an unsliced router")

        # 3. Draft MTP experts (512) against the target-sized trivial map (48 x 448).
        class Row:
            def __init__(self, e): self.e = e
            def tolist(self): return [self.e]

        class Map:
            shape = (48, 448, 1)
            def __getitem__(self, key):
                layer, e = key
                if layer >= 48 or e >= 448:
                    raise IndexError(key)
                return Row(e)

        def lookup(src):
            lns = {"List": typing.List}
            exec(src, lns)
            meta = SimpleNamespace(logical_to_all_physical_map_cpu=Map(), num_logical_experts=448)
            return lambda layer, e: lns["logical_to_all_physical"](meta, layer, e)

        old, new = lookup(original_lookup), lookup(function_source(location, "logical_to_all_physical"))
        try:
            old(0, 500)
            raise AssertionError("unpatched lookup should fail for draft expert 500")
        except IndexError:
            pass
        assert new(0, 500) == [500] and new(0, 447) == [447] and new(0, 10) == [10] and new(60, 300) == [300]
        print("PASS draft fallback: unpatched map raises on draft expert 500, patched maps 0..511 to identity")


if __name__ == "__main__":
    main()
