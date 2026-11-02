"""Offline checks for lever B3, solution/kaggle/franzen_overlays/nvfp4kv_serve.py.

    python3 solution/sim/test_franzen_nvfp4kv.py              # offline, no GPU
    python3 solution/sim/test_franzen_nvfp4kv.py --kv-scales  # network: re-check that the target has no k/v scales

Uses the real SGLang that Franzen's notebook installs, read from the sglang wheel in his pennyroyal-v253
dataset (cached and sha-checked by franzen_tree.pennyroyal_wheel(); NVFP4KV_SGLANG_WHEEL overrides it).
  1. The vendored files match PROVENANCE.json and the module's pins.
  2. The cell-15 block the builder inserts, run against a temp venv of the eight wheel files, leaves exactly
     what `git apply -p2` leaves, sets the QSA switch and hands CFG the split KV dtypes; a rerun is a no-op;
     a drifted or half-patched file raises before anything is written, with the switch still off.
  3. It composes with the p448 and fastdec serving patches in every overlay order.
  4. Both proposed arms build on the base notebook, with and without wastefix; the argv cell 15 assembles
     carries --kv-cache-dtype nvfp4, --speculative-draft-kv-cache-dtype fp8_e4m3 and --max-total-tokens
     with 20 streams and 100 Mamba slots; the probe boots FP8 first, and only outside a scored rerun.
  5. parity() against fake servers that answer in SGLang's logprob shape passes on a small NLL change and on
     unbiased noise, and raises past each gate, on a comparison that never read the pool, on a broken decode
     path, and when the NVFP4 boot never logs the QSA switch.
  6. Cell 15's own launch and READY wait, in a scored run, relaunches an NVFP4 server that dies before READY
     once on FP8 KV and waits for it; a healthy boot, a second death and a bench run are left alone.
No torch, no GPU.
"""
from __future__ import annotations

import ast
import contextlib
import hashlib
import io
import json
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import tempfile
import time
import types
import urllib.request
import zipfile
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "solution" / "kaggle"))
import build_franzen_arms as builder  # noqa: E402
import franzen_tree  # noqa: E402
from overlay_orders import orders  # noqa: E402

OVERLAYS = REPO / "solution/kaggle/franzen_overlays"
NV = OVERLAYS / "nvfp4kv"
PATCHES = ("0001-qsa-nvfp4-kv.patch", "0002-optional-mamba-pingpong-fix.patch")
P448_FILES = ("sglang/srt/models/qwen4_exp.py", "sglang/srt/eplb/expert_location.py")
FASTDEC_FILES = ("sglang/srt/layers/hyperconnection.py", "sglang/srt/models/qwen4_exp.py",
                 "sglang/srt/model_executor/runner/flashinfer_autotune.py")
QSA_INIT = "sglang/srt/layers/attention/qsa/__init__.py"  # 0001 adds a module to this package
POOL = 2_000_000

# The integrator pastes this into build_franzen_arms.py (helpers beside p448/fastdec, the ARMS entries
# into ARMS). Until then the test execs it into the builder module, so it checks exactly this text.
BUILDER_SNIPPET = r'''
NVFP4KV = OVERLAYS / "nvfp4kv"
NVFP4KV_PATCHES = ("0001-qsa-nvfp4-kv.patch", "0002-optional-mamba-pingpong-fix.patch")


def nvfp4kv(max_total_tokens: int) -> list[tuple[int, str, str]]:
    """B3: Lesilva's QSA NVFP4 KV patches (vendored with provenance in franzen_overlays/nvfp4kv/) and the
    flags they need. Cell 15 passes one KVDTYPE to both KV flags; this splits them: target nvfp4, MTP draft
    fp8_e4m3 (the draft flag rejects nvfp4). A token drops from 14,080 to 8,704 bytes, so the pool goes from
    1.53M FP8 tokens to max_total_tokens (the wheel's own ceiling is about 2.1M at 100 Mamba slots; the cap
    only lowers it). apply() patches SGLang before it returns the dtype, so a failed patch never launches
    nvfp4. In a scored run, an NVFP4 server that exits before READY is relaunched once on FP8 KV inside the
    cell's READY wait, so the draw is not played against a dead server. franzen_overlays/nvfp4kv_serve.py."""
    code = (OVERLAYS / "nvfp4kv_serve.py").read_text()
    patches = {name: (NVFP4KV / name).read_text() for name in NVFP4KV_PATCHES}
    died = "    if proc.poll() is not None:\n        show_log_tail()\n        break"  # cell 15's READY wait
    return [
        (SERVE, '    KVDTYPE="fp8_e4m3",',
         '    KVDTYPE="fp8_e4m3",  # arm nvfp4kv: nvfp4 once apply() below has patched SGLang\n'
         '    DRAFT_KVDTYPE="fp8_e4m3",  # arm nvfp4kv: the MTP draft keeps FP8 (its flag rejects nvfp4)\n'
         '    MAX_TOTAL_TOKENS=None,  # arm nvfp4kv: KV pool cap, set by apply()'),
        (SERVE, '"--speculative-draft-kv-cache-dtype", CFG["KVDTYPE"],',
         '"--speculative-draft-kv-cache-dtype", CFG["DRAFT_KVDTYPE"],'),
        (SERVE, 'if CFG["GDN_MTP_CACHE_MODE"]: args += ["--gdn-mtp-cache-mode", CFG["GDN_MTP_CACHE_MODE"]]',
         'if CFG["GDN_MTP_CACHE_MODE"]: args += ["--gdn-mtp-cache-mode", CFG["GDN_MTP_CACHE_MODE"]]\n'
         'if CFG["MAX_TOTAL_TOKENS"]: args += ["--max-total-tokens", str(CFG["MAX_TOTAL_TOKENS"])]  # arm nvfp4kv'),
        (SERVE, TARGET_ANCHOR, TARGET_ANCHOR + "\n# arm nvfp4kv: QSA NVFP4 KV patches, then the split KV dtypes and the pool cap\n"
         + "_nvfp4kv = {}\nexec(compile(" + repr(code) + ", 'franzen_overlays/nvfp4kv_serve.py', 'exec'), _nvfp4kv)\n"
         + f"CFG.update(_nvfp4kv['apply'](VENV, env, {int(max_total_tokens)}, " + repr(patches) + "))"),
        (SERVE, died,
         "    if proc.poll() is not None:\n        show_log_tail()\n"
         '        if TRUE_SUBMISSION and "nvfp4" in args:  # arm nvfp4kv: NVFP4 died before READY, relaunch once on FP8 KV\n'
         "            proc, args = _nvfp4kv['relaunch_fp8'](proc, args, env, LOG, int(SERVED_MODEL_PORT))\n"
         "            continue\n"
         "        break"),
    ]


def nvfp4kv_parity() -> list[tuple[int, str, str]]:
    """Probe only, never in a scored rerun: cell 15 boots the FP8 reference first; parity() scores 32 recorded
    prompts on it, restarts the server on NVFP4, scores them again and stops the run unless teacher-forced
    logprobs agree: NVFP4 NLL at most 0.02 nats/token above FP8 (the signed change Lesilva reported; mean
    |dlogp| is printed, not gated), top-1 agreement at least 95%, greedy decode agreeing for 2+ tokens."""
    prompts = (NVFP4KV / "parity_prompts.json").read_text()
    launch = "precache_model_thread.start()"
    return [(SERVE, launch, "args_nvfp4 = args\nif not TRUE_SUBMISSION:  # arm nvfp4kv probe: the FP8 reference boots first\n"
                            "    args = _nvfp4kv['reference_args'](args)\n" + launch),
            (SERVE, SERVE_END, SERVE_END + "\nif not TRUE_SUBMISSION:  # arm nvfp4kv probe: FP8 vs NVFP4 logprob parity, raises on failure\n"
             "    proc, log_offset = _nvfp4kv['parity'](proc, args_nvfp4, env, LOG, url, json.loads(" + repr(prompts) + "))\n"
             "    args = args_nvfp4")]


def s20() -> list[tuple[int, str, str]]:
    """B3 serving component: NVFP4 KV with a 2.0M-token pool, 20 admitted streams, 100 Mamba slots (SGLang
    runs max_mamba_cache_size // 5 requests). Replaces streams()/mamba() in whatever base it joins."""
    return nvfp4kv(2_000_000) + streams(20) + mamba(100)


ARMS.update({
    "s20probe": ("PROBE of B3 on the p448s14dpwfp stack: NVFP4 KV (2.0M-token pool), 20 streams, 100 Mamba slots; "
                 "boots FP8 first and stops unless FP8 and NVFP4 logprobs agree on 32 recorded prompts (300 s per game)",
                 lambda: STARTUP + p448() + s20() + PROBE + dprime() + overlay("wastefix") + nvfp4kv_parity()),
    "s20": ("B3 replica: p448s14dpwf with NVFP4 KV (2.0M-token pool), 20 admitted streams, 100 Mamba slots; add the B1 "
            "reply-cache overlay when it lands to make L1 + S20",
            lambda: STARTUP + p448() + s20() + dprime() + overlay("wastefix")),
})
'''

FAKE_SERVER = r'''
import json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
port, mode = int(sys.argv[1]), sys.argv[2]
if mode == "crash" or (mode == "crash-nvfp4" and "nvfp4" in sys.argv[3:]):
    sys.exit(3)  # exits before READY, like a boot that dies in graph capture
if mode.startswith("nvfp4") and mode != "nvfp4-noswitch":
    print("[fake] KV4: QSA NVFP4 KV enabled via SGLANG_QSA_NVFP4_KV=1", flush=True)
print("[fake] KV Cache is allocated. dtype: torch.uint8, #tokens: 1999936", flush=True)
warm = set()

class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def reply(self, obj):
        data = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self.reply({"ok": True})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/tokenize":
            return self.reply({"tokens": [ord(c) % 997 for c in body["prompt"][::50]], "count": 0, "max_model_len": 0})
        ids = body["input_ids"]
        start = body.get("logprob_start_len", len(ids))
        cached = start // 64 * 64 if tuple(ids[:start]) in warm and mode != "nvfp4-vacuous" else 0
        warm.add(tuple(ids))
        shift = {"nvfp4": 0.013, "nvfp4-drift": 0.05}.get(mode, 0.0)  # 0.013: Lesilva's worst published change
        noise = 0.05 if mode == "nvfp4-noisy" else 0.0  # |dlogp| 0.05 per token, signed mean about 0
        lp = [-((t * 7919 + i) % 1000) / 1000 - shift - noise * (-1) ** i for i, t in enumerate(ids)][start:]
        top = [(t * 31 + i) % 997 for i, t in enumerate(ids)][start:]
        if mode == "nvfp4-flip":
            top = [5 if i % 10 == 0 else x for i, x in enumerate(top)]
        if mode == "nvfp4-ties":  # about 4% of argmaxes flip, as near-ties do on a healthy pool
            top = [5 if (x + i) % 30 == 0 else x for i, x in enumerate(top)]
        gen = [(sum(ids) + k) % 997 for k in range(8)]
        if mode == "nvfp4-decode":
            gen = gen[:1] + [g + 1 for g in gen[1:]]
        # SGLang's shape (logprob_result_processor.py): the first scored position has no logprob and no top row.
        self.reply({"text": "", "meta_info": {
            "cached_tokens": cached,
            "input_token_logprobs": [[None, ids[start], None]] + [[x, t, None] for x, t in zip(lp[1:], ids[start + 1:])]
                                    if start < len(ids) else [],
            "input_top_logprobs": [None] + [[[0.0, x, None]] for x in top[1:]] if start < len(ids) else [],
            "output_token_logprobs": [[-0.1, g, None] for g in gen]}})

HTTPServer(("127.0.0.1", port), H).serve_forever()
'''


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load(path: Path) -> dict:
    ns: dict = {}
    exec(compile(path.read_text(), str(path.relative_to(REPO)), "exec"), ns)
    return ns


def wheel_files(rels) -> dict[str, bytes]:
    wheel = Path(os.environ.get("NVFP4KV_SGLANG_WHEEL") or franzen_tree.pennyroyal_wheel())
    assert sha(wheel.read_bytes()) == franzen_tree.WHEEL_SHA256, f"{wheel} is not Franzen's pennyroyal-v253 build"
    with zipfile.ZipFile(wheel) as z:
        return {rel: z.read(rel) for rel in rels}


def make_venv(root: Path, files: dict[str, bytes]) -> tuple[Path, Path]:
    site = root / "venv/lib/python3.12/site-packages"
    for rel, data in files.items():
        (site / rel).parent.mkdir(parents=True, exist_ok=True)
        (site / rel).write_bytes(data)
    return root / "venv", site


def snapshot(site: Path) -> dict[str, bytes]:
    return {str(p.relative_to(site)): p.read_bytes() for p in sorted(site.rglob("*.py"))}


def built_cell(arm: str) -> str:
    nb = json.loads((builder.BASE / "submission.ipynb").read_text())
    builder.apply_edits(nb, builder.REPLICA + builder.ARMS[arm][1]())
    return builder.cell_text(nb["cells"][builder.SERVE])


def block(cell: str, start: str, end: str) -> str:
    """Text from the start marker through the end marker, which must each occur once."""
    assert cell.count(start) == 1 and cell.count(end) == 1, (start[:40], end[:40])
    i = cell.index(start)
    return cell[i:cell.index(end, i) + len(end)]


def nvfp4kv_block(cell: str) -> str:
    start = "# arm nvfp4kv: QSA NVFP4 KV patches"
    line_start = cell.index("CFG.update(_nvfp4kv['apply'](VENV, env, ")
    return block(cell, start, cell[line_start:cell.index("\n", line_start)])


def cfg_of(cell: str) -> dict:
    """The cell's `CFG = dict(...)` statement, evaluated."""
    tree = ast.parse(cell)
    node = next(n for n in tree.body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "CFG")
    ns = {"SERVED_MODEL_NAME": "flashnext"}
    exec(ast.get_source_segment(cell, node), ns)
    return ns["CFG"]


def launch_argv(cell: str, cfg: dict) -> list[str]:
    """Run cell 15's argument assembly with stubs for the files it checks."""
    code = block(cell, "# ---- assemble Pennyroyal / AutoRound launch arguments ----",
                 "# ---- launch detached and wait for health ----")
    ns = {"CFG": cfg, "SGLANG": "sglang", "MODEL_DIR": "/m", "SERVED_MODEL_PORT": 8001, "DRAFT_VIEW": "/d",
          "WHEELHOUSE_DIR": "/w", "Path": Path, "TOKEN_MAP_SHA": "t", "TOKENIZER_SHA": "k",
          "find_unique": lambda *a, **k: Path("/w/hot_tokens_64k.pt"),
          "sha256": lambda p: "t" if str(p).endswith(".pt") else "k"}
    exec(code, ns)
    return ns["args"]


def flag(argv: list[str], name: str) -> str | None:
    at = [i for i, a in enumerate(argv) if a == name]
    assert len(at) <= 1, (name, at)
    return argv[at[0] + 1] if at else None


def check_patch(ns: dict, files: dict[str, bytes], patch_rels: list[str]) -> dict[str, bytes]:
    """Section 2. Returns the patched file bytes for the composition check."""
    probe_cell = built_cell("s20probe")
    code = nvfp4kv_block(probe_cell)
    assert code == nvfp4kv_block(built_cell("s20")), "probe and replica must install the same patch"
    assert probe_cell.index(builder.TARGET_ANCHOR) < probe_cell.index(code) < probe_cell.index("# ---- assemble Pennyroyal"), \
        "the patch must run after target validation and before the launch arguments are built"
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        reference = root / "git"
        for rel, data in files.items():
            (reference / rel).parent.mkdir(parents=True, exist_ok=True)
            (reference / rel).write_bytes(data)
        for name in PATCHES:
            subprocess.run(["git", "apply", "-p2", str(NV / name)], cwd=reference, check=True)
        venv, site = make_venv(root / "a", files)
        cfg, env = cfg_of(probe_cell), {}
        before = dict(cfg)
        cell_ns = {"VENV": str(venv), "env": env, "CFG": cfg}
        exec(compile(code, "cell15-nvfp4kv", "exec"), cell_ns)
        assert env == {"SGLANG_QSA_NVFP4_KV": "1"}, env
        assert cfg == dict(before, KVDTYPE="nvfp4", DRAFT_KVDTYPE="fp8_e4m3", MAX_TOTAL_TOKENS=POOL), cfg
        patched = {rel: (site / rel).read_bytes() for rel in patch_rels}
        assert patched == {rel: (reference / rel).read_bytes() for rel in patch_rels}, "differs from git apply"
        for rel, data in patched.items():
            compile(data.decode("utf-8"), rel, "exec")
        changed = sorted(rel for rel in patch_rels if files.get(rel) != patched[rel])
        assert changed == sorted(patch_rels), changed
        text = {rel: data.decode("utf-8") for rel, data in patched.items()}
        assert ns["KV4_LINE"] in text["sglang/srt/server_args.py"]
        assert "quant_method.__class__ = QSANVFP4KVCacheMethod" in text["sglang/srt/mem_cache/kv_cache_configurator.py"]
        assert "if not self._mamba_admission_ok(req, adder.can_run_list):" in text["sglang/srt/managers/scheduler.py"]
        print(f"ok   inserted cell-15 block: {len(patch_rels)} files equal `git apply -p2` byte for byte "
              f"({sum(len(ns['hunks'](NV.joinpath(n).read_text())) for n in PATCHES)} hunks) and compile; "
              "switch on; CFG gets nvfp4 / fp8_e4m3 / pool cap")

        state = snapshot(site)
        exec(compile(code, "cell15-nvfp4kv", "exec"), {"VENV": str(venv), "env": {}, "CFG": dict(before)})
        assert snapshot(site) == state
        print("ok   rerun leaves every file unchanged")

        # A drifted hunk anchor, and a file with one hunk of several applied, both stop the cell before any write.
        conf, attn = "sglang/srt/mem_cache/kv_cache_configurator.py", "sglang/srt/layers/attention/qwen_sparse_attn_backend.py"
        old_conf = next(o for r, o, _ in ns["hunks"](NV.joinpath(PATCHES[0]).read_text()) if r == conf)
        old_attn, new_attn = next((o, n) for r, o, n in ns["hunks"](NV.joinpath(PATCHES[0]).read_text()) if r == attn)
        for label, rel, edit, why in (
                ("drifted", conf, lambda t: t.replace(old_conf, old_conf.replace("QSATokenToKVPool", "QSATokenToKVPoolX", 1)),
                 "does not match the Pennyroyal v2.5.3 source"),
                ("half patched", attn, lambda t: t.replace(old_attn, new_attn), "half patched")):
            bad = dict(files)
            bad[rel] = edit(files[rel].decode("utf-8")).encode("utf-8")
            assert bad[rel] != files[rel]
            venv2, site2 = make_venv(root / label.replace(" ", "-"), bad)
            state, env2, cfg2 = snapshot(site2), {}, dict(before)
            try:
                exec(compile(code, "cell15-nvfp4kv", "exec"), {"VENV": str(venv2), "env": env2, "CFG": cfg2})
                raise AssertionError(f"a {label} {Path(rel).name} must raise")
            except RuntimeError as exc:
                assert why in str(exc), exc
            assert snapshot(site2) == state and env2 == {} and cfg2 == before
        print("ok   a drifted anchor or a half-patched file raises before any write; switch off, CFG unchanged")
    return patched


def check_composition(ns: dict, files: dict[str, bytes], patched: dict[str, bytes]) -> None:
    """Section 3: nvfp4kv, p448 and fastdec patch the installed SGLang in any order."""
    p448, fastdec = load(OVERLAYS / "p448_serve.py"), load(OVERLAYS / "fastdec_serve.py")
    patches = {name: NV.joinpath(name).read_text() for name in PATCHES}
    run = {"nvfp4kv": lambda v: ns["patch_sglang"](v, patches), "p448": p448["patch_sglang"],
           "fastdec": fastdec["patch_sglang"]}
    with tempfile.TemporaryDirectory() as tmp:
        def state(order, tag):
            venv, site = make_venv(Path(tmp) / tag, files)
            for name in order:
                run[name](venv)
            return snapshot(site)
        for n, order in enumerate(orders(run, "nvfp4kv")):
            got, without = state(order, f"order{n}"), state([x for x in order if x != "nvfp4kv"], f"without{n}")
            assert set(got) == set(without) | set(patched), order
            for rel, data in got.items():
                assert data == (patched[rel] if rel in patched else without[rel]), (order, rel)
        print(f"ok   composes with p448 and fastdec in {n + 1} overlay orders: its files are the same in every order "
              "and every other file is exactly what those patches leave without it")


def check_arms(ns: dict) -> None:
    """Section 4: the arms build and cell 15's argv carries the split dtypes."""
    for arm in ("s20probe", "s20"):
        assert arm in builder.ARMS, f"missing arm {arm}"
    stacks = {"s20 without wastefix": builder.STARTUP + builder.p448() + builder.s20() + builder.dprime(),
              "s20 with wastefix": builder.STARTUP + builder.p448() + builder.s20() + builder.dprime() + builder.overlay("wastefix")}
    for label, edits in stacks.items():
        nb = json.loads((builder.BASE / "submission.ipynb").read_text())
        builder.apply_edits(nb, builder.REPLICA + edits)
    for arm in ("s20probe", "s20"):
        cell = built_cell(arm)
        compile(cell, f"{arm}:cell15", "exec")
        cfg = cfg_of(cell)
        cfg.update(ns["SERVER_CFG"], MAX_TOTAL_TOKENS=POOL)  # what the inserted apply() returns
        argv = launch_argv(cell, cfg)
        assert flag(argv, "--kv-cache-dtype") == "nvfp4"
        assert flag(argv, "--speculative-draft-kv-cache-dtype") == "fp8_e4m3"
        assert flag(argv, "--max-total-tokens") == str(POOL)
        assert flag(argv, "--max-running-requests") == "20" and flag(argv, "--cuda-graph-max-bs-decode") == "20"
        assert flag(argv, "--max-mamba-cache-size") == "100"
        graphs = argv[argv.index("--cuda-graph-bs-decode") + 1:argv.index("--mamba-ssm-dtype")]
        assert set(map(str, range(11, 21))) <= set(graphs), graphs
        reference = ns["reference_args"](argv)
        assert [i for i, (a, b) in enumerate(zip(argv, reference)) if a != b] == [argv.index("--kv-cache-dtype") + 1]
        assert flag(reference, "--kv-cache-dtype") == "fp8_e4m3" and len(reference) == len(argv)
    pinned = built_cell("p448s14dp")
    argv = launch_argv(pinned, cfg_of(pinned))
    assert flag(argv, "--kv-cache-dtype") == flag(argv, "--speculative-draft-kv-cache-dtype") == "fp8_e4m3"
    assert flag(argv, "--max-total-tokens") is None
    print("ok   arms build with and without wastefix; argv: --kv-cache-dtype nvfp4, --speculative-draft-kv-cache-dtype "
          f"fp8_e4m3, --max-total-tokens {POOL}, 20 streams, graphs 11-20, 100 Mamba slots; the FP8 reference differs "
          "only in the target dtype; the pinned arm is untouched")

    cell = built_cell("s20probe")
    launch = block(cell, "args_nvfp4 = args\n", "precache_model_thread.start()")
    parity_block = cell[cell.index("if not TRUE_SUBMISSION:  # arm nvfp4kv probe: FP8 vs NVFP4"):]
    body = ast.parse(cell).body
    assert isinstance(body[-2], ast.While) and "NOTEBOOK_START_TIME" in ast.unparse(body[-2].test)
    assert ast.unparse(body[-1]).startswith("if not TRUE_SUBMISSION:"), "parity must follow cell 15's READY wait"
    assert "args_nvfp4" not in built_cell("s20"), "the replica arm must not carry the probe"
    nv_argv = ["sglang", "--kv-cache-dtype", "nvfp4", "--speculative-draft-kv-cache-dtype", "fp8_e4m3"]
    for scored in (False, True):
        calls = []
        stub = {"reference_args": ns["reference_args"],
                "parity": lambda *a: calls.append(a) or ("new-proc", 123)}
        cell_ns = {"TRUE_SUBMISSION": scored, "args": list(nv_argv), "_nvfp4kv": stub, "json": json,
                   "precache_model_thread": type("T", (), {"start": lambda self: None})(),
                   "proc": "fp8-proc", "env": {}, "LOG": "serve.log", "url": "http://127.0.0.1:8001/health", "log_offset": 0}
        exec(launch, cell_ns)
        assert flag(cell_ns["args"], "--kv-cache-dtype") == ("nvfp4" if scored else "fp8_e4m3")
        exec(parity_block, cell_ns)
        assert flag(cell_ns["args"], "--kv-cache-dtype") == "nvfp4"
        if scored:
            assert calls == [] and cell_ns["proc"] == "fp8-proc"
        else:
            (proc, args, env, log, url, prompts), = calls
            assert proc == "fp8-proc" and args == nv_argv and len(prompts["prompts"]) == 32
            assert (cell_ns["proc"], cell_ns["log_offset"]) == ("new-proc", 123)
    print("ok   probe: FP8 boots first and parity() runs after the READY wait on 32 prompts, then the cell keeps the "
          "NVFP4 server; a scored rerun launches NVFP4 directly and skips both")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextlib.contextmanager
def fake_tools():
    """A temp dir with the fake server and an nvidia-smi on PATH that reports a free GPU (the module waits for
    the GPU to give its memory back before it relaunches a server)."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "fake_sglang.py").write_text(FAKE_SERVER)
        smi = root / "bin/nvidia-smi"
        smi.parent.mkdir()
        smi.write_text("#!/bin/sh\necho 0\n")
        smi.chmod(0o755)
        old_path = os.environ["PATH"]
        os.environ["PATH"] = f"{smi.parent}{os.pathsep}{old_path}"
        try:
            yield root, root / "fake_sglang.py"
        finally:
            os.environ["PATH"] = old_path


def kill(procs, fake: Path, port: int) -> None:
    for p in procs:
        if p is not None and p.poll() is None:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
    subprocess.run(["pkill", "-f", f"{fake} {port} "], check=False)  # servers the module started itself


def check_parity(ns: dict) -> None:
    """Section 5: parity() end to end against fake servers launched the way cell 15 launches SGLang."""
    prompts = json.loads(NV.joinpath("parity_prompts.json").read_text())
    assert len(prompts["prompts"]) == 32 and len({p["source"] for p in prompts["prompts"]}) == 32
    ns["POLL_S"] = 0.05
    with fake_tools() as (root, fake):
        for mode, want in (("nvfp4", None), ("nvfp4-noisy", None), ("nvfp4-ties", None),
                           ("nvfp4-drift", r"FAILED: .*NLL change \(nvfp4 minus fp8\) \+0\.0500"),
                           ("nvfp4-flip", r"FAILED: .*top-1 agreement (8\d|9[0-4])\.\d\d%"),
                           ("nvfp4-decode", r"FAILED: .*median of 1 tokens"), ("nvfp4-vacuous", "vacuous"),
                           ("nvfp4-noswitch", "never logged")):
            port, log = free_port(), root / f"serve-{mode}.log"
            with open(log, "ab", buffering=0) as logf:
                ref = subprocess.Popen([sys.executable, str(fake), str(port), "fp8"], stdout=logf,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            procs = [ref]
            try:
                result = ns["parity"](ref, [sys.executable, str(fake), str(port), mode], dict(os.environ), log,
                                      f"http://127.0.0.1:{port}/health", prompts, timeout=30)
                assert want is None, f"{mode} must raise"
                proc, offset = result
                procs.append(proc)
                assert ref.poll() is not None and proc.poll() is None and offset == 0
                kv4 = ns["KV4_LINE"]
                reference = log.with_name(log.stem + "-fp8-reference.log").read_text()
                assert "#tokens" in reference and kv4 not in reference and kv4 in log.read_text()
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
                    assert r.status == 200
            except RuntimeError as exc:
                assert want is not None and re.search(want, str(exc)), (mode, exc)
            finally:
                kill(procs, fake, port)
    print("ok   parity(): passes at +0.013 nats/token (Lesilva's worst), on unbiased noise of |dlogp| 0.05 and at 96% "
          "top-1 agreement, keeps the NVFP4 server and leaves serve.log to its boot (FP8 boot in "
          "serve-fp8-reference.log); raises at +0.05 nats/token, at 92% agreement, on a broken decode path, on an "
          "uncached (vacuous) comparison, and when the NVFP4 boot never logs the QSA switch")


def check_fallback(ns: dict) -> None:
    """Section 6: cell 15's own launch and READY-wait code, with fake servers, in a scored run and a bench run."""
    assert "_nvfp4kv['relaunch_fp8']" in built_cell("s20probe"), "cell 15's READY wait has no FP8 relaunch"
    code = block(built_cell("s20"), "# ---- launch detached and wait for health ----", builder.SERVE_END)
    ns["POLL_S"] = 0.05
    with fake_tools() as (root, fake):
        # (scored run, fake mode, expect a relaunch on FP8, expect a live READY server at the end)
        for scored, mode, relaunch, up in ((True, "crash-nvfp4", True, True), (True, "nvfp4", False, True),
                                           (True, "crash", True, False), (False, "crash-nvfp4", False, False)):
            port, log, out = free_port(), root / f"serve-{scored}-{mode}.log", io.StringIO()
            cell_ns = {"TRUE_SUBMISSION": scored, "_nvfp4kv": ns, "env": dict(os.environ), "LOG": str(log),
                       "args": [sys.executable, str(fake), str(port), mode, "--kv-cache-dtype", "nvfp4"],
                       "SERVED_MODEL_PORT": port, "NOTEBOOK_START_TIME": time.time(), "SERVER_STARTUP_TIMEOUT": 20,
                       "precache_model_thread": types.SimpleNamespace(start=lambda: None), "subprocess": subprocess,
                       "urllib": urllib, "Path": Path, "shlex": shlex,
                       "time": types.SimpleNamespace(time=time.time, sleep=lambda s: time.sleep(0.05))}
            try:
                with contextlib.redirect_stdout(out):
                    exec(compile(code, "cell15-launch", "exec"), cell_ns)
                dtype = flag(cell_ns["args"], "--kv-cache-dtype")
                assert dtype == ("fp8_e4m3" if relaunch else "nvfp4"), (scored, mode, dtype)
                assert ("relaunched once on FP8 KV" in out.getvalue()) == relaunch, out.getvalue()
                assert ("READY after" in out.getvalue()) == up and (cell_ns["proc"].poll() is None) == up, out.getvalue()
            finally:
                kill([cell_ns.get("proc")], fake, port)
    print("ok   scored run: an NVFP4 server that dies before READY is relaunched once on FP8 KV and the cell waits for "
          "it; a healthy boot, a second death and a bench run are left to the cell's own loop")


def kv_scales_check() -> None:
    """--kv-scales: the target checkpoint carries no k/v scales (network, about 24 MB, nothing written)."""
    record = json.loads(NV.joinpath("PROVENANCE.json").read_text())["checkpoint_kv_scales"]
    repo, rev = (s.strip() for s in record["source_checkpoint"].split("@"))
    get = lambda name: urllib.request.urlopen(f"https://huggingface.co/{repo}/resolve/{rev}/{name}", timeout=600).read()
    cfg, qcfg, index = get("config.json"), get("quantization_config.json"), get("model.safetensors.index.json")
    assert (sha(cfg), sha(qcfg), sha(index)) == (record["config_json_sha256"], record["quantization_config_json_sha256"],
                                                 record["index_sha256"])
    names = list(json.loads(index)["weight_map"])
    hits = [n for n in names if re.search(r"(k_scale|v_scale|kv_scale|output_scale|q_scale|prob_scale)", n)]
    quant = json.loads(cfg)["quantization_config"]
    assert not hits and len(names) == record["tensors"], hits[:5]
    assert quant["quant_method"] == "auto-round" and quant.get("act_bits") is None and "static_kv_dtype" not in quant
    token = REPO / "ARC-AGI-3-Kaggle-Starter/.kaggle/access_token"
    token = token if token.is_file() else Path.home() / ".kaggle/access_token"
    request = urllib.request.Request(
        "https://www.kaggle.com/api/v1/models/dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/transformers/default/1/files"
        "?pageSize=200", headers={"Authorization": f"Bearer {token.read_text().strip()}"})
    with urllib.request.urlopen(request, timeout=120) as r:
        listing = json.loads(r.read())
    assert not listing.get("nextPageToken"), "listing has more pages"
    sizes = {f["name"]: f["size"] for f in listing["files"]}
    assert sizes["config.json"] == len(cfg) and sizes["quantization_config.json"] == len(qcfg)
    assert not [n for n in sizes if re.search(r"scale|kv", n, re.I)]
    print(f"ok   {repo} @ {rev[:10]}: 0 of {len(names):,} tensors are k/v scales, auto-round int4 with no KV quantization; "
          f"Franzen's Kaggle copy lists {len(sizes)} files with the same config sizes and no scale file")


def main() -> None:
    prov = json.loads(NV.joinpath("PROVENANCE.json").read_text())
    for name, entry in prov["files"].items():
        assert sha(NV.joinpath(name).read_bytes()) == entry["sha256"] and entry["url"].endswith(name), name
    assert sha(NV.joinpath("parity_prompts.json").read_bytes()) == prov["parity_prompts.json"]["sha256"]
    ns = load(OVERLAYS / "nvfp4kv_serve.py")
    assert ns["PATCH_SHA256"] == {name: prov["files"][name]["sha256"] for name in PATCHES}
    print(f"ok   vendored files match PROVENANCE.json ({prov['repo']} @ {prov['revision'][:10]}, {prov['license'][:10]})")
    if "--kv-scales" in sys.argv:
        kv_scales_check()

    if not hasattr(builder, "nvfp4kv"):
        exec(BUILDER_SNIPPET, vars(builder))
    patch_rels = sorted({rel for name in PATCHES for rel, _, _ in ns["hunks"](NV.joinpath(name).read_text())})
    assert len(patch_rels) == 8, patch_rels
    present = [rel for rel in patch_rels if not rel.endswith("qsa/nvfp4_kv.py")]  # 0001 creates that one
    files = wheel_files(sorted(set(present) | set(P448_FILES) | set(FASTDEC_FILES) | {QSA_INIT}))
    patched = check_patch(ns, {rel: files[rel] for rel in present + [QSA_INIT]}, patch_rels)
    check_composition(ns, files, patched)
    check_arms(ns)
    check_parity(ns)
    check_fallback(ns)
    print("PASS")


if __name__ == "__main__":
    main()
