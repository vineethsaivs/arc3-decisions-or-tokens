"""Offline test of the lora arm's edits (build_franzen_arms.lora()). Run: python3 solution/sim/test_franzen_lora_arm.py

1. The hook block lands in cell 15 after the p448 apply and before the launch anchor; the post-READY check is after SERVE_END.
2. Only the named LoRA arms carry lora text (never the pin or its replica).
3. Run against a fake venv holding the real Pennyroyal qwen4_exp.py: installs the hook and sets SGLANG_LORA_DIR; a missing
   adapter file raises; a server log without 'lora: merged' raises.
4. A W arm (adapter_arms) mounts its dataset and raises unless every merge line says scale 1.0. Per member: the wave
   arm is l1a1y8sv2r3rq plus lora(), the (d) probe l1a1y8sv2r3kp plus lora(), the held-out arm l1a1y8sv2r3hob plus
   lora(), and each fails closed on a missing adapter.
5. The opt-in SR arms (wasr, wasrp, hobwasr) are the W arms plus lora(sr_seed=0): the installed SGLang carries the
   stochastic rounding, and the post check refuses a log without 'lora: stochastic rounding, seed 0'.
"""
import json
import os
import re
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "solution/kaggle"))
import build_franzen_arms as B  # noqa: E402

ARM = "l1a1y8sv2r3hm32lora"
base = json.loads((B.BASE / "submission.ipynb").read_text())


def cell15(arm):
    nb = json.loads(json.dumps(base))
    B.apply_edits(nb, B.REPLICA + B.ARMS[arm][1]())
    return B.cell_text(nb["cells"][B.SERVE])


END = "merge check deferred to READY', flush=True)"


def post_of(text):
    """The post-release check block: from def _lora_check to its last print."""
    return text[text.index("def _lora_check("):text.index(END) + len(END)]


class Health:
    """A stand-in for the model server's /health on 127.0.0.1:8001: 200 while up, connection refused while down."""
    def __init__(self):
        self.srv = None

    def up(self):
        import http.server, threading
        class H(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200); self.end_headers(); self.wfile.write(b"ok")
            def log_message(self, *a):
                pass
        self.srv = http.server.HTTPServer(("127.0.0.1", 8001), H)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def down(self):
        if self.srv:
            self.srv.shutdown(); self.srv.server_close(); self.srv = None


HEALTH = Health()
t = cell15(ARM)
assert t.index("_p448['apply']") < t.index("# arm lora") < t.index(B.LAUNCH_ANCHOR) < t.index("def _lora_check(")
assert t.index(B.SERVE_END) < t.index("def _lora_check(")  # the check runs right after the READY wait
LORA_ARMS = {ARM, "l1a1y8sv2r3hobg1", "l1a1y8sv2r3g1", "l1a1y8sv2r3g1p", "l1a1y8sv2r3hobr16"} | {
    a for a in B.ARMS if re.fullmatch(r"l1a1y8sv2r3(w[a-z]3?(sr)?p?|hobw[a-z]3?(sr)?)", a)}
assert {"l1a1y8sv2r3wa", "l1a1y8sv2r3wap", "l1a1y8sv2r3hobwa", "l1a1y8sv2r3we", "l1a1y8sv2r3hobwe"} <= LORA_ARMS
for arm in B.ARMS:
    if arm not in LORA_ARMS:
        assert "arm lora" not in "".join(n for _, _, n in B.ARMS[arm][1]()), arm

# 3. execute the pre-launch block on a fake venv
pre = t[t.index("# arm lora: fold"):t.index(B.LAUNCH_ANCHOR)]
post = post_of(t)
with tempfile.TemporaryDirectory() as d:
    d = Path(d)
    mdl = d / "venv/lib/python3.12/site-packages/sglang/srt/models"
    mdl.mkdir(parents=True)
    (mdl / "qwen4_exp.py").write_text((REPO / "solution/sim/fixtures/pennyroyal-v253/qwen4_exp.py").read_text())
    ad = d / "adapter"
    ad.mkdir()
    code = pre.replace(repr(B.LORA_ADAPTER), repr(str(ad)))
    assert str(ad) in code

    def run(env):
        exec(compile(code, "pre", "exec"), {"os": os, "VENV": str(d / "venv"), "env": env})

    env = {}
    try:
        run(env)
        raise SystemExit("FAIL: ran with no adapter files")
    except RuntimeError as e:
        assert "missing" in str(e), e
    assert "SGLANG_LORA_DIR" not in env
    (ad / "adapter_model.safetensors").write_bytes(b"x")
    (ad / "adapter_config.json").write_text("{}")
    run(env)
    assert env["SGLANG_LORA_DIR"] == str(ad) and "arm lora" in (mdl / "qwen4_exp.py").read_text()
    run({})  # idempotent: a second launch (watchdog, re-boot) does not double-patch
    log = d / "serve.log"
    import time as _time, types
    killed = []
    fake_os = types.SimpleNamespace(kill=lambda pid, sig: killed.append(sig), getpid=os.getpid)
    g = {"LOG": str(log), "Path": Path, "os": fake_os, "time": _time}
    HEALTH.up()
    log.write_text("server up\n")
    try:
        exec(post, dict(g))
        raise SystemExit("FAIL: READY with no merge line, no raise")
    except RuntimeError as e:
        assert "merge" in str(e), e
    log.write_text("lora: merged 300 tensors, scale 2.0\n")
    exec(post, dict(g))  # r16 arm: no scale pinned
    # still loading at release (no health, no line): no raise, a guard thread waits for READY
    HEALTH.down()
    log.write_text("loading\n")
    exec(post, dict(g))
    log.write_text("loading\nlora: merged 300 tensors, scale 2.0\n")
    _time.sleep(6)
    assert not killed, "guard killed a run whose merge landed after release"
    log.write_text("loading\n")
    exec(post, dict(g))
    HEALTH.up()
    _time.sleep(6)
    assert killed == [9], f"guard did not kill a run that came READY without the merge line: {killed}"
    killed.clear()

    # 4. W arms: wave arm on rq, (d) probe on kp, held-out arm on hob; dataset mounted, scale pinned to 1.0, fail closed
    family = []
    for tag in ("wa", "wb", "wc", "wd", "we"):
        wpath = f"/kaggle/input/datasets/vineethsaivs/arc3-lora-grpo-{tag}/"
        assert B.OUR_DATASETS[wpath] == f"vineethsaivs/arc3-lora-grpo-{tag}"
        for arm, base_arm in ((f"l1a1y8sv2r3{tag}", "l1a1y8sv2r3rq"), (f"l1a1y8sv2r3{tag}p", "l1a1y8sv2r3kp"),
                              (f"l1a1y8sv2r3hob{tag}", "l1a1y8sv2r3hob")):
            assert B.ARMS[arm][1]() == B.ARMS[base_arm][1]() + B.lora(wpath, 1.0), arm
            family.append((arm, wpath))
    empty = d / "no-adapter"
    empty.mkdir()
    for arm, wpath in family:
        tw = cell15(arm)
        assert repr(wpath) in tw
        wpre = tw[tw.index("# arm lora: fold"):tw.index(B.LAUNCH_ANCHOR)].replace(repr(wpath), repr(str(empty) + "/"))
        try:
            exec(compile(wpre, arm, "exec"), {"os": os, "VENV": str(d / "venv"), "env": {}})
            raise SystemExit(f"FAIL: {arm} ran with no adapter files")
        except RuntimeError as e:
            assert "missing" in str(e), e
        wpost = post_of(tw)
        assert "{1.0} is not None and _s != {1.0}" in wpost
        for text, ok in (("lora: merged 300 tensors, scale 1.0\n", True), ("lora: merged 300 tensors, scale 2.0\n", False),
                         ("lora: merged 300 tensors, scale 1.0\nlora: merged 300 tensors, scale 2.0\n", False),
                         ("server up\n", False)):
            log.write_text(text)
            try:
                exec(wpost, dict(g))
                assert ok, f"{arm}: no raise on {text!r}"
            except RuntimeError:
                assert not ok, f"{arm}: raised on {text!r}"
    # a W arm still loading at release, then READY at the wrong scale: the guard kills the run
    HEALTH.down()
    log.write_text("loading\n")
    exec(post_of(cell15("l1a1y8sv2r3wbp")), dict(g))
    log.write_text("loading\nlora: merged 300 tensors, scale 2.0\n")
    HEALTH.up()
    _time.sleep(6)
    assert killed == [9], f"W guard did not kill a mis-scaled run: {killed}"
    HEALTH.down()

    # 5. SR arms: W arm plus the stochastic-rounding patch, installed and checked; sr_seed=None is the plain lora()
    wapath = "/kaggle/input/datasets/vineethsaivs/arc3-lora-grpo-wa/"
    assert B.lora(wapath, 1.0) == B.lora(wapath, 1.0, None)
    for arm, base_arm in (("l1a1y8sv2r3wasr", "l1a1y8sv2r3rq"), ("l1a1y8sv2r3wasrp", "l1a1y8sv2r3kp"),
                          ("l1a1y8sv2r3hobwasr", "l1a1y8sv2r3hob")):
        assert B.ARMS[arm][1]() == B.ARMS[base_arm][1]() + B.lora(wapath, 1.0, 0), arm
    ts = cell15("l1a1y8sv2r3wasr")
    smdl = d / "srvenv/lib/python3.12/site-packages/sglang/srt/models"
    smdl.mkdir(parents=True)
    (smdl / "qwen4_exp.py").write_text((REPO / "solution/sim/fixtures/pennyroyal-v253/qwen4_exp.py").read_text())
    senv = {}
    exec(compile(ts[ts.index("# arm lora: fold"):ts.index(B.LAUNCH_ANCHOR)].replace(repr(wapath), repr(str(ad))), "srpre",
                 "exec"), {"os": os, "VENV": str(d / "srvenv"), "env": senv})
    inst = (smdl / "qwen4_exp.py").read_text()
    assert senv["SGLANG_LORA_DIR"] == str(ad) and "_lora_sr_bf16(tensor.float() + delta, 0, name)  # arm lora sr" in inst
    compile(inst, "qwen4_exp_sr", "exec")
    for text, ok in (("lora: stochastic rounding, seed 0\nlora: merged 300 tensors, scale 1.0\n", True),
                     ("lora: merged 300 tensors, scale 1.0\n", False),
                     ("lora: stochastic rounding, seed 1\nlora: merged 300 tensors, scale 1.0\n", False)):
        log.write_text(text)
        try:
            exec(post_of(ts), dict(g))
            assert ok, f"SR arm: no raise on {text!r}"
        except RuntimeError:
            assert not ok, f"SR arm: raised on {text!r}"

    # 6. set_scale (round C WE3): the merge reads a /tmp config copy with lora_alpha = 3r; the weights are a symlink
    wepath = "/kaggle/input/datasets/vineethsaivs/arc3-lora-grpo-we/"
    assert B.ARMS["l1a1y8sv2r3we3sr"][1]() == B.ARMS["l1a1y8sv2r3rq"][1]() + B.lora(wepath, 3.0, 0, set_scale=True)
    assert B.ARMS["l1a1y8sv2r3we3srp"][1]() == B.ARMS["l1a1y8sv2r3kp"][1]() + B.lora(wepath, 3.0, 0, set_scale=True)
    import json
    (ad / "adapter_config.json").write_text(json.dumps({"r": 16, "lora_alpha": 16.0}))
    t3 = cell15("l1a1y8sv2r3we3sr")
    s3 = d / "s3venv/lib/python3.12/site-packages/sglang/srt/models"
    s3.mkdir(parents=True)
    (s3 / "qwen4_exp.py").write_text((REPO / "solution/sim/fixtures/pennyroyal-v253/qwen4_exp.py").read_text())
    env3 = {}
    exec(compile(t3[t3.index("# arm lora: fold"):t3.index(B.LAUNCH_ANCHOR)].replace(repr(wepath), repr(str(ad))), "s3pre",
                 "exec"), {"os": os, "VENV": str(d / "s3venv"), "env": env3})
    sd = Path(env3["SGLANG_LORA_DIR"])
    assert sd != ad and json.loads((sd / "adapter_config.json").read_text())["lora_alpha"] == 48.0
    assert (sd / "adapter_model.safetensors").resolve() == (ad / "adapter_model.safetensors").resolve()
    for text, ok in (("lora: stochastic rounding, seed 0\nlora: merged 300 tensors, scale 3.0\n", True),
                     ("lora: stochastic rounding, seed 0\nlora: merged 300 tensors, scale 1.0\n", False)):
        log.write_text(text)
        try:
            exec(post_of(t3), dict(g))
            assert ok, f"scale-3 arm: no raise on {text!r}"
        except RuntimeError:
            assert not ok, f"scale-3 arm: raised on {text!r}"
print("PASS test_franzen_lora_arm")
