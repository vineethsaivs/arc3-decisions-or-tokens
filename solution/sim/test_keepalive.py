"""The warmup-only keepalive (franzen_overlays/keepalive.py) on the transport the scored run uses.

    python3 solution/sim/test_keepalive.py      # about 5 minutes, no GPU, no network

Re-runs itself under reference/re-arc-3/.venv/bin/python (or $KEEPALIVE_PYTHON) when this interpreter
lacks the engine's dependencies; arc_agi 0.9.8 and arcengine 0.9.3 come straight from the Kaggle wheels
in solution/dev/cloud_public25_2026_09_13/engine-wheels.

The gateway is taaf's localhost CompetitionArcadeServer (the arc_agi REST app in competition mode) with
STALE_MINUTES=1 and arc_agi's own idle close (api.scorecard_cleanup_loop, polled every second instead of
every minute). The harness is Franzen's patched tree with the arm's overlays applied in the order its
notebook runs them. The analyzer is the real ToolAgent over HTTP against a stub model server that refuses
connections until READY and then answers every request with one python call that plays a move. The retry
loop is the real _post_with_retries under the arm's settings (ARC3_HTTP_RETRY_INITIAL_SECONDS=900 from
cell 7, ARC3_HTTP_RETRIES=360 from the watchdog).

Scaled clock. STALE_MINUTES=1 makes 60 real seconds stand for the gateway's 15 minutes, so everything runs
15 times faster and a minute below is 4 real seconds: the keepalive interval, the retry loop's sleeps and
clock, and the solver's 10 s backoff. Nothing else changes.

Scenarios, in parallel subprocesses (minutes from the release):
  control  the pin (r3p) without the keepalive, server never READY: the card closes 15 min after the warmup
  never    r3kp, server never READY: RESETs at 9, 18, 27, 36, 45, then the cap; the card closes 15 min later
  ready38  r3kp, READY at 38 (50 from notebook start): 4 RESETs, none after READY, every game plays
  before   r3kp, READY before the release: no keepalive RESET, every game plays
  raises   r3kp whose keepalive RESET raises, READY at 12: the error is logged, every game plays
Checks (docs/research/2026-10-19-top/PLAN.md 2.5): (1) RESETs at most 9 minutes apart and at most 5;
(2) none after READY; (3) only on the warmup game; (4) the card closes without the keepalive and stays open
with it; (5) READY at 50 minutes: no game gives up (the analyzer give-up branch, solver_note "analyzer
failed ..."; taaf's own "gave_up" is just the end state of any unwon game) and none crashes; (6) an exception
in the thread is logged and play continues; (7) READY before the release sends none. Before those: the
overlay is all or nothing and never raises, and the builder's r3kp is r3p plus this one overlay, run first.
"""
from __future__ import annotations

import errno
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
ENGINE_PY = Path(os.environ.get("KEEPALIVE_PYTHON", REPO / "reference/re-arc-3/.venv/bin/python"))
WHEELS = REPO / "solution/dev/cloud_public25_2026_09_13/engine-wheels"
ENV_DIR = REPO / "ARC-AGI-3-Kaggle-Starter/environment_files"
OVERLAYS = REPO / "solution/kaggle/franzen_overlays"
SOLVER = Path("src/ARC3-Inference/inference/framework/solver.py")
SCALE = 15.0  # STALE_MINUTES=1: 60 real seconds stand for 15 minutes
MINUTE = 60.0 / SCALE  # real seconds per scaled minute
GAMES = ("cd82", "sp80", "ls20")  # plain moves, no clicks
ARM, BASE_ARM = "l1a1y8sv2r3kp", "l1a1y8sv2r3p"
SCENARIOS = {  # tree, READY (scaled minutes after release, "before", or None), keepalive RESET raises, end
    "control": dict(tree=BASE_ARM, ready=None, raises=False, until=25.0),
    "never": dict(tree=ARM, ready=None, raises=False, until=70.0),
    "ready38": dict(tree=ARM, ready=38.0, raises=False, until=60.0),
    "before": dict(tree=ARM, ready="before", raises=False, until=15.0),
    "raises": dict(tree=ARM, ready=12.0, raises=True, until=30.0),
}
# the stub model's only reply: one python call that plays the first plain move
CODE = ("moves = [a for a in valid_actions if a in ('UP', 'DOWN', 'LEFT', 'RIGHT', 'SPACE')]\n"
        "action([moves[0] if moves else 'SPACE'])\n")


def _reexec() -> None:
    missing = []
    for name in ("numpy", "pydantic", "flask", "matplotlib", "imageio", "dotenv", "requests", "PIL"):
        try:
            __import__(name)
        except ImportError:
            missing.append(name)
    if not missing:
        return
    if os.environ.get("_KEEPALIVE_CHILD") or not ENGINE_PY.exists():
        raise SystemExit(f"{sys.executable} lacks {', '.join(missing)}; set KEEPALIVE_PYTHON to a python 3.12 "
                         f"with arc_agi's dependencies ({ENGINE_PY} not usable)")
    env = dict(os.environ, _KEEPALIVE_CHILD="1", PYTHONDONTWRITEBYTECODE="1")
    raise SystemExit(subprocess.run([str(ENGINE_PY), str(Path(__file__).resolve()), *sys.argv[1:]],
                                    env=env).returncode)


# ---------------------------------------------------------------- parent: structure and trees

def builder():
    sys.path.insert(0, str(REPO / "solution/kaggle"))
    import build_franzen_arms as B  # noqa: PLC0415
    return B


def arm_cells(B, arm: str) -> list[str]:
    nb = json.loads((B.BASE / "submission.ipynb").read_text())
    B.apply_edits(nb, B.REPLICA + B.ARMS[arm][1]())
    return [B.cell_text(cell) for cell in nb["cells"]]


def overlay_order(B, cells: list[str]) -> list[str]:
    """The bundle overlays in the order the notebook's cell 7 runs them."""
    return re.findall(r"^# arm overlay: (\w+)$", cells[B.ENV], re.M)


def apply_overlay(name: str, tree: Path) -> str:
    """exec one overlay the way cell 7 does; returns what it printed."""
    import contextlib  # noqa: PLC0415
    import io  # noqa: PLC0415
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile((OVERLAYS / f"{name}.py").read_text(), f"franzen_overlays/{name}.py", "exec"),
             {"BUNDLE_DIR": tree})
    return out.getvalue()


def copy_tree(pristine: Path, dst: Path) -> Path:
    shutil.copytree(pristine, dst, ignore=shutil.ignore_patterns("__pycache__"))
    return dst


def structure_checks(B, pristine: Path, tmp: Path) -> int:
    # the overlay: three edits, refuses a second application, all or nothing, never raises
    tree = copy_tree(pristine, tmp / "unit")
    before = (tree / SOLVER).read_text()
    said = apply_overlay("keepalive", tree)
    after = (tree / SOLVER).read_text()
    assert said.startswith("arm overlay keepalive:"), said
    assert after.count("_keepalive_start(self)") == 1 and after.count("_keepalive_halt(self)") == 1, "edits"
    compile(after, str(SOLVER), "exec")
    assert "NOT applied (already applied)" in apply_overlay("keepalive", tree) and (tree / SOLVER).read_text() == after
    broken = before.replace("        previous_grid = _grid_from_state(self.game.current_state)\n",
                            "        previous_grid = _grid_from_state(self.game.current_state)  # moved\n")
    (tree / SOLVER).write_text(broken)
    said = apply_overlay("keepalive", tree)
    assert "NOT applied (anchor matches 0 times" in said and (tree / SOLVER).read_text() == broken, said
    shutil.rmtree(tree)
    print("ok  overlay: three edits that compile; a second application and a missing anchor each leave solver.py "
          "unchanged and print why, with no exception")

    # the builder: r3kp is r3p plus this overlay, run first; r3k is r3 plus it
    keep = B.overlay("keepalive")
    assert B.ARMS[ARM][1]() == B.ARMS[BASE_ARM][1]() + keep
    assert B.ARMS["l1a1y8sv2r3k"][1]() == B.ARMS["l1a1y8sv2r3"][1]() + keep
    arm, base = arm_cells(B, ARM), arm_cells(B, BASE_ARM)
    assert [i for i, (a, b) in enumerate(zip(arm, base)) if a != b] == [B.ENV]
    assert overlay_order(B, arm) == ["keepalive"] + overlay_order(B, base), overlay_order(B, arm)
    marker = "print('harness patch applied successfully')\n# arm overlay: keepalive\n"
    assert marker in arm[B.ENV] and arm[B.ENV].replace(keep[0][2], keep[0][1]) == base[B.ENV]
    print(f"ok  builder: {ARM} differs from {BASE_ARM} only in cell 7, by the keepalive exec right after the "
          f"harness patch (it runs first, then {', '.join(overlay_order(B, base))})")
    return 2


def build_tree(B, pristine: Path, tmp: Path, arm: str) -> Path:
    tree = copy_tree(pristine, tmp / arm)
    for name in overlay_order(B, arm_cells(B, arm)):
        said = apply_overlay(name, tree)
        assert "NOT applied" not in said, (name, said)
    return tree


def arm_env(B, arm: str) -> dict[str, str]:
    """What the arm's cells 7 and 15 put in the environment before the benchmark starts."""
    cells = arm_cells(B, arm)
    cell7 = cells[B.ENV]
    for line in ("os.environ['ARC3_WARMUP_ACTION_GAMES'] = '1'\n",
                 "os.environ['ARC3_HTTP_RETRY_INITIAL_SECONDS'] = '900'\n",
                 'os.environ["ONLY_RESET_LEVELS"] = "true"\n'):
        assert line in cell7, line
    start, end = "setup_env = {", "os.environ.update({k: str(v) for k,v in setup_env.items()})"
    ns = {"SERVER_BASE_URL": "http://127.0.0.1:9/v1", "SERVED_MODEL_NAME": "flashnext",
          "USE_PRIORITY_SCHEDULING": True}
    exec(cell7[cell7.index(start):cell7.index(end)], ns)
    env = {k: str(v) for k, v in ns["setup_env"].items()}
    env.update(ARC3_WARMUP_ACTION_GAMES="1", ARC3_HTTP_RETRY_INITIAL_SECONDS="900", ONLY_RESET_LEVELS="true",
               TAAF_RUN_AS_SUBMISSION="1", TAAF_MINIMAL_DIAGNOSTICS="1", MPLBACKEND="Agg")
    assert "_watchdog['start'](" in cells[B.SERVE], "the watchdog sets ARC3_HTTP_RETRIES"
    env["ARC3_HTTP_RETRIES"] = re.search(r'HTTP_RETRIES = "(\d+)"', (OVERLAYS / "watchdog.py").read_text()).group(1)
    return env


# ---------------------------------------------------------------- the stub model server

class Model:
    """Nothing on its port until ready(), so every connection is refused at once, as while SGLang loads (a
    socket bound but not listening is no stand-in: macOS lets the connect hang); then /health is 200 and
    every chat request gets one python call that plays a move."""

    def __init__(self) -> None:
        from http.server import BaseHTTPRequestHandler  # noqa: PLC0415

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self._send(200 if self.path == "/health" else 404, b"{}")

            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length") or 0))
                call = {"id": f"call_{time.monotonic_ns()}", "type": "function",
                        "function": {"name": "python", "arguments": json.dumps({"code": CODE})}}
                reply = {"id": "stub", "object": "chat.completion", "model": "flashnext",
                         "choices": [{"index": 0, "finish_reason": "tool_calls",
                                      "message": {"role": "assistant", "content": "",
                                                  "reasoning_content": "Play the first plain move.",
                                                  "tool_calls": [call]}}],
                         "usage": {"prompt_tokens": 1000, "completion_tokens": 20, "total_tokens": 1020}}
                self._send(200, json.dumps(reply).encode())

            def _send(self, status: int, body: bytes) -> None:
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self.handler = Handler
        with socket.socket() as spare:  # a free port, released again until READY
            spare.bind(("127.0.0.1", 0))
            self.port = spare.getsockname()[1]
        self.server = None
        self.ready_at: float | None = None

    def ready(self) -> None:
        from http.server import ThreadingHTTPServer  # noqa: PLC0415
        self.server = ThreadingHTTPServer(("127.0.0.1", self.port), self.handler)
        self.server.daemon_threads = True
        self.ready_at = time.monotonic()
        threading.Thread(target=self.server.serve_forever, daemon=True, name="stub-model").start()


def stub_check() -> int:
    model = Model()
    with socket.socket() as probe:
        assert probe.connect_ex(("127.0.0.1", model.port)) == errno.ECONNREFUSED, "a loading server must refuse"
    model.ready()
    import urllib.request  # noqa: PLC0415
    with urllib.request.urlopen(f"http://127.0.0.1:{model.port}/health", timeout=5) as response:
        assert response.status == 200
    model.server.shutdown()
    print("ok  stub model server: connections refused until READY, then /health is 200")
    return 1


# ---------------------------------------------------------------- child: one scenario

def scenario(name: str, tree: Path, out: Path) -> None:
    """Play one scenario and write what the gateway and the harness saw. Exits without teardown: games
    stuck retrying a server that never comes up do not return for minutes."""
    spec = SCENARIOS[name]
    sys.path[:0] = [str(p) for p in sorted(WHEELS.glob("*.whl"))]  # the exact Kaggle engine
    sys.path.insert(0, str(tree / "src/tufa-arc-agi-framework/src"))
    sys.path.insert(0, str(tree / "src/ARC3-Inference"))
    import logging  # noqa: PLC0415
    import types  # noqa: PLC0415

    import arc_agi  # noqa: PLC0415
    import arc_agi.api  # noqa: PLC0415
    import arcengine  # noqa: F401, PLC0415
    import taaf.benchmark  # noqa: PLC0415
    import taaf.competition_arcade as ca  # noqa: PLC0415
    import taaf.game  # noqa: PLC0415
    import taaf.game_api as ga  # noqa: PLC0415

    assert "arc_agi-0.9.8-py3-none-any.whl" in arc_agi.__file__, arc_agi.__file__
    import inference.agent.tool_agent as ta  # noqa: PLC0415
    import inference.framework.solver as so  # noqa: PLC0415

    assert Path(so.__file__).resolve().is_relative_to(tree.resolve()), so.__file__
    t0 = time.monotonic()  # reset to the release below

    def minutes(t: float) -> float:
        return round((t - t0) / MINUTE, 3)

    records: list[dict] = []

    class Capture(logging.Handler):
        def emit(self, record):
            records.append({"t": time.monotonic(), "logger": record.name, "msg": record.getMessage()})

    logging.getLogger().addHandler(Capture(level=logging.WARNING))
    hooked: list[str] = []
    threading.excepthook = lambda args: hooked.append(f"{args.thread.name}: {args.exc_type.__name__}: {args.exc_value}")

    # the scaled clock
    if hasattr(so, "KEEPALIVE_INTERVAL_S"):
        assert so.KEEPALIVE_INTERVAL_S == 540.0 and so.KEEPALIVE_MAX_RESETS == 5
        so.KEEPALIVE_INTERVAL_S = 540.0 / SCALE
    assert so.ANALYZER_RETRY_BACKOFF_SECONDS == 10.0 / SCALE, so.ANALYZER_RETRY_BACKOFF_SECONDS
    real_post = ta._post_with_retries
    ta._post_with_retries = lambda post_fn, **kw: real_post(
        post_fn, sleep=lambda s: time.sleep(s / SCALE), monotonic=lambda: time.monotonic() * SCALE, **kw)
    arc_agi.api.time = types.SimpleNamespace(sleep=lambda s: time.sleep(1.0))  # idle-close poll: 1 s, not 60

    if spec["raises"]:  # the keepalive's RESET fails inside its thread
        real_execute = taaf.game.Game.execute_action

        def execute_action(self, action, *args, **kwargs):
            if threading.current_thread().name == "arc3-keepalive":
                raise RuntimeError("injected keepalive failure")
            return real_execute(self, action, *args, **kwargs)

        taaf.game.Game.execute_action = execute_action

    # the gateway, with every command recorded and arc_agi's idle close running
    server = ca.CompetitionArcadeServer(game_ids=GAMES, environments_dir=str(ENV_DIR)).start()
    api, calls, closes = server._api, [], []
    real_cmd = api.cmd

    def cmd(action):
        import flask  # noqa: PLC0415
        data = flask.request.get_json(silent=True) or {}
        response = real_cmd(action=action)
        calls.append({"t": time.monotonic(), "action": action.name, "game": data.get("game_id"),
                      "guid": bool(data.get("guid")), "status": response[1] if isinstance(response, tuple) else 200})
        return response

    api.cmd = cmd
    server._arcade._on_scorecard_close = lambda card: closes.append(time.monotonic())
    threading.Thread(target=api.scorecard_cleanup_loop, daemon=True, name="idle-close").start()

    model = Model()
    os.environ["LOCAL_ANALYZER_BASE_URL"] = os.environ["OPENAI_BASE_URL"] = f"http://127.0.0.1:{model.port}/v1"
    if spec["ready"] == "before":
        model.ready()

    games = [ga.GameAPI(env_name=game_id, arcade_spec=server.arcade_spec) for game_id in server.exposed_game_ids]
    solver = so.HarnessSolver(label="keepalive", model="flashnext", concurrency=len(GAMES), analyzer_timeout=900.0,
                              max_runtime_s_per_game=532 * 60 / SCALE, save_request_logs=False)
    job = Path.cwd() / f"job-{name}"  # inside the parent's temp dir
    job.mkdir()
    bm = taaf.benchmark.Benchmark(label="keepalive", games=games, solver=solver, job_dir=job)
    run: dict = {}

    def play():
        import asyncio  # noqa: PLC0415
        run["loop"] = loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(bm.run(minimal_diagnostics=True))  # the scored run's flag
            run["ended"] = "returned"
        except asyncio.CancelledError:
            run["ended"] = "stopped"
        except BaseException as exc:  # noqa: BLE001
            run["ended"] = f"{type(exc).__name__}: {exc}"

    t0 = time.monotonic()
    runner = threading.Thread(target=play, daemon=True, name="benchmark")
    runner.start()

    def moved(game_run) -> bool:
        return any(record.action.id.name != "RESET" for record in list(game_run.history))

    def keepalive_stopped() -> bool:
        return any("keepalive stopped after" in r["msg"] for r in records) or not hasattr(so, "_keepalive_start")

    stop_sent = None
    while time.monotonic() - t0 < spec["until"] * MINUTE:
        now = minutes(time.monotonic())
        if isinstance(spec["ready"], float) and model.ready_at is None and now >= spec["ready"]:
            model.ready()
        if spec["ready"] is None and closes and now >= minutes(closes[-1]) + 1.0:
            break  # never READY: the card has closed
        if (spec["ready"] is not None and stop_sent is None and len(bm.game_runs) == len(GAMES)
                and all(moved(r) for r in bm.game_runs) and keepalive_stopped()):
            stop_sent = time.monotonic()
            run["loop"].call_soon_threadsafe(bm.request_stop)
        if stop_sent is not None and "ended" in run:
            break
        time.sleep(0.05)
    result = {
        "scenario": name, "keepalive_code": hasattr(so, "_keepalive_start"),
        "ready": minutes(model.ready_at) if model.ready_at else None,
        "calls": [{**c, "t": minutes(c["t"])} for c in calls],
        "closes": [minutes(t) for t in closes],
        "logs": [{**r, "t": minutes(r["t"])} for r in records
                 if r["logger"].startswith("inference") or "keepalive" in r["msg"]],
        "threads": hooked, "ended": run.get("ended"),
        "runs": [{"game": r.game_id, "state": r.state, "note": r.solver_note,
                  "actions": [rec.action.id.name for rec in list(r.history)]} for r in bm.game_runs],
    }
    out.write_text(json.dumps(result, indent=1))
    sys.stdout.flush()
    os._exit(0)


# ---------------------------------------------------------------- parent: judge

def judge(results: dict[str, dict]) -> int:
    def guid_resets(res):
        return [c for c in res["calls"] if c["action"] == "RESET" and c["guid"]]

    def warmup_game(res):
        (line,) = [r["msg"] for r in res["logs"] if "warmup RESET issued" in r["msg"]]
        game = line.split(":")[0]
        first = guid_resets(res)[0]
        assert first["game"] == game and first["status"] == 200, (first, game)
        return game, first["t"]

    def keepalive_resets(res):
        game, _ = warmup_game(res)
        later = guid_resets(res)[1:]
        assert all(c["game"] == game for c in later), [c for c in later if c["game"] != game]  # (3)
        assert all(c["status"] == 200 for c in later), later
        return [c["t"] for c in later]

    def stopped(res) -> str:
        (line,) = [r["msg"] for r in res["logs"] if "keepalive stopped after" in r["msg"]]
        return line

    def played(res) -> None:  # (5) and (6): every game plays after READY; none gives up or crashes
        assert res["ended"] == "stopped", res["ended"]
        assert len(res["runs"]) == len(GAMES), res["runs"]
        for run in res["runs"]:
            assert any(a != "RESET" for a in run["actions"]), run
            assert run["state"] in {"cancelled", "gave_up", "won"}, run
            assert not str(run["note"]).startswith(("analyzer failed", "error:")), run
        assert not res["threads"], res["threads"]
        assert not res["closes"], res["closes"]

    def gaps(times: list[float]) -> list[float]:
        return [round(b - a, 3) for a, b in zip(times, times[1:])]

    checks = 0
    # control: the pin as shipped, server never READY
    res = results["control"]
    assert not res["keepalive_code"]
    game, warm = warmup_game(res)
    assert keepalive_resets(res) == [], "the pin sends one warmup RESET"
    (closed,) = res["closes"]
    assert 15.0 <= closed - warm <= 15.6, (warm, closed)
    print(f"ok  (4) control, the pin without the keepalive: one warmup RESET at {warm:.2f} min, then the card "
          f"closed at {closed:.2f} min ({closed - warm:.2f} idle): the run would bank its score so far")
    checks += 1

    # never READY: five RESETs at most 9 minutes apart, then the cap; the card lives to 15 minutes after the last
    res = results["never"]
    game, warm = warmup_game(res)
    times = keepalive_resets(res)
    spacing = gaps([warm] + times)
    assert len(times) == 5 and all(8.95 <= g <= 9.05 for g in spacing), spacing  # (1)
    assert "keepalive stopped after 5 RESET(s): the cap of 5 RESETs is spent" in stopped(res), stopped(res)
    (closed,) = res["closes"]
    assert 15.0 <= closed - times[-1] <= 15.6, (times[-1], closed)  # (4)
    assert not res["threads"], res["threads"]
    print(f"ok  (1)(3)(4) never READY: keepalive RESETs on {game} only at "
          f"{', '.join(f'{t:.2f}' for t in times)} min (gaps {min(spacing):.3f} to {max(spacing):.3f}), then the "
          f"cap; the card stayed open to {closed:.2f} min, {closed - times[-1]:.2f} after the last RESET")
    checks += 1

    # READY at 38 minutes after the release, 50 after notebook start
    res = results["ready38"]
    game, warm = warmup_game(res)
    times = keepalive_resets(res)
    spacing = gaps([warm] + times)
    assert len(times) == 4 and all(8.95 <= g <= 9.05 for g in spacing), spacing  # (1)
    assert res["ready"] is not None and max(times) < res["ready"], (times, res["ready"])  # (2)
    assert "keepalive stopped after 4 RESET(s): the game is playing" in stopped(res), stopped(res)
    failed = [r["t"] for r in res["logs"] if r["msg"].startswith("analyzer request failed")]
    assert len(failed) == len(GAMES) and all(30.0 <= t <= 37.0 for t in failed), failed  # one in-place retry run each
    played(res)
    first_moves = [min(c["t"] for c in res["calls"] if c["game"] == run["game"] and c["action"] != "RESET")
                   for run in res["runs"]]
    assert all(t >= res["ready"] for t in first_moves), first_moves
    print(f"ok  (1)(2)(5) READY at {res['ready']:.2f} min: 4 keepalive RESETs, the last at {times[-1]:.2f}, none "
          f"after READY; each game's first request gave up in place at {min(failed):.1f} to {max(failed):.1f} min "
          f"(360 retries) and the next one got through; all {len(GAMES)} games played, none gave up or crashed")
    checks += 1

    # READY before the release
    res = results["before"]
    game, warm = warmup_game(res)
    assert keepalive_resets(res) == []  # (7)
    line = stopped(res)
    assert re.search(r"stopped after 0 RESET\(s\): (the model server is READY|the game is playing)$", line), line
    played(res)
    print(f"ok  (7) READY before the release: no keepalive RESET ({line.split(': ', 2)[-1]}); all games played")
    checks += 1

    # the keepalive's RESET raises inside its thread
    res = results["raises"]
    game, warm = warmup_game(res)
    assert keepalive_resets(res) == []
    line = stopped(res)
    assert line.endswith("stopped after 0 RESET(s): RuntimeError: injected keepalive failure"), line
    (t_line,) = [r["t"] for r in res["logs"] if "keepalive stopped after" in r["msg"]]
    assert 8.95 <= t_line - warm <= 9.1, (warm, t_line)
    played(res)
    print(f"ok  (6) a RESET that raises in the thread at {t_line:.2f} min is logged ({line.split(': ', 1)[1]}); "
          f"no thread exception escaped and all games played after READY at {res['ready']:.2f}")
    checks += 1
    return checks


def main() -> None:
    _reexec()
    if len(sys.argv) == 5 and sys.argv[1] == "--scenario":
        return scenario(sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4]))
    sys.path.insert(0, str(REPO / "solution/sim"))
    from franzen_tree import pristine_tree  # noqa: PLC0415
    B = builder()
    pristine = pristine_tree()
    with tempfile.TemporaryDirectory(prefix="keepalive-") as tmp:
        tmp = Path(tmp)
        checks = structure_checks(B, pristine, tmp) + stub_check()
        trees = {arm: build_tree(B, pristine, tmp, arm) for arm in (BASE_ARM, ARM)}
        env = {**os.environ, **arm_env(B, ARM), "STALE_MINUTES": "1", "ARC3_MAX_ACTIVE_STREAMS": str(len(GAMES)),
               "ARC3_ANALYZER_RETRY_BACKOFF_SECONDS": str(10.0 / SCALE), "NO_PROXY": "127.0.0.1,localhost",
               "no_proxy": "127.0.0.1,localhost", "PYTHONDONTWRITEBYTECODE": "1", "_KEEPALIVE_CHILD": "1"}
        assert arm_env(B, BASE_ARM) == arm_env(B, ARM), "the keepalive changes no setting"
        procs = {}
        for name, spec in SCENARIOS.items():
            log = open(tmp / f"{name}.log", "w")
            procs[name] = (subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--scenario", name,
                                             str(trees[spec["tree"]]), str(tmp / f"{name}.json")],
                                            env=env, cwd=tmp, stdout=log, stderr=subprocess.STDOUT), log)
        print(f"... {len(procs)} scenarios running in parallel, about "
              f"{max(s['until'] for s in SCENARIOS.values()) * MINUTE / 60:.0f} min at most", flush=True)
        results = {}
        for name, (proc, log) in procs.items():
            code = proc.wait(timeout=SCENARIOS[name]["until"] * MINUTE + 240)
            log.close()
            if code != 0 or not (tmp / f"{name}.json").exists():
                print((tmp / f"{name}.log").read_text()[-4000:])
                raise SystemExit(f"scenario {name} exited {code}")
            results[name] = json.loads((tmp / f"{name}.json").read_text())
        checks += judge(results)
    print(f"\n{checks} checks passed")


if __name__ == "__main__":
    main()
