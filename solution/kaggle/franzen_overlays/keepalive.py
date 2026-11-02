"""Warmup-only keepalive: while the model server loads, the warmup game re-sends its RESET every 9 minutes.

Why. "If you don't do any move for 15 minutes, the run will end with the score so far" (Jeroen Cottaar of
Tufa, forum 750173), and arc_agi 0.9.8 closes a scorecard idle for STALE_MINUTES, default 15 (scorecard.py
lines 24 and 912-940, api.py 445-497). The pin releases the benchmark at cell 15's 12-minute deadline and the
first game to start sends one engine RESET before it touches the analyzer (cell 7 lines 15-17; solver.py
636-651 of the patched tree built by solution/sim/franzen_tree.py). Every game then waits on the analyzer,
so a server that is not serving by about 27 minutes leaves the card idle for 15 minutes and the run banks
a complete 0.00 (docs/research/2026-10-19-top/PLAN.md section 2.5).

What. Three edits to inference/framework/solver.py:
(a) after WARMUP_ACTION_GAMES (line 102), the keepalive helpers;
(b) in _play_inner, right after the warmup RESET (line 651), one daemon thread for that game;
(c) at the top of _HarnessGameSession._execute_action (line 1244), the game's own first action stops the
    keepalive under its lock, so a keepalive RESET never runs during or after a real action.
The thread re-sends RESET on the warmup game only, through that game's own taaf Game (its arc_agi wrapper and
requests.Session, its guid, the run's one card), at most 9 minutes after the previous RESET and at most 5
times, to about 57 minutes after notebook start. Before each RESET it checks /health on the analyzer's own
base URL and the analyzer's token count; it stops at the first 200, the first analyzer reply, the game's
first action or the game's end. Each RESET lands on an untouched level: the gateway returns the same frame
and bills one action on that level (api.py 316-334, scorecard.py inc_reset_count), as the warmup RESET does.
It skips the harness's own history and runtime-state file, which the game thread rewrites through one shared
.tmp path (runtime_state.py 138), so the model's history shows only the warmup RESET. An error in the thread
is logged and ends the keepalive. Nothing here raises, and a server that never comes up ends as it does
today, 15 minutes after the last RESET.

Grace. Each analyzer request retries in place for 360 attempts of about 5.5 s (watchdog.py sets
ARC3_HTTP_RETRIES=360; tool_agent.py 557-660 and 5283-5297), about 33 minutes, and a game gives up only
after 10 failed requests in a row (solver.py 105 and 695), so no game gives up during a 57-minute boot.

Run by build_franzen_arms.py right after Franzen's harness patch, as exec(code, {'BUNDLE_DIR': BUNDLE_DIR}).
Stdlib only. All or nothing: if an anchor does not match exactly once or the patched file does not compile,
solver.py is left as it was and this prints why, so the arm plays as its base. Checked on taaf's localhost
competition arcade by solution/sim/test_keepalive.py.
"""
from pathlib import Path

_SOLVER = Path(BUNDLE_DIR) / "src" / "ARC3-Inference" / "inference" / "framework" / "solver.py"
_MARK = "overlay keepalive"

_HELPERS = '''
# overlay keepalive: while the model server loads, the warmup game re-sends its RESET so the gateway never
# sees 15 idle minutes. Warmup game only, before its first analyzer reply only. Logs, never raises.
KEEPALIVE_INTERVAL_S = 540.0  # 9 minutes, inside the 15-minute idle close
KEEPALIVE_MAX_RESETS = 5  # to about 57 minutes after notebook start


def _keepalive_ready(health_url: str) -> bool:
    import urllib.request

    try:
        with urllib.request.urlopen(health_url, timeout=5) as response:
            return response.status == 200
    except Exception:
        return False


def _keepalive_start(session: Any) -> None:
    """Start the warmup game's keepalive thread (overlay keepalive). Never raises."""
    try:
        base = str(getattr(getattr(session.analyzer, "_model", None), "base_url", "") or "").rstrip("/")
        if not base.startswith(("http://", "https://")):
            log.warning("keepalive off: the analyzer has no http base URL (%r)", base)
            return
        health_url = (base[:-3] if base.endswith("/v1") else base) + "/health"
        session._keepalive_lock = threading.Lock()
        session._keepalive_stop = threading.Event()
        threading.Thread(target=_keepalive_loop, args=(session, health_url),
                         name="arc3-keepalive", daemon=True).start()
    except Exception as exc:
        log.warning("keepalive not started: %s: %s", type(exc).__name__, exc)


def _keepalive_halt(session: Any) -> None:
    """The game is about to act: stop its keepalive, waiting out a RESET already in flight."""
    stop = getattr(session, "_keepalive_stop", None)
    if stop is not None and not stop.is_set():
        with session._keepalive_lock:
            stop.set()


def _keepalive_loop(session: Any, health_url: str) -> None:
    sent, reason, game_id, last = 0, "", "?", time.monotonic()
    try:
        game_id = session.game.game_run.game_id
        tokens = _analyzer_reported_tokens(session.analyzer)

        def over() -> str:
            if session._keepalive_stop.is_set() or _analyzer_reported_tokens(session.analyzer) > tokens:
                return "the game is playing"
            if session.should_stop():
                return "the game ended"
            if _keepalive_ready(health_url):
                return "the model server is READY"
            return ""

        reason = over()
        while not reason and sent < KEEPALIVE_MAX_RESETS:
            session._keepalive_stop.wait(max(0.0, last + KEEPALIVE_INTERVAL_S - time.monotonic()))
            reason = over()
            if reason:
                break
            with session._keepalive_lock:
                if session._keepalive_stop.is_set():
                    reason = "the game is playing"
                    break
                # the game's own Game and transport; not _execute_action, whose history and state-file
                # writes belong to the game thread
                session.game.execute_action(arcengine.ActionInput(id=arcengine.GameAction.RESET, data={}))
            sent, last = sent + 1, time.monotonic()
            log.warning("%s: keepalive RESET %d of %d, the model server is still not READY",
                        game_id, sent, KEEPALIVE_MAX_RESETS)
        reason = reason or f"the cap of {KEEPALIVE_MAX_RESETS} RESETs is spent"
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
    log.warning("%s: keepalive stopped after %d RESET(s): %s", game_id, sent, reason)
'''

_EDITS = [
    # (a) the helpers, beside the warmup setting they extend
    ('WARMUP_ACTION_GAMES = _env_number("ARC3_WARMUP_ACTION_GAMES", 0, int)\n',
     'WARMUP_ACTION_GAMES = _env_number("ARC3_WARMUP_ACTION_GAMES", 0, int)\n' + _HELPERS),
    # (b) right after the warmup RESET, on the warmup game only
    ("            self._execute_auto_reset()\n"
     "        try:\n"
     "            retry_analysis_step: int | None = None\n",
     "            self._execute_auto_reset()\n"
     "            _keepalive_start(self)  # overlay keepalive: RESET again every 9 min until the server answers\n"
     "        try:\n"
     "            retry_analysis_step: int | None = None\n"),
    # (c) every action of the game's own goes through here: the first one ends the keepalive
    ("    ) -> dict[str, Any]:\n"
     "        previous_grid = _grid_from_state(self.game.current_state)\n",
     "    ) -> dict[str, Any]:\n"
     "        _keepalive_halt(self)  # overlay keepalive: from the game's first action on, never RESET under it\n"
     "        previous_grid = _grid_from_state(self.game.current_state)\n"),
]


def _patched(text: str) -> tuple[str, str]:
    """The patched solver.py and "", or the original text and why it was left alone."""
    if _MARK in text:
        return text, "already applied"
    new = text
    for old, replacement in _EDITS:
        count = new.count(old)
        if count != 1:
            return text, f"anchor matches {count} times, expected 1: {old.strip()[:70]!r}"
        new = new.replace(old, replacement)
    try:
        compile(new, str(_SOLVER), "exec")
    except SyntaxError as exc:
        return text, f"patched solver.py does not compile: {exc}"
    return new, ""


_text = _SOLVER.read_text()
_new, _why = _patched(_text)
if _why:
    print(f"arm overlay keepalive NOT applied ({_why}); this run plays without it", flush=True)
else:
    _SOLVER.write_text(_new)
    print("arm overlay keepalive: the warmup game re-sends RESET every 9 min while /health is not 200, "
          "at most 5 times; it stops at READY, the first analyzer reply or the game's first action", flush=True)
