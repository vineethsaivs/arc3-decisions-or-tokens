"""Offline check of the B3 overlay (franzen_overlays/handover.py): early handover of stuck games.

Copies Franzen's patched bundle to a temp dir, applies the overlay, compiles it, then drives the
real _maybe_handover and the real _PriorityGate: the threshold logic, empty-queue skips, who gets
a freed slot, and a threaded stress run (more games than slots) for deadlock and slot accounting.
Last, every overlay in franzen_overlays/ is applied in every order and compiled.

Run: python3 solution/sim/test_franzen_handover.py
 or: uv --directory ARC-AGI-3-Agents run python ../solution/sim/test_franzen_handover.py
FRANZEN_TREE overrides the pristine bundle path.
"""
from __future__ import annotations

import itertools
import overlay_orders
import logging
import os
import pathlib
import py_compile
import random
import shutil
import sys
import tempfile
import threading
import time
import types

ROOT = pathlib.Path(__file__).resolve().parents[2]
OVERLAYS = ROOT / "solution" / "kaggle" / "franzen_overlays"
PRISTINE = pathlib.Path(os.environ.get(
    "FRANZEN_TREE",
    str(__import__("franzen_tree").pristine_tree()),  # vendored bundle plus the notebook patch
))
REL = pathlib.Path("src") / "ARC3-Inference" / "inference" / "agent" / "tool_agent.py"
BAND = 1_000_000

# Franzen's scored-run priority settings (notebook cell 7)
FRANZEN_ENV = {
    "ARC3_PRIORITY_REFRESH_QUEUE": "1",
    "ARC3_PRIORITY_PACE": "0",
    "ARC3_PRIORITY_TAIL_FADE": "1",
    "ARC3_PRIORITY_TAIL_FADE_FRACTION": "0.2",
    "ARC3_PRIORITY_TAIL_LOOKUP": "remaining",
    "ARC3_PRIORITY_SCORE_NORMALIZATION": "1",
}


def apply_overlay(bundle: pathlib.Path, name: str) -> None:
    code = (OVERLAYS / f"{name}.py").read_text()
    exec(compile(code, f"franzen_overlays/{name}.py", "exec"), {"BUNDLE_DIR": bundle})


def copy_tree(tmp: pathlib.Path, name: str) -> pathlib.Path:
    dest = tmp / name
    shutil.copytree(PRISTINE, dest)
    return dest


def import_tool_agent(bundle: pathlib.Path):
    """Import the patched module; stub requests and PIL when this interpreter lacks them."""
    class _Meta(type):
        def __getattr__(cls, name):
            if name.startswith("__"):
                raise AttributeError(name)
            return _Meta(name, (), {})

    class _Stub(types.ModuleType):
        def __getattr__(self, name):
            if name.startswith("__"):
                raise AttributeError(name)
            return _Meta(name, (), {})

    sys.path.insert(0, str(bundle / "src" / "ARC3-Inference"))
    for _ in range(10):
        try:
            import inference.agent.tool_agent as ta
            return ta
        except ModuleNotFoundError as exc:
            if not exc.name or exc.name.startswith("inference"):
                raise
            for key in [m for m in sys.modules if m.startswith("inference")]:
                del sys.modules[key]
            parts = exc.name.split(".")
            for i in range(1, len(parts) + 1):
                sys.modules.setdefault(".".join(parts[:i]), _Stub(".".join(parts[:i])))
    raise AssertionError("tool_agent did not import")


def wait_until(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "timed out waiting for a gate state"
        time.sleep(0.002)


def make_agent(ta, name: str, dispatch: int = 0):
    """A ToolAgent with only the fields _maybe_handover reads; the real constructor needs a server."""
    agent = object.__new__(ta.ToolAgent)
    agent._context_was_trimmed = False
    agent._last_step_summary = {"level": 1, "end_action_num": 0}
    agent._actions_at_level_start = 0
    agent._tokens_at_level_start = 0
    agent._session_generated_tokens = 0
    agent._progress_pace = ta.ProgressPace()
    agent._diag_name = name
    agent._game_elapsed_seconds = None
    agent._priority_current = agent._priority_next = ta._PRIORITY_UNTRIMMED_BASE - dispatch
    agent._priority_level_count_warned = True
    return agent


def level_up(agent) -> None:
    """What _update_summarized_knowledge_from_step_summary does on a level transition."""
    summary = agent._last_step_summary
    agent._last_step_summary = {"level": summary["level"] + 1, "end_action_num": summary["end_action_num"]}
    agent._actions_at_level_start = summary["end_action_num"]
    agent._tokens_at_level_start = agent._session_generated_tokens


def new_gate(ta, slots: int):
    os.environ["ARC3_MAX_ACTIVE_STREAMS"] = str(slots)
    gate = ta._PriorityGate(slots)
    now = time.monotonic()
    gate.configure_clock(now, now + 3600.0)
    ta._PRIORITY_GATE = gate
    return gate


class _Messages(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(record.getMessage())  # raises on a bad format string


def check_threshold_logic(ta) -> None:
    """Deterministic: a spy stands in for the gate's handover, a fake entry for a waiter."""
    gate = new_gate(ta, 1)
    calls: list[tuple[int, object]] = []
    gate.handover = lambda priority, snapshot=None: calls.append((priority, snapshot))
    messages = _Messages()
    ta.log.addHandler(messages)
    agent = make_agent(ta, "g00")

    def step(tokens: int, waiting: bool, trimmed: bool = False) -> int:
        agent._session_generated_tokens = tokens
        agent._context_was_trimmed = trimmed
        gate._waiting = [(-5, 999)] if waiting else []
        before = len(calls)
        agent._maybe_handover()
        assert agent._context_was_trimmed is False
        return len(calls) - before

    # first threshold at 25,000 on the level, then 40,000 after the last handover
    assert step(24_999, True) == 0
    assert step(25_000, True) == 1
    estimate, snapshot = calls[-1]
    assert (snapshot.level, snapshot.tokens, snapshot.actions) == (1, 25_000, 0), snapshot
    assert 0 < estimate < BAND, estimate  # same raw (decayed) band as a trim handover
    assert agent._priority_next == estimate + BAND
    assert step(64_999, True) == 0
    assert step(65_000, True) == 1
    assert any("early handover: g00 yields after 65000 tokens on level 1, 1 waiting" in m for m in messages.lines)

    # empty queue: skipped, and the threshold stays armed until someone waits
    assert step(115_000, False) == 0
    assert step(125_000, False) == 0
    assert agent._early_handover_at == 105_000
    assert step(126_000, True) == 1
    assert agent._early_handover_at == 166_000

    # a trim still hands over, queue or not (Franzen's path), and pushes the next early one
    assert step(145_000, False, trimmed=True) == 1
    assert agent._early_handover_at == 185_000
    assert step(180_000, True) == 0
    assert step(185_000, True) == 1

    # a level-up restarts the count
    level_up(agent)
    assert step(185_000 + 24_999, True) == 0
    assert step(185_000 + 25_000, True) == 1
    assert calls[-1][1].level == 2 and calls[-1][1].tokens == 25_000

    # knobs read at call time: 0 disables, step is tunable
    os.environ["ARC3_EARLY_HANDOVER_TOKENS"] = "0"
    assert step(400_000, True) == 0
    os.environ["ARC3_EARLY_HANDOVER_TOKENS"] = "20000"
    os.environ["ARC3_EARLY_HANDOVER_STEP"] = "5000"
    level_up(agent)
    assert step(400_000 + 20_000, True) == 1
    assert step(400_000 + 24_999, True) == 0
    assert step(400_000 + 25_000, True) == 1

    # step 0: one early handover per level, not one before every request
    os.environ["ARC3_EARLY_HANDOVER_STEP"] = "0"
    level_up(agent)
    base = agent._session_generated_tokens
    assert sum(step(base + 20_000 + i * 1_000, True) for i in range(20)) == 1
    level_up(agent)  # a level-up re-arms it
    base = agent._session_generated_tokens
    assert step(base + 20_000, True) == 1
    assert step(base + 900_000, True) == 0
    level_up(agent)  # a trim also uses up the level's one handover
    base = agent._session_generated_tokens
    assert step(base + 10_000, False, trimmed=True) == 1
    assert step(base + 900_000, True) == 0
    del os.environ["ARC3_EARLY_HANDOVER_TOKENS"], os.environ["ARC3_EARLY_HANDOVER_STEP"]

    # no gate: nothing to hand over to
    os.environ["ARC3_MAX_ACTIVE_STREAMS"] = "0"
    assert step(10_000_000, True) == 0
    os.environ["ARC3_MAX_ACTIVE_STREAMS"] = "1"

    # its own failure is logged and keeps the slot; the game does not crash
    agent._session_generated_tokens = None
    agent._context_was_trimmed = False
    gate._waiting = [(-5, 999)]
    before = len(calls)
    agent._maybe_handover()
    assert len(calls) == before
    assert any("early handover check failed, keeping the slot" in m for m in messages.lines)
    ta.log.removeHandler(messages)


def check_highest_priority_admitted(ta) -> None:
    """Real gate, real threads. 3 slots: A, W1, W2 hold; U1, U2 (never trimmed) wait."""
    gate = new_gate(ta, 3)
    order: list[str] = []
    agents = {name: make_agent(ta, name, i) for i, name in enumerate(["A", "W1", "W2", "U1", "U2"])}
    for name in ("A", "W1", "W2"):
        gate.acquire(ta._PRIORITY_UNTRIMMED_BASE - 0)
    hold = {"U1": threading.Event(), "U2": threading.Event()}

    def untrimmed(name: str, priority: int) -> None:
        gate.acquire(priority)
        order.append(name)
        hold[name].wait(10)
        gate.release()

    def hand_over(name: str) -> None:
        agents[name]._maybe_handover()
        order.append(name)

    threads = [threading.Thread(target=untrimmed, args=("U1", ta._PRIORITY_UNTRIMMED_BASE - 50), daemon=True),
               threading.Thread(target=untrimmed, args=("U2", ta._PRIORITY_UNTRIMMED_BASE - 51), daemon=True)]
    for t in threads:
        t.start()
    wait_until(lambda: len(gate._waiting) == 2)

    # W1 trims deep into level 1 (low P), W2 trims fresh on level 4 (high P): each loses to an
    # untrimmed game, highest first
    agents["W1"]._session_generated_tokens = 100_000
    agents["W1"]._context_was_trimmed = True
    agents["W2"]._last_step_summary = {"level": 4, "end_action_num": 40}
    agents["W2"]._actions_at_level_start = 40
    agents["W2"]._context_was_trimmed = True
    for name, expect in (("W1", ["U1"]), ("W2", ["U1", "U2"])):
        threads.append(threading.Thread(target=hand_over, args=(name,), daemon=True))
        threads[-1].start()
        wait_until(lambda: order == expect and len(gate._waiting) == 2)

    # A is stuck at 60,000 tokens with W1 and W2 waiting: early handover, and the freed slot goes
    # to the highest re-priced waiter. Expected order from the gate's own priority function.
    agents["A"]._session_generated_tokens = 60_000
    now = time.monotonic()
    prio = {name: gate._snapshot_priority(ta.PrioritySnapshot(
        int(agents[name]._last_step_summary["level"]), 0,
        agents[name]._session_generated_tokens - agents[name]._tokens_at_level_start, 1.0, 10), now)
        for name in ("A", "W1", "W2")}
    expected = sorted(prio, key=prio.get, reverse=True)
    assert expected == ["W2", "A", "W1"], prio
    threads.append(threading.Thread(target=hand_over, args=("A",), daemon=True))
    threads[-1].start()
    wait_until(lambda: order[2:] == ["W2"] and len(gate._waiting) == 2)
    hold["U1"].set()  # U1 finishes: its slot goes to A, the higher of the two left
    wait_until(lambda: order[3:] == ["A"] and len(gate._waiting) == 1)
    hold["U2"].set()
    for t in threads:
        t.join(10)
        assert not t.is_alive(), "deadlock in the priority test"
    assert order == ["U1", "U2", "W2", "A", "W1"], order
    for _ in range(3):
        gate.release()
    assert gate._free == 3 and not gate._waiting and not gate._admitted


def check_stress(ta, slots: int = 3, games: int = 10, requests: int = 30) -> None:
    """More games than slots, stuck and progressing, trims mixed in. No deadlock, no lost slot,
    never more active than slots, and only stuck games hand over early."""
    gate = new_gate(ta, slots)
    rng = random.Random(7)
    lock = threading.Lock()
    state = {"active": 0, "max_active": 0}
    early: dict[str, int] = {}

    def enter() -> None:
        with lock:
            state["active"] += 1
            state["max_active"] = max(state["max_active"], state["active"])
            assert state["active"] <= slots, state

    def leave() -> None:
        with lock:
            state["active"] -= 1

    real_handover = gate.handover

    def counted_handover(priority, snapshot=None):
        leave()
        real_handover(priority, snapshot)
        enter()

    gate.handover = counted_handover
    errors: list[BaseException] = []

    def play(index: int) -> None:
        name = f"g{index:02d}"
        agent = make_agent(ta, name, index)
        real_due = agent._early_handover_due

        def due() -> bool:
            fired = real_due()
            if fired:
                early[name] = early.get(name, 0) + 1
            return fired

        agent._early_handover_due = due
        progressing = index % 2 == 0
        delays = [rng.uniform(0.0005, 0.003) for _ in range(requests)]
        gate.acquire(ta._PRIORITY_UNTRIMMED_BASE - index * 100)  # as solver.play does
        try:
            enter()
            for n in range(requests):
                agent._context_was_trimmed = n > 0 and n % 9 == 0
                agent._maybe_handover()
                time.sleep(delays[n])  # the request
                agent._session_generated_tokens += 9_000
                agent._last_step_summary["end_action_num"] += 3
                if progressing and n % 3 == 2:
                    level_up(agent)
        except BaseException as exc:
            errors.append(exc)
        finally:
            leave()
            gate.release()

    threads = [threading.Thread(target=play, args=(i,), daemon=True) for i in range(games)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
        assert not t.is_alive(), "deadlock: a game never finished"
    assert not errors, errors
    assert state["max_active"] == slots, state  # saturated, never over
    assert state["active"] == 0
    assert gate._free == slots and not gate._waiting and not gate._admitted, "slot lost or leaked"
    stuck = [f"g{i:02d}" for i in range(games) if i % 2]
    assert sum(early.get(name, 0) for name in stuck) > 0, early
    assert all(f"g{i:02d}" not in early for i in range(0, games, 2)), early  # checked at 18K at most < 25K


def check_composition(tmp: pathlib.Path) -> None:
    names = sorted(p.stem for p in OVERLAYS.glob("*.py") if not p.stem.endswith("_serve"))  # serving patches never touch the bundle
    assert "handover" in names, names
    pristine = {p.relative_to(PRISTINE): p.read_bytes() for p in (PRISTINE / "src").rglob("*.py")}
    for perm in overlay_orders.orders(names, "handover"):
        bundle = copy_tree(tmp, "combo-%08x" % __import__("zlib").crc32("-".join(perm).encode()))
        for name in perm:
            apply_overlay(bundle, name)
        # changed files and files an overlay created (levelmem writes a new module)
        touched = [p.relative_to(bundle) for p in (bundle / "src").rglob("*.py")
                   if pristine.get(p.relative_to(bundle)) != p.read_bytes()]
        assert REL in touched, touched
        for rel in touched:
            py_compile.compile(str(bundle / rel), doraise=True)
        # and the composed harness imports, new modules included
        for key in [m for m in sys.modules if m.startswith("inference")]:
            del sys.modules[key]
        ta = import_tool_agent(bundle)
        sys.path.remove(str(bundle / "src" / "ARC3-Inference"))
        assert ta.__file__.startswith(str(bundle)), ta.__file__
        assert hasattr(ta.ToolAgent, "_early_handover_due")
        print(f"composed {' -> '.join(perm)}: {len(touched)} files touched, all compile and import")


def main() -> None:
    assert PRISTINE.is_dir(), f"pristine Franzen bundle missing: {PRISTINE}"
    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = pathlib.Path(tmp_name)
        bundle = copy_tree(tmp, "handover")
        apply_overlay(bundle, "handover")
        py_compile.compile(str(bundle / REL), doraise=True)
        try:
            apply_overlay(bundle, "handover")
            raise AssertionError("a second application must fail loudly")
        except RuntimeError as exc:
            assert "anchor matches 0 times" in str(exc), exc
        for key in ("ARC3_EARLY_HANDOVER_TOKENS", "ARC3_EARLY_HANDOVER_STEP", "ARC3_DIAG_CONCURRENCY"):
            os.environ.pop(key, None)
        os.environ.update(FRANZEN_ENV)
        ta = import_tool_agent(bundle)
        assert ta.__file__.startswith(str(bundle)), ta.__file__
        check_threshold_logic(ta)
        print("threshold logic ok")
        ta.log.setLevel(logging.ERROR)  # the stress run logs every early handover
        check_highest_priority_admitted(ta)
        print("highest-priority waiter admitted ok")
        check_stress(ta)
        print("stress ok: no deadlock, active never above slots, no slot lost")
        check_composition(tmp)
    print("PASS")


if __name__ == "__main__":
    main()
