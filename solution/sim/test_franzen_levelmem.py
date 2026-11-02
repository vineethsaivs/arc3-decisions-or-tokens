"""Offline checks for the B1 overlay, solution/kaggle/franzen_overlays/levelmem.py.

    python3 solution/sim/test_franzen_levelmem.py

Copies Franzen's patched tree to a temp dir, applies the overlay there, and
drives the new hooks through the real ToolAgent: two level-ups, the rule read
back from a reply, a token drain, a turn-cap drain, the size bound, swallowed
errors, and every order of every overlay in franzen_overlays/. No network, no
GPU. requests and PIL are stubbed when this interpreter lacks them.
"""
from __future__ import annotations

import contextlib
import io
import itertools
import overlay_orders
import os
import py_compile
import shutil
import sys
import tempfile
import types
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
OVERLAYS = REPO / "solution" / "kaggle" / "franzen_overlays"
TREE = Path(os.environ.get(
    "FRANZEN_TREE",
    str(__import__("franzen_tree").pristine_tree()),  # vendored bundle plus the notebook patch
))
AGENT = Path("src/ARC3-Inference/inference/agent")


def fresh_bundle(root: Path) -> Path:
    shutil.copytree(TREE / "src" / "ARC3-Inference", root / "src" / "ARC3-Inference",
                    ignore=shutil.ignore_patterns("__pycache__"))
    return root


def apply(bundle: Path, name: str) -> str:
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile((OVERLAYS / f"{name}.py").read_text(), f"franzen_overlays/{name}.py", "exec"),
             {"BUNDLE_DIR": bundle})
    return out.getvalue()


def compile_tree(bundle: Path, scratch: Path) -> None:
    for path in sorted((bundle / AGENT).glob("*.py")):
        py_compile.compile(str(path), cfile=str(scratch / "check.pyc"), doraise=True)


def stub_missing() -> None:
    class _Any:
        def __getattr__(self, name):
            return _Any()

        def __call__(self, *args, **kwargs):
            return _Any()

    def stub(name: str) -> types.ModuleType:
        module = types.ModuleType(name)
        module.__getattr__ = lambda attr: _Any()
        sys.modules[name] = module
        return module

    for name in ("requests", "PIL"):
        try:
            __import__(name)
        except ImportError:
            module = stub(name)
            if name == "requests":
                module.RequestException = type("RequestException", (Exception,), {})
            else:
                module.Image = stub("PIL.Image")


def entry(HistoryEntry, Frame, action: str, level: int, step: int):
    return HistoryEntry(action=action, frame=Frame(grid=(), step=step, level=level))


def main() -> None:
    work = Path(tempfile.mkdtemp(prefix="levelmem-"))
    try:
        run(work)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print("PASS")


def run(work: Path) -> None:
    # (1)-(3) apply to a copy, compile what it touched
    bundle = fresh_bundle(work / "one")
    printed = apply(bundle, "levelmem")
    assert printed.count("\n") == 1 and "levelmem" in printed, printed
    assert (bundle / AGENT / "levelmem.py").exists()
    compile_tree(bundle, work)
    try:
        apply(bundle, "levelmem")
        raise AssertionError("a second application must refuse")
    except RuntimeError as exc:
        assert "already exists" in str(exc)

    # (4) the new logic, through the real ToolAgent
    stub_missing()
    sys.path.insert(0, str(bundle / "src" / "ARC3-Inference"))
    os.environ.update({"ARC3_CONTEXT_DRAIN_TOKENS": "8000", "ARC3_LEVELMEM_MAX_TOKENS": "3000"})
    import inference.agent.levelmem as lm
    import inference.agent.tool_agent as ta
    from inference.agent.runtime_state import Frame, HistoryEntry

    agent = ta.ToolAgent(model="flashnext")
    base = agent._system_prompt
    agent._context_budget_tokens = 20000

    # level 1: warmup RESET, a failed attempt, then UP x3, SPACE clears it
    entries = [entry(HistoryEntry, Frame, "", 1, 0), entry(HistoryEntry, Frame, "RESET", 1, 0)]
    for step, (action, level) in enumerate(
            [("UP", 1), ("LEFT", 1), ("RESET", 1), ("UP", 1), ("UP", 1), ("UP", 1), ("SPACE", 2)], 1):
        entries.append(entry(HistoryEntry, Frame, action, level, step))
    clearing = {"role": "assistant", "reasoning_content": "The block sits under the target, so UP three times then SPACE.",
                "content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "python", "arguments": "{}"}}]}
    agent._history_messages = [{"role": "user", "content": "opener 0"}, clearing,
                               {"role": "tool", "tool_call_id": "c1", "content": "level_completed"}]
    agent._last_step_summary = {"level": 2, "level_transition": True}
    opener1 = lm.on_opener(agent, "OPENER", Frame(grid=(), step=7, level=2), entries)
    assert opener1.startswith("OPENER\n\n"), opener1
    assert "Level 1: cleared after 6 actions in 2 attempts." in opener1, opener1
    assert "Winning attempt, 4 actions: UP x3, SPACE." in opener1, opener1
    assert "starts exactly with 'Rule for level 1:'" in opener1, opener1
    assert agent._system_prompt == base, "a level-up must not touch the system prompt"
    record = agent._levelmem["records"][0]
    assert "UP three times then SPACE" in record["reasoning"], record

    # the reply quotes the ask, then answers it; the answer is what is kept
    reply = {"role": "assistant", "content": None, "reasoning_content":
             "I must write 'Rule for level 1:' first.\nRule for level 1: the goal is to push the blue block "
             "onto the yellow target; UP moves it one cell, SPACE locks it in place."}
    agent._history_messages = [*agent._history_messages, {"role": "user", "content": opener1}, reply]
    agent._last_step_summary = {"level": 2, "level_transition": False}
    opener2 = lm.on_opener(agent, "OPENER2", Frame(grid=(), step=8, level=2), entries)
    assert opener2 == "OPENER2", "no record and no ask once the rule is in"
    assert record["rule"].startswith("the goal is to push the blue block"), record["rule"]

    # level 2 clears: its record rides on the next level-up opener; still no system change
    entries += [entry(HistoryEntry, Frame, a, lv, 8 + i) for i, (a, lv) in
                enumerate([("DOWN", 2), ("MOUSE(row=4, col=7)", 2), ("MOUSE(row=4, col=7)", 3)])]
    agent._history_messages = [*agent._history_messages, {"role": "user", "content": opener2},
                               {"role": "assistant", "content": "DOWN then click twice."}]
    agent._last_step_summary = {"level": 3, "level_transition": True}
    opener3 = lm.on_opener(agent, "OPENER3", Frame(grid=(), step=11, level=3), entries)
    assert "Level 2: cleared after 3 actions in 1 attempt. Winning attempt, 3 actions: DOWN, MOUSE(row=4, col=7) x2." in opener3, opener3
    assert "'Rule for level 2:'" in opener3 and "Level 1:" not in opener3, opener3
    assert agent._system_prompt == base

    # the level-up turn fails and is rolled back: the retry carries the record and
    # the ask again, and the failed ask is not counted
    retry = lm.on_opener(agent, "OPENER3", Frame(grid=(), step=11, level=3), entries)
    assert retry == opener3, retry
    assert [r["level"] for r in agent._levelmem["records"]] == [1, 2]
    assert agent._levelmem["records"][1]["asks"] == 1
    # a resumption after a yield restates the turn: no ask, no check spent
    committed = [*agent._history_messages, {"role": "user", "content": opener3},
                 {"role": "assistant", "content": "Let me look first."}]
    agent._history_messages = committed
    agent._resume_after_yield = True
    assert lm.on_opener(agent, "RESUME", Frame(grid=(), step=11, level=3), entries) == "RESUME"
    assert agent._levelmem["records"][1]["asks"] == 1 and agent._levelmem["records"][1]["checks"] == 0
    agent._resume_after_yield = False

    # the rule parser: reasoning is read before the reply, and the model restating
    # the ask is never taken as the rule
    echo = ("The host wants a paragraph starting with Rule for level 1: then the goal, the controls "
            "and the winning plan. Fine.")
    real = "push the blue block onto the yellow target; UP moves it one cell and SPACE locks it."
    assert lm.find_rule(lm._text({"reasoning_content": echo, "content": f"Rule for level 1: {real}"}), 1) == real
    assert lm.find_rule(echo, 1) is None
    assert lm.find_rule("They ask me to write Rule for level 1: with the goal, what each control did, "
                        "and the plan that won. First let me look at the new board.", 1) is None
    assert lm.find_rule("Rule for level 1: the goal, what each control did, and the plan that won.", 1) is None
    assert lm.find_rule("**Rule for level 1:** The goal is to fill every white cell; a click paints one.",
                        1) == "The goal is to fill every white cell; a click paints one."
    assert lm.find_rule("**Rule for level 1**: The goal is to fill every white cell; a click paints one.",
                        1).startswith("The goal is")

    # one action reads as one action; ARC3_LEVELMEM_ACTIONS=0 keeps none
    one = dict(record, n_actions=1, attempts=1, n_winning=1, winning=["SPACE"])
    assert lm._facts(one) == "Level 1: cleared after 1 action in 1 attempt." and lm._winning_line(one) == "Winning attempt, 1 action: SPACE"
    os.environ["ARC3_LEVELMEM_ACTIONS"] = "0"
    assert lm._new_record(1, entries, [])["winning"] == []
    del os.environ["ARC3_LEVELMEM_ACTIONS"]

    # a drain that can drop nothing (one long turn behind the only user message)
    # must not fold: the system prompt and the message list stay as they were
    lone = ta.ToolAgent(model="flashnext")
    lone._context_budget_tokens = 20000
    lone._levelmem = {"records": [dict(record)], "folded": ""}
    long_turn = [{"role": "system", "content": lone._system_prompt}, {"role": "user", "content": "opener"}]
    for i in range(40):
        long_turn += [{"role": "assistant", "content": None, "tool_calls": [{"id": f"t{i}", "type": "function", "function": {"name": "python", "arguments": "{}"}}]},
                      {"role": "tool", "tool_call_id": f"t{i}", "content": "y" * 10000}]
    assert lone._trim_messages_for_context(long_turn, force_drain=True) == long_turn
    assert lone._levelmem["folded"] == "" and lm.HEADER not in lone._system_prompt

    # a trim that drops nothing must not fold
    small = [{"role": "system", "content": agent._system_prompt}, {"role": "user", "content": opener3}]
    assert agent._trim_messages_for_context(small) == small and agent._system_prompt == base

    # the drain: history well over budget, the level-up openers at its head
    filler = "x" * 6000
    history = [{"role": "user", "content": opener1}, reply, {"role": "user", "content": opener3}]
    for i in range(30):
        history += [{"role": "assistant", "content": f"thinking {i} {filler}"}, {"role": "user", "content": f"turn {i}"}]
    messages = [{"role": "system", "content": agent._system_prompt}, *history]
    out = agent._trim_messages_for_context(messages)
    assert len(out) < len(messages) and opener1 not in [m.get("content") for m in out], "drain must drop the head"
    pinned = agent._levelmem["folded"]
    assert pinned and agent._system_prompt == base.rstrip("\n") + "\n\n" + pinned and out[0]["content"] == agent._system_prompt
    assert "Level 1: cleared after 6 actions in 2 attempts." in pinned and "Your rule: the goal is to push" in pinned
    assert "Level 2: cleared after 3 actions" in pinned and "DOWN, MOUSE(row=4, col=7) x2" in pinned
    assert agent._estimate_request_input_tokens(out) <= agent._context_budget_tokens, "drain must make room for the block"
    assert messages[0]["content"] == base, "the caller's message list is not mutated"

    # a later drain with nothing new re-sends the same system prompt
    folded_prompt = agent._system_prompt
    again = [out[0], *history]
    agent._trim_messages_for_context(again)
    assert agent._system_prompt == folded_prompt

    # the turn-cap drain (no token trim) also folds, through _persistent_history_messages
    os.environ.update({"ARC3_HISTORY_ASSISTANT_TURNS": "4", "ARC3_HISTORY_TURN_DRAIN": "2"})
    capped = ta.ToolAgent(model="flashnext")
    capped._levelmem = {"records": [dict(record)], "folded": ""}
    turns = [{"role": "system", "content": capped._system_prompt}]
    for i in range(6):
        turns += [{"role": "user", "content": f"u{i}"}, {"role": "assistant", "content": f"a{i}"}]
    kept = capped._persistent_history_messages(turns)
    assert kept[0]["content"] != "u0" and lm.HEADER in capped._system_prompt
    for name in ("ARC3_HISTORY_ASSISTANT_TURNS", "ARC3_HISTORY_TURN_DRAIN"):
        del os.environ[name]

    # size bound: many long records keep only the newest that fit
    os.environ["ARC3_LEVELMEM_MAX_TOKENS"] = "250"
    many = types.SimpleNamespace(_text_chars_per_token=3.0)
    records = []
    for level in range(1, 9):
        rec = dict(record, level=level, rule=f"rule {level} " + "y" * 150, reasoning="z" * 500)
        records.append(rec)
    text = lm.block(many, {"records": records})
    assert len(text) / 3.0 <= 250, len(text)
    assert "Level 8:" in text and "Level 1:" not in text and text.startswith(lm.HEADER) and text.endswith(lm.FOOTER)
    os.environ["ARC3_LEVELMEM_MAX_TOKENS"] = "3000"

    # its own errors are logged and swallowed
    import logging
    logging.disable(logging.WARNING)
    assert lm.on_opener(object(), "P", Frame(grid=(), step=1, level=2), None) == "P"
    broken = types.SimpleNamespace(_levelmem={"records": [None], "folded": ""})
    sys_msg = {"role": "system", "content": "S"}
    assert lm.fold(broken, sys_msg, []) is sys_msg
    logging.disable(logging.NOTSET)

    # (5) every overlay present, in every order, still applies and compiles
    names = sorted(p.stem for p in OVERLAYS.glob("*.py") if not p.stem.endswith("_serve"))  # serving patches never touch the bundle
    for order in overlay_orders.orders(names, "levelmem"):
        combo = fresh_bundle(work / ("combo-%08x" % __import__("zlib").crc32("-".join(order).encode())))
        for name in order:
            apply(combo, name)
        compile_tree(combo, work)
        text = (combo / AGENT / "tool_agent.py").read_text()
        assert text.count("# overlay B1") == 4, order
    print(f"overlay orders checked: {len(overlay_orders.orders(names, "levelmem"))} ({', '.join(names)})")


if __name__ == "__main__":
    main()
