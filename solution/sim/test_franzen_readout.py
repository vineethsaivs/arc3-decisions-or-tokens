"""Offline check for solution/analysis/franzen_readout.py against the fork-franzen-v1 run.

Expected numbers are the ones docs/research/2026-10-01-top5-plan/milestone2-teardown.md quotes.
Tolerances, and why they are not zero:
  - decode tok/s, KV mean/p90: the teardown averaged a hand-picked 04:12 to 04:34 steady window;
    the readout uses the whole benchmark play window (ramp and drain included), which reads about
    650 / 0.548 / 0.80 against 665 / 0.57 / 0.81.
  - token curve: the teardown's 5.9/30.2/41.7/50.9 came from a token accounting we cannot rebuild;
    benchmark history deltas give 5.7/30.2/41.4/49.9 (tu93's level 5 lands at 100,459 tokens).
  - prefills > 20K: the teardown says none. That was per serve.log line, and a chunked prefill logs
    one line per 8K chunk. ReqTimeStats shows 13 trim re-prefills of up to 58,524 new tokens.

benchmark.json and kernel.log live in the repo. The full serve.log only exists in the teardown
scratchpad (override with FRANZEN_READOUT_SERVE_LOG); without it the whole-run server checks are
skipped and say so. The parsers themselves always run against SERVE_FIXTURE, nine real lines of it.

    python3 solution/sim/test_franzen_readout.py
"""
from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUN = REPO / "solution/kaggle/generated/kernel-outputs/fork-franzen-v1"
SCRATCH = Path("<scratch>/"
               "c8309e29-5c7b-46fe-8bbb-77b013bc9554/scratchpad/teardown/fork-out")
SERVE_LOG = Path(os.environ.get("FRANZEN_READOUT_SERVE_LOG", SCRATCH / "serve.log"))
TOOL = REPO / "solution/analysis/franzen_readout.py"

sys.path.insert(0, str(TOOL.parent))
import franzen_readout as fr  # noqa: E402


# fork serve.log 1663-1670 and 1687: one trim re-prefill chunked 6 x 8192 + 448 (4,672 cached, then
# none), the Decode line after it, and the request's ReqTimeStats (54,223 in, 4,672 cached).
SERVE_FIXTURE = """\
[2026-10-01 04:23:22] Prefill batch, #new-seq: 1, #new-token: 8192, #cached-token: 4672, full token usage: 0.76, mamba usage: 0.67, #running-req: 9, #queue-req: 0, #pending-token: 41359, cuda graph: False, input throughput (token/s): 4437.67
[2026-10-01 04:23:23] Prefill batch, #new-seq: 1, #new-token: 8192, #cached-token: 0, full token usage: 0.77, mamba usage: 0.67, #running-req: 9, #queue-req: 0, #pending-token: 33167, cuda graph: False, input throughput (token/s): 12486.70
[2026-10-01 04:23:24] Prefill batch, #new-seq: 1, #new-token: 8192, #cached-token: 0, full token usage: 0.78, mamba usage: 0.67, #running-req: 9, #queue-req: 0, #pending-token: 24975, cuda graph: False, input throughput (token/s): 11970.87
[2026-10-01 04:23:24] Prefill batch, #new-seq: 1, #new-token: 8192, #cached-token: 0, full token usage: 0.79, mamba usage: 0.67, #running-req: 9, #queue-req: 0, #pending-token: 16783, cuda graph: False, input throughput (token/s): 12901.04
[2026-10-01 04:23:25] Prefill batch, #new-seq: 1, #new-token: 8192, #cached-token: 0, full token usage: 0.80, mamba usage: 0.67, #running-req: 9, #queue-req: 0, #pending-token: 8591, cuda graph: False, input throughput (token/s): 12002.62
[2026-10-01 04:23:25] Prefill batch, #new-seq: 1, #new-token: 8192, #cached-token: 0, full token usage: 0.80, mamba usage: 0.67, #running-req: 9, #queue-req: 0, #pending-token: 399, cuda graph: False, input throughput (token/s): 84602.48
[2026-10-01 04:23:25] Prefill batch, #new-seq: 1, #new-token: 448, #cached-token: 0, full token usage: 0.80, mamba usage: 0.67, #running-req: 9, #queue-req: 0, #pending-token: 0, cuda graph: False, input throughput (token/s): 13744.71
[2026-10-01 04:23:26] Decode batch, #running-req: 10, #full token: 805184, full token usage: 0.80, mamba num: 40, mamba usage: 0.67, accept len: 2.58, accept rate: 0.53, cuda graph: True, gen throughput (token/s): 162.49, #queue-req: 0
[2026-10-01 04:23:34] ReqTimeStats(rid=1be0926978684eaf9339dee3f6ec5bfc, input_len=54223, cached_input_len=4672, output_len=734, attempts=0, type=unified): queue_duration=0.69ms, initial_prefill_elapsed=4140.87ms, post_prefill_elapsed=8433.38ms, forward_duration=12574.24ms, entry_time=1790828601.440
"""

# a transcript in franzen-rep-v1's layout: 2 openers, 3 responses (1 without a tool call after
# `length`, 1 `stop`), 2 tool results, one of them with two KeyError lines (counted once). The
# KeyError in the USER PROMPT and the indented JSON in the META block must not count.
TX_FIXTURE = """\
[SYSTEM PROMPT]
You are playing a game.
[USER PROMPT]
No previous sequence has been executed yet. A KeyError here is prompt text.
[THINKING]
look first
[MODEL RESPONSE META]
finish_reason: tool_calls
tool_call_count: 1
raw_tool_calls:
[
  {"finish_reason: stop": 1}
]
[TOOL CALL: python]
print(n['area'])
[TOOL RESULT: python]
Traceback (most recent call last):
KeyError: 'area'
KeyError: 'area'
[THINKING]
too long
[MODEL RESPONSE META]
finish_reason: length
tool_call_count: 0
[ANALYZER STATUS]
retrying
[SYSTEM PROMPT]
You are playing a game.
[USER PROMPT]
The code executed 3 actions in the previous sequence.
[action] DOWN -> effect
[THINKING]
act
[MODEL RESPONSE META]
finish_reason: stop
tool_call_count: 0
[TOOL CALL: python]
action(['UP'])
[TOOL RESULT: python]
[(1, 2)] ok
"""


def near(got, want, tol, what):
    assert got is not None and abs(got - want) <= tol, f"{what}: got {got}, want {want} +/- {tol}"


def cut(run: dict, budget: int) -> dict:
    """The same trajectory stopped at `budget` generated tokens, as if the game lost its slot."""
    r = copy.deepcopy(run)
    cum, keep = 0, 0
    for h in r["history"]:
        if cum + (h.get("generated_tokens") or 0) > budget:
            break
        cum += h.get("generated_tokens") or 0
        keep += 1
    r["history"] = r["history"][:keep]
    r["levels_completed"] = sum(1 for e in fr.level_tokens(run) if e <= cum)
    r["final_score"], r["solver_note"] = None, f"tokens={budget}"
    return r


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="franzen-readout-"))
    try:
        run_dir = tmp / "fork-franzen-v1"
        run_dir.mkdir()
        for name in ("benchmark.json", "kernel.log"):
            shutil.copy(RUN / name, run_dir / name)
        have_serve = SERVE_LOG.is_file()
        if have_serve:
            shutil.copy(SERVE_LOG, run_dir / "serve.log")

        before = sorted(p.name for p in run_dir.iterdir())
        r = fr.readout(run_dir)
        assert sorted(p.name for p in run_dir.iterdir()) == before, "readout() must not write into its input"

        # benchmark.json
        bm = json.loads((RUN / "benchmark.json").read_text())
        assert r["games"] == 10 and r["levels"] == 49 and r["zero_level_games"] == 0, r
        near(r["mean_score"], 51.0, 0.05, "mean score")
        assert r["per_game"]["sb26-7fbdac44"]["levels"] == 8
        near(r["tok_s"], 587, 2, "harness tok/s")
        for t, want in ((25_000, 5.9), (50_000, 30.2), (75_000, 41.7), (100_000, 50.9)):
            near(r["score_at_tokens"][str(t)], want, 1.25, f"score at {t} tokens")
        near(r["score_at_tokens"]["150000"], r["mean_score"], 1e-9, "curve ends at the final score")
        assert r["levels_at_tokens"] == {"25000": 13, "50000": 35, "75000": 43, "100000": 48, "150000": 49}, r
        # level_tokens: sb26 clears level 1 on its 9th action, at 1,692 cumulative tokens
        sb26 = next(g for g in bm["game_runs"] if g["game_id"] == "sb26-7fbdac44")
        assert fr.level_tokens(sb26)[0] == 1692, fr.level_tokens(sb26)

        # hidden baselines: taaf writes final_score 0.0, which must fall back to the completion cap
        hid = copy.deepcopy(bm)
        for g in hid["game_runs"]:
            g["base_actions_per_level"], g["final_score"] = None, 0.0
        h = fr.bench_metrics(hid)
        cap = [sum(range(1, g["levels_completed"] + 1)) / sum(range(1, g["number_of_levels"] + 1)) * 100
               for g in bm["game_runs"]]
        near(h["mean_score"], sum(cap) / len(cap), 1e-9, "hidden-baseline mean score")
        sc = [h["score_at_tokens"][str(t)] for t in fr.TOKEN_POINTS]
        assert sc == sorted(sc) and sc[-1] == h["mean_score"], sc

        # n_passes > 1: repeated game ids must not overwrite each other
        twice = dict(bm, game_runs=bm["game_runs"] * 2)
        t2 = fr.bench_metrics(twice)
        assert t2["games"] == 20 and t2["levels"] == 98 and len(t2["per_game"]) == 20, t2["per_game"].keys()

        # kernel.log
        near(r["ready_s"], 640, 5, "READY time")
        assert r["server_boot_s"] == 556

        # transcripts: counters, and actions per opener once they sit next to benchmark.json
        assert "openers" not in r, "no transcripts, no transcript metrics"
        tx = tmp / "with-tx"
        shutil.copytree(run_dir, tx)
        (tx / "g1-abc_p0.txt").write_text(TX_FIXTURE)
        c = fr.transcript_metrics([tx / "g1-abc_p0.txt"])
        assert c == {"openers": 2, "responses": 3, "no_tool_call": 2, "finish_stop": 1, "finish_length": 1,
                     "tool_results": 2, "tool_result_keyerrors": 1}, c
        (tx / "serve.log").write_text("[2026-10-01 03:50:00] FlashInfer autotune completed.\n" + SERVE_FIXTURE)
        rt = fr.readout(tx)
        actions = sum(len(g["history"]) for g in bm["game_runs"])
        assert rt["actions"] == actions and rt["openers"] == 2 and rt["actions_per_opener"] == actions / 2, rt
        assert rt["autotune_completed"] == 1 and rt["prefill_new_tokens"] == 49551, rt  # boot line counts unwindowed

        # serve.log parsers, always: the fixture is one chunked re-prefill that no per-line count sees
        f = fr.serve_metrics(SERVE_FIXTURE, None)
        assert f["requests"] == 1 and f["prefills_over_20k"] == 1 and f["max_prefill_new_tokens"] == 49551, f
        assert f["prefill_new_tokens"] == 49551, f
        near(f["prefill_ms_per_1k_new"], 4140.87 / 49.551, 1e-9, "fixture prefill ms per 1K new tokens")
        near(f["cache_hit_pct"], 100 * 4672 / (4672 + 6 * 8192 + 448), 1e-9, "fixture cache hit %")
        assert (f["decode_tok_s_mean"], f["running_mean"], f["kv_mean"], f["accept_len_mean"]) == (162.49, 10, 0.80, 2.58), f
        w = fr._time("2026-10-01T04:23:25"), fr._time("2026-10-01T04:23:30")
        assert fr.serve_metrics(SERVE_FIXTURE, w)["requests"] is None, "ReqTimeStats at :34 is outside the window"

        # serve.log, whole run
        if have_serve:
            near(r["decode_tok_s_mean"], 665, 665 * 0.03, "server decode tok/s")
            near(r["kv_mean"], 0.57, 0.03, "KV mean")
            near(r["kv_p90"], 0.81, 0.015, "KV p90")
            assert r["kv_max"] == 0.91, r["kv_max"]
            near(r["cache_hit_pct"], 94.4, 0.2, "prefix cache hit %")
            near(r["accept_len_mean"], 2.71, 0.02, "accept length")
            near(r["running_mean"], 8.65, 0.3, "running requests")
            assert r["prefills_over_20k"] == 13 and r["max_prefill_new_tokens"] == 58524, r

        # gates, on synthetic arms built from the real readout
        assert fr.serving_gate(r, r).startswith("SERVING FAIL" if have_serve else "SERVING n/a")
        assert fr.agent_gate(r, r).startswith("AGENT FAIL")
        b = copy.deepcopy(r)
        b["tok_s"] *= 1.06
        b["levels"] -= 3
        b["cache_hit_pct"] = (b.get("cache_hit_pct") or 0) - 1.9
        a = dict(r, cache_hit_pct=r.get("cache_hit_pct") or 0)
        assert fr.serving_gate(a, b).startswith("SERVING PASS"), fr.serving_gate(a, b)
        b["levels"] -= 1
        assert fr.serving_gate(a, b).startswith("SERVING FAIL")
        # an arm that plays the same per token but lp85 loses its slot at 65K: throughput, not agent
        short = dict(bm, game_runs=[cut(g, 65_000) if g["game_id"].startswith("lp85") else g
                                    for g in bm["game_runs"]])
        a = fr.bench_metrics(short)
        assert a["levels"] == 48 and r["levels"] - a["levels"] == 1
        gate = fr.agent_gate(a, r)
        assert gate.startswith("AGENT FAIL") and "75K levels +0" in gate and "100K levels +0" in gate, gate
        assert "(2 of 10 games at full 100K)" in fr.agent_gate(r, r), fr.agent_gate(r, r)
        # an arm that clears every level on half the tokens: a real agent gain at equal tokens
        b = copy.deepcopy(r)
        for g in b["per_game"].values():
            g["level_tokens"] = [e // 2 for e in g["level_tokens"]]
        assert fr.agent_gate(r, b).startswith("AGENT PASS"), fr.agent_gate(r, b)
        b["levels"] -= 1
        assert fr.agent_gate(r, b).startswith("AGENT FAIL")
        assert fr.agent_gate({"levels": 0}, r).startswith("AGENT n/a")

        # CLI, both modes
        out = subprocess.run([sys.executable, str(TOOL), str(run_dir)], capture_output=True, text=True, check=True)
        assert "levels 49 over 10 games" in out.stdout and "100K 48 lv" in out.stdout, out.stdout
        assert sorted(p.name for p in run_dir.iterdir()) == before
        twin = tmp / "twin"
        shutil.copytree(run_dir, twin)
        dest = tmp / "readout.json"
        out = subprocess.run([sys.executable, str(TOOL), "--compare", str(run_dir), str(twin), "--out", str(dest)],
                             capture_output=True, text=True, check=True)
        assert "SERVING" in out.stdout and "AGENT FAIL" in out.stdout, out.stdout
        saved = json.loads(dest.read_text())
        assert saved["arm"]["levels"] == 49 and saved["agent_gate"].startswith("AGENT FAIL"), saved.keys()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("PASS" if have_serve else f"PASS (server-log checks skipped: no serve.log at {SERVE_LOG})")


if __name__ == "__main__":
    main()
