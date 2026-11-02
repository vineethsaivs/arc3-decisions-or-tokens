"""Readout for one Franzen-fork run, or a gate between two (plan 2026-10-01, section 4).

Input is a kernel output directory as written by `solution/kaggle/kapi.sh output SLUG DIR`
(files land flat by basename): benchmark.json, kernel.log (JSON array of {stream_name,time,data}),
serve.log (SGLang) and, if fetched (`kapi.sh output SLUG DIR transcripts/`, ~50 MB), the per-game
transcripts <game>_p<N>.txt. Missing files only blank the metrics that need them.

    python3 solution/analysis/franzen_readout.py DIR [--out readout.json]
    python3 solution/analysis/franzen_readout.py --compare CONTROL_DIR ARM_DIR [--out readout.json]

Input directories are never written (kernel outputs under generated/ are read-only evidence);
the JSON goes only where --out says.

Where each number comes from:
  levels, scores      benchmark.json game_runs (final_score, else score_benchmark.game_score; taaf
                      writes final_score 0.0 when baselines are hidden, so that case is recomputed)
  tok/s               sum of per-game `tokens=N` / benchmark start_time..end_time (harness side)
  ready_s             kernel.log time of the 'READY after Ns' line (seconds after notebook start)
  decode tok/s, running, KV, accept len
                      SGLang 'Decode batch' lines inside the play window, one sample per line
  cache hit           sum #cached-token / sum (#cached + #new) over 'Prefill batch' lines
  openers, responses  transcripts: [USER PROMPT] sections (one per analyzer turn opener) and
                      finish_reason lines; no-tool-call = tool_call_count 0; KeyErrors = TOOL RESULT
                      sections containing one; actions/opener = benchmark actions / openers.
                      Baseline franzen-rep-v1: 1,561 openers, 2,247 responses, 27 without a tool
                      call, 3.09 actions per opener.
  prefills > 20K      ReqTimeStats input_len - cached_input_len (a chunked prefill logs one line
                      per 8K chunk, so per-line counts can never see a big one)
  prefill ms/1K new   sum of ReqTimeStats initial_prefill_elapsed / sum of new prefill tokens
                      (input_len - cached_input_len), times 1,000: what a new prompt token costs
  autotune completed  'FlashInfer autotune completed.' lines anywhere in serve.log (boot, so unwindowed)
  levels/score at tokens
                      per game, levels whose completing action falls at or below T cumulative
                      generated tokens (benchmark history deltas), scored with the real formula;
                      a game that ended before T keeps its final levels and score (one run only)

Gates (A = control, B = arm):
  SERVING PASS  tok/s up >= 5%, cache hit not down more than 2 points, levels not down more than 3
  AGENT PASS    levels at matched tokens higher at both 75K and 100K, total levels not lower.
                Matched: each game is cut at min(T, tokens_A, tokens_B) in BOTH arms, so a game
                that stopped early (clock, starved by the priority gate) in one arm is not compared
                against tokens it never had. That strips out throughput luck, which the plan's
                serving gate measures instead.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_benchmark import game_score  # noqa: E402

TOKEN_POINTS = (25_000, 50_000, 75_000, 100_000, 150_000)
BIG_PREFILL = 20_000
SERVE_TOK_GAIN = 0.05
SERVE_CACHE_DROP_PTS = 2.0
SERVE_LEVEL_DROP = 3
AGENT_POINTS = (75_000, 100_000)

_TS = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\] ")
_FIELD = re.compile(r"(#?[a-z][a-z \-]*(?:\(token/s\))?): ([0-9.]+)")
_REQ = re.compile(r"\b(input_len|cached_input_len|output_len)=(\d+)")
_PREFILL_MS = re.compile(r"\binitial_prefill_elapsed=([0-9.]+)ms")
_READY = re.compile(r"READY after (\d+(?:\.\d+)?)s")
_SECTION = re.compile(r"^\[(SYSTEM PROMPT|USER PROMPT|THINKING|ASSISTANT|ANALYZER STATUS|MODEL RESPONSE META"
                      r"|TOOL CALL:[^\]]*|TOOL RESULT:[^\]]*)\]$")


def _p90(xs: list) -> float | None:
    if not xs:
        return None
    return statistics.quantiles(xs, n=10, method="inclusive")[-1] if len(xs) > 1 else xs[0]


def _mean(xs: list) -> float | None:
    return statistics.mean(xs) if xs else None


def _time(s: str | None) -> float | None:
    return dt.datetime.fromisoformat(s).timestamp() if s else None


def _game_tokens(run: dict) -> int:
    m = re.search(r"tokens=(\d+)", run.get("solver_note") or "")
    return int(m.group(1)) if m else sum(h.get("generated_tokens") or 0 for h in run.get("history") or [])


def _score(run: dict, done: int) -> float:
    n = int(run.get("number_of_levels") or 0)
    s = game_score(run.get("actions_per_level") or [], run.get("base_actions_per_level") or [], done, n)
    if s is None:  # no baselines (submission mode): completion cap only, i.e. efficiency 1.0
        s = sum(range(1, done + 1)) / max(1, sum(range(1, n + 1))) * 100.0
    return s


def level_tokens(run: dict) -> list[int]:
    """Cumulative generated tokens at the action that completed each cleared level."""
    cum, toks = 0, []
    for h in run.get("history") or []:
        cum += h.get("generated_tokens") or 0
        toks.append(cum)
    out, a = [], 0
    for i in range(int(run.get("levels_completed") or 0)):
        a += run["actions_per_level"][i]
        out.append(toks[min(a, len(toks)) - 1] if toks else 0)
    return out


def bench_metrics(bm: dict) -> dict:
    runs = bm.get("game_runs") or []
    per_game, finals, tokens = {}, [], 0
    curve = {t: [] for t in TOKEN_POINTS}
    at_levels = {t: 0 for t in TOKEN_POINTS}
    reached = {t: 0 for t in TOKEN_POINTS}
    for i, r in enumerate(runs):
        done = int(r.get("levels_completed") or 0)
        final = r.get("final_score")
        hidden = r.get("base_actions_per_level") is None  # taaf then writes 0.0, not None
        final = float(final) if final is not None and not hidden else _score(r, done)
        finals.append(final)
        gt = _game_tokens(r)
        tokens += gt
        ends = level_tokens(r)
        by_k = [_score(r, k) for k in range(done)] + [final]
        key = r["game_id"] if r["game_id"] not in per_game else f"{r['game_id']}#{i}"  # n_passes > 1
        per_game[key] = {"levels": done, "of": r.get("number_of_levels"), "score": round(final, 2),
                         "tokens": gt, "level_tokens": ends, "score_by_levels": by_k}
        for t in TOKEN_POINTS:
            k = sum(1 for e in ends if e <= t)
            curve[t].append(by_k[k])
            at_levels[t] += k
            reached[t] += gt >= t
    start, end = _time(bm.get("start_time")), _time(bm.get("end_time"))
    window = (end - start) if start and end else max((r.get("final_wallclock_seconds") or 0 for r in runs), default=0)
    return {
        "games": len(runs),
        "levels": sum(g["levels"] for g in per_game.values()),
        "zero_level_games": sum(1 for g in per_game.values() if g["levels"] == 0),
        "mean_score": _mean(finals),
        "per_game": per_game,
        "generated_tokens": tokens,
        "actions": sum(len(r.get("history") or []) for r in runs),
        "play_window_s": window,
        "tok_s": tokens / window if window else None,
        "score_at_tokens": {str(t): _mean(v) for t, v in curve.items()},
        "levels_at_tokens": {str(t): v for t, v in at_levels.items()},
        "games_reaching_tokens": {str(t): v for t, v in reached.items()},
        "_window": (start, end),
    }


def ready_metrics(text: str) -> dict:
    try:
        events = json.loads(text)
    except ValueError:
        events = [{"time": None, "data": text}]
    for e in events:
        m = _READY.search(e.get("data") or "")
        if m:
            return {"ready_s": e.get("time"), "server_boot_s": float(m.group(1))}
    return {"ready_s": None, "server_boot_s": None}


def serve_metrics(text: str, window: tuple) -> dict:
    lo, hi = window if window and all(window) else (float("-inf"), float("inf"))
    dec, pre, req = [], [], []
    for line in text.splitlines():
        m = _TS.match(line)
        if not m:
            continue
        t = dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").timestamp()
        if not lo - 1 <= t <= hi + 1:  # serve.log stamps are whole seconds
            continue
        if "Decode batch" in line:
            dec.append({k.strip(): float(v) for k, v in _FIELD.findall(line)})
        elif "Prefill batch" in line:
            pre.append({k.strip(): float(v) for k, v in _FIELD.findall(line)})
        elif "ReqTimeStats" in line:
            r = {k: int(v) for k, v in _REQ.findall(line)}
            ms = _PREFILL_MS.search(line)
            if ms:
                r["prefill_ms"] = float(ms.group(1))
            req.append(r)
    col = lambda rows, k: [r[k] for r in rows if k in r]  # noqa: E731
    gen, run, kv = col(dec, "gen throughput (token/s)"), col(dec, "#running-req"), col(dec, "full token usage")
    cached, new = sum(col(pre, "#cached-token")), sum(col(pre, "#new-token"))
    req = [r for r in req if "input_len" in r and "cached_input_len" in r]
    timed = [r for r in req if "prefill_ms" in r]
    new_prefill = sum(r["input_len"] - r["cached_input_len"] for r in timed)
    return {
        "decode_tok_s_mean": _mean(gen), "decode_tok_s_p90": _p90(gen),
        "running_mean": _mean(run), "running_p90": _p90(run),
        "kv_mean": _mean(kv), "kv_p90": _p90(kv), "kv_max": max(kv) if kv else None,
        "cache_hit_pct": 100.0 * cached / (cached + new) if cached + new else None,
        "accept_len_mean": _mean(col(dec, "accept len")),
        "requests": len(req) if req else None,
        "prefills_over_20k": sum(1 for r in req if r["input_len"] - r["cached_input_len"] > BIG_PREFILL) if req else None,
        "max_prefill_new_tokens": max((r["input_len"] - r["cached_input_len"] for r in req), default=None),
        "prefill_new_tokens": new_prefill if timed else None,
        "prefill_ms_per_1k_new": 1000.0 * sum(r["prefill_ms"] for r in timed) / new_prefill if new_prefill else None,
    }


def transcript_metrics(paths: list[Path]) -> dict:
    """Counters over per-game transcripts: turn openers, model responses and tool results."""
    c = dict.fromkeys(("openers", "responses", "no_tool_call", "finish_stop", "finish_length",
                       "tool_results", "tool_result_keyerrors"), 0)
    for path in paths:
        section, seen = "", False
        for line in path.read_text(errors="replace").splitlines():
            m = _SECTION.match(line)
            if m:
                section, seen = m.group(1), False
                c["openers"] += section == "USER PROMPT"
                c["tool_results"] += section.startswith("TOOL RESULT")
            elif section == "MODEL RESPONSE META":
                c["responses"] += line.startswith("finish_reason:")
                c["finish_stop"] += line == "finish_reason: stop"
                c["finish_length"] += line == "finish_reason: length"
                c["no_tool_call"] += line == "tool_call_count: 0"
            elif section.startswith("TOOL RESULT") and not seen and "KeyError" in line:
                seen = True
                c["tool_result_keyerrors"] += 1
    return c


def readout(d: Path) -> dict:
    """Read-only: returns the metrics, writes nothing."""
    d = Path(d)
    out: dict = {"dir": str(d)}
    window = None
    if (d / "benchmark.json").is_file():
        b = bench_metrics(json.loads((d / "benchmark.json").read_text()))
        window = b.pop("_window")
        out.update(b)
    if (d / "kernel.log").is_file():
        out.update(ready_metrics((d / "kernel.log").read_text(errors="replace")))
    if (d / "serve.log").is_file():
        text = (d / "serve.log").read_text(errors="replace")
        sm = serve_metrics(text, window)
        if sm["decode_tok_s_mean"] is None:  # clocks disagree (timezone): fall back to the whole log
            sm = dict(serve_metrics(text, None), serve_window="whole log")
        out.update(sm, autotune_completed=text.count("FlashInfer autotune completed."))
    transcripts = sorted(d.glob("*_p[0-9]*.txt"))
    if transcripts:
        out.update(transcript_metrics(transcripts))
        out["actions_per_opener"] = out["actions"] / out["openers"] if out.get("actions") and out["openers"] else None
    return out


def _f(v, nd=1, pct=False) -> str:
    if v is None:
        return "n/a"
    return f"{v:.{nd}f}{'%' if pct else ''}" if isinstance(v, float) else str(v)


def report(r: dict) -> str:
    g = r.get("per_game") or {}
    curve = r.get("score_at_tokens") or {}
    lv = r.get("levels_at_tokens") or {}
    reach = r.get("games_reaching_tokens") or {}
    return "\n".join([
        f"{Path(r['dir']).name}",
        f"  levels {r.get('levels', 'n/a')} over {r.get('games', 'n/a')} games, {r.get('zero_level_games', 'n/a')} at 0 levels"
        f" | mean score {_f(r.get('mean_score'), 2)}",
        "  " + ", ".join(f"{k.split('-')[0]} {v['levels']}/{v['of']} {v['score']:.0f}" for k, v in g.items()),
        f"  READY {_f(r.get('ready_s'))} s (server boot {_f(r.get('server_boot_s'), 0)} s)"
        f" | FlashInfer autotune completed {_f(r.get('autotune_completed'))}x",
        f"  tok/s {_f(r.get('tok_s'))} harness ({r.get('generated_tokens', 'n/a')} tokens / {_f(r.get('play_window_s'), 0)} s)"
        f" | decode {_f(r.get('decode_tok_s_mean'))} mean, {_f(r.get('decode_tok_s_p90'))} p90",
        f"  running {_f(r.get('running_mean'), 2)} mean, {_f(r.get('running_p90'))} p90"
        f" | KV {_f(r.get('kv_mean'), 3)} mean, {_f(r.get('kv_p90'), 2)} p90, {_f(r.get('kv_max'), 2)} max",
        f"  cache hit {_f(r.get('cache_hit_pct'), 1, True)} | accept len {_f(r.get('accept_len_mean'), 2)}"
        f" | prefills >20K new: {_f(r.get('prefills_over_20k'))} of {_f(r.get('requests'))}"
        f" (max {_f(r.get('max_prefill_new_tokens'))})"
        f" | prefill {_f(r.get('prefill_ms_per_1k_new'), 2)} ms per 1K new tokens ({_f(r.get('prefill_new_tokens'))} new)",
        f"  openers {_f(r.get('openers'))}, {_f(r.get('actions_per_opener'), 2)} actions each"
        f" | responses {_f(r.get('responses'))}, {_f(r.get('no_tool_call'))} without a tool call"
        f" (stop {_f(r.get('finish_stop'))}, length {_f(r.get('finish_length'))})"
        f" | KeyErrors in {_f(r.get('tool_result_keyerrors'))} of {_f(r.get('tool_results'))} tool results",
        "  at tokens/game, levels and mean score: " + " | ".join(
            f"{int(t) // 1000}K {lv.get(t, 'n/a')} lv {_f(v)} ({reach.get(t, 0)} reached)" for t, v in curve.items()),
    ])


def serving_gate(a: dict, b: dict) -> str:
    if not a.get("tok_s") or not b.get("tok_s") or a.get("cache_hit_pct") is None or b.get("cache_hit_pct") is None:
        return "SERVING n/a: needs benchmark.json and serve.log on both arms"
    gain = b["tok_s"] / a["tok_s"] - 1
    dc = b["cache_hit_pct"] - a["cache_hit_pct"]
    dl = b["levels"] - a["levels"]
    ok = gain >= SERVE_TOK_GAIN and dc >= -SERVE_CACHE_DROP_PTS and dl >= -SERVE_LEVEL_DROP
    return (f"SERVING {'PASS' if ok else 'FAIL'}: tok/s {gain:+.1%} (need >= +{SERVE_TOK_GAIN:.0%}),"
            f" cache {dc:+.1f} pts (need >= -{SERVE_CACHE_DROP_PTS:g}), levels {dl:+d} (need >= -{SERVE_LEVEL_DROP})")


def matched(a: dict, b: dict, t: int) -> dict:
    """Levels and score per arm, each game cut at min(t, tokens_A, tokens_B) in both arms."""
    ga, gb = a.get("per_game") or {}, b.get("per_game") or {}
    m = {"games": 0, "full": 0, "levels_a": 0, "levels_b": 0, "score_a": 0.0, "score_b": 0.0}
    for g in ga.keys() & gb.keys():
        tg = min(t, ga[g]["tokens"], gb[g]["tokens"])
        m["games"] += 1
        m["full"] += tg == t
        for side, x in (("a", ga[g]), ("b", gb[g])):
            k = sum(1 for e in x["level_tokens"] if e <= tg)
            m["levels_" + side] += k
            m["score_" + side] += x["score_by_levels"][k]
    return m


def agent_gate(a: dict, b: dict) -> str:
    ms = {t: matched(a, b, t) for t in AGENT_POINTS}
    if not all(m["games"] for m in ms.values()):
        return "AGENT n/a: needs benchmark.json on both arms with games in common"
    dl = b["levels"] - a["levels"]
    ok = all(m["levels_b"] > m["levels_a"] for m in ms.values()) and dl >= 0
    shown = ", ".join(
        f"{t // 1000}K levels {m['levels_b'] - m['levels_a']:+d} score {(m['score_b'] - m['score_a']) / m['games']:+.1f}/game"
        f" ({m['full']} of {m['games']} games at full {t // 1000}K)" for t, m in ms.items())
    return (f"AGENT {'PASS' if ok else 'FAIL'}: at matched tokens {shown} (levels must be higher at both),"
            f" total levels {dl:+d} (need >= 0)")


def compare(a: dict, b: dict) -> str:
    rows = [("levels", "levels", 0), ("zero-level games", "zero_level_games", 0), ("mean score", "mean_score", 2),
            ("READY s", "ready_s", 1), ("tok/s harness", "tok_s", 1), ("decode tok/s mean", "decode_tok_s_mean", 1),
            ("running mean", "running_mean", 2), ("KV mean", "kv_mean", 3), ("KV p90", "kv_p90", 3),
            ("KV max", "kv_max", 2), ("cache hit %", "cache_hit_pct", 2), ("accept len", "accept_len_mean", 2),
            ("prefills >20K", "prefills_over_20k", 0), ("prefill new tokens", "prefill_new_tokens", 0),
            ("prefill ms/1K new", "prefill_ms_per_1k_new", 2), ("autotune completed", "autotune_completed", 0),
            ("openers", "openers", 0),
            ("actions/opener", "actions_per_opener", 2), ("no tool call", "no_tool_call", 0),
            ("tool KeyErrors", "tool_result_keyerrors", 0)]
    rows += [(f"levels @ {t // 1000}K", ("levels_at_tokens", str(t)), 0) for t in TOKEN_POINTS]
    rows += [(f"score @ {t // 1000}K", ("score_at_tokens", str(t)), 1) for t in TOKEN_POINTS]

    def get(r, k):
        return (r.get(k[0]) or {}).get(k[1]) if isinstance(k, tuple) else r.get(k)

    out = [f"{'':20s} {Path(a['dir']).name[:14]:>14s} {Path(b['dir']).name[:14]:>14s} {'delta':>10s}"]
    for label, k, nd in rows:
        va, vb = get(a, k), get(b, k)
        d = f"{vb - va:+.{nd}f}" if va is not None and vb is not None else "n/a"
        fmt = lambda v: "n/a" if v is None else f"{v:.{nd}f}"  # noqa: E731
        out.append(f"{label:20s} {fmt(va):>14s} {fmt(vb):>14s} {d:>10s}")
    return "\n".join(out + [serving_gate(a, b), agent_gate(a, b)])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dir", nargs="?", type=Path)
    ap.add_argument("--compare", nargs=2, type=Path, metavar=("CONTROL", "ARM"))
    ap.add_argument("--out", type=Path, help="write the readout JSON here (nothing is written otherwise)")
    args = ap.parse_args()
    if args.compare:
        a, b = (readout(p) for p in args.compare)
        print(report(a), report(b), "", compare(a, b), sep="\n")
        data = {"control": a, "arm": b, "serving_gate": serving_gate(a, b), "agent_gate": agent_gate(a, b)}
    elif args.dir:
        data = readout(args.dir)
        print(report(data))
    else:
        ap.error("give DIR or --compare CONTROL ARM")
    if args.out:
        args.out.write_text(json.dumps(data, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
