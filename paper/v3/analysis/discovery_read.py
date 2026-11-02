#!/usr/bin/env python3
"""Discovery readings from the Franzen-stack analyzer transcripts (research digest 2026-11-01, [transcripts] section).

Run from the repo root (stdlib only, no network):
    python3 -I -B paper/v3/analysis/discovery_read.py

Reads solution/kaggle/generated/kernel-outputs/<run>/*_p0.txt (one analyzer transcript per game) and <run>/benchmark.json
for three run groups: "baseline" (the 11 lf15 runs the digest parsed), "ustv" (the three replay-check runs of 11-01) and
"pre_unstick2" (the r1 to r3 runs behind the digest's "4% before unstick2"). Writes
paper/v3/analysis/figs/data/discovery_levels.csv (one row per level play in a transcript) and
paper/v3/analysis/figs/data/discovery_summary.json (headline numbers per group), and prints each headline next to the
digest's value.

Definitions are ported from the digest's scratch scripts: turn parsing (parse.py), clock and executed-action count
(common.py), level segments and per-turn flags (levels.py), first-action delay from benchmark.json (m1_gap.py), long
cleared levels (m2_phases.py, segment clock over 900 s), winning-code proxy (m2b_approach.py, first earlier turn on the
level whose code identifiers have Jaccard >= 0.5 with the clearing turn's code; cleared levels 2+ with 4+ turns),
multi-action outcomes (m3_plans.py) and the simulator-check detector (m6_validate.py). Four headlines had no saved
script, so their definitions here were fixed before comparing with the digest: idle turn = a turn that executed no action
(levels.py nexec), over all turns; human ratio = actions / human baseline on cleared levels, pooled median; level opening
reuse = the first turn of a level 2+ calls a function defined on an earlier turn of the game; tool error rate = responses
whose result holds a traceback or an XError line, over responses with code (levels.py flags). The m6 detector needs a
loop over transitions or history in the model's own code, so it cannot see a call to the ustv harness helper
check_model(); check_model_called_* in the summary counts those calls separately, and helper_called_* (with
sim_checked_and_helper_cleared, sim_checked_or_helper_cleared) counts a call to either replay-check helper, check_model()
or check_goal().

CSV columns: cleared is the transcript rule of levels.py (a later turn shows a higher level), so a level cleared on the
transcript's last turn reads 0 there (97 level-2+ levels in the baseline); bench_cleared is benchmark.json's
levels_completed. seconds_on_level is the transcript clock (first turn of the level to the first turn of the next).
seconds_before_first_action and bench_seconds_on_level come from benchmark.json (previous level's last action to this
level's first and last action) and are empty when the level has no action there; the 17.3% headline is their ratio
summed over level-2+ rows with bench_cleared 1. win_code_elapsed_frac is empty when the proxy did not match or the level
is not eligible. helper_called is 1 when a model code block on the level calls check_model( or check_goal(.
A turn without a "Current state" line has an empty level and splits its level into two rows.
"""
import builtins
import csv
import json
import keyword
import re
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
K = ROOT / "solution" / "kaggle" / "generated" / "kernel-outputs"
DATA = Path(__file__).resolve().parent / "figs" / "data"
GROUPS = {
    "baseline": [f"kaggle-lf15uns2-1030-{i}" for i in (1, 2, 3)] + [f"kaggle-lf15us8-1031-{i}" for i in (1, 2, 3)]
    + [f"kaggle-lf15usc-1031-{i}" for i in (1, 2, 3)] + [f"kaggle-lf15ust-1031-{i}" for i in (1, 2)],
    "ustv": [f"kaggle-lf15ustv-1101-{i}" for i in (1, 2, 3)],
    "pre_unstick2": [f"franzen-l1a1y8sv2r{i}-v1" for i in (1, 2, 3)],
}
DIGEST = {  # the digest's values for the baseline group (pre_unstick2 for sim_checked_pre)
    "turns": 16750, "plays": 275, "levels_cleared": 1387, "long_share_of_clears": .27, "long_share_of_clock": .66,
    "win_code_left_median": .13, "win_code_matched": "94 of 272", "first_action_clock_share": .173,
    "first_action_median_s": 83, "idle_turn_share": .13, "idle_clock_share": .30, "idle_cut_share": .22,
    "plan_n": 7494, "plan_guard_stop": .07, "plan_game_over": .01, "human_ratio_median": .64, "opener_reuse": .81,
    "tool_error_rate": .082, "sim_checked_cleared": .06,
}

# parse.py
HEAD = re.compile(r"^--- analysis_step=(\d+) \| action=(\d+) \| (\d\d):(\d\d):(\d\d) \| tool-agent ---$", re.M)
STATE = re.compile(r"Current state: step (\d+), level (\d+)\.")
SEC = re.compile(r"^\[(SYSTEM PROMPT|USER PROMPT|THINKING|TOOL CALL: python|TOOL RESULT: python|MODEL RESPONSE META|"
                 r"ANALYZER STATUS|ASSISTANT)\]\n", re.M)
CODE = re.compile(r"<parameter=code>\n?(.*?)\n?</parameter>", re.S)
ACT = re.compile(r"^\[action\] (.*)$", re.M)
BATCH = re.compile(r"batch of (\d+) actions, (\d+) executed")
# levels.py
SIM_DEF = re.compile(r"def \w*(sim|simulate|step|apply|move|predict|model|transition)\w*\s*\(", re.I)
DEF = re.compile(r"^def (\w+)\s*\(", re.M)
CALL = re.compile(r"\b([A-Za-z_]\w*)\s*\(")
# m6_validate.py
LOOP = re.compile(r"for\s+[^\n]*\bin\s+[^\n]*(transitions|history)\b")
AFTER = re.compile(r"after_frame|\.frame\b|after\b")
CMP = re.compile(r"!=|==|mismatch|match|ok\b|assert", re.I)
SIMCALL = re.compile(r"\b(\w*(sim|predict|step|apply|model|move|simulate|transition|forward)\w*)\s*\(", re.I)
# m2b_approach.py
STOP = set(keyword.kwlist) | set(dir(builtins)) | {
    "current_frame", "segmentation", "nodes", "ascii", "action", "history", "transitions", "previous_frame",
    "last_transition", "frame_diff", "print", "len", "range", "i", "j", "k", "r", "c", "n", "x", "y", "a", "b", "f", "e",
    "s", "t", "p", "d", "m", "v", "g", "color", "bbox", "pixels", "id", "append", "items", "get", "split", "join",
    "last_action_call_result", "level_completed", "no_op", "row", "col", "MOUSE", "UP", "DOWN", "LEFT", "RIGHT", "SPACE",
    "RESET", "UNDO", "fr", "res", "out"}
IDENT = re.compile(r"\b[A-Za-z_]\w+\b")


def nexec(lines):  # common.py
    n = 0
    for a in lines:
        m = BATCH.search(a)
        if m:
            n += int(m.group(2))
        elif "not executed" in a:
            pass
        elif " -> " in a:
            n += 1
    return n


def outcome(res):  # m3_plans.py
    n, o = 0, "ran_out"
    for a in ACT.findall(res):
        m = BATCH.search(a)
        if m:
            n += int(m.group(2))
        elif "not executed" not in a and " -> " in a:
            n += 1
        if "LEVEL COMPLETED" in a or "stop_reason: level_completed" in a or "RUN COMPLETE" in a or "run_complete" in a:
            o = "level_completed"
        elif "GAME OVER" in a or "game_over" in a:
            o = "game_over"
        elif "stale_state" in a or "no_op_action" in a:
            o = "guard_stop" if o == "ran_out" else o
    if o == "ran_out" and "StaleStateActionError" in res:
        o = "guard_stop"
    return n, o


def validates(code, defined):  # m6_validate.py
    if not LOOP.search(code) or not AFTER.search(code) or not CMP.search(code):
        return False
    return any(m.group(1) in defined for m in SIMCALL.finditer(code))


def ids(code):  # m2b_approach.py
    return {w for w in IDENT.findall(code) if w not in STOP and not w.isdigit()}


def parse(path):
    """Turns of one transcript: level, clock t and dur (common.py), executed actions, responses (code, result, finish)."""
    text = path.read_text(errors="replace")
    heads = list(HEAD.finditer(text))
    turns = []
    for i, h in enumerate(heads):
        parts = SEC.split(text[h.end(): heads[i + 1].start() if i + 1 < len(heads) else len(text)])
        user, resp, cur = "", [], None
        for name, body in zip(parts[1::2], parts[2::2]):
            if name == "USER PROMPT":
                user = body
            elif name == "MODEL RESPONSE META":
                fr = re.search(r"finish_reason: (\S+)", body)
                cur = dict(finish=fr.group(1) if fr else "", code="", result="")
                resp.append(cur)
            elif name == "THINKING" and cur is None:
                cur = dict(finish="", code="", result="")
                resp.append(cur)
            elif name == "TOOL CALL: python" and cur is not None:
                m = CODE.search(body)
                cur["code"] = m.group(1) if m else body
            elif name == "TOOL RESULT: python" and cur is not None:
                cur["result"] = body
        m = STATE.search(user)
        turns.append(dict(clock=int(h.group(3)) * 3600 + int(h.group(4)) * 60 + int(h.group(5)), resp=resp,
                          level=int(m.group(2)) if m else None,
                          nexec=nexec([a for r in resp for a in ACT.findall(r["result"])])))
    prev = off = 0
    for t in turns:  # seconds since the first turn, midnight unwrapped
        s = t["clock"] - turns[0]["clock"] + off
        if s < prev - 3600:
            off += 86400
            s += 86400
        t["t"] = prev = s
    for i, t in enumerate(turns):
        t["dur"] = turns[i + 1]["t"] - t["t"] if i + 1 < len(turns) else 0
    return turns


def bench_levels(run):
    """m1_gap.py: per level with an action, the delay from the previous level's last action to its first action."""
    out = {}
    for g in json.loads((K / run / "benchmark.json").read_text())["game_runs"]:
        h, cum, prev_end = g["history"], 0, 0.0
        for k, n in enumerate(g["actions_per_level"]):
            if n == 0:
                break
            acts = h[cum:cum + n]
            cum += n
            first, end = acts[0]["wallclock_seconds"], acts[-1]["wallclock_seconds"]
            out[(g["game_id"][:4], k + 1)] = dict(cleared=k < g["levels_completed"], gap=first - prev_end,
                                                  gap_tok=acts[0]["generated_tokens"], dur=end - prev_end, nact=n,
                                                  base=g["base_actions_per_level"][k])
            prev_end = end
    return out


def read_game(path, tally):
    """Level rows of one game play; adds turn-level counts to tally."""
    ts = parse(path)
    tally["plays"] += 1
    tally["turns"] += len(ts)
    for t in ts:
        idle = t["nexec"] == 0
        tally["clock"] += t["dur"]
        tally["idle_turns"] += idle
        tally["idle_clock"] += t["dur"] if idle else 0
        tally["idle_cut"] += idle and any(r["finish"] == "length" for r in t["resp"])
        for r in t["resp"]:
            if r["code"]:
                tally["calls"] += 1
                tally["errors"] += "Traceback" in r["result"] or bool(re.search(r"^\w+Error\b", r["result"], re.M))
            n, o = outcome(r["result"])
            if n >= 3:
                tally["plan_n"] += 1
                tally["plan_" + o] += 1
    rows, defined, i = [], set(), 0
    while i < len(ts):
        L, j = ts[i]["level"], i
        while j < len(ts) and ts[j]["level"] == L:
            j += 1
        seg = ts[i:j]
        cleared = j < len(ts) and ts[j]["level"] is not None and L is not None and ts[j]["level"] > L
        t0, t1 = ts[i]["t"], ts[j]["t"] if j < len(ts) else ts[-1]["t"] + ts[-1]["dur"]
        dur = t1 - t0
        opener_reuse = bool({n for r in seg[0]["resp"] for n in CALL.findall(r["code"])} & defined)
        sim = val = cm = hc = False
        for t in seg:
            for r in t["resp"]:
                defined |= set(DEF.findall(r["code"]))
                sim = sim or bool(SIM_DEF.search(r["code"]))
                val = val or validates(r["code"], defined)
                cm = cm or "check_model(" in r["code"]
                hc = hc or "check_model(" in r["code"] or "check_goal(" in r["code"]
        frac = None
        if cleared and L is not None and L >= 2 and len(seg) >= 4:
            wc = ids("\n".join(r["code"] for r in seg[-1]["resp"]))
            for t in seg[:-1]:
                c = ids("\n".join(r["code"] for r in t["resp"]))
                if wc and c and len(wc & c) / len(wc | c) >= 0.5:
                    frac = (t["t"] - t0) / max(1, dur)
                    break
        rows.append(dict(level=L, cleared=cleared, dur=dur, nturns=len(seg), frac=frac, sim=sim, val=val, cm=cm, hc=hc,
                         opener_reuse=opener_reuse, idle_turns=sum(t["nexec"] == 0 for t in seg),
                         idle_clock=sum(t["dur"] for t in seg if t["nexec"] == 0)))
        i = j
    return rows


def share(a, b):
    return round(a / b, 4) if b else None


def read_group(runs):
    tally = dict.fromkeys(["plays", "turns", "clock", "idle_turns", "idle_clock", "idle_cut", "calls", "errors", "plan_n",
                           "plan_ran_out", "plan_level_completed", "plan_guard_stop", "plan_game_over"], 0)
    rows, bench = [], []
    for run in runs:
        b = bench_levels(run)
        bench += [dict(run=run, k=k, **v) for (g, k), v in b.items()]
        for f in sorted((K / run).glob("*_p0.txt")):
            game = f.name.split("-")[0]
            for r in read_game(f, tally):
                bl = b.get((game, r["level"]))
                rows.append(dict(run=run, game=game, bench=bl, **r))
    cl2 = [r for r in rows if r["cleared"] and r["level"] and r["level"] >= 2]
    long = [r for r in cl2 if r["dur"] > 900]
    elig = [r for r in long if r["nturns"] >= 4]
    left = [1 - r["frac"] for r in elig if r["frac"] is not None]
    b2 = [x for x in bench if x["k"] >= 2]
    b2c = [x for x in b2 if x["cleared"]]
    ratios = [x["nact"] / x["base"] for x in bench if x["cleared"] and x["base"]]
    by_run = [st.median([x["nact"] / x["base"] for x in bench if x["run"] == u and x["cleared"] and x["base"]])
              for u in runs]
    open2 = [r for r in rows if r["level"] and r["level"] >= 2]
    stuck2 = [r for r in rows if not r["cleared"] and r["level"] and r["level"] >= 2]
    s = dict(
        runs=runs, plays=tally["plays"], turns=tally["turns"],
        levels_cleared=sum(1 for x in bench if x["cleared"]), cleared_l2_segments=len(cl2),
        cleared_l2_bench=len(b2c),
        long_n=len(long), long_share_of_clears=share(len(long), len(cl2)),
        long_share_of_clock=share(sum(r["dur"] for r in long), sum(r["dur"] for r in cl2)),
        win_code_eligible=len(elig), win_code_matched=len(left),
        win_code_left_median=round(st.median(left), 4) if left else None,
        first_action_clock_share=share(sum(x["gap"] for x in b2c), sum(x["dur"] for x in b2c)),
        first_action_median_s=round(st.median(x["gap"] for x in b2), 1), first_action_n=len(b2),
        first_action_median_s_cleared=round(st.median(x["gap"] for x in b2c), 1),
        first_action_median_tokens=st.median(x["gap_tok"] for x in b2),
        idle_turn_share=share(tally["idle_turns"], tally["turns"]),
        idle_clock_share=share(tally["idle_clock"], tally["clock"]),
        idle_cut_share=share(tally["idle_cut"], tally["idle_turns"]),
        idle_turn_share_cleared_l2=share(sum(r["idle_turns"] for r in cl2), sum(r["nturns"] for r in cl2)),
        idle_clock_share_cleared_l2=share(sum(r["idle_clock"] for r in cl2), sum(r["dur"] for r in cl2)),
        plan_n=tally["plan_n"], **{k: share(tally[k], tally["plan_n"]) for k in
                                   ("plan_ran_out", "plan_level_completed", "plan_guard_stop", "plan_game_over")},
        human_ratio_median=round(st.median(ratios), 4), human_ratio_median_run_mean=round(st.mean(by_run), 4),
        opener_n=len(open2), opener_reuse=share(sum(r["opener_reuse"] for r in open2), len(open2)),
        tool_calls=tally["calls"], tool_error_rate=share(tally["errors"], tally["calls"]),
        sim_checked_cleared=share(sum(r["val"] for r in cl2), len(cl2)),
        sim_checked_stuck=share(sum(r["val"] for r in stuck2), len(stuck2)),
        sim_checked_long=share(sum(r["val"] for r in long), len(long)),
        sim_defined_cleared=share(sum(r["sim"] for r in cl2), len(cl2)),
        check_model_called_cleared=share(sum(r["cm"] for r in cl2), len(cl2)),
        check_model_called_stuck=share(sum(r["cm"] for r in stuck2), len(stuck2)),
        sim_checked_or_check_model_cleared=share(sum(r["val"] or r["cm"] for r in cl2), len(cl2)),
        helper_called_cleared=share(sum(r["hc"] for r in cl2), len(cl2)),
        helper_called_stuck=share(sum(r["hc"] for r in stuck2), len(stuck2)),
        sim_checked_and_helper_cleared=share(sum(r["val"] and r["hc"] for r in cl2), len(cl2)),
        sim_checked_or_helper_cleared=share(sum(r["val"] or r["hc"] for r in cl2), len(cl2)),
    )
    return rows, s


def main():
    out, summary = [], {}
    for name, runs in GROUPS.items():
        rows, summary[name] = read_group(runs)
        for r in rows:
            b = r["bench"] or {}
            out.append(dict(group=name, run=r["run"], game=r["game"], level="" if r["level"] is None else r["level"],
                            cleared=int(r["cleared"]), seconds_on_level=r["dur"],
                            seconds_before_first_action="" if not b else round(b["gap"], 1),
                            bench_seconds_on_level="" if not b else round(b["dur"], 1),
                            bench_cleared="" if not b else int(b["cleared"]),
                            win_code_elapsed_frac="" if r["frac"] is None else round(r["frac"], 4),
                            sim_defined=int(r["sim"]), sim_checked=int(r["val"]), helper_called=int(r["hc"])))
    with open(DATA / "discovery_levels.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    summary["digest"] = dict(DIGEST, sim_checked_cleared_pre_unstick2=.04)
    (DATA / "discovery_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    base, ustv, pre = summary["baseline"], summary["ustv"], summary["pre_unstick2"]
    print("%-28s %10s %10s %10s %10s" % ("metric", "digest", "baseline", "ustv", "pre_uns2"))
    for k, v in DIGEST.items():
        if k == "win_code_matched":
            got = ["%d of %d" % (s["win_code_matched"], s["win_code_eligible"]) for s in (base, ustv, pre)]
        else:
            got = [s[k] for s in (base, ustv, pre)]
        print("%-28s %10s %10s %10s %10s" % (k, v, *got))
    print("%-28s %10s %10s %10s %10s" % ("sim_checked (pre_unstick2)", .04, "", "", pre["sim_checked_cleared"]))
    for k in ("first_action_median_s_cleared", "first_action_median_tokens", "idle_turn_share_cleared_l2",
              "idle_clock_share_cleared_l2", "human_ratio_median_run_mean", "sim_checked_stuck", "sim_checked_long",
              "sim_defined_cleared", "check_model_called_cleared", "check_model_called_stuck",
              "sim_checked_or_check_model_cleared", "cleared_l2_segments", "cleared_l2_bench", "helper_called_cleared",
              "helper_called_stuck", "sim_checked_and_helper_cleared", "sim_checked_or_helper_cleared"):
        print("%-28s %10s %10s %10s %10s" % (k, "", base[k], ustv[k], pre[k]))
    print("\nwrote", (DATA / "discovery_levels.csv").relative_to(ROOT), "and",
          (DATA / "discovery_summary.json").relative_to(ROOT))


if __name__ == "__main__":
    main()
