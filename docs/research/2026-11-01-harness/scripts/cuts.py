import re, sys, glob, os, statistics as st
from collections import Counter

HDR = re.compile(r"^--- (?:analysis_step=\d+ \| )?action=(\d+) \| (\d\d):(\d\d):(\d\d) \| tool-agent ---$")
SEC = re.compile(r"^\[([A-Z][A-Za-z :_]*?)(?:: [a-z_]+)?\]$")
LVL = re.compile(r"Current state: step (\d+), level (\d+)")


def parse(path):
    invs = []
    cur = None
    sec = None
    buf = []

    def flush():
        if cur is not None and sec is not None:
            cur["secs"].append((sec, "\n".join(buf)))

    with open(path, errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            m = HDR.match(line)
            if m:
                flush()
                sec, buf = None, []
                cur = {"t": int(m.group(2)) * 3600 + int(m.group(3)) * 60 + int(m.group(4)), "secs": []}
                invs.append(cur)
                continue
            if cur is None:
                continue
            s = SEC.match(line)
            if s and line in ("[SYSTEM PROMPT]", "[USER PROMPT]", "[MODEL RESPONSE META]", "[THINKING]",
                              "[ASSISTANT]", "[ANALYZER STATUS]") or (s and line.startswith("[TOOL ")):
                flush()
                sec, buf = line.strip("[]"), []
                continue
            buf.append(line)
    flush()
    # monotone day rollover
    off = 0
    prev = None
    for inv in invs:
        if prev is not None and inv["t"] + off < prev - 3600:
            off += 86400
        inv["t"] += off
        prev = inv["t"]
        opener = next((b for n, b in inv["secs"] if n == "USER PROMPT"), "")
        inv["opener"] = opener
        m = LVL.search(opener)
        inv["level"] = int(m.group(2)) if m else None
        inv["exec"] = any(n == "ANALYZER STATUS" and "step_executed: True" in b for n, b in inv["secs"])
        # responses: list of (meta, thinking, toolcall)
        resp = []
        for i, (n, b) in enumerate(inv["secs"]):
            if n == "MODEL RESPONSE META":
                fr = re.search(r"finish_reason: (\S+)", b)
                tc = re.search(r"tool_call_count: (\d+)", b)
                rc = re.search(r"reasoning_chars: (\d+)", b)
                call = ""
                for n2, b2 in inv["secs"][i + 1:i + 4]:
                    if n2.startswith("TOOL CALL"):
                        call = b2
                        break
                    if n2 == "MODEL RESPONSE META":
                        break
                resp.append({"fr": fr.group(1) if fr else "", "tc": int(tc.group(1)) if tc else 0,
                             "rc": int(rc.group(1)) if rc else 0, "call": call})
        inv["resp"] = resp
    return invs


rows = []
for d in sys.argv[1:]:
    for p in sorted(glob.glob(os.path.join(d, "*_p0.txt"))):
        invs = parse(p)
        game = os.path.basename(p)[:4]
        maxlvl = max((i["level"] or 0) for i in invs) if invs else 0
        # level of each invocation, filling None forward
        last = None
        for i in invs:
            if i["level"] is None:
                i["level"] = last
            last = i["level"]
        t_end = invs[-1]["t"] if invs else 0
        for k, inv in enumerate(invs):
            for j, r in enumerate(inv["resp"]):
                if r["fr"] == "length" and r["tc"] == 0:
                    lvl = inv["level"]
                    # idle count on this level before this invocation
                    idle = 0
                    for q in range(k - 1, -1, -1):
                        if invs[q]["level"] != lvl or invs[q]["exec"]:
                            break
                        idle += 1
                    nxt = invs[k + 1] if k + 1 < len(invs) else None
                    # time to next executing invocation end (header of the invocation after it)
                    t_exec = None
                    for q in range(k + 1, len(invs)):
                        if invs[q]["exec"]:
                            t_exec = (invs[q + 1]["t"] if q + 1 < len(invs) else None)
                            break
                    cleared = any((x["level"] or 0) > (lvl or 0) for x in invs[k:])
                    nr = nxt["resp"][0] if nxt and nxt["resp"] else None
                    rows.append({
                        "run": d[-1], "game": game, "lvl": lvl, "k": k, "j": j, "idle": idle,
                        "last_in_inv": j == len(inv["resp"]) - 1,
                        "stall_here": "Stall check:" in inv["opener"],
                        "stall_next": bool(nxt and "Stall check:" in nxt["opener"]),
                        "next_noexec_line": bool(nxt and "produced no tool call" in nxt["opener"]),
                        "next_fr": nr["fr"] if nr else None, "next_tc": nr["tc"] if nr else None,
                        "next_call_def": bool(nr and "def " in nr["call"]),
                        "next_call_action": bool(nr and "action(" in nr["call"]),
                        "next_exec": bool(nxt and nxt["exec"]),
                        "dt_exec": (t_exec - inv["t"]) if t_exec else None,
                        "cleared": cleared, "rc": r["rc"],
                    })

n = len(rows)
print("cuts", n, Counter(r["run"] for r in rows))
print("last response in its invocation:", sum(r["last_in_inv"] for r in rows))
print("median reasoning chars", st.median(r["rc"] for r in rows))
print("idle-turn index at cut:", Counter(min(r["idle"], 9) for r in rows))
print("stall line in cut turn's own opener:", sum(r["stall_here"] for r in rows))
print("stall line in NEXT opener:", sum(r["stall_next"] for r in rows))
print("next opener says 'produced no tool call':", sum(r["next_noexec_line"] for r in rows))
print("next first reply fr:", Counter(r["next_fr"] for r in rows))
nx = [r for r in rows if r["next_fr"] is not None]
print("next first reply has tool call:", sum(1 for r in nx if r["next_tc"] > 0), "of", len(nx))
print("next first call has def:", sum(r["next_call_def"] for r in nx), " action(:", sum(r["next_call_action"] for r in nx),
      " def or action(:", sum(1 for r in nx if r["next_call_def"] or r["next_call_action"]))
print("next invocation executed a step:", sum(r["next_exec"] for r in nx))
print("back-to-back cut (next first reply also cut, no call):", sum(1 for r in nx if r["next_fr"] == "length" and r["next_tc"] == 0))
d = [r["dt_exec"] for r in rows if r["dt_exec"] is not None]
print("cut-inv start to end of next executing inv: median", st.median(d), "mean", round(st.mean(d)), "n", len(d))
print("on levels never cleared:", sum(1 for r in rows if not r["cleared"]))
# by stall-next split
for flag in (True, False):
    sub = [r for r in nx if r["stall_next"] == flag]
    if sub:
        print(f"stall_next={flag}: n={len(sub)} next-call {sum(1 for r in sub if r['next_tc']>0)}, next exec {sum(r['next_exec'] for r in sub)}, b2b cut {sum(1 for r in sub if r['next_fr']=='length' and r['next_tc']==0)}")
print("by game", Counter(r["game"] for r in rows).most_common(10))

# extra: next-start to end of first executing invocation; base rates over all invocations
import collections
allfirst = []
gap = []
for d in sys.argv[1:]:
    for p in sorted(glob.glob(os.path.join(d, "*_p0.txt"))):
        invs = parse(p)
        for k, inv in enumerate(invs):
            if inv["resp"]:
                allfirst.append(inv["resp"][0])
            for j, r in enumerate(inv["resp"]):
                if r["fr"] == "length" and r["tc"] == 0 and k + 1 < len(invs):
                    t0 = invs[k + 1]["t"]
                    for q in range(k + 1, len(invs)):
                        if invs[q]["exec"]:
                            if q + 1 < len(invs):
                                gap.append(invs[q + 1]["t"] - t0)
                            break
fc = [r for r in allfirst if r["tc"] > 0]
print("ALL invocations: first reply has call", len(fc), "of", len(allfirst),
      "; def in first call", sum("def " in r["call"] for r in fc), "; def or action(", sum(("def " in r["call"]) or ("action(" in r["call"]) for r in fc))
print("after cut: next-start to end of first executing inv: median", st.median(gap), "mean", round(st.mean(gap)), "n", len(gap))
