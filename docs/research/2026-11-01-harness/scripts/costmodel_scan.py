import re, sys, glob, os, json
from datetime import datetime, timedelta

HDR = re.compile(r"^--- analysis_step=(\d+) \| action=(\d+) \| (\d\d:\d\d:\d\d) \| tool-agent ---$", re.M)
STATE = re.compile(r"Current state: step (\d+), level (\d+)\.")
EXEC = re.compile(r"^step_executed: (True|False)", re.M)

# phrases that mark action-thrift deliberation (the failure costmodel targets)
THRIFT = re.compile(
    r"(actions? left|remaining (?:actions|moves|budget)|budget (?:is|bar) (?:low|limited|tight)|"
    r"(?:don't|do not|can't|cannot) (?:afford|waste)|waste (?:an? )?actions?|wasting actions|"
    r"before spending|conserve|limited (?:actions|moves|budget)|every action counts|think (?:really|very) carefully|"
    r"minimi[sz]e (?:the )?(?:number of )?actions|as few actions|fewest actions|costs? (?:an? )?actions?)",
    re.I)
PROBE = re.compile(r"(just (?:try|test)|let me (?:just )?(?:try|test) (?:it|one|a single)|test (?:it|this) (?:empirically|directly)|try one action)", re.I)

def parse(path):
    txt = open(path, errors="replace").read()
    heads = list(HDR.finditer(txt))
    turns = []
    day = 0
    prev_t = None
    for i, m in enumerate(heads):
        body = txt[m.end(): heads[i + 1].start() if i + 1 < len(heads) else len(txt)]
        t = datetime.strptime(m.group(3), "%H:%M:%S") + timedelta(days=day)
        if prev_t and t < prev_t:
            day += 1; t += timedelta(days=1)
        prev_t = t
        st = STATE.search(body)
        level = int(st.group(2)) if st else None
        ex = EXEC.findall(body)
        think = "\n".join(re.findall(r"\[THINKING\]\n(.*?)(?=\n\[[A-Z ]+\]\n|\Z)", body, re.S))
        turns.append(dict(step=int(m.group(1)), action=int(m.group(2)), t=t, level=level,
                          executed=(ex[-1] == "True") if ex else None,
                          thrift=len(THRIFT.findall(think)), probe=len(PROBE.findall(think)),
                          think_chars=len(think)))
    return turns

def main(dirs):
    agg = dict(turns=0, idle=0, idle_s=0.0, total_s=0.0, thrift_turns=0, thrift_idle=0, thrift_idle_s=0.0,
               stuck_s=0.0, stuck_idle_s=0.0, stuck_thrift_turns=0, stuck_turns=0)
    examples = []
    for d in dirs:
        for p in sorted(glob.glob(os.path.join(d, "*_p0.txt"))):
            turns = parse(p)
            if len(turns) < 2:
                continue
            last_level = turns[-1]["level"]
            for i, tr in enumerate(turns[:-1]):
                dur = (turns[i + 1]["t"] - tr["t"]).total_seconds()
                acted = turns[i + 1]["action"] > tr["action"]
                agg["turns"] += 1; agg["total_s"] += dur
                stuck = tr["level"] == last_level
                if stuck:
                    agg["stuck_s"] += dur; agg["stuck_turns"] += 1
                if not acted:
                    agg["idle"] += 1; agg["idle_s"] += dur
                    if stuck: agg["stuck_idle_s"] += dur
                if tr["thrift"]:
                    agg["thrift_turns"] += 1
                    if stuck: agg["stuck_thrift_turns"] += 1
                    if not acted:
                        agg["thrift_idle"] += 1; agg["thrift_idle_s"] += dur
                    examples.append((os.path.basename(d), os.path.basename(p)[:4], tr["step"], tr["level"], int(dur), acted, tr["thrift"]))
    a = agg
    print(json.dumps({k: (round(v, 1) if isinstance(v, float) else v) for k, v in a.items()}, indent=1))
    print("idle clock share", round(a["idle_s"] / a["total_s"], 3), "mean idle turn s", round(a["idle_s"] / max(1, a["idle"]), 1))
    print("thrift-phrase turns share", round(a["thrift_turns"] / a["turns"], 3),
          "thrift idle clock share of total", round(a["thrift_idle_s"] / a["total_s"], 3))
    print("stuck-level clock share", round(a["stuck_s"] / a["total_s"], 3), "stuck idle share of stuck", round(a["stuck_idle_s"] / max(1, a["stuck_s"]), 3))
    print("examples (run, game, step, level, dur_s, acted, n_phrases):")
    for e in sorted(examples, key=lambda e: -e[4])[:25]:
        print(" ", e)

if __name__ == "__main__":
    main(sys.argv[1:])
