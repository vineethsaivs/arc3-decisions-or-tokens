"""Run the patched unstick2 helpers on engine-replayed frames: new-types at each level start, coverage after each
death and at each level's end. python3 check.py FRAMES_DIR (from solution/sim)"""
import glob, json, os, shutil, sys, tempfile, time, statistics as st
from pathlib import Path
sys.path.insert(0, '<repo>/solution/sim')
from test_franzen_sandbox import FORK_ENV, TREE, apply, import_tool_agent
import test_franzen_unstick2 as T
names = T.order()
tmp = tempfile.mkdtemp(prefix="u3-"); bundle = Path(tmp) / "bundle"
shutil.copytree(TREE, bundle, ignore=shutil.ignore_patterns("__pycache__"))
for n in names: apply(n, bundle)
os.environ.update(FORK_ENV)
ta = import_tool_agent(bundle)
from inference.agent.runtime_state import Frame, HistoryEntry
times_cov, times_new, newcount, newfired, covfired, covtypes = [], [], 0, 0, 0, []
show = sys.argv[2:] if len(sys.argv) > 2 else []
for f in sorted(glob.glob(sys.argv[1] + '/*.json')):
    d = json.load(open(f)); tag = Path(f).stem
    valid = ["RESET"] + [f"ACTION{a}" for a in d["valid"]]
    agent = type("A", (), {})()
    hist = []
    seq = d["seq"]
    for i, e in enumerate(seq):
        fr = Frame(grid=tuple(tuple(int(ch, 16) for ch in row) for row in e["g"]), step=i, level=e["lv"])
        hist.append(HistoryEntry(action=e["a"], frame=fr, result={"automatic": True} if e.get("auto") else {}))
        last_of_level = i + 1 == len(seq) or seq[i + 1]["lv"] != e["lv"]
        if i and e["lv"] != seq[i - 1]["lv"] and e["lv"] > 1:
            t0 = time.perf_counter(); lines = ta._unstick_new_types(agent, fr, hist); times_new.append(time.perf_counter() - t0)
            newcount += 1; newfired += bool(lines)
            if tag in show or (show and show[0] == 'all'): print(f"{tag} L{e['lv']} start: {' | '.join(lines)[:400]}")
        if e.get("over") or (last_of_level and i + 1 < len(seq)):
            prev_fr = hist[-2].frame if e.get("over") else fr
            t0 = time.perf_counter(); lines = ta._unstick_coverage(prev_fr if not e.get("over") else fr, hist, valid); times_cov.append(time.perf_counter() - t0)
            u = [l for l in lines if l.startswith("Untested")]
            covfired += bool(u); covtypes.append(len(u[0].split("; ")) if u else 0)
            if tag in show or (show and show[0] == 'all'):
                print(f"{tag} L{e['lv']} {'death' if e.get('over') else 'end'}: {' | '.join(lines)[:400]}")
print(f"new-types: {newfired}/{newcount} level starts fired; ms mean {1000*st.mean(times_new):.1f} max {1000*max(times_new):.1f}")
print(f"coverage: {covfired}/{len(times_cov)} calls listed untested types (mean types {st.mean(covtypes):.1f}); "
      f"ms mean {1000*st.mean(times_cov):.1f} max {1000*max(times_cov):.1f}")
shutil.rmtree(tmp)
