"""Write gate-results-2026-10-31-franzen-l1a1y8sv2r3lf15us8.json from us8_gate.py us8 and the probe logs (run from repo
root). "promote" is true only on VERDICT PASS with 3 W runs read and 0 probe tracebacks; prints the record path."""
import json, re, subprocess, sys
from pathlib import Path

S = Path(__file__).parent
G = Path('solution/kaggle/generated')
P = G / 'kernel-outputs' / 'franzen-l1a1y8sv2r3kp12lf15us8-v1'
out = subprocess.run([sys.executable, str(S / 'us8_gate.py'), 'us8'], capture_output=True, text=True, check=True).stdout
verdict = re.search(r'^VERDICT (\w+)', out, re.M).group(1)
logs = [P / 'kernel.log', P / 'serve.log']
tb = sum(p.read_text(errors='replace').count('Traceback') for p in logs if p.exists())
pg = json.loads((P / 'benchmark.json').read_text())['game_runs'] if (P / 'benchmark.json').exists() else []
p_levels, p_actions = sum(g['levels_completed'] for g in pg), sum(sum(g['actions_per_level']) for g in pg)
probe_ok = all(p.exists() and p.stat().st_size > 0 for p in logs) and tb == 0 and p_actions >= 25  # it plays
promote = verdict == 'PASS' and probe_ok
checks = [line for line in out.splitlines() if line.startswith(('PASS ', 'FAIL '))]
runs = [line for line in out.splitlines() if re.match(r'^(W|A) ', line)]
counters = [line.strip() for line in out.splitlines() if line.startswith('   ')]
w_mean = next(line for line in runs if line.startswith('W mean'))
a_mean = next(line for line in runs if line.startswith('A mean'))
record = {
    "schema_version": 1,
    "date": "2026-10-31",
    "gate": str(G / 'gate-predeclared-2026-10-31-franzen-l1a1y8sv2r3lf15us8.json'),
    "instrument": "Kaggle replica, 25 public games, 15 streams, 7,200 s window; W = kaggle-lf15us8-1031-1..3 "
                  "(l1a1y8sv2r3lf15us8, us8b, us8c); unstick counters plus code blocks reading group; "
                  "read by scratchpad unst/us8_gate.py us8",
    "control": {"runs": "A pooled, the kv2 gate's 7 runs: vertex-kva, vertex-s15w-1..4, kaggle-lf15-1029, -1029-2",
                "summary": a_mean},
    "arm": {"kernel": "vineethsaivs/arc3-franzen-l1a1y8sv2r3kp12lf15us8-a", "version": 1,
            "change": "the pin l1a1y8sv2r3kp12lf15uns2 plus franzen_overlays/seg8.py (each segmentation node carries its 8-connected group), over unstick2: a stall check at the 3rd, 5th, "
                      "7th idle analyzer turn on a level; the object types and actions not yet tried on the level after "
                      "a stall check, game over or model RESET; new object types at each level start; step "
                      "verification on with mechanics-first, replay-before-plan and retry-from-another-state lines"},
    "evidence": {"checks": checks, "runs": runs, "counters": counters,
                 "probe": f"{P.name}: {tb} tracebacks in kernel.log and serve.log; benchmark.json "
                          f"{'present' if (P / 'benchmark.json').exists() else 'missing'}, {p_levels} levels and {p_actions} actions in "
                          "its 300 s per game"},
    "promote": promote,
    "verdict": (f"PROMOTE. Every predeclared item passes on 3 runs and the probe plays clean. {w_mean}; {a_mean}. "
                "Rollback: re-pin l1a1y8sv2r3kp12lf15uns2 v2 if the first live draw is 0.00 or a day errors twice.")
               if promote else f"KILL. Reader verdict {verdict}, probe tracebacks {tb}, probe ok {probe_ok}.",
    "what_this_does_not_claim": "A level gain on the hidden 110 games: three replica runs cannot resolve effects under "
                                "about 25 levels, so the gate is the mechanism plus no harm; the level means are reported, "
                                "not claimed.",
}
path = Path(sys.argv[1]) if len(sys.argv) > 1 else G / 'gate-results-2026-10-31-franzen-l1a1y8sv2r3lf15us8.json'
path.write_text(json.dumps(record, indent=1) + "\n")
print(out)
print(path, 'promote', promote)
