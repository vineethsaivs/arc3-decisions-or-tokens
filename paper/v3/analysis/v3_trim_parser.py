"""Post-hoc check of the trimtest first-action parser (not part of the predeclared gate).

    python3 -I -B paper/v3/analysis/v3_trim_parser.py

trim_compare.first_action takes the first quoted identifier inside the first action( call. For the dict form
action([{'action': 'UP'}, ...]) that identifier is the key 'action', so the parsed name is ACTION whatever the
move. This script counts the parse classes and recomputes the gate's agreement numbers with the dict value read
instead (same bootstrap, same seed), to show what the gate measured.
"""
import collections
import json
import random
import re
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "solution/analysis"))
import trim_compare as tc  # noqa: E402

DICT = re.compile(r"""["']action["']\s*:\s*["']([A-Za-z][A-Za-z0-9_]*)["']""")


def fixed(code):
    m = tc.ACTION_CALL.search(code)
    if not m:
        return None
    d = DICT.search(m.group(1))
    if d:
        return d.group(1).upper()
    n = tc.NAME.search(m.group(1))
    return n.group(1).upper() if n else "VAR"


by = collections.defaultdict(dict)
for line in open(Path.home() / ".cache/arc3/trimtest-2026-10-20/trimtest_replies.jsonl"):
    r = json.loads(line)
    if r.get("error") is None:
        by[r["state"]][(r["variant"], r["sample"])] = tc.code_of(r)
states = [s for s, d in by.items() if len(d) == 4]
for name, f in (("gate parser", tc.first_action), ("dict value read", fixed)):
    cls = collections.Counter(f(c) for s in states for c in by[s].values())
    acting = sum(v for k, v in cls.items() if k is not None)
    print(f"== {name}: {acting} acting replies of {4 * len(states)}; classes {cls.most_common(12)}")

    def agree(cross, f=f):
        def g(ss):
            vals = []
            for s in ss:
                fa = [f(by[s][("full", k)]) for k in (0, 1)]
                la = [f(by[s][("lean", k)]) for k in (0, 1)]
                vals += [a == b for a in fa for b in la] if cross else [fa[0] == fa[1]]
            return st.mean(vals)
        return g

    for label, stat in (("full-full", agree(False)), ("full-lean", agree(True)),
                        ("difference", lambda ss: agree(True)(ss) - agree(False)(ss))):
        p, lo, hi = tc.bootstrap(states, stat)
        print(f"  {label:12s} {p:+.3f}  90% [{lo:+.3f}, {hi:+.3f}]")
