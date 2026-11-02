"""M5: what thinking text is spent on. Each thinking line is classed (first match wins):
 trace   - hand simulation: >=2 arrows (->, =>, unicode arrow) or 'after <ACTION>' with coordinates
 board   - board description: >=2 coordinate mentions ((r,c), [r,c], 'row N', 'col N', 'rows N-M')
 arith   - inline arithmetic: >=2 'N op N' expressions
 mech    - mechanics/goal reasoning keywords
 other
Char shares overall, on new-level opening turns, in yield (8K, no action) turns, and in turns before vs after
a level's last 3 turns. Also counts the tool code that prints whole segmentation listings."""
from levels import *
import collections, re, statistics as st

ARROW = re.compile(r'->|=>|→|⇒')
COORD = re.compile(r'\(\s*\d+\s*,\s*\d+\s*\)|\[\s*\d+\s*,\s*\d+|\brows? \d+|\bcols? \d+')
ARITH = re.compile(r'\d+\s*[-+*/]\s*\d+')
MECH = re.compile(r'\b(rule|mechanic|means|because|hypothes|goal|win|objective|implies|therefore|so the|cause|trigger|must|should|likely|perhaps|maybe|test|plan|strategy|idea|wait|hmm|actually)\w*', re.I)

def classify(line):
    if len(ARROW.findall(line)) >= 2: return 'trace'
    if len(COORD.findall(line)) >= 2: return 'board'
    if len(ARITH.findall(line)) >= 2: return 'arith'
    if MECH.search(line): return 'mech'
    return 'other'

tot = collections.Counter(); bucket = collections.defaultdict(collections.Counter)
wait_n = collections.Counter(); nchars = collections.Counter()
for r, g, L, seg, cleared, t0, t1, ts, i0 in segments():
    for k, t in enumerate(seg):
        tags = ['all']
        if k == 0 and L and L >= 2: tags.append('level_open')
        if t['yield']: tags.append('yield_turn')
        if t['stall']: tags.append('stall_turn')
        if cleared and k >= len(seg) - 3: tags.append('last3_cleared')
        if not cleared and t is seg[-1]: pass
        for x in t['resp']:
            for line in x['think'].splitlines():
                if not line.strip(): continue
                c = classify(line)
                for tg in tags:
                    bucket[tg][c] += len(line)
            th = x['think']
            for tg in tags:
                nchars[tg] += len(th)
                wait_n[tg] += len(re.findall(r'\b(Wait|Hmm|Actually)\b', th))

for tg in ['all', 'level_open', 'yield_turn', 'stall_turn', 'last3_cleared']:
    B = bucket[tg]; T = sum(B.values())
    print(f"{tg:14s} chars {T/1e6:7.1f}M  " + '  '.join(f"{c} {B[c]/T:.0%}" for c in ['board', 'trace', 'arith', 'mech', 'other']) +
          f"  | 'Wait/Hmm/Actually' per 1K chars {wait_n[tg]/nchars[tg]*1000:.2f}")
