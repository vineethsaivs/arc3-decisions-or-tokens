"""Write-off phrases in THINKING per turn: cleared vs never-cleared levels (4 Kaggle runs with transcripts)."""
import json, re, glob, collections, sys
RUNS = ['kaggle-lf15unst-1030-1', 'kaggle-lf15unst-1030-2', 'kaggle-lf15uns2-1030-1', 'kaggle-lf15uns2-1030-2']
HEADER = re.compile(r"^--- analysis_step=\d+ \| action=\d+ \| \S+ \| tool-agent ---$", re.M)
LEVEL = re.compile(r"Current state: step \d+, level (\d+)\.")
THINK = re.compile(r"^\[THINKING\]\n(.*?)(?=^\[[A-Z :]+\]$)", re.S | re.M)
PH = {k: re.compile(v, re.I) for k, v in {
    'decoy': r"\bdecoys?\b", 'decor': r"\bdecorat(ion|ive)\b", 'inert': r"\binert\b", 'herring': r"red herring",
    'nothing': r"\b(does|did) nothing\b", 'noeffect': r"\bno effect\b", 'useless': r"\buseless\b", 'irrelevant': r"\birrelevant\b",
    'impossible': r"\bimpossible\b", 'unsolvable': r"\bunsolvable\b", 'notmech': r"\bnot a mechanic\b"}.items()}
turns = collections.Counter(); hits = collections.defaultdict(collections.Counter); levels = collections.Counter()
for run in RUNS:
    done = {g['game_id'][:4]: g['levels_completed'] for g in json.load(open(f'{run}/benchmark.json'))['game_runs']}
    for f in glob.glob(f'{run}/*_p0.txt'):
        gid = f.split('/')[-1][:4]; t = open(f, errors='replace').read()
        parts = HEADER.split(t)[1:]
        seen = set()
        for p in parts:
            m = LEVEL.search(p)
            if not m: continue
            lv = int(m.group(1)); kind = 'cleared' if lv <= done[gid] else 'stuck'
            if (gid, lv) not in seen: seen.add((gid, lv)); levels[kind] += 1
            turns[kind] += 1
            th = " ".join(THINK.findall(p))
            for k, rx in PH.items():
                if rx.search(th): hits[k][kind] += 1
print('levels', dict(levels), 'turns', dict(turns))
for k in PH:
    c, s = hits[k]['cleared'], hits[k]['stuck']
    print(f"{k:11s} cleared {c:4d} ({c / turns['cleared']:.3f}/turn)  stuck {s:4d} ({s / turns['stuck']:.3f}/turn)  ratio {(s / turns['stuck']) / max(1e-9, c / turns['cleared']):.1f}")
