"""Parse Franzen transcripts into per-turn records (pickle). Usage: parse.py OUT.pkl DIR..."""
import re, sys, glob, os, pickle, json
from datetime import datetime

H = re.compile(r'^--- analysis_step=(\d+) \| action=(\d+) \| (\d\d:\d\d:\d\d) \| tool-agent ---$', re.M)
L = re.compile(r'Current state: step (\d+), level (\d+)\.')
SEC = re.compile(r'^\[(SYSTEM PROMPT|USER PROMPT|THINKING|TOOL CALL: python|TOOL RESULT: python|MODEL RESPONSE META|ANALYZER STATUS|ASSISTANT)\]\n', re.M)
CODE = re.compile(r'<parameter=code>\n?(.*?)\n?</parameter>', re.S)
ACT = re.compile(r'^\[action\] (.*)$', re.M)


def parse_file(f):
    t = open(f, errors='replace').read()
    hs = list(H.finditer(t))
    turns = []
    for i, h in enumerate(hs):
        body = t[h.end(): hs[i + 1].start() if i + 1 < len(hs) else len(t)]
        parts = SEC.split(body)
        rec = dict(step=int(h.group(1)), act=int(h.group(2)), ts=h.group(3), user='', resp=[], status='')
        cur = None
        for j in range(1, len(parts) - 1, 2):
            name, text = parts[j], parts[j + 1]
            if name == 'USER PROMPT':
                rec['user'] = text
            elif name == 'MODEL RESPONSE META':
                m = re.search(r'reasoning_chars: (\d+)', text)
                fr = re.search(r'finish_reason: (\S+)', text)
                cc = re.search(r'content_chars: (\d+)', text)
                cur = dict(rchars=int(m.group(1)) if m else 0, finish=fr.group(1) if fr else '',
                           cchars=int(cc.group(1)) if cc else 0, think='', code='', result='')
                rec['resp'].append(cur)
            elif name == 'THINKING':
                if cur is None:
                    cur = dict(rchars=len(text), finish='', cchars=0, think='', code='', result=''); rec['resp'].append(cur)
                cur['think'] = text
            elif name == 'TOOL CALL: python':
                m = CODE.search(text)
                if cur is not None: cur['code'] = m.group(1) if m else text
            elif name == 'TOOL RESULT: python':
                if cur is not None: cur['result'] = text
            elif name == 'ANALYZER STATUS':
                rec['status'] = text
            elif name == 'ASSISTANT':
                if cur is not None: cur['assistant'] = text
        m = L.search(rec['user'])
        rec['level'] = int(m.group(2)) if m else None
        rec['gstep'] = int(m.group(1)) if m else None
        hm = re.search(r'history_messages: (\d+)', rec['status'])
        rec['hist'] = int(hm.group(1)) if hm else None
        rec['executed'] = 'step_executed: True' in rec['status']
        mm = re.search(r'message: (.*)', rec['status'])
        rec['msg'] = mm.group(1) if mm else ''
        rec['actions'] = [a for r in rec['resp'] for a in ACT.findall(r['result'])]
        turns.append(rec)
    return turns


out, dirs = sys.argv[1], sys.argv[2:]
data = {}
for d in dirs:
    run = os.path.basename(d.rstrip('/')).replace('kaggle-', '')
    bj = {}
    try:
        bj = json.load(open(f'{d}/benchmark.json'))
    except Exception:
        pass
    for f in sorted(glob.glob(f'{d}/*_p0.txt')):
        g = os.path.basename(f).split('-')[0]
        data[(run, g)] = parse_file(f)
    data[(run, '_bench')] = bj
pickle.dump(data, open(out, 'wb'))
print(len(data))
