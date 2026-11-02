"""Per game: wall minutes on levels it cleared vs on the final level (stuck when the window closed), from the
'Current state: step N, level L.' line of each analyzer turn. Also minutes per cleared level."""
import re, sys, glob, os
from datetime import datetime
H = re.compile(r'^--- analysis_step=(\d+) \| action=(\d+) \| (\d\d:\d\d:\d\d) \| tool-agent ---$', re.M)
L = re.compile(r'Current state: step (\d+), level (\d+)\.')
for d in sys.argv[1:]:
    clear = tail = nclr = 0; per = []
    for f in sorted(glob.glob(f'{d}/*_p0.txt')):
        txt = open(f, errors='replace').read()
        hs = list(H.finditer(txt)); pts = []
        for i, h in enumerate(hs):
            body = txt[h.end(): hs[i + 1].start() if i + 1 < len(hs) else len(txt)]
            m = L.search(body)
            if m: pts.append((datetime.strptime(h.group(3), '%H:%M:%S'), int(m.group(2))))
        if len(pts) < 2: continue
        t0, tend, lv_end = pts[0][0], pts[-1][0], pts[-1][1]
        first_final = next(t for t, lv in pts if lv == lv_end)
        c = (first_final - t0).seconds; k = (tend - first_final).seconds
        clear += c; tail += k; nclr += lv_end - pts[0][1]
        per.append(f"{os.path.basename(f).split('-')[0]}:{lv_end - pts[0][1]}clr/{c//60}m+{k//60}m")
    print(f"{d}: cleared-level clock {clear//60} min over {nclr} clears ({clear/60/max(nclr,1):.1f} min each), "
          f"final-level tail {tail//60} min ({tail/(clear+tail):.0%})")
    print('   ', ' '.join(per))
