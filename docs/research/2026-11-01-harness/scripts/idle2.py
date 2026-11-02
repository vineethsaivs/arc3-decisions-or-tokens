import re,glob,sys,collections
from datetime import datetime,timedelta
HDR=re.compile(r"^--- analysis_step=(\d+) \| action=(\d+) \| (\d\d:\d\d:\d\d) \| tool-agent ---$",re.M)
STATE=re.compile(r"Current state: step (\d+), level (\d+)\.")
SCORE=re.compile(r"(fewer actions (?:is|=|are) better|as few (?:in-game )?actions|minimi[sz]e (?:the )?(?:number of )?(?:in-game )?actions|minimi[sz]ing actions|action efficiency|efficiency (?:score|matters|counts|penalty)|score (?:is|depends on) .{0,30}actions|optimi[sz]e for (?:as few|the shortest)|fewest (?:possible )?actions|(?:save|saving) actions)",re.I)
BAR=re.compile(r"(actions? left|moves? left|steps? left|remaining (?:actions|moves|steps|budget)|budget bar|step budget|action budget|bar (?:is )?(?:almost |nearly )?(?:depleted|empty|low))",re.I)
tot=0;idle=collections.Counter();n=collections.Counter()
ex=[]
for d in sys.argv[1:]:
  for p in sorted(glob.glob(d+"/*_p0.txt")):
    txt=open(p,errors="replace").read(); hs=list(HDR.finditer(txt))
    rows=[];day=0;pt=None
    for i,m in enumerate(hs):
      body=txt[m.end():hs[i+1].start() if i+1<len(hs) else len(txt)]
      t=datetime.strptime(m.group(3),"%H:%M:%S")+timedelta(days=day)
      if pt and t<pt: day+=1;t+=timedelta(days=1)
      pt=t
      th="\n".join(re.findall(r"\[THINKING\]\n(.*?)(?=\n\[[A-Z ]+\]\n|\Z)",body,re.S))
      st=STATE.search(body)
      rows.append((int(m.group(2)),t,int(st.group(2)) if st else None,bool(SCORE.search(th)),bool(BAR.search(th))))
    last=rows[-1][2]
    for i in range(len(rows)-1):
      a,t,l,s,b=rows[i]; dur=(rows[i+1][1]-t).total_seconds()
      if i==0: continue  # first turn includes startup queue
      tot+=dur; acted=rows[i+1][0]>a
      k=("idle" if not acted else "act")
      n[k]+=dur
      if s: n[k+"_score"]+=dur; idle[k+"_score_turns"]+=1
      if b: n[k+"_bar"]+=dur; idle[k+"_bar_turns"]+=1
      if s and not acted: ex.append((d[-1],p.split('/')[-1][:4],l,l==last,int(dur)))
print("total s",int(tot))
for k,v in sorted(n.items()): print(k, int(v), round(v/tot,3))
print(dict(idle))
print("idle turns citing score-efficiency:",len(ex)," on stuck level:",sum(1 for e in ex if e[3]))
for e in sorted(ex,key=lambda e:-e[4])[:15]: print(e)
