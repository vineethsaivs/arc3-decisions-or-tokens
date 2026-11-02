"""Where an ARC-AGI-3 replica run loses points, from the recorded benchmark.json files (2026-10-26).

Reads the 34 pin-family replica runs on the 25 demonstration games (rq1020-*, ra1-rq-*, ra2h-a2..a5, wsrh-a1..a5,
yw-y8-*, we1h-a*; fetched with solution/gcp/vertex_fetch.sh) and prints four things:
 1. efficiency of the levels the agent clears (actions over the human baseline) and what perfect efficiency would add;
 2. the replica points lost to uncleared levels, by how often each level is cleared;
 3. the chance a run reaches each level, and when its last clear happens in the 7,200 s window;
 4. level clears and weighted points per 720 s time bin (the marginal value of clock).
Run from the repo root: python3 paper/v3/analysis/clock_marginal.py > paper/v3/analysis/out/clock_marginal_2026-10-26.txt
Replica points: the shipped scoring formula on the demonstration set; the board reads about 0.6 to 0.66 of them.
"""
import json, glob, collections, statistics as st

def section_eff():
    print("\n== 1. Efficiency of cleared levels")
    runs=[]
    for pat in ('rq1020-*','ra1-rq-*','ra2h-a[2-5]-1','wsrh-a[1-5]-*','yw-y8-*','we1h-a*'):
        runs+=sorted(glob.glob(f'solution/kaggle/generated/kernel-outputs/vertex-{pat}/benchmark.json'))
    tot=dict(S=0,S_eff100=0,S_allclear=0,n=0)
    rat=[]; per_lvl=collections.defaultdict(list); gcomp=collections.defaultdict(list); lossbylvl=collections.defaultdict(float)
    for f in runs:
        d=json.load(open(f))
        for g in d['game_runs']:
            b=g['base_actions_per_level']; a=g['actions_per_level']; n=len(b)
            W=sum(range(1,n+1)); lc=g.get('levels_completed',0)
            sc=[]; sc100=[]
            for i in range(n):
                done = i<lc
                if done and a[i]>0:
                    r=b[i]/a[i]; s=min(115,100*r*r); rat.append(a[i]/b[i]); per_lvl[i+1].append(a[i]/b[i])
                else: s=0
                sc.append(s*(i+1)); sc100.append((100 if done else 0)*(i+1))
            raw=sum(sc)/W; cap=sum(i+1 for i in range(lc))/W*100
            tot['S']+=min(raw,cap); tot['S_eff100']+=min(sum(sc100)/W,cap); tot['n']+=1
            # all-levels-cleared at current mean efficiency of cleared levels (upper bound shape)
    print('runs',len(runs),'game-runs',tot['n'])
    n=tot['n']; print('mean game score %.2f ; if every CLEARED level were at human baseline (a=b): %.2f'%(tot['S']/n,tot['S_eff100']/n))
    rat.sort(); q=lambda p: rat[int(p*(len(rat)-1))]
    print('cleared levels:',len(rat),' actions/baseline quantiles  p10 %.2f p25 %.2f p50 %.2f p75 %.2f p90 %.2f'%(q(.1),q(.25),q(.5),q(.75),q(.9)))
    print('share of cleared levels with a<=b: %.2f ; a<=1.5b: %.2f ; a<=2b: %.2f'%tuple(sum(1 for x in rat if x<=t)/len(rat) for t in (1,1.5,2)))
    print('median a/b by level index:',{k:round(st.median(v),2) for k,v in sorted(per_lvl.items())})

def section_loss():
    print("\n== 2. Where the uncleared points sit")
    runs=[]
    for pat in ('rq1020-*','ra1-rq-*','ra2h-a[2-5]-1','wsrh-a[1-5]-*','yw-y8-*','we1h-a*'):
        runs+=sorted(glob.glob(f'solution/kaggle/generated/kernel-outputs/vertex-{pat}/benchmark.json'))
    cl=collections.defaultdict(list); nl={}
    for f in runs:
        for g in json.load(open(f))['game_runs']:
            gid=g['game_id'].split('-')[0]; n=len(g['base_actions_per_level']); nl[gid]=n
            for i in range(n): cl[(gid,i+1)].append(1 if i<g.get('levels_completed',0) else 0)
    tot=0; buckets=collections.defaultdict(float); rows=[]
    for (gid,i),v in cl.items():
        n=nl[gid]; W=n*(n+1)/2; p=sum(v)/len(v); lost=(1-p)*i/W*100/25   # board points lost (per game /25 games)
        tot+=lost; rows.append((lost,gid,i,p,n))
        b='never(<=10%)' if p<=.1 else 'rare(10-40%)' if p<.4 else 'mixed(40-80%)' if p<.8 else 'usual(>=80%)'
        buckets[b]+=lost
    print('replica points lost to uncleared levels: %.1f of 100'%tot)
    for k,v in sorted(buckets.items()): print(f'  {k}: {v:.1f}')
    rows.sort(reverse=True)
    print('top lost (game level rate nlevels):'); 
    for l,g,i,p,n in rows[:14]: print(f'  {g} L{i}/{n} clear {p:.2f} lost {l:.2f}')
    byg=collections.defaultdict(float)
    for l,g,i,p,n in rows: byg[g]+=l
    print('lost by game:',', '.join(f'{g} {v:.1f}' for g,v in sorted(byg.items(),key=lambda x:-x[1])[:10]))

def section_reach():
    print("\n== 3. How deep runs get, and when")
    runs=[]
    for pat in ('rq1020-*','ra1-rq-*','ra2h-a[2-5]-1','wsrh-a[1-5]-*','yw-y8-*','we1h-a*'):
        runs+=sorted(glob.glob(f'solution/kaggle/generated/kernel-outputs/vertex-{pat}/benchmark.json'))
    reach=collections.defaultdict(lambda:[0,0]); spent=collections.defaultdict(list); lastclear=[]; tail=[]
    for f in runs:
        for g in json.load(open(f))['game_runs']:
            a=g['actions_per_level']; lc=g.get('levels_completed',0); n=len(a); H=g['history']
            cum=[]; t=0
            for h in H: cum.append(h.get('wallclock_seconds',0))
            # time at the last action of each level (cumulative action counts)
            ends=[]; c=0
            for i in range(n):
                c+=a[i]; ends.append(cum[min(c,len(cum))-1] if c>0 and cum else None)
            for i in range(n):
                reached = i<=lc   # level i+1 reached when i levels done
                reach[i+1][1]+=1; reach[i+1][0]+= 1 if reached else 0
            if lc>0 and ends[lc-1] is not None: lastclear.append(ends[lc-1])
            if lc<n:  # stuck level: time from its start to window end
                start = ends[lc-1] if lc>0 else 0
                if start is not None: spent[lc+1].append((cum[-1]-start if cum else 0, a[lc], g['game_id']))
    print('P(level reached) by index:',{k:round(v[0]/v[1],2) for k,v in sorted(reach.items())})
    lc=sorted(lastclear); print('time of last cleared level (s): p25 %d p50 %d p75 %d p90 %d (window 7200)'%tuple(lc[int(p*(len(lc)-1))] for p in (.25,.5,.75,.9)))
    allspent=[x[0] for v in spent.values() for x in v]; allspent.sort()
    print('time sunk in the final unsolved level (s): p25 %d p50 %d p75 %d ; share of games ending unsolved: %.2f'%(allspent[len(allspent)//4],allspent[len(allspent)//2],allspent[3*len(allspent)//4],len(allspent)/sum(v[1] for v in [reach[1]])))

def section_marg():
    print("\n== 4. Points per time bin (the marginal value of clock)")
    runs=[]
    for pat in ('rq1020-*','ra1-rq-*','ra2h-a[2-5]-1','wsrh-a[1-5]-*','yw-y8-*','we1h-a*'):
        runs+=sorted(glob.glob(f'solution/kaggle/generated/kernel-outputs/vertex-{pat}/benchmark.json'))
    bins=[0]*10; w_bins=[0.0]*10; n_runs=len(runs); tok_bins=[0]*10
    for f in runs:
        for g in json.load(open(f))['game_runs']:
            a=g['actions_per_level']; lc=g.get('levels_completed',0); H=g['history']; n=len(a); W=n*(n+1)/2
            c=0
            for i in range(lc):
                c+=a[i]
                if c>0 and c<=len(H):
                    t=H[c-1]['wallclock_seconds']; b=min(9,int(t/720)); bins[b]+=1; w_bins[b]+=(i+1)/W*100/25
            for k,h in enumerate(H):
                tok_bins[min(9,int(h['wallclock_seconds']/720))]+=h.get('generated_tokens',0)
    print('level clears per 720 s bin (all %d runs, per run):'%n_runs,[round(x/n_runs,1) for x in bins])
    print('weighted board points per bin (per run):',[round(x/n_runs,2) for x in w_bins])
    print('generated tokens per bin per run (K):',[round(x/n_runs/1000) for x in tok_bins])
    last=w_bins[-1]/n_runs
    print('points from the last 10%% of the window: %.2f of %.2f total (%.0f%%); one extra 720 s bin at that rate would add about +%.2f replica points'%(last,sum(w_bins)/n_runs,100*last/(sum(w_bins)/n_runs),last))

if __name__ == "__main__":
    section_eff(); section_loss(); section_reach(); section_marg()
