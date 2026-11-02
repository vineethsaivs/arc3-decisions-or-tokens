import re,glob,sys,collections
pats = {
 "score_eff": r"(efficien\w*|fewer actions|fewest actions|as few (?:in-game )?actions|minimi[sz]e (?:the )?(?:number of )?(?:in-game )?actions|action count|optimi[sz]e for (?:as few|the shortest))",
 "budget_bar": r"(actions? left|moves? left|steps? left|remaining (?:actions|moves|steps|budget)|budget bar|step budget|action budget|bar (?:is )?(?:almost |nearly )?(?:depleted|empty|low))",
 "waste": r"(waste|wasting|wasted|afford|conserve|precious)",
 "careful": r"(think (?:really |very )?carefully|be careful|careful(?:ly)? (?:plan|think))",
 "deliberate_vs_test": r"(just (?:try|test) it|test (?:it|this) empirically|cheaper to (?:test|try)|quicker to (?:test|try)|faster to (?:test|try))",
 "time": r"(running out of time|wall.?clock|time is limited|limited time|time budget)",
}
C=collections.Counter(); T=collections.Counter(); n=0
for d in sys.argv[1:]:
  for p in glob.glob(d+"/*_p0.txt"):
    txt=open(p,errors="replace").read()
    for th in re.findall(r"\[THINKING\]\n(.*?)(?=\n\[[A-Z ]+\]\n|\Z)", txt, re.S):
      n+=1
      for k,v in pats.items():
        m=re.findall(v,th,re.I)
        if m: C[k]+=len(m); T[k]+=1
print("thinking blocks",n)
for k in pats: print(k, "blocks",T[k], "hits",C[k])
