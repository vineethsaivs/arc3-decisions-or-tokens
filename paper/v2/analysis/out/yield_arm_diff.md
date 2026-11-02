# l1a1 -> l1a1y8: 26 vs 26 cells
## cell 7: 2 changed lines
  -    'LOCAL_ANALYZER_YIELD_TOKENS': '2048',
  +    'LOCAL_ANALYZER_YIELD_TOKENS': '8192',  # arm y8: yield 4x less often
## cell 15: 3 changed lines
  +        if TRUE_SUBMISSION:  # arm failclosed: never play a scored run against a dead server
  +            raise RuntimeError('failclosed: the model server exited before READY; raising so this '
  +                               'submission errors (no slot spent) and the daily job retries')
## cell 23: 7 changed lines
  +# arm failclosed: a dead scored run raises (an error does not spend the slot); bench runs print only
  +_failclosed = {}
  +exec(compile('"""Fail closed on a dead scored run: raise at the end of the run so Kaggle records an error.\n\nWhy. taaf\'s Benchmark.run swallows any exception from the solver task (R2.15), marks every game\nstill playi
  +if TRUE_SUBMISSION:
  +    _failclosed['check'](bm.game_runs)
  +else:
  +    print('failclosed (not enforced off the scored path):', _failclosed['summarize'](bm.game_runs), flush=True)
# changed cells: 3
# l1a1tbfc -> l1a1y8: 26 vs 26 cells
## cell 7: 2 changed lines
  -    'LOCAL_ANALYZER_YIELD_TOKENS': '2048',
  +    'LOCAL_ANALYZER_YIELD_TOKENS': '8192',  # arm y8: yield 4x less often
## cell 15: 5 changed lines
  -# arm thinkbudget: strict thinking budget of 10240 tokens, think exclusions emptied
  -_thinkbudget = {}
  -exec(compile('"""Arm thinkbudget (B5): a strict thinking budget that holds under NEXTN. Research:\ndocs/research/2026-10-05-top5/throughput.md section 2.1.\n\nRuns inside the launcher cell (cell 15): args = apply(VENV, 
  -args = _thinkbudget['apply'](VENV, env, args, 10240)
  -_thinkbudget['check'](url, proc, CFG['SERVED_NAME'], strict=not TRUE_SUBMISSION)  # arm thinkbudget: 64-token self-test on a bench run
# changed cells: 2
# l1a1y8 -> l1a1y8r: 26 vs 26 cells
# changed cells: 0
# l1a1y8sv2 -> l1a1y0sv2: 26 vs 26 cells
## cell 7: 2 changed lines
  -    'LOCAL_ANALYZER_YIELD_TOKENS': '8192',  # arm y8: yield 4x less often
  +    'LOCAL_ANALYZER_YIELD_TOKENS': '0',  # arm y0: never yield on tokens
# changed cells: 1
# p448s14 -> l1a1: 26 vs 26 cells
## cell 7: 7 changed lines
  +# arm overlay: agentfix
  +exec(compile('"""Agent fixes A1: RESET behind a level-start guard, the stale-guard catch contract, and\nlevel/attempt index globals. docs/research/2026-10-05-top5/PLAN.md section 3 item 5;\nthe measured failures are in 
  +# arm overlay: replycache
  +exec(compile('"""B1 reply-cache round trip: re-send the model\'s reasoning as the server returned it.\n\ndocs/research/2026-10-05-top5/top-movers.md section 4 measured that the next request\ngets the previous reply from
  +# arm overlay: wastefix
  +exec(compile('"""Transcript-waste fixes: corrections and lossless repairs, no new advice text.\n\ndocs/research/2026-10-04-climb/transcript-waste.md measured where the pinned config\n(p448s14, two 25-game runs) wastes r
  +    'EXPOSE_RESET': 'on',  # arm agentfix: RESET behind the level-start guard
## cell 15: 5 changed lines
  +# arm replycache: one serve.log line per finished request, bench runs only
  +env['ARC3_REPLYCACHE_DIAG'] = '0' if TRUE_SUBMISSION else '1'
  +_replycache = {}
  +exec(compile('"""Arm replycache, serving side: one serve.log line per finished request saying how deep it\nleft its reply in the radix cache. The fix itself is harness-side (franzen_overlays/replycache.py).\n\nRuns insi
  +_replycache['apply'](VENV, env, MODEL_DIR)
## cell 19: 9 changed lines
  +# arm dprime: D-prime slot priority (shiiin9 / AFF AI CLUB public notebook), verbatim; scored run too
  +import sys as _fp_sys, types as _fp_types
  +_fp_mod = _fp_types.ModuleType('ours_form_priority')
  +_fp_sys.modules['ours_form_priority'] = _fp_mod
  +exec(compile('"""dfranzen の配分の優先度を、模擬で見つけた式に差し替える（本番でも効かせる）。\n\n彼の配分の流れ（まだ始めていないゲームが先、動いているゲームは履歴を削るときだけ\n入れ替える）はそのまま使い、**優先度の値を出す関数だけ**を替える。\n\n    優先度 = e^{lp·(ℓ-1)}·norm(N) × 2^{-a/ec} × ペース × 2^{-(t/ts)²} × (1 + 伸びし
  +import inference.agent.tool_agent as _fp_ta
  +import inference.framework.solver as _fp_solver
  +_fp_mod.install_d(_fp_ta, _fp_solver)
  +
# changed cells: 3
