Source: board-submissions-2026-10-13T150841.912452Z.json (created 2026-10-13T15:08:41.911562+00:00, complete_history True).

| date (UTC) | score | config | change from the previous pin |
| --- | ---: | --- | --- |
| 2026-10-03 | 0.00 | fork-franzen (unmodified fork) | the unmodified public fork |
| 2026-10-04 | 28.97 | franzen-p448s14 | REAP keep-448, 14 streams (Franzen's priority rule) |
| 2026-10-05 | 31.90 | franzen-p448s14dp | + D-prime slot priority |
| 2026-10-06 | 29.44 | franzen-p448s14dp | + D-prime slot priority |
| 2026-10-07 | 30.56 | franzen-p448s14dp | + D-prime slot priority |
| 2026-10-08 | 28.19 | franzen-p448s14dp | + D-prime slot priority |
| 2026-10-09 | 27.15 | franzen-l1a1p | + waste fixes + reply-cache round trip + agent fixes |
| 2026-10-10 | 0.00 | franzen-l1a1p | + waste fixes + reply-cache round trip + agent fixes |
| 2026-10-11 | 28.51 | franzen-l1a1fc2p | + two fail-closed checks (inert on a healthy run) |
| 2026-10-12 | 34.34 | franzen-l1a1y8sv2p | + YIELD 2,048 -> 8,192 + QSA ring fix + async M-RoPE upload |
| 2026-10-13 | 40.09 | franzen-l1a1y8sv2r3p | + watchdog, MEMFRAC 0.955, memcap, MMHASH, toolfast, D3 drafter, promptobs lite |

Before the yield change (positive draws on the fork, 10-04 to 10-11): n=7, scores [28.97, 31.9, 29.44, 30.56, 28.19, 27.15, 28.51], mean 29.25, sd 1.58, range 27.15 to 31.9.
One config, four draws (p448s14dp, 10-05 to 10-08): [31.9, 29.44, 30.56, 28.19], mean 30.02, sd 1.58.
After: [34.34, 40.09] (y8sv2p 10-12, r3p 10-13). Both exceed every earlier draw (previous best 31.9).
Exact rank probability that two specific draws out of 9 are the top two if the order carried no information: 1/C(9,2) = 0.0278. This ignores that the draws are time-ordered and that both carry other changes.
- 2026-10-12 franzen-l1a1y8sv2p: 34.34 is +5.09 above the pre-yield mean; 3.0 predictive sd (pre-yield sd 1.58, n=7).
- 2026-10-13 franzen-l1a1y8sv2r3p: 40.09 is +10.84 above the pre-yield mean; 6.4 predictive sd (pre-yield sd 1.58, n=7).

### Replica mean game score to board score, by config

| config | replica runs (mean game score) | board draw(s) | board / replica |
| --- | --- | --- | ---: |
| franzen-p448s14 | 53.5, 52.1 | 28.97 | 0.548 |
| franzen-p448s14dp | 57.8, 44.6 | 31.90, 29.44, 30.56, 28.19 | 0.586 |
| franzen-l1a1p | 53.8, 49.2 | 27.15 | 0.527 |
| franzen-l1a1fc2p | 53.8, 49.2 | 28.51 | 0.554 |
| franzen-l1a1y8sv2p | 54.2 | 34.34 | 0.633 |
| franzen-l1a1y8sv2r3p | 58.5 | 40.09 | 0.686 |

For reference the y8 gate pair's replica mean game score is 59.1; at the pre-yield ratio (mean 0.554) a replica of 54.2 (y8sv2) predicts 30.0 and 58.5 (r3) predicts 32.4; the draws were 34.34 and 40.09.
