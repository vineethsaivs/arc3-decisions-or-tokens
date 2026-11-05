# Gates and board (computed by analysis/v3_gates_board.py)

## 1. False-ship simulation (computed)

Rule read from `solution/kaggle/generated/gate-predeclared-2026-10-30..11-01-*.json` and V3-W19: ship if arm mean levels >= control mean - 2.5 AND arm mean score >= control mean - 1.5 (plus mechanism and action checks, not simulated). Control 7 runs, arm 3 runs. Normal run noise; control independent.
Score model for the joint column, fitted on 18 harness runs: 0.521 score points per level, r = 0.93, residual SD 1.68.
I judge SD 5.95 (eight same-day runs) the central frame: 8.25 includes a shared session shift (pair SD about the shift is about 5.6, diag_skeptic), 4.7 is the 18-run pool across days and platforms. Control SD in the 7-run set itself is 7.76.

P(ship), 3 arm runs vs 7 control, levels rule only / levels and score:

| SD | true -6 | -3 | 0 | +3 | +6 | true +6 missed |
|---|---|---|---|---|---|---|
| 4.7 | 0.14 / 0.11 | 0.44 / 0.37 | 0.78 / 0.70 | 0.95 / 0.92 | 1.00 / 0.99 | 0.00 / 0.01 |
| 5.95 | 0.20 / 0.16 | 0.45 / 0.39 | 0.73 / 0.67 | 0.91 / 0.87 | 0.98 / 0.96 | 0.02 / 0.04 |
| 8.25 | 0.27 / 0.23 | 0.47 / 0.42 | 0.67 / 0.62 | 0.83 / 0.80 | 0.93 / 0.91 | 0.07 / 0.09 |

Other designs (levels rule only), SD 5.95: 3v7: -6 0.20, 0 0.73, +6 0.98; 2v7: -6 0.23, 0 0.70, +6 0.96; 3v3: -6 0.24, 0 0.70, +6 0.96

The gate screens out large losers only. It ships a null change about 7 times in 10 and a true -6 about 1 in 5 (SD 5.95), and it misses a true +6 rarely; it cannot tell +3 from 0. The score condition (-1.5) lowers every column a few points (see the second number) but does not change this reading.

## 2. Board selection (computed)

Definition of the Writeup's 17: nonzero draws 2026-10-13 to 2026-11-02 of the agent's family (r3p 3, hm32p 4, kp 3, kp12 2, kp12lf15 3, two harness builds 1 each) = 17; excludes the 3 wesrp adapter draws, the 10-12 draw of the pre-r3 build y8sv2p (34.34) and the 10-18 zero. Mean 34.593, SD 3.389, range 28.97 to 40.98.
The insight lens's 21 = the same 17 + 3 adapter draws + the 10-12 draw: mean 34.957, SD 3.254 (check 34.957). Both numbers are right; they are different sets. 'Best of 28 live draws' = those 21 + 7 pre-yield nonzero draws (10-04 to 10-11, mean 29.246).
Within-build SD (builds with 2 or more draws) 2.767; one-way ANOVA across builds F 2.335, p 0.113; between-build SD (method of moments) 2.084.

Expected maximum of n draws and chance of reaching 40.98:

| model | n | E[max] | P(max >= 40.98) |
|---|---|---|---|
| normal, family SD | 17 | 40.67 | 0.401 |
| normal, within-build SD | 17 | 39.56 | 0.165 |
| bootstrap | 17 | 40.4 | 0.643 |
| normal, 21 post-yield, family SD | 21 | 41.1 | 0.495 |
| normal, 21, within-build SD | 21 | 40.18 | 0.267 |
| bootstrap, 21 | 21 | 40.55 | 0.64 |
| 7 pre-yield + 21 post-yield | 28 | 40.18 | 0.267 |

A single draw of the family reaches 40.98 with p 0.0298 (z 1.88). Against an expected best of about 40.67 to 41.1, 40.98 is inside selection noise: it is a typical best of the draws, not evidence that r3p is better than the family.
r3p's three draws average 38.43; with reliability 0.63 the shrunk true mean is 37.01, 2.4 above the family mean 34.59; the build effect is not established (ANOVA p 0.113).
Expected private score for the two auto-selected draws: between 34.59 (family mean) and 37.01 (shrunk r3p mean), per-draw SD 3.39, not 40.98. Public and private halves hold different games, so a config's true mean can differ between halves by a game-split term not estimable from these data. Expected private for the best two public draws: between the family mean and the shrunk r3p mean, with the draw SD around it.
