## 1. Run levels by family (25-game replica, 7,200 s window)

| family | n | levels | mean | sd |
| --- | ---: | --- | ---: | ---: |
| Y8 gate pair | 2 | 128, 124 | 126.0 | 2.8 |
| Y8 later (post-gate) | 7 | 119, 120, 128, 122, 129, 120, 126 | 123.4 | 4.2 |
| Y8 all | 9 | 128, 124, 119, 120, 128, 122, 129, 120, 126 | 124.0 | 3.9 |
| 2048 gate control (l1a1, GCP) | 2 | 113, 109 | 111.0 | 2.8 |
| 2048 l1a1-lineage | 9 | 123, 101, 104, 116, 113, 113, 109, 115, 109 | 111.4 | 6.6 |
| 2048 p448-lineage | 12 | 122, 106, 117, 111, 111, 110, 108, 105, 107, 115, 113, 113 | 111.5 | 4.9 |
| 2048 all | 21 | 123, 101, 104, 116, 113, 113, 109, 115, 109, 122, 106, 117, 111, 111, 110, 108, 105, 107, 115, 113, 113 | 111.5 | 5.5 |

Platform check inside the 2,048 controls: Kaggle n=13 mean 111.3; GCP n=8 mean 111.8.
Lineage check: p448-lineage mean 111.5 vs l1a1-lineage mean 111.4.
No-yield arm (y0sv2, one run, server OOM at 6,893 s): 115 levels; same-stack y8sv2: 119.

## 2. Run-to-run noise from identical-code replicate pairs

| code | run A | run B | diff |
| --- | ---: | ---: | ---: |
| p448 | 122 | 106 | +16 |
| p448s14 | 117 | 111 | +6 |
| dpctl | 123 | 101 | +22 |
| L1 | 116 | 113 | +3 |
| l1a1 | 113 | 109 | +4 |
| y8 | 128 | 124 | +4 |

Within-code sd = sqrt(sum d^2 / 2m): 8.3 with all 6 pairs (df 6); 8.9 with the 5 2,048-yield pairs only.
Sd across all 21 2,048 runs (different configs pooled): 5.5; across the 9 y8-family runs: 3.9.

## 3. Difference in mean levels: exact permutation (run labels) and run-level bootstrap

| comparison | n1 vs n2 | mean diff | exact p (one-sided) | exact p (two-sided) | splits | bootstrap 95% CI |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Y8 gate pair vs gate control (l1a1) | 2 vs 2 | +15.0 | 0.1667 | 0.3333 | 6 | +11.0 to +19.0 |
| Y8 gate pair vs l1a1-lineage 2048 | 2 vs 9 | +14.6 | 0.0182 | 0.0182 | 55 | +9.6 to +19.4 |
| Y8 gate pair vs all 2048 | 2 vs 21 | +14.5 | 0.0040 | 0.0040 | 253 | +11.0 to +18.1 |
| Y8 later (7) vs all 2048 | 7 vs 21 | +12.0 | 2.7e-05 | 2.7e-05 | 1,184,040 | +8.3 to +15.7 |
| Y8 later (7) vs l1a1-lineage 2048 | 7 vs 9 | +12.0 | 6.1e-04 | 0.0012 | 11,440 | +7.0 to +17.0 |
| Y8 all (9) vs all 2048 | 9 vs 21 | +12.5 | 2.4e-06 | 2.4e-06 | 14,307,150 | +9.2 to +15.8 |
| Y8 all (9) vs l1a1-lineage 2048 | 9 vs 9 | +12.6 | 1.4e-04 | 2.9e-04 | 48,620 | +7.8 to +17.2 |
| Y8 all (9) vs p448-lineage 2048 | 9 vs 12 | +12.5 | 2.4e-05 | 3.4e-05 | 293,930 | +8.8 to +16.0 |
| Y8 all (9) vs pinned-code replicate groups only (12) | 9 vs 12 | +11.1 | 1.2e-04 | 3.1e-04 | 293,930 | +6.9 to +15.3 |
| Y8 gate pair vs pinned-code groups (12) | 2 vs 12 | +13.1 | 0.0110 | 0.0220 | 91 | +8.8 to +17.5 |
| Y8 all (9) vs Kaggle-run 2048 controls only (13) | 9 vs 13 | +12.7 | 1.4e-05 | 1.8e-05 | 497,420 | +9.2 to +16.0 |
| Y8 gate pair vs Kaggle-run 2048 controls (13) | 2 vs 13 | +14.7 | 0.0095 | 0.0095 | 105 | +11.1 to +18.3 |

Config-level (replicates averaged; 8 y8 configs vs 16 2,048 configs): diff +12.8, Monte Carlo one-sided p 1.0e-05 or smaller (100,000 shuffles, 0 hits).

Day-level clustering (UTC start date; unit = day mean):

| family | days | day means (runs) |
| --- | ---: | --- |
| y8 family | 3 | 10-10: 123.7 (3), 10-11: 120.0 (1), 10-12: 125.0 (5) |
| 2,048 controls | 6 | 10-03: 110.8 (9), 10-04: 113.7 (3), 10-06: 119.5 (2), 10-07: 108.2 (4), 10-08: 111.0 (2), 10-10: 109.0 (1) |

Day-level exact permutation, 3 y8 days vs 6 control days: diff of day-means +10.9, one-sided p 0.0119 over 84 splits (lowest y8 day 120.0, highest control day 119.5).
Rank check: the gate pair's levels [128, 124] against the 21 controls' maximum 123; controls at or above 124: 0; at or above 119: 2 of 21; y8-family runs below 119: 0 of 9.

Parametric cross-check, gate pair vs the 2 l1a1 runs (the gate's own comparison): diff +15.0, se 8.3 (within-code sd 8.3, df 6), t = 1.82, one-sided p = 0.059.

Parametric cross-check, gate pair vs all 21 controls: diff +14.5, se 6.1 (within-code sd 8.3, df 6), t = 2.38, one-sided p = 0.027.

Selection check (assumes noise sd 8.3, so a 2-run mean has sd 5.8; arms have no true effect): best of k arms exceeds +14.5 over the control mean with probability, and averages:

| k arms | P(best 2-run mean gap >= +14.5) | mean best gap |
| ---: | ---: | ---: |
| 1 | 0.007 | +0.0 |
| 5 | 0.033 | +6.8 |
| 10 | 0.066 | +9.0 |
| 20 | 0.125 | +10.9 |

## 4. What each design could detect (one-sided alpha 0.05, power 0.8; MDE = 2.49 * sd * sqrt(1/n1 + 1/n2))

| design | sd used | MDE (levels) |
| --- | ---: | ---: |
| gate: 2 vs 2 | 8.3 | 20.5 |
| gate pair vs 21 controls | 8.3 | 15.2 |
| 9 y8 vs 21 controls | 5.5 | 5.5 |
| 9 y8 vs 21 controls | 8.3 | 8.2 |

## 5. Per game (levels completed), run labels permuted

| comparison | games y8 > control | y8 < control | tie | permutation p for (wins - losses) |
| --- | ---: | ---: | ---: | ---: |
| Y8 pair vs all 2048 | 21 | 3 | 1 | 0.0042 |
| Y8 pair vs l1a1 (GCP) | 12 | 2 | 11 | 0.1704 |
| Y8 later (7) vs all 2048 | 20 | 3 | 2 | 0.0015 |
| Y8 all (9) vs all 2048 | 22 | 2 | 1 | 0.0003 |

Gate record says the y8 mean beats 'the four control runs' on 14 games, loses 2, ties 9. Quadruples of the 21 controls that reproduce exactly 14/2/9: 95; of those, containing both l1a1 GCP runs: 4 (e.g. [('dpctl', 'l1a1', 'l1a1', 'franzen-p448h14'), ('h16dp', 'l1a1', 'l1a1', 'l1tb'), ('l1a1', 'l1a1', 'l1tb', 'franzen-p448s14h71')]).

Per-game mean level difference, Y8 all (9 runs) minus 2,048 all (21 runs), top and bottom 5: g50t +2.16, vc33 +1.22, tu93 +1.13, tn36 +0.89, sp80 +0.84 | sb26 +0.10, bp35 +0.06, lp85 +0.00, ft09 -0.02, s5i5 -0.51
Total of the per-game differences = +12.5 levels; the 22 positive games carry +13.0, the negative -0.5.

## 6. Is it tokens? Throughput, matched-token levels, and what a token is worth

| family | n | generated tokens (M) | tok/s | actions | new prefill (M) | prefill ms / 1K new | levels @75K | @100K | @150K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Y8 gate pair | 2 | 5.61 | 770 | 6162 | 9.27 | 130 | 62.0 | 80.0 | 100.5 |
| Y8 later (7) | 7 | 6.20 | 855 | 6611 | 10.02 | 112 | 60.3 | 71.1 | 92.0 |
| Y8 all (9) | 9 | 6.07 | 836 | 6511 | 9.85 | 116 | 60.7 | 73.1 | 93.9 |
| l1a1 GCP (gate control) | 2 | 5.62 | 775 | 5904 | 10.52 | 120 | 57.0 | 68.0 | 87.5 |
| 2048 l1a1-lineage | 9 | 5.51 | 760 | 5925 | 11.41 | 122 | 57.9 | 70.3 | 90.1 |
| 2048 p448-lineage | 12 | 5.12 | 705 | 5790 | 12.83 | 121 | 59.3 | 71.2 | 92.9 |
| 2048 all | 21 | 5.29 | 729 | 5848 | 12.22 | 121 | 58.7 | 70.9 | 91.7 |

Levels at 100K tokens per game, Y8 gate pair vs all 2,048: 80.0 vs 70.9 (diff +9.1, exact one-sided p 0.0158).

Levels at 100K tokens per game, Y8 later (7) vs all 2,048: 71.1 vs 70.9 (diff +0.3, exact one-sided p 0.4630).

Levels at 150K tokens per game, Y8 gate pair vs all 2,048: 100.5 vs 91.7 (diff +8.8, exact one-sided p 0.0277).

Levels at 150K tokens per game, Y8 later (7) vs all 2,048: 92.0 vs 91.7 (diff +0.3, exact one-sided p 0.4525).

OLS on the 21 2,048 runs: levels = 120.2 + -1.7 * tokens(M); slope se 3.9, residual sd 5.7, df 19.
OLS on all 30 runs with a y8 indicator: levels = 114.1 + -0.5 * tokens(M) + 12.9 * y8; y8 coefficient se 3.1 (t = 4.2), token slope se 2.9, residual sd 5.2, df 27.
Token volume: y8 all minus 2,048 all = +0.78M tokens (+14.8%); level gap +12.5. Controls' slope times the token gap = -1.3 levels (-10% of the gap).
Gate pair only: token gap +0.32M (+6.1%); level gap +14.5; slope times token gap = -0.5 levels (-4%).

Tail value of a token: the 21 controls clear 19.8 levels on 1.76M tokens spent beyond 150K per game = 11.2 levels per M tokens (an average, so a generous upper bound on the marginal value).
Prefill ceiling: l1a1 (GCP) minus the y8 pair new prefill = 1.25M tokens; at 130 ms per 1K new tokens that is 163 s of GPU time = 2.3% of the 7,200 s window, i.e. at most 2.3% more tokens = 0.13M tokens = +1.4 levels at the tail rate.
Measured token gap, gate pair minus l1a1 (GCP): -0.01M (-0.2%); gate pair minus all 2,048: +0.32M = +3.6 levels at the tail rate (25% of +14.5). Y8 all minus all 2,048: +0.78M = +8.8 levels at the tail rate (70% of +12.5).
Correlation of run levels with run tokens: 2,048 controls r = -0.10 (n 21); y8 family r = +0.18 (n 9).
Calendar: control runs start 0.0 to 7.2 days after the first; y8 runs 7.2 to 9.6. Correlation of control levels with start time r = -0.12 (n 21), so no drift is visible inside the controls. The one same-time pair: l1a1tbfc (2,048, +strict thinking budget) started 04:11 and l1a1y8 started 04:08 on 2026-10-10: 109 vs 128 levels.

## 7. Levels at matched generated tokens per game (games that stopped earlier keep their final levels)

| tokens per game | gate pair (2) | later 7 | all 9 | 2048 all (21) | pair - ctl | later7 - ctl | later7 exact one-sided p | games reaching: pair / later 7 / ctl |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 25K | 25.5 | 21.6 | 22.4 | 22.3 | +3.2 | -0.7 | 7.8e-01 | 25.0 / 25.0 / 25.0 |
| 50K | 46.0 | 42.9 | 43.6 | 42.9 | +3.1 | -0.0 | 5.3e-01 | 25.0 / 24.4 / 24.8 |
| 75K | 62.0 | 60.3 | 60.7 | 58.7 | +3.3 | +1.6 | 2.3e-01 | 24.5 / 23.9 / 23.9 |
| 100K | 80.0 | 71.1 | 73.1 | 70.9 | +9.1 | +0.3 | 4.6e-01 | 22.5 / 23.1 / 23.2 |
| 150K | 100.5 | 92.0 | 93.9 | 91.7 | +8.8 | +0.3 | 4.5e-01 | 19.5 / 20.9 / 19.6 |
| 200K | 114.0 | 106.9 | 108.4 | 104.9 | +9.1 | +2.0 | 1.8e-01 | 15.0 / 17.9 / 14.6 |
| 250K | 122.5 | 115.3 | 116.9 | 109.7 | +12.8 | +5.6 | 1.5e-02 | 10.0 / 13.9 / 8.0 |
| 300K | 124.5 | 120.0 | 121.0 | 111.1 | +13.4 | +8.9 | 4.5e-04 | 4.0 / 8.3 / 2.8 |
| 400K | 126.0 | 123.3 | 123.9 | 111.5 | +14.5 | +11.8 | 2.7e-05 | 0.5 / 0.6 / 0.0 |

The gate pair is already ahead at 100K to 150K tokens per game. The later seven match the controls through 150K and open their gap past 200K, where they also have many more games still running (reaching 300K: 8.3 vs 2.8 games per run). Past 200K the later seven are bought with tokens (+17% generated) as much as with decisions; the pair, whose token volume equals the l1a1 control's, is not.

## 8. Mean game score (the metric the board uses), secondary

| family | n | mean score | sd |
| --- | ---: | ---: | ---: |
| Y8 gate pair | 2 | 59.09 | 2.96 |
| Y8 later (7) | 7 | 55.95 | 2.46 |
| Y8 all (9) | 9 | 56.65 | 2.75 |
| 2048 all (21) | 21 | 49.72 | 3.61 |

Score, Y8 all vs 2048 all: diff +6.93 points, exact one-sided p 0.00002 (14,307,150 splits).

Score, Y8 later vs 2048 all: diff +6.23 points, exact one-sided p 0.00019 (1,184,040 splits).
Score per level: Y8 all 0.457 vs 2048 all 0.446 (efficiency per completed level is not obviously different; the score gain tracks the level gain).
