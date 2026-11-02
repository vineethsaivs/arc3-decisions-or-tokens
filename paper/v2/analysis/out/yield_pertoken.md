## A. Indirectly standardized clearance ratio (observed / expected levels given exposure)

Expected levels for a run = sum over (game, level) strata of (controls' pooled clearance rate per token in that stratum) x (the run's own tokens at risk there).
Controls are scored leave-one-run-out, so their ratio is 1 in expectation: that is the placebo spread.

| run | family | levels | tokens (M) | observed / expected |
| --- | --- | ---: | ---: | ---: |
| l1a1y8 | y8 | 128 | 5.68 | 1.075 |
| l1a1y8r | y8 | 124 | 5.54 | 1.066 |
| l1a1y8sv2 | y8 | 119 | 5.79 | 0.992 |
| l1a1y8sv2o16 | y8 | 120 | 5.86 | 0.919 |
| l1a1y8sv2r1 | y8 | 128 | 6.13 | 1.158 |
| l1a1y8sv2r2 | y8 | 122 | 6.43 | 0.978 |
| l1a1y8sv2r3 | y8 | 129 | 6.43 | 0.991 |
| l1a1y8sv2r3s4 | y8 | 120 | 6.27 | 0.926 |
| l1a1y8sv2r3hm32 | y8 | 126 | 6.49 | 1.044 |

Controls leave-one-out: mean 0.966, sd 0.082, range 0.846 to 1.129.
- y8 gate pair: mean ratio 1.070; vs controls' placebo ratios diff +0.104, exact one-sided p 0.05534 (253 splits).
- y8 later 7: mean ratio 1.001; vs controls' placebo ratios diff +0.035, exact one-sided p 0.17207 (1,184,040 splits).
- y8 all 9: mean ratio 1.017; vs controls' placebo ratios diff +0.050, exact one-sided p 0.06713 (14,307,150 splits).

## B. Stratified clearance rate per generated token (game x level strata, Mantel-Haenszel rate ratio)

| comparison | MH rate ratio | permutation p (one-sided, run labels) | run-cluster bootstrap 95% CI |
| --- | ---: | ---: | --- |
| y8 all (9) vs all 2048 (21) | 1.073 | 0.0323 | 1.029 to 1.138 |
| y8 gate pair (2) vs all 2048 (21) | 1.192 | 0.0523 | 1.146 to 1.327 |
| y8 later (7) vs all 2048 (21) | 1.071 | 0.0817 | 1.030 to 1.147 |

Rate ratio by level index (y8 all vs all 2048; strata restricted to that level):

| level | MH rate ratio | events y8 / events ctl (per run) |
| --- | ---: | ---: |
| 1 | 1.045 | 24.8 / 24.5 |
| 2 | 1.094 | 22.9 / 21.3 |
| 3 | 0.964 | 21.4 / 18.8 |
| 4-5 | 1.109 | 34.1 / 29.1 |
| 6+ | 1.151 | 20.8 / 17.8 |

Crude decomposition of the level ratio into token volume and levels per token:

| family | levels | tokens (M) | levels per M tokens |
| --- | ---: | ---: | ---: |
| y8 gate pair | 126.0 | 5.61 | 22.46 |
| y8 later 7 | 123.4 | 6.20 | 19.91 |
| y8 all 9 | 124.0 | 6.07 | 20.43 |
| l1a1 GCP (gate control) | 111.0 | 5.62 | 19.75 |
| 2048 l1a1-lineage | 111.4 | 5.51 | 20.23 |
| 2048 all | 111.5 | 5.29 | 21.08 |
