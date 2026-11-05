# Run-to-run SD frames (computed by paper/v3/analysis/v3_variance_frames.py)

All SDs are in levels on the 25 development games. "computed" = this script; "read" = taken from a repo file and asserted equal.

## 1. The frames

| id | frame | n | mean | SD | df | 95% chi-square interval | runs and source |
|---|---|---|---|---|---|---|---|
| a | six identical pairs, about pair means (includes the first-to-second shift) | 12 |  | 8.25 | 6 | 5.3 to 18.2 | p448 (Kaggle), p448s14 (Kaggle), l1a1y8 (Kaggle), dpctl (GCP), L1 (GCP), l1a1 (GCP); 12 runs, 2.3 to 26.7 h between members; sources yield_runs.json, e4_noise.json |
| a2 | same pairs, SD about the common shift (-9.17 removed) | 12 |  | 5.59 | 5 | 3.5 to 13.7 | same 12 runs; computed here from the six signed gaps |
| b | eight same-day identical runs | 8 | 124.6 | 5.95 | 7 | 3.9 to 12.1 | vertex-rq1020-1 to 8, one build, Vertex AI, all started 2026-10-20 07:29 to 09:59 UTC; sources kernel-outputs benchmark.json, bench_pool18_levels.json |
| c | nine same-family builds | 9 | 124.0 | 3.91 | 8 | 2.6 to 7.5 | 9 Kaggle y8-family builds (y8, y8r, sv2, sv2o16, r1, r2, r3, r3s4, r3hm32), 10-10 to 10-12 (58 h span), NOT byte-identical; source yield_runs.json, lora_pergame.csv |
| d | pooled 18 no-adapter runs | 18 | 124.4 | 4.69 | 17 | 3.5 to 7.0 | frame c + vertex-cal1 + frame b; source bench_pool18_levels.json |
| e1 | interleaved test, yield 8,192 arm | 6 | 120.7 | 6.53 | 5 | 4.1 to 16.0 | yw-y8-* six runs, Vertex, 2026-10-21; source read_2026-10-24.json |
| e2 | interleaved test, yield 2,048 arm | 6 | 123.2 | 2.23 | 5 | 1.4 to 5.5 | yw-y2-* six runs, same wave |
| e | interleaved test, pooled within arm |  |  | 4.88 | 10 | 3.4 to 8.6 | e1 + e2, 10 df |
| P1 | pooled, four disjoint sets (a, b, c, e) |  |  | 5.73 | 31 | 4.6 to 7.6 | df-weighted, 31 df, shift inside via a |
| P2 | pooled, sets without the cross-day pairs (b, c, e) |  |  | 4.93 | 25 | 3.9 to 6.8 | 25 df; c spans 58 h |
| P3 | pooled, identical-build same-session only (b, e) |  |  | 5.35 | 17 | 4.0 to 8.0 | 17 df, no config differences |

## 2. Why the frames differ

- **Order shift is the main cause of 8.25.** The second run was lower in 6 of 6 pairs (second minus first: [-16, -6, -4, -22, -3, -4]; mean -9.17; sign test p 0.031; paired t on 5 df p 0.036; 95% interval for the shift -17.5 to -0.9). The mean square of the pairs, 68.1, is 62% common shift (42.0) and 38% spread. Removing the shift gives SD 5.59 (5 df, interval 3.5 to 13.7). Kaggle pairs shifted [-16, -6, -4], GCP pairs [-22, -3, -4].
- **Day drift and platform are not separable from the shift.** Pairs ran 2.3 to 26.7 h apart on two platforms; start-to-READY timing does not explain it (V3-W37). Only one pair (l1a1y8, gap 4) ran within 3 h; the six pairs cannot split day, order and platform.
- **Same-day identical runs (b, 5.95) match the SD about the shift (5.6).** No trend with start time inside the 2.5 h (Spearman 0.10, p 0.82).
- **The nine-family SD (c, 3.9) is the lowest** and its interval (2.6 to 7.5) contains 5.35 and 5.6. The builds differ, so it could in principle be larger, not smaller, than pure repeat noise; with 8 df nothing says it is truly lower than b. The F test of pairs against c is p 0.028 one-sided (4.46 on 6, 8 df), against b p 0.206.
- **The pool (d, 4.69) mixes b and c**, so it is between them. Its two platforms agree (Kaggle 124.0, Vertex 124.8).
- **The interleaved arms (e) differ by 3x** (y8 6.53, y2 2.23; variance ratio 8.6, two-sided F p 0.034). Pooled within-arm SD is 4.88 (10 df, 3.4 to 8.6). Heterogeneity across the four disjoint sets is not significant (Bartlett p 0.24), so one SD per frame is not demanded by the data except for the shift.

## 3. Pooled estimate

- Per-run SD without the cross-day shift: **4.93 (25 df, interval 3.9 to 6.8)** pooling b, c, e; **5.35 (17 df, interval 4.0 to 8.0)** with identical builds only (b, e). Point value for planning: 5 to 5.5.
- Including the cross-day pairs, all four disjoint sets: **5.73 (31 df, interval 4.6 to 7.6)**.
- Cross-day, not interleaved: the error of a difference also carries a shift that the pairs put at about -9 (second minus first) with an interval of -17.5 to -0.9; 8.25 (interval 5.3 to 18.2) is the RMS that covers it.

## 4. Runs needed (two-sample t, equal n per arm, noncentral t, 80% power, 5% two-sided)

Script: this file (functions power, n_needed, mde). Cells are n per arm.

| SD | d = 5 | d = 10 | d = 15 | d = 23 |
|---|---|---|---|---|
| 4.7 | 15 | 5 | 3 | 3 |
| 5.95 | 24 | 7 | 4 | 3 |
| 8.25 | 44 | 12 | 6 | 4 |

Normal approximation, unrounded (the paper convention), for contrast: SD 4.7: 13.9, 3.5, 1.5, 0.7; SD 5.95: 22.2, 5.6, 2.5, 1.1; SD 8.25: 42.7, 10.7, 4.8, 2.0 (d = 5, 10, 15, 23).

Minimum detectable gain at 80% power, n per arm (t | normal):

| SD | n=2 | n=3 | n=6 | n=12 | n=20 |
|---|---|---|---|---|---|
| 4.7 | 26.6 / 13.2 | 14.4 / 10.8 | 8.4 / 7.6 | 5.6 / 5.4 | 4.3 / 4.2 |
| 5.95 | 33.6 / 16.7 | 18.3 / 13.6 | 10.7 / 9.6 | 7.1 / 6.8 | 5.4 / 5.3 |
| 8.25 | 46.6 / 23.1 | 25.3 / 18.9 | 14.8 / 13.3 | 9.9 / 9.4 | 7.5 / 7.3 |

3 against 7 (the harness gate design), MDE with t on 8 df: SD 4.7 10.4, 5.95 13.1, 8.25 18.2 (the paper quotes 15.9 at 8.25 by normal approximation).

Interleaved yield test difference (-2.50, six per arm), 95% interval by SD: 8.25 (pairs, paper convention, t on 6 df): -14.15 to 9.15; 5.95 (same-day 8, t on 7 df): -10.62 to 5.62; 4.88 (this test own pooled within-arm SD, t on 10 df): -8.78 to 3.78.

## 5. Per-game decomposition

Sum of per-game variances against the variance of the run total (covariances are the difference). "null" is the total variance when each game column is shuffled across runs (independent games).

| frame | runs | sum per-game var | total var | total SD | sum of covariances x2 | games with zero variance | top 5 share of per-game var |
|---|---|---|---|---|---|---|---|
| nine same-family Kaggle builds (frame c) | 9 | 26.2 | 15.3 | 3.91 | -11.0 | 6 (ar25 cd82 lp85 r11l sb26 tr87) | 0.60 |
| eight same-day identical Vertex runs (frame b) | 8 | 39.3 | 35.4 | 5.95 | -3.9 | 6 (ar25 cd82 dc22 ft09 lp85 sb26) | 0.57 |
| pooled 18 (frame d), about the grand mean | 18 | 30.4 | 22.0 | 4.69 | -8.4 | 4 (ar25 cd82 lp85 sb26) | 0.55 |
| interleaved y8 arm, 6 runs | 6 | 27.7 | 42.7 | 6.53 | 15.0 | 6 (cd82 dc22 lp85 sb26 tu93 vc33) | 0.63 |
| interleaved y2 arm, 6 runs | 6 | 45.5 | 5.0 | 2.23 | -40.5 | 5 (dc22 ft09 lp85 sb26 tr87) | 0.64 |
| six identical pairs, centred on each pair mean (12 runs, 6 df) | 12 | 38.8 | 68.1 | 8.25 | 29.3 | 3 (ft09 lp85 sb26) | 0.58 |

- Independent games predict a total SD of 5.12 to 6.27 in every non-pair frame (sum of per-game SDs: family 5.12, same-day 6.27, pool 5.51, y8 5.26, y2 6.75); observed total SD is 3.9 to 6.5 there (the y2 arm, 2.2, aside), and the permutation null is matched within noise (family9 p 0.20, same-day8 p 0.48, pool18 p 0.21 for observed total variance at or below null). Same-session run totals behave like a sum of independent game outcomes; no shared run effect is detectable in them.
- The pairs frame is the exception: per-game variances sum to 38.8 (SD 6.22) but the total is 68.1 (SD 8.25), a positive covariance term of +29.3, which is the common shift. The y8 arm of the interleaved test also shows positive covariance (+15.0), the y2 arm strongly negative (-40.5); 6 runs each, so treat both as noise.
- Who carries the variance (pool of 18): tn36 5.6 (range 0 to 7, 0.18 of total), sc25 3.6 (range 0 to 6, 0.12 of total), su15 2.9 (range 3 to 8, 0.09 of total), sk48 2.4 (range 0 to 4, 0.08 of total), s5i5 2.2 (range 1 to 6, 0.07 of total). Five games hold about 55 to 64% of per-game variance in every frame; tn36 is first in all but the nine-family frame (top there: sc25), then s5i5, su15, sc25, sk48 or wa30.
- Games that never move: lp85 and sb26 have zero variance in all four same-session frames; ar25 and cd82 are zero in the nine-family and pool-18 sets (ar25 8 of 8 levels, cd82 6 of 6, cleared every time). In the pool of 18, 4 games never move (ar25 cd82 lp85 sb26); 21 of 25 move at all, 10 have variance of 1 or more.
- lora_pergame.csv reproduces the nine-family matrix exactly (asserted: True).

## 6. Recommended wording

Writeup (replace the SD sentence in Identical code, different scores and What to copy item 1; about +20 words, cut elsewhere):

> Run-to-run SD depends on the comparison. Without a day gap (eight same-day identical runs, nine near-identical builds, the interleaved test; 4.7 pooled over 18) it is about 5 levels (4.9 to 5.3; 95% interval 3.9 to 8.0). Across days it is 8.25 (interval 5.3 to 18.2), because the second run of all six identical pairs scored lower (mean -9.2), a shift we cannot assign to day, order or platform. Interleave arms; if you cannot, plan on 8.25. Two runs per arm detect only gains above 13 (SD 4.7) to 23 (SD 8.25) by normal approximation, and above 27 to 47 with the t distribution; detecting 10 levels takes 5 to 12 runs per arm.

Short Writeup version (about 55 words, if the word budget is tight):

> SD across runs is about 5 levels within a day and 8.25 across days (the second run of six identical pairs scored 9 lower on average). Two runs per arm detect only gains above 13 to 23 levels (normal approximation; 27 to 47 by t); 10 levels need 5 to 12 runs per arm.

PDF (Identical-runs and Limits sections, one sentence each):

> The pairs SD of 8.25 is a root-mean-square of the within-pair gaps and includes a common first-to-second shift (mean -9.17, t on 5 df p 0.036, all six pairs); the SD about the shift is 5.6 (5 df). Same-session frames give 3.9 (nine builds), 5.95 (eight identical runs), 4.7 (18 pooled) and 4.9 (interleaved arms, pooled); their df-weighted pool is 4.9 (25 df, 3.9 to 6.8). We use 8.25 for cross-day comparisons and about 5 to 6 for same-session ones, and we state minimum detectable effects as a range.

MDE as a range (state both ends, say which SD): two runs per arm, 80% power, 5% two-sided: 13 to 23 levels by normal approximation (SD 4.7 to 8.25), 27 to 47 with the t distribution on 2 df; six runs per arm: 8 to 15 (t); twelve: 5.6 to 9.9 (t). Also note the paper currently quotes normal-approximation figures (V3-W12, V3-W19: 23, 15.9); the t-based 3-against-7 value at 8.25 is 18.2.

## 7. Contradictions with current claims

- Writeup/PDF treat 8.25 as the one noise SD. It is the largest of the frames, and about 62% of its mean square is a mean shift, not spread. Truthful, but the "noise" label hides that the shift is directional (second lower, 6 of 6).
- The "two runs per arm detect only gains above about 23" line is a normal-approximation figure at SD 8.25. At 2 df the t-based figure is 47; at SD 5 it is 13 (normal) or 27 (t). So 23 is neither the cautious nor the optimistic end.
- V3-W36 interval for the yield test (-14.2 to +9.2) uses SD 8.25 on 6 df; the test own within-arm SD is 4.9 (10 df) and gives -8.78 to 3.78. The paper chose the widest; say that it is the conservative choice.
- The 4.69 pooled SD (frame d) is derived from bench_pool18_levels.json (18 runs, 17 df, interval 3.5 to 7.0); this script reproduces it from the 18 totals.
- Frame c (3.9) is not "identical code": nine different builds. Its same-day subset (10-12 only, 5 builds) gives 3.87 on 4 df (interval 2.3 to 11.1), so it neither supports nor refutes 5.95.
