# Second population for the noise recipe (zero GPU)

Script: `paper/v3/analysis/v3_second_population.py` (stdlib, `python3 -I -B`). Data: `out/second_population.json`.
Recipe: MDE = 2.8016 x SD x sqrt(2/n), two-sided 5%, 80% power, SD known. Tags: [computed] = by the script from repo files or numbers read; [read] = taken from a public page the same day (2026-11-04).

## 1. Repo: Franzen copies (forum 745062)

- Pham's 25.77 / SD 3.93 / median 26.19: [read] in `paper/v2/evidence/board.md` line 58 from the stored forum JSON. His n is only in a screenshot ("hundreds"), never stated in text. Mintu Kumar asked for n on 10-06, no reply in the stored dump (v2 `new_repeat_sets.md` section 6).
- The repo DOES hold the snapshots to rebuild a cross-section: `solution/kaggle/generated/daily/board-2026-10-01.csv` and `-10-02.csv`. [computed] Rule: submission count up by exactly 1, prior best < 10, new best >= floor.

| Floor | n | mean | SD | median | MDE n=1 / 2 / 3 / 5 (score points) |
|---|---|---|---|---|---|
| 10 | 220 | 26.03 | 3.67 | 26.22 | 14.5 / 10.3 / 8.4 / 6.5 |
| 15 | 215 | 26.35 | 3.04 | 26.25 | 12.0 / 8.5 / 7.0 / 5.4 |
| 20 | 208 | 26.63 | 2.65 | 26.38 | 10.5 / 7.4 / 6.1 / 4.7 |

- Medians match Pham's to 0.1 to 0.2. SD runs 2.65 to 3.67 with the floor, Pham's 3.93 fits a lower floor that keeps failures. This reproduces v2's N6-8 numbers exactly (n=215, SD 3.04 asserted in the script).
- Limits: it is a cross-section of different teams (edits, hardware, failed runs included), not repeats of one build, and "unchanged copy" is not checked per team. The snapshots only show each team's best score, so per-team multi-draw repeats of a fork cannot be built. No team in the repo has several scored draws of byte-identical code except ours.
- Within that limit it is a usable second population for BOARD draws: single-draw differences between two teams' configs are unresolvable below about 10 to 12 points at n=1.

## 2. Other ARC-AGI-3 repeat sets (levels or scores)

| Set | n | mean | SD [computed] | MDE n=1 / 2 / 3 / 5 | Caveat |
|---|---|---|---|---|---|
| Tufa Labs, 20 passes, 25 public games, Qwen3.6-27B, one job (levels of 183) | 20 | 15.45 | 2.44 | 9.7 / 6.8 / 5.6 / 4.3 (63% / 44% / 36% / 28% of mean) | Within one job, no session shift. SD/mean 15.8%. |
| Franzen v3, 4 passes, one job (levels) | 4 | 104.25 | 1.71 | 6.8 / 4.8 / 3.9 / 3.0 | Within one job, floor only. VIF 0.06 (v2). |
| guoyi12311125 README, "identical code" Kaggle scores 24.38, 29.41, 29.60 | 3 | 27.80 | 2.96 | 11.7 / 8.3 / 6.8 / 5.3 | [read] README scores only. Unit, dates, hardware not stated. Adding the byte-identical pair 29.89 and 30.21: n=5, SD 2.43. |

Our own for comparison [read, `evidence/data/mde_by_runs.csv`]: SD 3.9 to 8.25 levels at about 111 to 124 levels, MDE at n=1 of 15.5 to 32.7 levels, relative SD 3.3% to 7.2%.

The only independent same-protocol population (Tufa) has a LARGER relative SD (15.8%) at a smaller scale: at n=2 per arm a gain must exceed 44% of the mean. The guoyi README is a second board-level set with SD 2.4 to 3.0 points, in line with the field cross-section (2.65 to 3.67) and above our 1.58 pre-yield draws.

## 3. Other agent benchmarks (public numbers)

| Source | What it publishes [read] | MDE by recipe [computed] | Versus typical gains |
|---|---|---|---|
| Bjarnason, Silva, Monperrus, arXiv 2602.07150 (SWE-Bench-Verified, 500 tasks, 10 runs per cell, 12 model/scaffold/temperature cells) | Table 1 pass@1 mean +/- std per cell: SD 0.7 to 1.8 points (median 1.45), means 16.4% to 63.8%, relative SD 1.7% to 8.1% [computed from the table]. Abstract quotes: "standard deviations exceeding 1.5 percentage points even at temperature 0"; "2.2 to 6.0" points depending on which run is picked | At SD 1.45: 5.7 / 4.1 / 3.3 / 2.6 points for n = 1/2/3/5. At SD 0.7: 2.8 / 2.0 / 1.6 / 1.2. At SD 1.8: 7.1 / 5.0 / 4.1 / 3.2 | A single-run SWE-bench gain of a few points (the paper's own 2.2 to 6.0 pick-a-run spread) is at or below the n=1 detection line |
| Anthropic, "infrastructure noise" post | Terminal-Bench 2.0 extremes gap 6 points (p < 0.01); moderate configs spread "just below 2"; recommends skepticism below 3 points; SWE-bench RAM 5x effect 1.54 points; "roughly 1 to 2" binomial CI. No SD, no runs per config | Not computable | Its 3-point skepticism line is close to our recipe's n=2 to 3 line at SD 1.45 (4.1 / 3.3) |
| Miller, arXiv 2411.00640 | Abstract only: formulas for comparing two models and planning evals, no repeat-run SD | Not computable | n/a |
| Kapoor et al., arXiv 2407.01502 | Abstract only: "pervasive lack of reproducibility", no SD | Not computable | n/a |

Note on units: Bjarnason's SD is over a fixed 500-task set, which is the same kind of quantity as ours (fixed games, repeat the agent). Relative SD overlaps ours (1.7% to 8.1% against 3.3% to 7.2%). Not apples to apples: SWE-bench is a per-task pass rate, ours is a sum of weighted all-or-nothing levels, and the SDs are within-config, one job family each.

## 4. Recommendation

Enough for ONE hedged sentence, not for a "universality" claim. What it may say truthfully:

> The same recipe on public repeat-run data from other settings gives similar sizes: 10 repeats of SWE-Bench-Verified agents (Bjarnason et al.) have SD 0.7 to 1.8 points, so one run per arm detects only about 6 points; Tufa Labs' 20 public ARC-AGI-3 passes have SD 2.4 levels on a mean of 15.

(Word count 49. Shorter: "Public repeat runs elsewhere give the same picture: SWE-Bench-Verified agents (10 runs per cell, Bjarnason et al.) have SD 0.7 to 1.8 points, about 6 points to detect at one run per arm.")

What it may NOT say:
- Not that the noise recipe "generalizes" or is "confirmed" on other benchmarks: both are within-config SDs I applied the formula to, no one has tested the recipe's predictions there, and Bjarnason's 2.2 to 6.0 is a pick-a-run spread, not a measured gain.
- Not that Tufa corroborates 8.25: it is within-job (no session shift), at 15 levels, 16% relative SD, a different model.
- Not "Pham's copies show SD 3.93 with n=N": n is unstated. The rebuilt cross-section (n=215, SD 3.04; 220 and 3.67 at floor 10) is cross-team and not repeats. It may be cited as "about 3 to 3.7 on 215 one-shot copies, rebuilt from snapshots", flagged as cross-section.
- The guoyi README (n=3, SD 2.96, conditions unknown) is too thin and unverified (scores read from a README, not from Kaggle) to cite as a number; mention only if space allows, as "a public repo reports 24.4, 29.4, 29.6 for identical code".
- Anthropic, Miller, Kapoor publish no usable SD: cite none for the recipe.

Current-claim check: nothing found contradicts the Writeup. Bjarnason's "2.2 to 6.0" in the Writeup matches the abstract; the Writeup's "run chosen" wording is right. The v2 verdict stands: no public set repeats a fresh session at our scale, so no public set tests the 8.25 or the 5.6-around-the-shift figure.
