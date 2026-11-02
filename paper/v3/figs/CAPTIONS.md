# Captions for the v3 figures

Each entry: the title (the message), then the caption. Levels are counted on the 25 public demonstration games
(development numbers, not held out); board scores are public scores on half of the 110 hidden games.

## F1 `fig_loop` (body; Writeup gallery)

**One turn of the agent, and where each lever acts.**
The agent sees a board image, reasons on the local server, and acts by calling `action()` from Python; an action ends
the turn and the next turn starts from the new board. Three key decisions are in colour: (1) a turn that generates 8,192
tokens without an action yields and resumes with its opener (the turn's opening message) and board image sent again
(Franzen's harness yielded at 2,048), so they are re-sent less often; (2) keeping 448 of 512 experts frees memory for 14
games at once; (3) fail-closed checks make a dead server or a run under 20 levels raise an error, which does not spend the
daily slot; neither has fired on a scored run, and they cannot catch a run killed from outside. Faster serving acts on the
server; fine-tuning would act on the model weights (our adapter was not shipped). Schematic, no data. Source: the component record `paper/v2/evidence/system.md`.

## F2 `fig_noise_draws` (body; Writeup gallery)

**Identical code, different scores.**
(a) Six pairs of byte-identical development runs (filled: first run, open: second; the second is lower in all six,
sign test p 0.031); spreads of 3 to 22 levels give a run SD of 8.25 levels (6 df, 95% CI 5.3 to 18.2). Bottom row: eight
byte-identical runs on Vertex AI on 2026-10-20, 115 to 132 levels (SD 5.95, 7 df, 95% CI 3.9 to 12.1). (b) All 17 valid
public-leaderboard draws of one agent, 2026-10-13 to 2026-11-02, grouped by build: shipped agent (3 draws); draft-model
variant (4; the same agent with a retuned vocabulary map for its draft model, which should leave the output distribution
unchanged); boot-keepalive variant (5; a keepalive that acts only while the server boots); later builds (5; the
15-stream and harness builds from 10-29). Not interleaved in time, so the gaps between groups are not attributed. The
one 0.00 draw (2026-10-18, a 3,411-byte output, cause unknown) is the open dot, off the scale. Dashed line: mean of the 17,
34.59. The mean of the first 12 draws (to 10-28) is also 34.59, by coincidence (415.09 / 12 and 588.08 / 17). 40.98 is
the best of these 17 and of all 28 live draws since the fork. Higher is better. No band: every run and draw is plotted.
Data: `paper/v2/analysis/out/e4_noise.json`, `paper/v3/evidence/data/vertex_vs_kaggle_runs.csv`,
`paper/v3/analysis/figs/data/board_draws_all_2026-11-02.csv`. Claims V3-W16, V3-W30.

## F3 `fig_power` (body)

**Two runs per arm resolve about 23 levels; a 12 vs 12 wave would resolve about 9 (not yet run).**
Smallest level shift that a two-arm test with n runs per arm detects at 80% power and two-sided 5%,
d = 2.80 x SD x sqrt(2/n). Solid line: SD 8.25 from six identical pairs; graded band: the 50%, 80% and 95% chi-square
intervals of that SD (6 df; 95%: 5.3 to 18.2). Dashed: SD 5.95 from eight concurrent identical runs. Each lever is placed
where its size meets the solid line, so its x is the runs per arm a test needs: the yield change, +12.5 levels (9 runs
against 21), about 7; the fine-tuned adapter, 7 to 15 levels below the pin mean of 124.0 (three runs), 5 to 22; +11%
serving tokens, priced at 4.2 levels by the budget curve (F4), about 61. An effect above the line at the runs used is
resolvable. Every runs-per-arm number in the paper and the Writeup is two-sided at 80% power (Table detect: 32 and 61
runs per arm at SD 5.95 and 8.25 for the serving price). Data: `e4_noise.json`, `yield_stats.json`, `e5_detect.json` (v2 out); `mde_by_runs.csv`,
`vertex_vs_kaggle_runs.csv`, `lora_replicas.csv` (v3 evidence).

## F4 `fig_budget` (body)

**A simulator puts the agent near its token ceiling: 11% more tokens is priced at about 4 levels.**
Simulated score against token budget, relative to today's serving speed (rho = 1), replaying recorded level attempts of 8
replica runs on the 25 demonstration games in the hidden run's regime (110 games, 14 streams, 31,920 s, our priority
rule); at this budget, not a held-out result. Line: all 8 runs pooled (480 seeds). Graded band: the middle 50%, 80% and
95% of 200 run-level bootstrap resamples (8 of the 8 runs drawn with replacement, 10 seeds each); at rho = 1 the 95%
range is 54.4 to 58.3. Resamples hold about 5.2 distinct runs, so near the plateau the band sits below the pooled line.
The agent (dot, 56.7) earns 94% of these 8 runs' plateau (dashed, 60.2) and about 90% of the ceiling extrapolated to more
runs (hatched, 62.1 to 66.4; 85% to 91% across estimates). The 8 runs are seven at the 8,192 yield and the one run
without a token yield. The simulator assumes that levels no attempt cleared stay stuck; no run had more than this budget. +11% serving tokens moves it along the curve by 1.4 points,
about 4 levels per run, less than the SD of one run of identical code (bar, 5.0 points). Higher is better. Data:
`f1_curves_n2.json`, `new_plateau_pool.json`, `e4_noise.json` (v2 out); `paper/v3/analysis/figs/data/budget_band.json`.

## F5 `fig_yield` (body; Writeup gallery)

**Re-sending context less often coincided with more levels across days, and not when interleaved.**
(a) Across days: for each of the 25 demonstration games, mean levels of 9 runs that yield at 8,192 tokens minus mean
levels of 21 runs that yield at 2,048 (one dot per game, sorted; open: the one game every run cleared fully). 22 games
up, 2 down, 1 tie. Line: mean, +0.50 levels per game (+12.5 per run); band: 95% run-level bootstrap CI (20,000
resamples of runs within each arm), +9.2 to +15.8 per run. These runs were not interleaved, and seven of the nine 8,192
runs also carry serving changes (+17% tokens, worth about 5 of the 12.5 levels), so the gap is a coincidence of
several things, not an attribution. (b) Resume openers per game (a turn resumed after a yield, opener and image
re-sent), in clean turns. Circles: three 2,048 runs, observed (x) against the same turns replayed at 8,192 (y), mean of
three, 74% fewer in total. Triangles: one 8,192 run, replayed at 2,048 (x) against observed (y), 70% fewer. Each replay
reproduces its own run's count exactly. Below the diagonal = fewer re-sends. (c) Interleaved test: six runs at 2,048
(123, 125, 119, 124, 123, 125; mean 123.17) and six at 8,192 (114, 129, 113, 127, 120, 121; mean 120.67), one build,
alternating regions, predeclared in a file dated 2026-10-21. Right: the two differences, 8,192 minus 2,048, with 95%
intervals at the six identical pairs' SD of 8.25 (6 df): across days +12.5 (+4.5 to +20.6); interleaved -2.5 (-14.2 to
+9.2). The interleaved interval excludes +12.5 and any gain above about 9 levels, and does not exclude smaller gains;
at the eight concurrent runs' SD of 5.95 it is -10.6 to +5.6. Drift between days and the extra tokens are candidate
causes of the non-reproduction. Four launched runs never ran (three cancelled while pending, one failed on a regional
stockout). Higher is better. Data: `yield_runs.json`, `yield_stats.json`, `yield_mech.json` (v2 out);
`paper/v3/analysis/out/yield_wave/read_2026-10-24.json`; per-game counts
`paper/v3/analysis/figs/data/yield_resume_by_game.csv`. Claims D1, D5, D8, D9, V3-W36.

## F6 `fig_finetune` (body; Writeup gallery)

**All three adapter runs scored low.**
A LoRA adapter trained on the agent's own cleared-level turns from 127 offline RE-ARC-3 games (the training games), served
at LoRA scale 0 (no adapter), half and full trained strength. (a) Levels per run: 9 of the 18 runs of the yield-8,192
family without the adapter are plotted (the nine on Kaggle: 119 to 129, mean 124.0, SD 3.9; the other nine, on Vertex
AI, are not drawn; all 18 span 115 to 132, mean 124.4, and the text's rank p of 0.004 is against all 18), one run at half
strength (117) and two at full (109, 115; GCP VM, not interleaved with the runs without it). Every run is plotted; bar:
no-adapter mean; dashed: mean by scale. One half-strength run cannot show a dose response. Higher is better. (b) Play counters per run on the same runs: reasoning characters per action
and actions per reply on cleared levels (8 pin runs with transcripts), RESET actions per run (9 pin runs), and deaths
(GAME OVER echoes in transcripts) per run (8 pin runs). Gray band: the agent's own play on the training games (two
datagen attempts; read from `docs/research/2026-10-19-top/lora.md`, not recomputed). Reasoning per action moves toward the training play; actions per reply
moves about a tenth of the way; RESETs double and deaths rise at full strength only, which weaker play would also
produce; NO-OP share (not drawn) moves away from the training play. RESETs and deaths have no training-play line: the
datagen attempts played 127 and 48 games at the same per-game budget as a development run, so their whole-run counts do
not compare with a 25-game run. Data: `paper/v3/evidence/data/lora_replicas.csv`, `lora_behaviour.csv`.

## F7 `fig_tried` (appendix)

**What we tried: almost every arm lands inside the noise of one run.**
One dot per measured arm (27 arms), levels minus its control, grouped by lever class. Control: the pooled byte-identical
pair where one exists, else the run the arm was gated against; the right column gives arm runs against control runs.
Filled: the mechanism counter the arm was built to move moved as its gate asked; cross: it did not; open: no counter was
recorded. Colour and bold label: shipped. Shaded: where 50% and 95% of the gaps between two single runs of identical code
fall (SD 8.25). No arm falls outside the 95% band; expert pruning (+19.5, 2 runs against its 2 startup-trim controls, 92 and 97) sits
nearest its edge. Not drawn:
BF16 draft KV (0 levels, out of memory) and arms with no level reading. Data:
`paper/v3/analysis/figs/data/tried_arms.csv` (sources: `docs/research/2026-10-19-top/ledger.md`, `e4_gates.txt`, the
named gate records), levels from `yield_runs.json` and `lora_replicas.csv`.

## F8 `fig_selection` (appendix)

**How much of 40.98 is selection: the expected best of 12 draws is 5.7 points over the mean.**
Points: the 12 valid board draws of one agent in date order, to 2026-10-28; orange step: best so far. Solid line: the
expected best of n draws resampled from these 12 (Dodge et al.'s closed form, E[max_n] = sum_i v_(i) ((i/N)^n -
((i-1)/N)^n), N = 12); dashed: a normal model with the draws' mean and SD (34.59, 3.59). E[max_12] = 40.33 (normal
40.43); E[max_23] = 40.82 (normal 41.51), 23 being the live draws since the fork to 2026-10-28. Check: the closed form matches a
simulation (200,000 sets of n draws with replacement, n = 1 to 23) to within 0.012 points at every n. The resampled form cannot exceed the observed best, which is why the normal model is shown.
Higher is better. Data: `paper/v3/evidence/data/board_draws.csv`.

## F9 `fig_trainnoise` (appendix)

**The training forward pass is not deterministic: raw k3 spans four decades on identical weights; clipped k3 does not.**
Five base-only forward passes over the same 24 validation windows (66,949 tokens; three VM processes, two Vertex hosts),
compared in all 10 pairs and in both directions (20 readings). Only about 12% of tokens match exactly between any two
passes. (a) Raw k3 (open) ranges from 0.099 to 1,525; clipped k3 (filled; each token's log ratio clamped to [-2, 2])
from 0.0629 to 0.0658. Every raw reading is above the predeclared stop (k3 above 0.02), so the stop would have fired on
noise. (b) The clipped readings on a linear axis: mean 0.0644, SD 0.0008, against the amended stop at the run's own floor
plus 0.02. VM with VM, VM with Vertex and Vertex with Vertex pairs cannot be told apart. Lower is closer to identical.
Data: `paper/v3/evidence/data/grpo_base_noise_pairs.csv`.

## Cover (Writeup)

**Which changes move the agent?** (FRAMING_v3 s9)
Schematic, no numbers printed. Score rises with tokens and saturates; the agent sits near the top. The arrows carry only
questions or tested facts: "faster serving: small, if any" (along the curve), "context re-send: under test" (up),
"imitating easy wins: lowered levels" (down). The gray band is the noise of one run, drawn at the measured ratio: its
half-width (SD 8.25 levels) against the vertical arrows (the +12.5-level yield gap). Files: `cover_1200x675.png`,
`cover_1600x900.png`, `cover.pdf`, `cover_thumb_200.png`.

## F10 `fig_draws` (the anonymised PDF uses `fig_draws_anon`, same figure without the rank text; `fig_draws.py --anon`) (body)

**Every daily draw: a jump from about 4 to about 30 at the fork, then a plateau of 27 to 41.**
(a) One point per scored submission, n = 81 (2026-07-17 to 2026-11-02; two errored submissions excluded), by configuration
family: early stack (50 draws before the Franzen fork, gray), Franzen fork with first pins (9), yield-8K family (9), later
pins (8), 15-stream pin (3), harness builds (2: 33.56 on 11-01, 36.38 on 11-02, the final draw). Open markers at the axis
are the three 0.00 draws (2026-10-03, 10-10, 10-18); the first two were field-wide early failures of the midnight wave, the
cause of the third is unknown. Rings: the two Final Submissions, 40.09 (2026-10-13) and 40.98 (2026-10-20). Higher is
better. (b) Our best public score so far (step) against the rank-1, 3 and 5 scores, one leaderboard snapshot per day from
2026-08-25 (rank 1 is recorded from 2026-09-12); last snapshot 2026-11-02 07:06Z: 40.98, rank 20 of 5,043 (live read at
about 12:15Z: rank 23 of 5,048, before the deadline). Not shown: rank over time (in September a fraction of a point moves
rank by hundreds), and the private score, which is not known until after 2026-11-02. Same-family draws differ by up to 12
points (28.97 to 40.98 for the yield-8K family and its variants), so a family's position is not an effect. Data:
`paper/v3/analysis/figs/data/board_draws_all_2026-11-02.csv`, `board_snapshots_2026-11-02.csv`.

## F11 `fig_clock` (body)

**A quarter of the clock goes to the level a game was stuck on when the window closed; idle turns hold 27%.**
Six 25-game replica runs (three with the stall check, three with stall check plus object groups), 150 game plays, 226
game-hours of turn clock. (a) Per game, mean minutes of the game's clock on levels it cleared (gray) and on the level it
was on when the window closed (vermillion); sorted by the stuck tail; the right column is the mean number of levels
cleared out of the game's levels. A game that cleared every level has no tail. (b) Share of the clock by the number of
actions executed in the turn (action counter difference between consecutive turns); bars pool the six runs, dots are
single runs (filled: stall check; open triangles: plus object groups). Idle turns (0 actions) hold 27% pooled (27.5% in the
stall-check runs); the first idle turn of a streak alone holds 20%. Time stamps are wall clock at the start of each turn
on shared serving, so a turn's clock includes queueing. Idle turns are mostly real work (simulators, history reads), so
the figure shows where time goes, not waste. It does not show what a faster turn would buy, and the last turn of a game
is dropped. Data: `paper/v3/analysis/figs/data/clock_by_game_run.csv`.

## F12 `fig_gates` (body; Writeup gallery)

**Each change moves the mean by less than one run-to-run SD of identical code.**
Levels cleared (a) and game score (b) of each 25-game development run, one dot per run (gray: the control; vermillion:
the change), bar: mean, dashed line: the control's mean (123.4 levels, 54.85 score), shading: one SD of identical runs
(8.25 levels, 4.98 score points). Control n = 7 (129, 120, 125, 122, 135, 123, 110; own SD 7.76); stall check n = 3 (141, 126,
118; mean 128.3, +4.9); corner-joined objects n = 3 (132, 128, 127; mean 129.0, +5.6); scroll detector n = 3 (111, 122, 130;
mean 121.0, -2.4); tool fixes n = 3 (123, 129, 115; mean 122.3, -1.1); replay check n = 3 (124, 128, 128; mean 126.7,
+3.2). Higher is better. The tool fixes and the replay check extend the corner-joined build, not the scroll build.
No test or significance marks are drawn: with this noise, 3 runs per arm resolve about 14 to 19 levels (13.6 at SD 5.95, 18.9 at 8.25), so the figure
cannot rank the changes and does not show that any lever helps or hurts. The control is mostly another platform (five of
its seven runs on Vertex AI, two on Kaggle); the changes are all Kaggle. Data: `paper/v3/analysis/figs/data/gate_runs.csv`.

## F13 `fig_glyph` (appendix)

**Four-connected objects turn a drawn glyph into single pixels; a `group` field turns it back into one object.**
bp35 level 3, start frame of a human replay (replay frame 55), a 7 by 18 cell crop around the three X marks, each X five cells
of colour 12. (a) the crop. (b) segmentation by the harness's 4-connected nodes: the 15 X cells are 15 one-pixel nodes, since
the cells of an X touch only at corners. (c) the same board with the new `group` field (nodes of one colour joined through
corner contacts): 3 objects of 3 by 3. The counts are the real patched `segmentation.py` output, and its nodes, hashes and
boundaries equal the unpatched module's. (d) over every level start of the 22 human replays (184 starts), 76 starts in 13 games
hold a `group` of 2 or more nodes with a bounding box of at least 3 by 3 cells (73 if sparse dot lattices are dropped). The
count is a property of the boards, not of play: it does not show that the model reads a group better or that any level was
cleared because of it. B and C are faded so the outlines read; colours are a fixed palette chosen for the picture. The seg8
docstring's 71 in 13 games used a different, unstated dot-texture rule and is not reproduced. Data:
`paper/v3/analysis/figs/data/frames/h_bp35.json`, `glyph_corpus.json`.

## F14 `fig_scroll` (appendix)

**A camera scroll reads as hundreds of changed cells and dozens of moved objects; the scroll detector reports it as one shift.**
Two human-replay frame pairs, one per game: bp35 level 1 (frames 4 to 5) and lf52 level 3 (frames 58 to 59). Columns: before,
after, changed cells under the unpatched diff, changed cells after undoing the detected shift (gray: cells that scrolled into
view, which the patched module counts as changed). bp35: 1,343 cells and 89 objects 'moved' unpatched; patched, scroll [24, 0],
1,668 cells (1,536 scrolled into view) and 11 moved. lf52: 1,633 cells and 48 moved unpatched; patched, scroll [0, -8],
743 cells (512 in view) and 10 moved. The cell count rises for bp35 because the strip that scrolled in is counted; the gain
is in the moved-object list. (C) the detector's firing rate over all 6,313 consecutive same-level pairs with a changed frame in
the 22 human replays: 103 of 363 in bp35, 40 of 772 in lf52, 0 of 5,178 in the other 20 games. Numbers are from the real
`frame_diff.py` before and after overlay scroll. Two pairs are chosen examples, not a sample, and the figure does not show that the
model plays scroll games better with the report. Data: `paper/v3/analysis/figs/data/frames/h_bp35.json`, `h_lf52.json`,
`scroll_corpus.json`.

## F15 `fig_discovery` (body)

**A quarter of the cleared levels hold two thirds of their clock, and on those levels the winning code shows up late.**
Analyzer transcripts of 25-game replica runs, cleared levels from level 2 on (cleared: a later turn shows a higher level,
so a level cleared on a game's last turn is missed; 97 of the 1,113 level-2+ clears in `benchmark.json` for the
stall-check family). Solid: the stall-check family, 11 runs of the 15-stream harness before the replay check (the stall-check
build and three builds that add object groups, the scroll report or the tool fixes), 275 game plays, 1,016 cleared levels. Dashed: the
replay-check build, 3 runs with the `check_model()` and `check_goal()` helpers, 279 cleared levels; its numbers are
readings, not claims. (a) Levels sorted longest first, cumulative share of levels against cumulative share of their
clock; the dot is the 15 minute point, 27% of levels holding 66% of the clock (replay-check build: 30% and 71%); gray
diagonal: equal shares. (b) On levels over 15 minutes with 4 or more turns, the share of the level's clock elapsed when
code like the clearing turn's (identifier Jaccard 0.5 or more) first appears; the proxy matches 94 of 272 levels (27 of
83); vertical line: median 87% elapsed, 13% left (replay-check build: 13% left too). (c) Cleared levels whose code checks
a model against the recorded history: 4.0% of 275 in 3 runs before the stall check, 6.0% of 1,016 in the stall-check
family, both by the hand-written detector (a loop over transitions or history that compares a simulator call with the
recorded frame). That detector cannot see a helper call, so the replay-check bar stacks hand-written only (2.9%), both
(1.8%) and helper only (5.4%), 10.0% of 279 in all. The proxy in (b) is a text match, not proof that the matched code is
what won. The figure shows where clock goes, not that any change clears more levels. Data:
`paper/v3/analysis/figs/data/discovery_levels.csv`, `discovery_summary.json`.
