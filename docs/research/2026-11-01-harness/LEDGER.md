# Research pass ledger (draft for the paper appendix)

Drafted 2026-11-02. Sources: the 2026-11-01 research digest (`scratchpad/research_digest.txt`, 411 lines; "Lnnn" below
is a digest line), claims rows V3-W28 and V3-W29 (`paper/v3/evidence/claims_addendum.md` lines 37 and 38), and
`docs/PROGRESS.md` entries 2026-11-01 18:45Z to 2026-11-02 12:50Z (lines 7142 to 7273). No paper .tex or docs file
was edited.

Provenance tags. A tag at the end of a cell or bullet covers every number in it unless a number carries its own tag.

- [reproduced]: recomputed on 2026-11-02 by rerunning `solution/analysis/harness_2026-10/gates/ustv_gate.py 122.3`,
  `ust_gate.py` and `verify_outcomes.py` on the local kernel outputs, by one new count (helper calls as a share of all
  code blocks), by arithmetic on per-run values, or by counting digest lines.
- [source-verified]: read in the named repo record, not recomputed.
- [from digest, not reproduced]: the research or skeptic agents' own measurement; their scripts sit in the session
  scratchpad and were not rerun.

## 1. Method

On 2026-11-01 (run wf_27428c3d-32e) five research agents each took one angle: public ARC-AGI-3 harnesses
(public-arc3), the agent literature, a first-principles read of our harness code (code-read), a quantitative read of
our transcripts (transcripts), and the competitive meta (PROGRESS L7143 to L7145 [source-verified]; the digest holds
exactly these five summaries, L300 to L411). A synthesizer merged their 26 proposals (5, 5, 8, 5 and 3 per angle,
L318 to L322, L335 to L339, L361 to L368, L388 to L392, L409 to L411 [reproduced]) into one model of where the clock
goes (L1 to L10), five ranked candidates, each with a spec, evidence, an offline test and a gate metric (L12 to
L202), and 15 rejections with reasons (L205 to L219 [reproduced]). One skeptic per top-four candidate (4 skeptics,
L221 to L295) re-measured the candidate's evidence on our own runs and returned a strongest objection, an adjusted
expected effect, what would change its mind and a list of build pitfalls; all four set `refuted= True`, and candidate
5 (groupdiff) got no skeptic. PROGRESS counts 10 agents (L7153 [source-verified]), which matches 5 + 1 + 4
[inference about the split]; we built only candidate 1 (verify), fixed the skeptic's build pitfalls, and gated it on
mechanism plus no harm, because every skeptic put the level effect below what three runs can resolve.

## 2. Candidates

### 2a. The five ranked candidates

| # | Candidate (angles, build estimate) | What it would change | Skeptic verdict | Skeptic's adjusted effect | Key measured fact behind the verdict | What happened |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | verify kit (public-arc3, agent-literature, transcripts, competitive-meta; 2 h; L12, L52 "four of five angles") | Two sandbox helpers: `check_model(step, observe)` replays the current level's recorded transitions through the model's own simulator and names the first mismatches; `check_goal(is_goal)` runs a goal guess on every recorded board of the level, none of which ended it; plus a host-time prompt clause (f) (L13 to L48) | Refuted (L221) | 0 to +1 level per 25-game run (0 to +0.5 score points); generous case about +0.2 levels (L225) [from digest, not reproduced] | In 3 us8 runs, 17 of 75 stuck plays had a simulator and 3 had a check; about 4 stuck levels (about 1.3 per run) had 10 or more minutes left after a simulator appeared with no check; 8 of the 10 false-rule stuck levels are consistent with every recorded transition, so replay cannot refute them; closest precedent GG6, a validate helper on Flash-Next, called in 1.5% of calls (L222, L223) [from digest, not reproduced]. GG6 1.5% of calls and arm V 1 of 50 runs [source-verified, PROGRESS L4620] | Built without clause (f), with the skeptic's pitfalls fixed; offline test 9 checks on both bases; 3 Kaggle runs PASS every predeclared item; promoted 2026-11-02 00:28Z; carried the final daily draw, 36.38 (Section 3) [source-verified, V3-W29, PROGRESS L7199 to L7250] |
| 2 | codefirst (code-read; 1 h; L83) | After a reply cut at the 12K output cap, the next opener says so and redirects the model to code or a 1 to 2 action probe (L84 to L96) | Refuted (L228) | About 0; best case under 1 level per run (L256) [from digest, not reproduced] | The post-cut turn already acts: its first call holds `def ` or `action(` in 66 of 79 (0.84), 68 of 84 next invocations execute an action, back-to-back cuts 5 of 84; the addressable time is about 1.6% of the clock; the gate's m2 bar (0.7) sat below that baseline; the closest arm (l1a1tbfc, a forced tool call at this moment) tied (L229, L231, L233, L235) [from digest, not reproduced] | Not built |
| 3 | coverage v3 (public-arc3, agent-literature, code-read, competitive-meta; 1.5 h; L116) | At stall and failure openers, list non-directional actions and click types tried from only one board state with no effect, plus an UNDO line where UNDO is valid and unused on the level (L117 to L131) | Refuted (L267) | About 0; the list's upper bound about +0.2 level per run; the UNDO line saves actions, not completions (L279) [from digest, not reproduced] | The spec's own prerequisite back-test fails: 14 of 154 listed items (9%) were later used with effect in a clearing run, against a 20% ship bar; on stuck levels 1 of 49 (2%); 150 game runs replayed on the local engine, all reproducing the benchmark level counts (L133, L268, L270, L273) [from digest, not reproduced] | Not built (failed its own CPU back-test bar) |
| 4 | costmodel (code-read; 0.75 h; L160) | Replace the prompt's only objective line ("Optimize for as few in-game actions as possible while still being reliable") with the true scoring and clock model: completion first, the clock binds (L163, L164) | Refuted (L290) | About 0, plausible range -6 to +3 levels, slightly negative expected score (L294) [from digest, not reproduced] | Thinking that reasons from the scoring objective is 16 idle turns, 2,815 s, 0.7% of the clock in 3 us8 runs; a uniform 1.15x action rise would cost about 4.5% of the score estimate yet pass the proposed guard (median ratio 0.757 against 0.774) (L291, L292) [from digest, not reproduced]. Arm EE deleted the same line in September: 30 levels, 3,144 actions [source-verified, PROGRESS L3718], against 35 for its control (L291) [from digest, not reproduced] | Not built |
| 5 | groupdiff (code-read; 2.25 h; L183) | `frame_diff` gains an additive `group_moves` key that reports each seg8 8-connected group whose parts all moved by one offset as one entry (L184 to L192) | No skeptic pass (no VERIFY block in the digest) | None given | Its own counter-evidence: scroll, the last `frame_diff` change, fired in its target games yet bp35 stayed at 1 level in all 3 runs and its mean was 8 levels under us8 (L199) [from digest, not reproduced]; scroll's runs 111, 122, 130 (mean 121.0, -8.0 on us8's 129.0) [source-verified, V3-W27] | Not built |

### 2b. Named but not ranked (synthesizer's disposition; no skeptic ran on these)

| Item | Proposed by | Disposition | Reason given, with its number |
| --- | --- | --- | --- |
| Aligned-window gate reading | transcripts (L388) | Not a harness change; recommended as the predeclared reader for the last wave (L205) | 6 of 19 lf15-family runs started play 12 to 22 min before the server was READY; 120.2 against 127.5 levels raw, 116.5 against 116.2 on a common 5,800 s window (L205, L385) [from digest, not reproduced]. What happened: the verify gate listed it as reported only and it was not computed [source-verified, predeclared JSON `reported_not_gated`, V3-W29] |
| `run_checked` halt-on-mismatch executor | public-arc3, agent-literature (L319, L337) | Rejected (L206) | The model already hand-writes checked loops (445 blocks in 3 runs); FF's host-enforced prediction gate lost 10 of 35 levels (L206) [from digest, not reproduced]. Its host-time sentence went into candidate 1 as clause (f), then was dropped at build (Section 3d) |
| Typed transitions | public-arc3 (L320) | Merged into `check_model` (L207) | No new behavior beyond the helper's own skip rules |
| Goal-consistency check (certified negatives) | public-arc3 (L318) | Became `check_goal` in candidate 1 | (none) |
| `search()` BFS/A* helper | transcripts (L391) | Rejected (L208) | Search snippets fail 15.5% against 7.8%, but the saving is tokens; fp8 KV gave +28.7% tokens for +0 levels; the model writes its own search on 170 of 272 long levels (L208) [from digest, not reproduced] |
| SWE-agent-style traceback hints | agent-literature (L338) | Rejected (L209) | Tool errors are 8.2% of calls, mostly the model's own bugs (L209) [from digest, not reproduced] |
| Trim ledger; drain-time opener compaction | transcripts, code-read (L390, L368) | Rejected (L210) | 71K retained history tied; 2.5 to 3 h build; about 300 tokens per trim (L210) [from digest, not reproduced] |
| Level-start object table; carry-over line | code-read, transcripts (L365, L392) | Rejected (L211) | 0.5 to 1.5K tokens per level start to save about 1% of the clock; 81% of openings already reuse earlier-level functions (L211) [from digest, not reproduced] |
| levelclock (minutes on this level) | code-read (L367) | Rejected (L212) | Sign uncertain: time pressure feeds the top stuck mode |
| Salient-event summary | public-arc3 (L322) | Rejected (L213) | No measurement on our stack; untuned thresholds |
| Turducken mechanic checklist | competitive-meta (L411) | Rejected (L214) | Dead-end family: strategy audit 110 levels (L214) [from digest, not reproduced] |
| Cross-level memory (ExpeL, AWM, Reflexion) | agent-literature (L339) | Rejected (L215) | Solved-level memory 111, strategy audit 110 (L215) [from digest, not reproduced] |
| notonce, undoprobe, EMPA untested list, tried-once ledger, one-try write-off line | code-read, agent-literature, public-arc3, competitive-meta (L366, L363, L336, L321, L410) | Folded into candidate 3 (L216, L129) | notonce's own back-test is null: write-off signals equally frequent in stuck and cleared levels (L216) |
| Structured notes, death ledger, stall cutoff, more streams, summary on trim, graph explorer | public evidence (L217) | Rejected | Stall cutoff cost 9.1 points; more streams 29.90 against 47.30 (forum 750885) (L217) [from digest, not reproduced] |
| Quadratic-scoring prompt plus temperature 0.55 | public evidence (L218) | Rejected | One offline run, LB 30.26 (L218) [from digest, not reproduced] |
| Shorter prompt, lower thinking effort, fine-tuning on our wins | public and our own evidence (L219) | Rejected | -12 and -5 points (Son Pham); r16 RFT 109 to 117 against about 124 (L219) [from digest, not reproduced] |

## 3. The built candidate: verify, predicted against measured

Base build: tool2 (`ust` = us8 + tool2), 3 Kaggle runs 123, 129, 115 levels (mean 122.3), score 54.63, actions 7,099,
every predeclared item PASS [reproduced, `ust_gate.py`]; promoted 2026-11-01 20:51Z [source-verified, V3-W28, PROGRESS
L7178]. Verify (`ustv` = ust + verify) was built on it after the research verdict (PROGRESS L7168 to L7172).

Why it was built despite the refutation: it was the only candidate four of five angles proposed (L52), it computes
something for the model rather than adding advice (L10, L51), and the external case is a fixed-weight replay check
(Twin 93.3 against 61.1 for the same model and agent without the harness, arXiv 2608.14490, L55, tagged [verified]
in the digest) [from digest, not reproduced]. The gate was set to mechanism plus no harm because the skeptic's level
effect (0 to +1) is below three-run resolution (L225; predeclared JSON `evidence` field [source-verified]).

### 3a. Helper use

- Skeptic's prediction: low. The precedent it named is GG6 on Flash-Next itself, a `validate(predict)` helper called
  in 1.5% of calls, and arm V on the 27B, called in 1 of 50 game runs (L222 point 3) [from digest, not reproduced;
  both numbers source-verified, PROGRESS L4620]. Its generous case assumed a 50% call rate on eligible plays (L225).
  Its pitfall 10 warned that a bar of 10 calls per run sits below the hand-validation rate of about 19 to 23 blocks
  per run, so the gate could pass by substitution alone (L226) [from digest, not reproduced].
- Spec bar: at least 10 calls per run in at least 5 games (L80). Predeclared gate bar: mean at least 10 per run, at
  least 5 in every run, at least 4 games per run on average [source-verified, predeclared JSON `m1_verify`].
- Measured: code blocks calling `check_model(` or `check_goal(` 33, 50 and 18 per run (mean 33.7), in 6, 6 and 5 of
  25 games (mean 5.7) [reproduced]. Probe: a helper called in 1 of 25 games (tu93), 10 of 25 probe transcripts empty
  [source-verified, V3-W29, PROGRESS L7205].
- New like-for-like reading: as a share of all model code blocks, 33 of 2,909, 50 of 3,020 and 18 of 2,913 (1.13%,
  1.66%, 0.62%); pooled 101 of 8,842 = 1.14% [reproduced, same regex as the gate reader]. This is about the GG6 rate
  (1.5%), so by the precedent's own metric the skeptic's low-use prediction held; the per-run count cleared the bar
  3.4 times over because each run has about 2,900 code blocks [inference: GG6's "calls" denominator may not equal
  code blocks]. Against the us8 hand-replay rate (57 of 8,486 blocks, 0.67%, L223 [from digest, not reproduced]) the
  helper calls alone are about 1.7 times higher [inference]; hand-replay blocks in the ustv runs were not counted, so
  substitution against addition is not measured.
- Skeptic's "would change if" (L224): none of its three conditions was tested before the build. Afterward, the probe
  condition ("calling the helper on most simulator-bearing level plays") was not met in the probe (1 of 25 games);
  the per-level-play share in the W runs was not computed.

### 3b. Level effect

- Skeptic: 0 to +1 level per 25-game run, about +0.2 under generous assumptions, SE about 4.5 levels for 3 runs
  (L225) [from digest, not reproduced].
- Spec harm items: levels at least A - 2.5 and score at least A - 1.5, read on the aligned 5,800 s window (L80). The
  predeclared gate read them on the raw 7,200 s window and added an incumbent bar (mean at least ust's 122.3 minus 5 =
  117.3) [source-verified, predeclared JSON].
- Measured: 128, 124, 128 levels (mean 126.7) against pooled A 123.4 (7 runs, sum 864, mean 123.43): +3.2; against
  ust 122.3: +4.3; score 56.74 against 54.85 (+1.89) and against ust 54.63 (+2.11); actions 6,983 against 7,105;
  median actions/baseline 0.679 against 0.674 [reproduced].
- Reading: +3.2 is above the skeptic's range but below its own 4.5-level SE, so it neither confirms nor refutes the
  prediction; the gate record claims no level gain (`what_this_does_not_claim`) [source-verified]. Aligned-window
  levels were not computed (V3-W29).
- Board: the final daily draw (submission 57893128, the ustv build) scored 36.38, against 33.56 for the 11-01 us8
  draw; both below 40.09, so the two auto-picked public draws (40.98, 40.09) were unchanged [source-verified, PROGRESS
  L7250 to L7252]. One draw is not a level measurement.

### 3c. Mechanism

- Skeptic's prediction: model simulators mostly use abstract state and other signatures (about 2 of 237 take a
  frame), so without an adapter every transition becomes a TypeError "mismatch" (L222 point 4, L226 pitfall 6); 8 of
  the 10 false-rule stuck levels are log-consistent, so `check_model` can certify a false rule (L222 point 1);
  `check_goal` has almost nothing to check where it matters, 3 goal-like defs in 1 of 75 stuck plays (L223) [from
  digest, not reproduced].
- Spec mechanism items: m2, the step function redefined within 2 turns after a mismatch in at least 0.5 of cases;
  m3, replay-validated share of level plays at least 15% against a 6% baseline (L80). The predeclared gate moved both
  to reported only [source-verified]; neither was computed (V3-W29).
- Measured: `check_model` printed 85 reports (23, 43, 19 per run): 33 matched every checked transition, 52 named a
  mismatch (42 a wrong prediction, 10 only exceptions raised by the model's own step or observe code), 8 stopped at
  the time limit. `check_goal` printed 25 reports (19, 5, 1), none with a counterexample [reproduced,
  `verify_outcomes.py`].
- Reading: the helpers ran and 42 reports named a concrete wrong prediction. Exceptions are 10 of 52 mismatch
  reports (19%) [reproduced] even with the adapter example in the prompt, a smaller share than the skeptic's
  "every transition" worst case. `check_goal` never found a counterexample, in line with the skeptic's view that it
  had little to check. Not shown: that the model fixed a rule after a mismatch, that any of the 33 all-match reports
  certified a false rule, or that any level cleared because of a report.

### 3d. Build pitfalls (L226) against the build (`solution/kaggle/franzen_overlays/verify.py`) [source-verified]

- Addressed: 1, the sandbox child source is compiled on its own before writing; 2, `import time` inside each
  closure; 3, `action` replaced by a blocking stub while model functions run, restored after; 4, only `Exception` is
  caught, and the cap is 8 s or 2 s before tool2's limit; 5, newest transitions first; 6, a lambda adapter example
  in the prompt; 7, a hidden-state note when one board and action led to two results; 8, clause (f) dropped (no
  host-time edit in the overlay's edit list); 9, model RESET and automatic RESET, deaths and level changes skipped;
  11, built on both bases (ustv run, usv built but not run) and applied after every other overlay via
  `overlay_last()` (PROGRESS L7171, predeclared JSON).
- Not addressed: 10, the m1 bar stayed at 10 calls per run. Not checked: 12, registration placement relative to
  `_available_names`.

## 4. Measured non-levers (from the research summaries)

All [from digest, not reproduced].

From code-read (L351 to L356, 3 us8 runs, 75 transcripts, 4,484 turns, L341):
- Context drains, 4 per game, show no burst of re-derivation afterwards.
- Wrong stale-state refusals cost about 0.27 turns per game.
- NameErrors, tool timeouts, truncations and retention rejections: "all small" (no number given).
- 3.7% of post-action turns open with a pure inspect-the-last-change call.
- "Written off" signals are equally common in stuck and cleared levels (keyboard no-op 5.6% against 5.6%, one-try
  clicks 16.5% against 13.9%, L144), so a list of them cannot tell the two apart.

From transcripts (L372 to L376, L381; 11 newest runs, 16,750 turns, 275 game plays, 1,387 cleared levels, L370):
- Plan failure: of 7,494 responses that ran 3 or more actions, 7% were stopped by the no-op guard and 1% hit game
  over.
- Action efficiency: median 0.64x human actions; the score sits 1.2 to 5.6 points under its cap. Code-read and the
  costmodel skeptic agree: 387 cleared levels, median ratio 0.659, 78.0% at or under 0.933 (L171, L292).
- Re-deriving mechanics: 81% of level openings call functions written on earlier levels.
- Tool errors: 8.2% of calls, mostly the model's own bugs.
- Context trims: 4.2 per game play, a median of 6 turns kept, 54% drop the current level's opening, and long levels
  run 44% of their clock without it; a natural experiment shows no harm.

Earlier measured non-levers the digest leans on: fp8 KV +28.7% tokens for +0 levels (L208); 71K retained history
tied (L210); a forced tool call that removed every cut reply tied (l1a1tbfc, L106, L233).

## 5. Competitive meta (as the digest tags it)

Tagged [verified] in the digest (L394, L396 to L398), not reproduced here unless marked:
- No team in the top 12 has published what it changed after Franzen's Milestone 2 notebook (forum read 11-01, ARC
  Daily repo, GitHub, Kaggle model lists).
- Turducken: credits prompt wording ("small changes in wording swing the score"); a short prompt cost them about 12
  points; temperature 0.6 and noborder; prunes experts by play traffic and runs a heal fine-tune, so not
  harness-only.
- xz: rank 3, 58.59 from 15 submissions; the same public Intel W4A16 weights; ported its own harness on 10-25;
  validates on public-25 plus Turducken's copycat-30; specialises in throughput (team name "faster machine is all
  you need"). Rank 3 at 58.59 [source-verified, `paper/v3/analysis/figs/data/leaderboard_live_2026-11-02T1215Z.txt`].

Tagged [claim]:
- Tufa says scored notebooks start about 40 minutes late and that time counts against the run (L399).

Tagged [inference]:
- xz's lead comes from serving speed plus its own harness (L398).

Untagged in the digest:
- Board at 11-01 18:31Z: we were 18th at 40.98; ranks 13 to 17 at 41.0 to 42.3, so the real gap is ranks 3 to 12
  (45 to 58.6) (L394). Rank 18 at 40.98 [source-verified, PROGRESS L7193].
- "Our config means fell from 38.4 (r3p) to 33.6 (us8)" (L399).
- From public-arc3 (L304 to L307): nothing public explains the gap to rank 5; no top-30 team has published; xz's
  posts are about infrastructure; Turducken's own note says it prunes experts, runs a short heal training round,
  coaches heavily by prompt and uses human-play recordings.
- Public results used for rejections: stall cutoff -9.1 and more streams 29.90 against 47.30 (Francise Liang, forum
  750885); EAnge1o's quadratic-scoring prompt, one offline run, LB 30.26 (L312, L314, L217, L218).

## 6. Mismatches found while drafting

1. Level delta: PROGRESS L7211 says +3.3 levels on the control; V3-W29 says +3.2. Unrounded, 126.67 - 123.43 = 3.24,
   so +3.2 is right; +3.3 subtracts two rounded means [reproduced].
2. Rank 3: public-arc3 calls Turducken rank 3 (L307); competitive-meta says xz (L398). The 11-02 12:15Z board has xz
   3rd (58.59) and Turducken 5th (56.02) [source-verified, leaderboard file above]; L307 is stale.
3. Publication scope: public-arc3 says no top-30 team has published (L305); competitive-meta says top 12 (L394);
   PROGRESS L7166 repeats top 12.
4. GG6 precedent: the verify skeptic says GG6's persistent prelude was adopted in 25 of 25 games (L222). PROGRESS
   L4622 to L4623: `memory['prelude']` in 18 of 25, `memory` used at all in 25 of 25 [source-verified]. The 1.5%
   call rate itself matches.
5. Cut replies: code-read says the model "usually picks the same deliberation back up" after a cut (L347); the
   codefirst skeptic measured the first post-cut call holding `def ` or `action(` in 66 of 79 (L229).
6. Gate against spec: the verify spec read harm items on the aligned 5,800 s window and asked for 5 games (L80), and
   the synthesizer said to adopt the aligned window as the predeclared reader (L205). The predeclared gate used the
   raw window, asked for 4 games per run on average, and made the aligned window, m2 and m3 reported only; none of
   the three was computed [source-verified, predeclared JSON, V3-W29].
7. The 728 s early-clock figure in V3-W29 and PROGRESS L7164 is attributed to this research pass but does not appear
   in the digest, which has only "12 to 22 minutes before the server was ready" (L205, L385) [reproduced, grep].
8. The competitive-meta proposal list tags `check_model` "dead-end" (L409) while the same agent ranks it first
   (L404).
9. Cross-angle counts on the same 3 seg8/us8 runs differ slightly: 53 replay blocks in 8,437 unique code blocks
   (L61) against 57 hand-replay blocks in 8,486 code blocks (L223); median cut length 35,244 chars over 93 cuts (L101)
   against 35,339 over 86 no-call cuts (L246); idle turns 26% of the clock (code-read, L344), 30% (transcripts, 11
   runs, L380), 0.264 (costmodel skeptic, L292). Different definitions or run sets, not contradictions.
10. UNDO: code-read says UNDO was used 8 times in 3 runs while UNDO-valid games log about 10 RESETs per run (L358);
    the coverage skeptic over 6 runs measured 0 to 6 presses per run (mean 1.8) and 5 to 11 model RESETs (mean 7.8)
    (L275).
