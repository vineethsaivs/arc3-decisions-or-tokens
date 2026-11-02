# LoRA track: why the r16 RFT lost 12 levels, and what post-training can still do (2026-10-19)

Tags: [verified] = read today in code, data, a log or an API record; [claim] = someone said it; [inference] = my reading.
Sources: `solution/lora/` (code and reports), `docs/PROGRESS.md` (10-19 entry), the VM (`~/runs/loraR1`, `~/runs/loraR2`,
`~/datagen/attempts/0{2,3,4}`, `~/lora_data/`, `~/lora_train_report.md`), 9 local Kaggle y8-family replicas under
`solution/kaggle/generated/kernel-outputs/franzen-l1a1y8*-v1`, `reference/re-arc-3/`. Scripts: section 7.

## Answer

1. The adapter learned the pin's RE-ARC-3 play style, not skill. On the public 25 it spends 20% less reasoning per action,
   takes 20% more actions, dies 2 to 3 times as often, resets twice as often, and loses its levels on the mid-difficulty
   games (tn36, sp80, wa30, vc33, cn04, g50t) [verified counters, section 2].
2. The cause is the data, not the merge and probably not the strength. The pin clears 91% of RE-ARC-3 levels, so "turns of
   cleared levels" selects almost nothing; 51% of the supervised turns come from levels that took more than 2x the baseline
   actions; 99% of the windows are a 12K-token suffix of contexts that were typically 21K to 53K tokens; targets were sampled
   at T 0.7 / top-k 20 and fit with a T 1 loss [verified facts; causal link inference].
3. No post-training recipe has positive expected value for the final board on one G4 by 11-02. The best remaining one
   (group-relative RFT, R1 below) has about a 10% chance of +6 replica levels (about +2 board points) for about $55 and 3
   days [inference]. Rank 3 (52.82) needs about +13 points over our best draw and about +20 over our recent mean; nothing in
   this track closes that.
   Stop after the half-strength read unless it reaches 130 levels.

Status at 22:12Z: the half-strength replica (`~/runs/loraH1`) was preempted at 21:48:15Z (GCP operation
`compute.instances.preempted`) [verified]; the Mac waiter restarted the VM at 21:56Z and the run restarted 22:00:51Z
[verified]; earliest result about 00:20Z on 10-20 if Spot holds [inference].

## 1. What was trained [verified unless tagged]

- **Plays:** arm `l1a1y8sv2r3hm32rl` = the pin's scored path + reqlog + RE-ARC-3 loader, 127 clean RE-ARC-3 games (no public
  family), 14 streams, 4,032 slot-seconds per game, sampling T 0.7, top-p 0.95, top-k 20 (candidate notebook). Three attempts:
  02 (all 127, Spot-preempted), 03 (60 unfinished games, preempted), 04 (48, complete). The pin cleared 548 of 601 levels
  (91%) and won 106 of 127 games: train 454/498 levels (90 of 107 won), validation 67/74 (12 of 15), test 27/29 (4 of 5).
- **Windows** (`build_rft_windows.py --reqlog`, `windows_15000.jsonl`): 13,652 windows (train 10,810, val 2,206, test 636).
  One window = the last request of a chain that fits 15,000 estimated tokens plus the reply; longer contexts are cut to the
  system prompt plus the latest suffix that starts at a user message. **99% of windows are cut**; mean 12,384 est tokens.
  Logged prompt sizes at sampling time (calibration lines in the transcripts): median 21,137 (attempt 02) and 53,346
  (attempt 04), p90 71K to 79K; public r3 median 53,418. So the model was trained on a quarter to a half of the context it saw.
- **Labels:** a reply is supervised (weight 1) when its level, read from the last "Current state: step S, level L" before it
  (1-based, checked on transcripts), is at most that run's `levels_completed`. Every turn of a cleared level counts:
  exploration, deaths, RESETs, failed attempts and the clearing turn. Weight 2.0 when the (game, level) was cleared in some
  runs and not others: 1,020 of 7,426 supervised turns (13.7%, 31 games). The "not cleared" runs are attempts cut by Spot
  preemption, so this doubling is a preemption artifact, not a bistability signal [inference].
- **Selection pressure:** 6,806 of 10,810 train windows (63%) carry a target. By level efficiency (actions / baseline) the
  supervised turns are: at most 1x 17%, 1 to 1.5x 20%, 1.5 to 2x 12%, 2 to 4x 30.5%, over 4x 20.4%. 69% come from won games.
- **Loss:** weighted token NLL at T 1 (raw logits) over every generated token of a supervised reply: reasoning, the python
  tool call and the `action()` calls in it, up to `<|im_end|>`. No separate treatment of thinking, code or actions.
- **Run:** 1,400 windows drawn at random (seed 0; 100 games, the top 10 games hold 34%), 1,507 supervised replies, 16.0M
  tokens. LoRA r16, alpha 32 (scale 2.0) on GDN `in_proj_qkv`, `in_proj_z`, `out_proj`, attention q/k/v/o and the shared
  expert (300 tensors merged at load, `lora: merged 300 tensors, scale 2.0`); routed int4 experts frozen. AdamW lr 1e-4,
  cosine to 1e-5, 3 warmup steps, accum 4, clip 1.0, 350 steps, one pass, 4.55 h at 977 tok/s, peak 78.4 GiB, about $8.
- **Val NLL:** unweighted T 1 NLL over the supervised tokens of 32 fixed windows from the 15 validation families. It measures
  how well the adapter reproduces the pin's own sampled RE-ARC-3 replies; it does not measure decisions. Curve: 0.7853,
  0.6966 at step 40 (81% of the total drop), 0.6759 at 350. Train NLL on fresh windows per 20 steps: 0.684, then 0.62 to
  0.70 throughout (no overfitting signal).
- **Parity:** merged-served vs trained adapter, 30 requests: mean |d| 0.029, argmax 97.8%, signed -0.005 (PASS).

## 2. What changed in play

Pin = 8 Kaggle y8-family replicas with transcripts (y8, y8r, y8sv2, r1, r2, r3, s4, o16); levels also include hm32 (9 runs).
LoRA = the two G4 replicas. RE-ARC-3 = the pin's datagen attempts 02 and 04 (the training distribution). [verified]

| counter (whole run unless noted) | pin, range | LoRA R1 / R2 | RE-ARC-3 data |
|---|---|---|---|
| levels | 119 to 129 (mean 124.0, sd 3.9) | 109 / 115 | 91% of levels |
| mean game score | 52.7 to 61.2 | 46.7 / 48.8 | 37.9 / 37.0 |
| games won | 8 to 11 | 8 / 8 | 106 of 127 |
| replies | 2,441 to 2,885 | 3,191 / 3,328 | n/a |
| generated tokens | 5.36M to 6.32M | 6.48M / 6.53M | n/a |
| tokens per reply | 2,097 to 2,277 | 2,031 / 1,962 | 1,783 / 2,152 |
| actions | 6,108 to 7,077 | 7,859 / 7,927 | n/a |
| actions spent on never-cleared levels | 777 to 1,592 | 2,965 / 1,980 | n/a |
| reasoning chars per action, cleared levels | 2,021 to 2,346 | 1,729 / 1,772 | 966 / 1,347 |
| actions per reply, cleared levels | 2.35 to 2.67 | 2.69 / 2.75 | 4.52 / 4.28 |
| median reasoning chars per reply | 2,916 to 3,343 | 2,837 / 2,779 | 2,555 / 3,146 |
| GAME OVER echoes | 8 to 13 | 26 / 27 | 0.3 per level |
| manual RESETs per 100 turns | 3.1 to 4.1 | 6.1 / 6.5 | 4.8 / 9.7 |
| automatic RESETs (after a death) | 10 to 15 | 28 / 28 | 97 / 100 |
| snippets with a stale-state refusal | 51 to 65 | 154 / 141 | 107 / 132 (per 3.1K to 3.5K turns) |
| NO-OP share of actions | 0.042 to 0.058 | 0.073 / 0.081 | 0.032 to 0.037 |
| cleared-level actions / baseline, median | 0.60 to 0.68 | 0.71 / 0.71 | 1.14 to 1.54 |
| tracebacks per 100 replies | 6.95 to 8.72 | 7.71 / 7.00 | 4.95 to 6.10 |
| length-capped replies | 26 to 57 | 50 / 23 | 57 to 69 |
| accept length at 8+ running | 2.838 (hm32), 2.830 (r3) | 3.037 / 3.032 | 2.794 |
| gen tok/s at 14 running | 987 (hm32) | 1,035 / 1,027 | 970 |
| mean wall per game | 6,410 to 6,666 s | 6,597 / 6,604 s | n/a |

Per-game levels where the LoRA moved (pin 9-run mean and range; LoRA R1, R2). Unchanged: ar25, cd82, lp85, sb26, r11l, tr87
(saturated), ka59, lf52, m0r0, re86, sc25, su15, dc22, tu93, bp35 within the pin's range.

| game | pin mean (min to max) | R1 | R2 |
|---|---|---|---|
| tn36 | 3.6 (1 to 7) | 1 | 1 |
| sp80 | 2.9 (1 to 4) | 1 | 1 |
| wa30 | 3.2 (1 to 4) | 2 | 2 |
| vc33 | 6.9 (6 to 7) | 3 | 7 |
| cn04 | 4.2 (2 to 5) | 4 | 1 |
| g50t | 4.4 (2 to 6) | 1 | 5 |
| ft09 | 5.9 (5 to 6) | 4 | 6 |
| ls20 | 4.6 (4 to 5) | 5 | 3 |

Five LoRA game-runs fall below the pin's 9-run minimum (vc33 3, ft09 4, g50t 1, cn04 1, ls20 3); tn36 and sp80 are stuck on
level 2 in both. The extra actions go to levels never cleared (wa30 R1: 705 actions on one open level; vc33 R1: 259).

Reading [inference]: every style counter moved from the pin's public-game value toward the RE-ARC-3 value (reasoning per
action, actions per reply, deaths, resets). Stale refusals are actions batched past a death or a level end, so they follow the
death rate. Coding ability is intact (tracebacks flat). Outputs got more predictable to the drafter (accept +7%), which also
fits shorter, more formulaic replies.

## 3. Diagnosis

| hypothesis | verdict | evidence |
|---|---|---|
| Data mismatch: imitating easy-game play transfers its style | **supported, main cause** | style counters move toward the RE-ARC-3 values (table above); the pin clears 91% of RE-ARC-3 levels vs 68% of public levels; losses sit on mid-difficulty public games [verified counters; cause inference] |
| Weak training signal (own trajectories, almost no filter) | **supported** | 63% of windows and 91% of levels pass the filter; 51% of targets from levels at >2x baseline actions, 20% at >4x; one play per game, so no within-state contrast [verified] |
| Truncated contexts teach acting without the evidence | **plausible, untested** | 99% of windows cut to about 12K est tokens of 21K to 53K; the reply is fit without the history it was conditioned on [verified data; effect inference] |
| Sampler sharpening (T 0.7 / top-k 20 targets, T 1 loss) | **plausible, weak evidence** | settings [verified]; accept length 2.84 to 3.03 at the same drafter [verified]; but greedy-token confidence on 30 T0 prompts is flat (mean logprob -0.150 base vs -0.142 adapter; common prefix -0.107 vs -0.113) [verified, small sample] |
| Over-strength (scale 2.0) | **open** | alpha/r = 2 is the standard scale, not an extreme one; 81% of the val NLL drop came by step 40, so a weaker adapter likely carries the same shift [inference]; the half-strength replica decides |
| Longer outputs | **refuted** | tokens per reply 2,031 / 1,962 vs 2,097 to 2,277; the lenprobe mean 652 to 844 is one 5,663-token reply in 20 prompts (median 194 to 148, none capped) [verified] |
| Overfitting after step 40 | **refuted as the cause** | val NLL fell to the last step; fresh-window train NLL flat at 0.62 to 0.70; the shift itself is the problem [verified curve; inference] |
| Merge or quantization mismatch | **refuted** | adapter parity 0.029 mean abs, 97.8% argmax, signed -0.005 [verified] |
| Forgetting | **refuted for skills** | tracebacks per 100 replies 7.0 / 7.7 vs 6.95 to 8.72 [verified] |

Platform caveat: the LoRA replicas ran on G4, the controls on Kaggle. CLAUDE.md records G4 serving within 1% of Kaggle
[claim]; the LoRA served 4% faster at 14 running, so the token budget was not smaller [verified].

## 4. Bench validity for model changes

- **Can a model change help hidden-like games while hurting the public 25?** Unlikely here. The public families were
  excluded from training, so the public 25 are novel games for the adapter, made by the same designers as the hidden 110;
  RE-ARC-3 is a Codex-generated imitation. The hidden set is harder than the public 25 (board about 0.55 to 0.6 of replica
  score) and the adapter loses most on the harder public games, so the hidden set would likely show a larger loss [inference].
- **Data that exist to test it:** the 20 held-out-family RE-ARC-3 games (15 validation, 5 test, never trained on) have one pin
  play each, but the pin clears 94 of their 103 levels: a ceiling, at most +9 levels of headroom [verified]. Their competition
  score has room (cleared-level actions are 1.14 to 1.54x the baseline), but efficiency is not where the hidden score is.
- **A usable held-out RE-ARC-3 bench** needs hard games: hold out about 25 games (the 21 the pin never won, plus a few with
  most levels above 2x baseline) and never train on them. Cost on one G4: 25 games x 288 s = 7,200 s window plus about 10 min
  boot = about 2h10m, about $3.6 per run; an A/B with 2 runs per arm is about 9 h and $15. Run it with the existing tooling:
  write the game ids to a JSON list, export `ARC3_REARC_GAMES` (and `ARC3_REQLOG_DIR` if logs are wanted), and run the
  `candidates-franzen-l1a1y8sv2r3hm32rl` notebook through papermill as `solution/lora/datagen/run_datagen.sh` does. For the
  adapter, add a dev arm to `build_franzen_arms.py` = `REPLICA + rearc() + reqlog() + lora(<adapter dir>)` (the server is
  launched inside cell 15, so the `lora()` edit is the right path, not `start_server_lora.py`). For several plays per game set
  `bm.n_passes` and multiply the window (`max_runtime_s_per_game = 288 * _REARC_N`) by it [inference from the arm code].
- **Cheaper and more decisive: an offline style-drift gate** before any replica. Take about 120 states from the public-25
  request logs the G4 replicas already hold (`~/runs/loraR*/working/*_p0_requests.jsonl`), sample 2 replies per state from
  the base server and 2 from the adapter server at the live sampling (T 0.7, top-k 20), and compare reasoning chars per
  action, `action()` calls per reply, RESET calls and reply tokens. About 1M tokens plus two boots: about 40 min, $1.1.
  Kill if reasoning per action falls or actions per reply rise by more than 10%. On this run it would likely have flagged the
  shift before $8 and 4.5 h of replicas [inference]. Pattern: `solution/lora/train/lenprobe.py`, `trimtest_replay.py`.

## 5. Recipes (ranked), honest about value

Expected value frame: +6 replica levels is about +5% of levels, about +2 to +2.5 board points (one more level on every
unfinished game is about +20% of score), and it must also win the public-score auto-selection of the two finals against
the 40.09 draw. Under 4 levels nothing is detectable by 11-02.

**R0. Stop the r16 line after the half-strength read (recommended).** Decision rule as written: 130+ earns a second replica,
otherwise stop. Also read the four style counters on it (`~/ana_tmp/playstats.py`, `split.py`, `agentfix_counters.py`,
`acc.py`): halfway values confirm the dose-response of the style shift (Paper Track evidence). Cost 0. Frees the G4.

**R1. Group-relative RFT on hard RE-ARC-3 games (the only RFT variant with real selection pressure).**
- Data: about 50 RE-ARC-3 train-family games where the pin left levels uncleared or cleared them above 2x baseline; play each
  4 times (`bm.n_passes = 4`), so selection compares plays of the same game and the same policy: no style import by
  construction. Hold out 10 of them as the bench above.
- Selection: per (game, level), positives = plays that cleared it at or under the group median actions and under 2x baseline;
  supervise only those plays' turns of that level; drop turns after a death within the level. Weight by the group advantage.
- Context: rebuild windows at 24,000 est tokens (the trainer peaks 88.8 GiB at 29K; 39K OOMs) and drop windows whose cut
  removed the current level's start.
- Loss: NLL on logits / 0.7 (match the sampler, so the T 1 model is not sharpened); lr 5e-5, about 300 steps, merge scale 1.0;
  checkpoint every 40 steps, pick by the style-drift gate, not by val NLL.
- Gates and kill points: style-drift gate (kill as above); held-out hard bench, 2 runs per arm (kill if levels or score fall);
  public replica (kill under 120, promote at a 2-run mean of 130+ with style counters inside the pin's range).
- Cost: about 16 h of plays ($27), 7 to 9 h training ($15), 1 h style gate, 4 h held-out bench, 2 public replicas ($8):
  about $55 and 34 G4-h, about 12 engineering hours, 3 days wall if Spot holds. Days to result: 3 (lands about 10-23, the freeze).
- Expected: -3 to +6 levels; P(>= +6) about 10% [inference]. Risk: the selected plays are still RE-ARC-3 plays; preemptions
  push it past the freeze; trajectory-level credit is coarse.

**R2. Branching GRPO from saved RE-ARC-3 states.** Feasible in principle: RE-ARC-3 envs deep-copy (its DSL agent
`ft09_close.py` branches with `copy.deepcopy(env)`; `reward_env.py` guards deepcopy) and only 7 of 193 env files mention
`random`, 6 of them with an explicit seed [verified], so a logged action prefix replays to the same state [inference]. The harness sandbox (the model's REPL namespace) is not copyable, so a branch must replay the logged
tool-call prefix into a fresh sandbox: 2 to 3 days of engineering. Budget on one G4: about 89M generated tokens per day, so
G=8 branches x 5 replies x 2K tokens is about 1,100 states per day. Reward: level cleared within the branch, minus deaths,
with an action-count term; KL to base. Cost about 6 G4-days ($240) and 40 to 60 engineering hours; result about 10-27 at best.
Expected -5 to +10 levels, P(>= +6) 10 to 15% [inference]. Not in time for the board; a Paper Track experiment.

**R3. Public-25 data (self-imitation across our 9+ replicas, or the 22 official human replays).** The human replays cover
public games only (22 files, no ft09, lp85, ls20) and are action-only [verified]; RE-ARC-3 games have solver action files
(192), also action-only [verified]. Training on public games removes our only hidden-like bench, the Kaggle replicas'
request logs (exact contexts) are not kept locally (transcripts only), and rationalizing actions into reasoning is a
project of its own. Not
recommended.

**Not rules-OK:** distilling Claude or other API-model traces into the open weights (provider terms restrict training
competing models on outputs [claim]; the project memory already bars Azure GPT output as a corpus [verified]). The eval path stays offline in every recipe above.

**With 4 or 8 GPUs** (GCP cap is 1, the 8-GPU request was denied; Kaggle adds 2 concurrent sessions, 30 GPU-h/week): R1's plays
take 4 h or 2 h instead of 16 h and its gates run in one 2h10m wave, so two R1 iterations fit before 10-26; R2 gets 4K to 8K
states per day, which is the first scale where trajectory RL has a fair chance. Engineering time stays the bottleneck.
Practical now: train on the G4 while Kaggle runs the replicas (the adapter must be our own private dataset, as
`vineethsaivs/arc3-lora-r16` already is).

## 6. What to do now

1. 10-20 about 00:20Z: read the half-strength replica (levels plus the counters). Under 130: stop the r16 line (R0).
2. Do not start R1 unless the G4 has no higher-value job from the other tracks; if run, start 10-20 morning, keep every kill
   point, and treat a miss at the freeze as final.
3. Record the negative result for the Paper Track: imitation of the agent's own easy-game play moved style toward the training
   regime and cost 12 levels; val NLL fell 14% while play got worse; MTP accept length rose 7% as an output-entropy signal.
4. Fix before any reuse of `build_rft_windows.py`: preempted partial plays likely count as "not cleared" (the 2.0 weight on
   13.7% of targets), and the 15K cut leaves 99% of contexts truncated.

## 7. Scripts and outputs (read-only analysis)

- Mac scratch (`<scratch>/fb519d0b-5ca4-49bf-b4d4-4ffb21661429/scratchpad/research/lora/scripts/`):
  `playstats.py` (run counters), `split.py` (cleared vs open levels), `acc.py` (accept and tok/s from serve.log),
  `sharp.py` (greedy confidence, base vs adapter), `wins.py` (window composition and the exact 1,400 sample),
  `eff.py` (supervised turns by level efficiency), `heldout.py` (pin levels by family split), `pergame.py`.
- Copies on the VM in `~/ana_tmp/` (plus `agentfix_counters.py`), ready for the half-strength run.
