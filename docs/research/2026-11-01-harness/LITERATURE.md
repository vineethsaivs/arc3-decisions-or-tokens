# Related-work verification for the 2026-11-01 research digest

Scope: the outside sources behind the digest's [public-arc3], [agent-literature] and [competitive-meta]
sections (`research_digest.txt` lines 298 to 411, plus the decision notes at lines 51 to 58 and 134 to 138
that cite the same papers). Checked 2026-11-02.

Tags: [source-verified] = read in a saved copy (location given). [reproduced] = arithmetic I redid from
source-verified numbers. [from digest, not reproduced] = only the digest states it; no saved copy holds it.
[repo record] = a prior session's evidence file in the repo holds it; not re-read from the original here.
Web use was limited to bibliographic metadata (the arXiv API record for 8 ids, the Retrodict repo page
title, one search for the EMPA journal version). Abstract text that came back with that metadata is
labelled [web abstract].

Saved copies used (all under the scratchpad): `pubharness/abs_*.html`, `pubharness/h_2607.15439.txt`,
`h_2607.28287.txt`, `h_2608.14490.txt`, `pdfs/twin.txt`, `pdfs/sweagent.pdf`, `pubharness/justexplore.txt`,
`pubharness/arcskill_SKILL.md`, `pubharness/readme/{Cveinnt_kepler,pbshgthm_arc-skill,raimondasl_persistgym}.md`,
`forum/t*.json` and `pubharness/topic_*.json` (dumped to text in `litv/`), `arxiv.xml`, `board.json`.
Character offsets (`@N`) refer to the `h_*.txt` files.

---

## 1. Fact sheets

### Twin (Skoutnev, Acharya, Longhitano, Udell, Ellis, Drori), arXiv 2608.14490, 2026. Key `skoutnev2026twin` (exists)

- Headline: 93.3 of 100 on the 25 public games, 23 of 25 games, 179 of 183 levels, base model GPT-5.6 Sol
  driven by OpenAI Codex [source-verified: abstract; Table 1 @28839].
- The 61.1 comparison [source-verified: "The harness earns the score: an ablation", @33508 to @34400]:
  "The ablation keeps the identical agent, base model, file bridge, and sandbox, with the entire Twin harness
  disabled. The deletion removes the executable world model, replay validation (Validate), diagnostic
  exploration (Explore), model-based planning (Plan), and the halt-on-mismatch executor." Score 93.3 to 61.1,
  full clears 23 to 13; Twin matches or beats the ablation on 24 of 25 games (exception sc25, 32.7 vs 44.8).
  So it is the same model and the same agent, and what was removed is the whole harness (five parts at once),
  not replay validation alone. Difference 32.2 points [reproduced]; the paper says "32 percentage points"
  (Appendix C, @64538).
- Ablation compute [source-verified: Table 4, @62260]: Twin 2.60B tokens, 11,614 scored actions, 224k
  tokens per action; no-twin ablation 1.06B tokens, 22,224 actions, 48k tokens per action, 148/183 levels.
- Direct play of the same base model 7.8; Prime Agent harness on the same model 78.3; Table 6 attributes
  "the final 15.0 points" (78.3 to 93.3) to "what a validated world model adds" [source-verified: Appendix F,
  @73958 to @75100].
- Enforcement [source-verified]: "The harness enforces that an action is not made until the program
  reproduces every previous observed game transition" (abstract). "A twin that fails to replay the past is
  blocked from acting" (Method, Validate, @19681).
- On harness enforcement vs prompting [source-verified: "Harness-enforced validation matters most on long
  games", @31454]: the comparison systems "rely on prompt instructions to request validation"; the hard
  constraint "may explain why Twin alone clears bp35, lf52, and sk48." This compares Twin to other systems
  (EWM, OPINE-World); no experiment isolates enforced vs prompted validation.
- Goal check, "certified negatives" [source-verified]: a tentative goal predicate "must evaluate false on
  every logged frame; because level completion replaces the winning frame, all logged frames are known
  non-goals" and a candidate reached without a level boundary is permanently excluded (Method, @24354 to
  @24700; Algorithm 1). Appendix D (@66851 to @67600): "The green check requires [the predicate] to reject
  every recorded frame, each one a certified negative ... The released harness enforces the green check as a
  hard constraint ...; during the reported runs the UpdateGoal prompt asked the agent to apply the same
  check." The Method paragraph after Algorithm 1 (@27633) says the harness enforces it. The paper is
  internally inconsistent; per Appendix D the 93.3 runs had the goal check by prompt, not by harness.
- Execution [source-verified]: Plan does BFS in the validated twin; ExecuteChecked submits one move at a
  time and a halted move returns a counterexample (@4841, @37111). 92.9% of scored actions execute plans
  tested in simulation, 7.1% are probes ("Every scored action starts from a replay-validated twin", @34300).
- Goal-discovery signals [source-verified: Appendix D, @69128]: color_gone, color_new, local_burst,
  big_change, frontier.
- Runs: "one scored run per game" [source-verified: Appendix I, @90193].
- Related work Twin states [source-verified, @12300 to @13700]: EWM (Rodionov), OPINE-World and Schema
  "converge on one control loop. Each logs every interaction, holds the game as an executable program,
  accepts the program only when it replays the log, plans inside it, and halts at the first misprediction."
  Graph exploration (Rudakov et al.) "reaches deep states through unbounded search, but at near-zero action
  efficiency" (@14321). WorldCoder "introduced the core pattern" of synthesizing a transition and reward
  model, planning through it and repairing it from counterexamples; PoE-World "scales it by composing
  multiple programmatic experts" (@14712 to @15100).
- Cross-source mismatch: Twin's Table 1 gives EWM 63.8 (GPT-5.5, 14 games, 146 levels); EWM's own arXiv
  abstract gives 58.12% with GPT-5.5 high and 15 full games [web abstract]; Tycho's table gives 63.74 (9/25)
  [source-verified, h_2607.28287 @65038]. Do not quote an EWM number without naming its source.

### Tycho (Lehmann, Aioanei, Vahdati), arXiv 2607.28287, 2026. Key `lehmann2026tycho` (exists)

- Four orchestration policies, Claude Opus 4.8, one matched public-set run per policy. Orchestrator
  (actor-requested delegation) 88.49 RHAE, the best; Trigger (automatic repair after verification failure)
  83.07; with Orchestrator, GPT-5.6 Sol and Opus 5 both reach 100.00 and all 183 levels; Opus 5 uses 61%
  fewer scored actions than the aggregate human baselines [source-verified: abs page abstract].
- "Transition match indicates whether a simulator reproduces observed dynamics, not whether it has
  identified the objective" [source-verified: abstract].
- Typed evidence contract [source-verified: contributions @10690; Sec 4.1 @38500]: frames tagged decision,
  transient, completion terminal, fatal terminal, reset initialization, next-level initialization; saved
  under `level_L/turn_NNN.{txt,png,json}`; the `wmlib` library gives parsed access to prior decision
  frames, transitions, archived reset attempts, terminal, fatal and animation events.
- Verification [source-verified: Sec 4.3, "Operational verification", @44685]: "Verification replays the
  most recent attempt for each level from its first frame"; outcomes are checked separately and "Ordinary
  decision states should remain ongoing." Every model save auto-verifies (prompt excerpt, @128660).
- Planning and execution [source-verified: Sec 4.3, @43400 to @44700]: plan.py writes a validated route
  with an expected-frame hash per step; "The actor receives the validated route as conditional advice and
  commits one action at a time"; the next action is surfaced only while the observed frame matches the
  stored hash. Tycho has no halting executor: "A guarded executor could instead carry out several verified
  actions and interrupt on a prediction mismatch, as discussed in Section 8" (@79795).

### Kepler (Wensen Wu), arXiv 2610.00834, 2026. New key `wu2026kepler`

- Title "Kepler: Auditable World Models for ARC-AGI-3"; accepted to the non-archival Interpreting Agent
  Behavior workshop at NeurIPS 2026 [source-verified: arxiv.xml; README].
- Claude Opus 5, one frozen configuration: 100.00, server-verified; GPT-5.6 Sol max 95.97; 181 of 183
  completed Opus levels used no more actions than the median-human baseline; $777.72 list-equivalent
  [source-verified: README "What is different" and table].
- Mechanism [source-verified: README "How it works" and fact 4]: "The daemon records transitions; the
  agent writes a model and is instructed to backtest the interaction history before planning, not
  mechanically forced to rerun that backtest before every commit." "A usable prediction's first mismatch
  voids the remaining plan and returns the counterexample. If prediction fails or covers no cells, the
  action still executes with a warning." Certified per-level action programs are played by a mechanical
  executor. A bundled planner "crashed on every invocation across five experimental boards".
- Lineage it credits: Tycho, baseline1 (Rodionov) and Retrodict [source-verified: README "Scope and credit"].

### arc-skill (Poobesh Gowtham), github.com/pbshgthm/arc-skill, August 2026. New key `gowtham2026arcskill`

- Claude Code on Claude Opus 5: 25 of 25 games, 183 of 183 levels, RHAE 100.00, 7,645 actions against a
  median human 17,135; ARC scorecard 24ddb219 [source-verified: README "The result"].
- Hard rule [source-verified: README "The rule"]: every press needs a written prediction; the harness
  refuses a press without one and grades it against the returned frame. 7,627 graded predictions, 443
  missed. Plans halt at the first miss ("commit"; SKILL.md "Batching proven mechanics").
- Log [source-verified: SKILL.md "Evidence tools"]: append-only `events.jsonl`; `arc python` preloads
  every settled board, frames, transitions, actions; any past press's animation can be viewed.
- Full-history check is optional [source-verified: SKILL.md "rules tier"; README ladder step 4]: write
  `rules.py`, "verify it against the entire recorded history", A* search. README: "Taken once, and it never
  fitted." Single test presses missed 37.1%, planned sequences 2.9%, 91.6% of presses went into plans.

### Retrodict (GitHub user ryanbbrown), github.com/ryanbbrown/Retrodict, 2026. New key `ryanbbrown2026retrodict`

- No saved copy. Repo description: "Frontier ARC-AGI-3 agent: 99.86% RHAE, all 183 levels solved for
  $654" [web metadata only, not a verified result]. Kepler's README credits it as part of the
  executable-world-model lineage and cites "Retrodict's $2,986 API-equivalent estimate for Tycho"
  [source-verified]. Its mechanisms are unverified.

### Rodionov ablation ("the 4x4 ablation"), arXiv 2607.15439, 2026. New key `rodionov2026ablation`

- Title "Do Coding Agents Need Executable World Models, Simplification, and Verification to Solve
  ARC-AGI-3?", Sergey Rodionov (SingularityNET) [source-verified: abs page; HTML v1 header].
- Design: 4 nested variants (textual, flexible-interface executable, + simplification, fixed-interface +
  simplification + verification) x 4 model-effort settings (gpt-5.4, gpt-5.5 at high, xhigh): "Each of the
  16 variant-model-effort combinations is evaluated once on each of the same 25 public games"
  [source-verified: @15205; Table 1 @22046]. This is the best match for the digest's "4x4 ablation"; the
  digest also lists "Rodionov's ablation" separately, so it may count one paper twice.
- "The verification variant ranks first in every setting" [source-verified: Sec 6.3, @41529; abstract v1
  "ranks first in all four settings"; abs page v2 "ranks first throughout, sometimes narrowly"].
- Table 2 RHAE (textual / executable / simplification / verification) [source-verified, @35700]:
  gpt-5.4 high 34.16 / 33.54 / 30.60 / 39.16; gpt-5.4 xhigh 40.67 / 44.72 / 53.10 / 53.72;
  gpt-5.5 high 58.85 / 51.16 / 58.35 / 65.64; gpt-5.5 xhigh 72.51 / 69.70 / 73.09 / 74.78.
  Margin of verification over simplification 0.62 to 8.56 points [source-verified, @36869].
- Caveats the author states [source-verified]: "The verification result applies to the complete treatment
  rather than to a single instruction ... The experiments therefore do not establish how much of its
  advantage comes from verification alone" (conclusion, @50300). "Verification is prompted and
  tool-supported, but it is not silently enforced by the external controller ... real actions are not
  required to pass through an online plan executor in the main ablation" (Sec 3.4, @25978). Verification
  used the most cost tokens in every setting (Sec 5.2).
- What the verifier checks [source-verified: Sec 3.4, @24583]: replays every recorded attempt from level 1
  to the current level, cell-exact settled frames, and "compares the predicted environment status with the
  recorded status" at every step; a second verifier checks the planner ends in LEVEL_COMPLETED on solved levels.

### EWM (Rodionov), arXiv 2605.05138, 2026, AGI-2026. New key `rodionov2026ewm`

- The predecessor system ("baseline1"). [web abstract]: maintains an executable Python world model,
  verifies it against previous observations, plans through it, uses "a scripted controller, predefined
  world-model interfaces, verifier programs, and a plan executor"; GPT-5.5 high 58.12%, 15 games.
  Twin's bibliography gives LNCS volume 16855, pages 198 to 210 [source-verified, h_2608.14490 @43694].

### WorldCoder (Tang, Key, Ellis), arXiv 2402.12275, NeurIPS 2024. New key `tang2024worldcoder`

- No saved copy of the paper. What the saved sources say is secondhand: Twin (above) and Tycho's
  bibliography (NeurIPS volume 37, pages 70148 to 70212) [source-verified, h_2607.28287 @119423]. Twin
  cites "WorldCoder's observed-data consistency requirement" as the origin of its non-goal filter (@24665).
  [web abstract]: "The world model tries to explain its interactions, while also being optimistic about what
  reward it can achieve"; gridworlds and task planning. Not an ARC-AGI-3 harness.

### PoE-World (Piriyakulkij, Liang, Tang, Weller, Kryven, Ellis), arXiv 2505.10819, 2025. New key `piriyakulkij2025poeworld`

- No saved copy. Twin: "scales it by composing multiple programmatic experts" and bibliography lists NeurIPS
  38 [source-verified, @15039, @43180]. [web abstract]: product of programmatic experts, evaluated on Atari
  Pong and Montezuma's Revenge. Nothing saved says it replays every logged transition before acting.

### EMPA (Tsividis, Loula, Burga, Foss, Campero, Pouncy, Gershman, Tenenbaum), arXiv 2107.12544, 2021. New key `tsividis2021empa`

- No saved copy. Tycho mentions it once: "EMPA performs program-based model inference, exploration, and
  planning" and cites a 2026 Phil. Trans. R. Soc. A version with 11 authors and a different title
  [source-verified, h_2607.28287 @21173, @119449]. One web search did not confirm that journal version.
- The digest's specific EMPA claims (an interaction counts as known only once seen in the relevant state;
  exploration goals reopen when the agent's state changes) are NOT verified: no saved text covers them.

### AutumnBench / WorldTest (Warrier, Nguyen, Naim, Jain, Liang, Schroeder, Yang, Tenenbaum, Vollmer, Ellis, Tavares), arXiv 2510.19788, 2025. New key `warrier2025autumnbench`

- Current title "Benchmarking World-Model Learning with Environment-Level Queries", ICML 2026 per the arXiv
  journal reference [web metadata]. AutumnBench: 43 grid-world environments, 129 tasks, 517 human
  participants, five frontier models [web abstract].
- The digest's "humans spend 12.5% of actions each on resets and no-ops, models under 7% on both" is NOT
  verified: no saved copy, and the abstract does not state it. Note the abstract says "five frontier models",
  not "reasoning models".

### persistgym (Raimondas Lencevicius), arXiv 2610.26632, 2026. New key `lencevicius2026persistgym`

- Paper title "When the harness decides success: state that cannot be re-observed"; repo raimondasl/persistgym
  [source-verified: abs page; README].
- Synthetic probe (pre-registered, one low-cost reasoning model, one replication): truncation costs nothing
  in success when the fact can be looked up again, and drops to chance when it cannot; penalty gap 92.5
  points, 90% CI [89.5, 95.5] [source-verified: abstract].
- ARC-AGI-3 part (one model, four games, one counted run per configuration): summarizing old turns instead
  of dropping them, with the note-taking invitation kept, recovers 86% of a 63-point gap between the
  Standard and Provider Adapter harnesses; without the invitation 43%; native reasoning replay recovers the
  remaining 57% [source-verified: abstract].

### Just Explore = Graph-Based Exploration (Rudakov, Shock, Cowley), arXiv 2512.24156, 2025. New key `rudakov2025graph`

- "Just Explore" is the code repo name (dolphin-in-a-coma/arc-agi-3-just-explore); the paper title is
  "Graph-Based Exploration for ARC-AGI-3 Interactive Reasoning Tasks"; AAAI 2026 Workshop on AI for
  Scientific Research [source-verified: justexplore.txt abstract; arxiv.xml journal_ref].
- "By tracking visited states and tested actions, the agent prioritizes actions that provide the shortest
  path to untested state-action pairs"; median 30 of 52 levels across six Preview games, 3rd on the Preview
  private board [source-verified: abstract]. Twin calls its action efficiency near zero (see Twin).

### SWE-agent (Yang, Jimenez, Wettig, Lieret, Yao, Narasimhan, Press), arXiv 2405.15793, NeurIPS 2024. New key `yang2024sweagent`

- ACI principles 3 ("Environment feedback should be informative but concise") and 4 ("Guardrails mitigate
  error propagation and hasten recovery") [source-verified: sweagent.pdf Sec 2]. Linter guardrail on edit:
  SWE-bench Lite 18.0 with linting, 15.0 without (down 3.0) [source-verified: Table 3 and Sec 5.1].
  "SWE-agent uses informative prompts, error messages, and history processors" (Sec 3, context management).

### Schema and PRO-LONG (named by Twin and Tycho; not in the digest's list). New keys `zeng2026schema`, `fox2026prolong`

- Schema (arXiv 2609.39140): per Tycho, "an append-only transition record, persistent notes, editable Python
  hypotheses, full-history backtesting, certified-model search, discriminating probes, and guarded action
  queues invalidated by prediction mismatches"; Twin calls it "the closest design to ours", self-reported
  98.98 via a best-of-two-models fallback [source-verified secondhand: h_2607.28287 @16455; h_2608.14490 @13102].
- PRO-LONG (arXiv 2607.20064): a lossless append-only log the agent queries with code; 76.1 pass@1 at its
  default 500-action budget [source-verified secondhand: Twin @11315; Tycho @16455].

### Forum and rival claims

- Francise Liang, forum 750885, 2026-10-20 [source-verified: litv/t750885.txt]. Franzen M2 notebook, Kaggle
  RTX Pro 6000, 25 games x 2 passes per run. Identical reruns 47.30 and 48.92, per-game difference sd 24.0,
  run-mean SE about 4.8; "we could only trust offline differences of about 10 points or more". 14 streams:
  29.90 vs 47.30, single run, and that machine also had slower disk (preload 837 s vs 219 s). Stall cutoff
  at 110k tokens: 38.97 vs 48.11 (baseline mean of two), -9.1 [reproduced: -9.14]; paired 10 wins, 11
  losses, 4 ties.
- Tufa (Jeroen Cottaar), forum 753843, 2026-10-29 [source-verified: litv/t753843.txt]: "all our submission
  are losing ~40 minutes"; the notebook starts about 40 minutes after the container; "These 40 minutes count
  against submission" because successful runs last 9 h, not 9 h 40 min. Follow-up 10-30: an empty notebook
  now takes 50 minutes to error.
- xz (Kaggle user xiaoz259, team "faster machine is all you need"):
  - Rank 3, 58.59, 15 entries on the 11-01 18:31Z board [source-verified: board.json].
  - 10-25: "I've ported my solution to Kaggle (accelerator: RTX PRO 6000)"; 10-29 reply to Mark Barney:
    "I've been using public-25 and the copycat-30 games you guys made for local validation!"
    [source-verified: forum 752485]. 10-29: "I just start the games at 10min mark ... and use RESET every
    7min" [source-verified: forum 753843]. 10-20: told Francise Liang his lanes were preempted for lack of
    KV memory [source-verified: forum 750885].
  - "Runs the same public Intel W4A16 weights": not in the saved copies [repo record: docs/PROGRESS.md
    10-29 entry, "xz's public input is the same Intel W4A16 weights with the PLE table resharded"].
- Team Turducken:
  - Rank 5, 56.02, 216 entries on the 11-01 board; members include markbarney and sonphamorg
    [source-verified: board.json].
  - "Short prompt cost about 12 points": sonpham-org/arc-3 commit bcbd6eb7 by Mark Barney, 2026-10-08,
    "restored full-content draft after draftB lost 12 points" [repo record:
    docs/research/2026-10-10-field/evidence/github/sonpham_commit_bcbd6eb7.json]. This predates the
    Turducken merger (10-18) and is a score on their own 25-game bench, not the board.
  - Expert pruning by play traffic plus a short heal training, prompt coaching ("Small changes in wording
    swing the score more than you'd believe"), human-play recordings: [repo record:
    docs/research/2026-10-23-push/field.md:188, citing arc-explainer commit 268c630,
    notes/mark-october-2026, 10-21]. Not in the saved copies; not re-read here.
  - Temperature 0.6 and noborder: [repo record: docs/research/2026-10-19-top/leaders.md:112, their 10-11
    repo stack, read before the repo closed]. Not in the saved copies.
- Side note: `board.json` has `"ourTeamId": "15605182"`, which is Team Turducken's id; our row is team
  16537511, rank 18, 40.98 [source-verified]. The snapshot's pinned-team field is mislabelled.

### Saved but not cited by the digest

- AERA, "Explore Before You Solve" (Liew Keong Han), arXiv 2605.25931: all 25 public games reachable by
  non-intelligent strategies; RHAE 0.2116 with Qwen2.5-0.5B [source-verified: abs page]. No digest claim
  rests on it.

---

## 2. BibTeX for sources not yet in refs.bib

Twin (`skoutnev2026twin`), Tycho (`lehmann2026tycho`) and the ARC-AGI-3 report (`arcprize2026arcagi3`)
already exist. arXiv metadata below is from the arXiv API (2026-11-02) or the saved abs pages.

```bibtex
@misc{rodionov2026ablation,
  author = {Rodionov, Sergey},
  title = {{Do Coding Agents Need Executable World Models, Simplification, and Verification to Solve ARC-AGI-3?}},
  year = {2026},
  eprint = {2607.15439},
  archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/2607.15439}
}

@misc{rodionov2026ewm,
  author = {Rodionov, Sergey},
  title = {{Executable World Models for ARC-AGI-3 in the Era of Coding Agents}},
  year = {2026},
  eprint = {2605.05138},
  archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/2605.05138},
  note = {AGI-2026, Lecture Notes in Computer Science vol. 16855, pp. 198-210 (venue per the arXiv comment; volume and pages per the Twin bibliography)}
}

@misc{wu2026kepler,
  author = {Wu, Wensen},
  title = {{Kepler: Auditable World Models for ARC-AGI-3}},
  year = {2026},
  eprint = {2610.00834},
  archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/2610.00834},
  note = {Non-archival Interpreting Agent Behavior workshop at NeurIPS 2026. Code: github.com/Cveinnt/kepler}
}

@misc{gowtham2026arcskill,
  author = {Gowtham, Poobesh},
  title = {{arc-skill: an agent skill that plays ARC-AGI-3}},
  year = {2026},
  note = {GitHub repository pbshgthm/arc-skill and write-up at arc-skill.vercel.app, August 2026; README accessed 2026-11-01},
  url = {https://github.com/pbshgthm/arc-skill}
}

@misc{ryanbbrown2026retrodict,
  author = {{ryanbbrown}},
  title = {{Retrodict: Frontier ARC-AGI-3 agent}},
  year = {2026},
  note = {GitHub repository ryanbbrown/Retrodict; title from the repository description, accessed 2026-11-02; contents not reviewed},
  url = {https://github.com/ryanbbrown/Retrodict}
}

@misc{tang2024worldcoder,
  author = {Tang, Hao and Key, Darren and Ellis, Kevin},
  title = {{WorldCoder, a Model-Based LLM Agent: Building World Models by Writing Code and Interacting with the Environment}},
  year = {2024},
  eprint = {2402.12275},
  archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/2402.12275},
  note = {NeurIPS 2024}
}

@misc{piriyakulkij2025poeworld,
  author = {Piriyakulkij, Wasu Top and Liang, Yichao and Tang, Hao and Weller, Adrian and Kryven, Marta and Ellis, Kevin},
  title = {{PoE-World: Compositional World Modeling with Products of Programmatic Experts}},
  year = {2025},
  eprint = {2505.10819},
  archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/2505.10819},
  note = {NeurIPS 2025 (venue per the Twin bibliography)}
}

@misc{tsividis2021empa,
  author = {Tsividis, Pedro A. and Loula, Joao and Burga, Jake and Foss, Nathan and Campero, Andres and Pouncy, Thomas and Gershman, Samuel J. and Tenenbaum, Joshua B.},
  title = {{Human-Level Reinforcement Learning through Theory-Based Modeling, Exploration, and Planning}},
  year = {2021},
  eprint = {2107.12544},
  archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/2107.12544}
}

@misc{warrier2025autumnbench,
  author = {Warrier, Archana and Nguyen, Dat and Naim, Michelangelo and Jain, Moksh and Liang, Yichao and Schroeder, Karen and Yang, Cambridge and Tenenbaum, Joshua B. and Vollmer, Sebastian and Ellis, Kevin and Tavares, Zenna},
  title = {{Benchmarking World-Model Learning with Environment-Level Queries}},
  year = {2025},
  eprint = {2510.19788},
  archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/2510.19788},
  note = {ICML 2026; introduces the WorldTest protocol and the AutumnBench benchmark}
}

@misc{lencevicius2026persistgym,
  author = {Lencevicius, Raimondas},
  title = {{When the harness decides success: state that cannot be re-observed}},
  year = {2026},
  eprint = {2610.26632},
  archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/2610.26632},
  note = {Code and pre-registration: github.com/raimondasl/persistgym}
}

@misc{rudakov2025graph,
  author = {Rudakov, Evgenii and Shock, Jonathan and Cowley, Benjamin Ultan},
  title = {{Graph-Based Exploration for ARC-AGI-3 Interactive Reasoning Tasks}},
  year = {2025},
  eprint = {2512.24156},
  archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/2512.24156},
  note = {AAAI 2026 Workshop on AI for Scientific Research. Code: github.com/dolphin-in-a-coma/arc-agi-3-just-explore}
}

@misc{yang2024sweagent,
  author = {Yang, John and Jimenez, Carlos E. and Wettig, Alexander and Lieret, Kilian and Yao, Shunyu and Narasimhan, Karthik and Press, Ofir},
  title = {{SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering}},
  year = {2024},
  eprint = {2405.15793},
  archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/2405.15793},
  note = {NeurIPS 2024}
}

@misc{zeng2026schema,
  author = {Zeng, Guanning and Wang, Jiani and Ma, Wenjie and Yin, Shaofeng and Wang, Chenyang and Liu, Shichen and Kanazawa, Angjoo and Ni, Wode and Li, Xiuyu and Zanette, Andrea and Feng, Haiwen},
  title = {{Schema: Discovering Unknown Environments via Agentic Program Induction}},
  year = {2026},
  eprint = {2609.39140},
  archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/2609.39140}
}

@misc{fox2026prolong,
  author = {Fox, Alexis and Wang, Junlin and Rosu, Paul and Dhingra, Bhuwan},
  title = {{PRO-LONG: Programmatic Memory Enables Long-Horizon Reasoning}},
  year = {2026},
  eprint = {2607.20064},
  archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/2607.20064}
}

@misc{liang2026forum750885,
  author = {Liang, Francise},
  title = {{How noisy is an offline eval of the Franzen M2 harness? Plus two negative results (more streams, stall cutoff)}},
  year = {2026},
  note = {Kaggle forum topic 750885, ARC Prize 2026 - ARC-AGI-3, post of 2026-10-20},
  url = {https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/750885}
}

@misc{cottaar2026forum753843,
  author = {Cottaar, Jeroen},
  title = {{Reply in ``Is everything back to normal, or is something else going on behind the scenes?''}},
  year = {2026},
  note = {Kaggle forum topic 753843, ARC Prize 2026 - ARC-AGI-3, posts of 2026-10-29 and 2026-10-30},
  url = {https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/753843}
}

@misc{xz2026forum752485,
  author = {{xz}},
  title = {{RTX PRO 6000 sessions: startup takes 10-60 min, and slow-starting instances run about 10\% slower}},
  year = {2026},
  note = {Kaggle forum topic 752485, ARC Prize 2026 - ARC-AGI-3, post of 2026-10-25 and replies of 2026-10-29 (Kaggle user xiaoz259)},
  url = {https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/752485}
}

@misc{barney2026promptcommit,
  author = {Barney, Mark},
  title = {{System prompt: restored full-content draft after draftB lost 12 points}},
  year = {2026},
  note = {Commit bcbd6eb7 in sonpham-org/arc-3, 2026-10-08; the repository later returned 404; copy in the repo evidence folder docs/research/2026-10-10-field/evidence/github},
  url = {https://github.com/sonpham-org/arc-3/commit/bcbd6eb7b14962924397f239aa2cd9880c41c94a}
}

@misc{barney2026turduckennote,
  author = {Barney, Mark},
  title = {{Team notes, October 2026}},
  year = {2026},
  note = {arc-explainer repository, notes/mark-october-2026, commit 268c630 (2026-10-21) per docs/research/2026-10-23-push/field.md; title is a placeholder and the note was not re-read in this pass},
  url = {https://github.com/82deutschmark/arc-explainer}
}
```

---

## 3. The five mechanisms by system (only what the sources support)

Mechanisms: (1) log every frame and let the agent query it; (2) check every hypothesis against all recorded
history; (3) plan inside a simulator; (4) run plans step by step and stop at the first prediction miss;
(5) treat every recorded board as a non-goal.

| System | (1) Log and query | (2) Check vs all history | (3) Plan in simulator | (4) Stop at first miss | (5) Recorded boards are non-goals |
|---|---|---|---|---|---|
| Twin | Yes: permanent transition log, file bridge | Yes, harness-enforced before every scored action | Yes: BFS in the validated twin; 92.9% of scored actions | Yes, harness executor (ExecuteChecked) | Yes, explicit goal filter; prompt-requested in the reported runs, enforced only in the released harness (App. D) |
| Tycho | Yes: typed frames per turn, `wmlib` query library | Partial: replays the most recent attempt per level; auto-verify on save; repair only in Trigger policy | Yes, optional: plan.py validated route | No: advisory only (route shown while frame hash matches; guarded executor named as future work) | Implicit: outcome check requires ordinary decision states to stay "ongoing" |
| Kepler | Yes: daemon records transitions | Partial: agent instructed to backtest full history, not forced | Yes (planner, certified per-level programs; planner crashed in five experimental boards) | Yes, but not blocking when prediction fails or covers no cells | Not stated in saved README |
| arc-skill | Yes: append-only event log; every settled board preloaded in `arc python` | Optional rules tier only; used once, never fitted | Optional (A* in rules tier; BFS/A* on grids) | Yes, hard: prediction required per press; plans halt at first miss | Not stated |
| Retrodict | Unverified (no saved source) | Unverified | Unverified | Unverified | Unverified |
| Rodionov 4x4 ablation (2607.15439) | Recorded attempts replayable; state-reconstruction tools in the verification variant | Verification variant only: replays every recorded attempt; prompted and tool-supported, not enforced | Executable variants keep planning code; planner verified to reach LEVEL_COMPLETED | No: "not required to pass through an online plan executor in the main ablation" | Implicit: verifier compares predicted vs recorded status each step |
| EWM (2605.05138), secondhand | Implied (verifies against previous observations) | Yes per abstract ("verifier programs") | Yes per abstract | Plan executor per abstract; Twin says it "halts at the first misprediction" | Not stated |

Reading: only Twin has all five, and only Twin enforces (2). (4) is absent from Tycho (advisory) and from
Rodionov's main ablation. (5) is explicit only in Twin; Tycho and Rodionov have it implicitly through outcome
or status replay. The "converged control loop" sentence in Twin's related work covers (1) to (4) for EWM,
OPINE-World and Schema, not Tycho, Kepler or arc-skill, and not (5).

---

## 4. Digest statements not supported or wrong

1. "Public ARC-AGI-3 harnesses have converged on five mechanisms ... Twin, Tycho, Kepler, arc-skill,
   Retrodict and the 4x4 ablation" (line 300). Overstated. Per the table, only Twin has all five; (4) is
   missing in Tycho and Rodionov's ablation; (5) is explicit only in Twin; Retrodict has no saved source;
   "the 4x4 ablation" is not a separate system (best match is Rodionov 2607.15439, which [agent-literature]
   lists again).
2. "The same model and agent with the harness removed score 61.1" (line 324), credited to "this loop"
   (replay against every logged transition). The 61.1 ablation removed the world model, Validate, Explore,
   Plan and the halt-on-mismatch executor together. The 32.2-point gap cannot be credited to replay
   validation alone. One run per game.
3. "The authors credit most of the gain to validation being enforced by the harness rather than requested
   in the prompt" (line 324). Not supported. Twin says enforcement "matters most on long games" and "may
   explain" three game clears, comparing against other systems; no experiment isolates enforced vs prompted
   validation. Twin attributes the final 15.0 points (78.3 to 93.3) to a validated world model.
4. "harness-enforced replay validation and the goal-consistency check score 93.3" (line 55). Half wrong:
   per Twin Appendix D, the goal check was asked for by prompt during the reported runs and is enforced only
   in the released harness. Twin's Method text contradicts its own appendix here.
5. "Rodionov's ablation (exact replay verification ranks first in every setting)" (line 324). The quote is
   accurate, but the winner is a bundle (fixed interface, tools, scheduled simplification, replay
   verification). The author says the experiments "do not establish how much of its advantage comes from
   verification alone", verification was prompted, not enforced, margins were 0.62 to 8.56 points, and it
   cost the most tokens in every setting.
6. "Kepler, ..., WorldCoder and PoE-World all use the same pattern" (line 324). Kepler only partly
   (backtest instructed, not forced; failed predictions do not block). WorldCoder and PoE-World support is
   secondhand through Twin's related work; they are offline or non-ARC world-model learners (PoE-World on
   Atari), and nothing saved shows a replay-every-transition gate before acting.
7. EMPA claims (state-conditioned "known" interactions; exploration goals reopen when state changes; lines
   134, 136, 324). Not verified: no saved text. Tycho cites a 2026 journal version (11 authors, other
   title) that one web search did not confirm; the digest's arXiv id is the 2021 preprint.
8. AutumnBench "12.5% of actions each on resets and no-ops for humans, under 7% for models" (lines 138,
   324). Not verified: no saved copy and not in the abstract. The abstract says "five frontier models", not
   reasoning models. Correct title: "Benchmarking World-Model Learning with Environment-Level Queries"
   (AutumnBench is the benchmark inside it).
9. "Turducken (rank 3)" (line 307). Stale: on the 11-01 18:31Z board Turducken is 5th (56.02); rank 3 is xz
   (58.59).
10. Turducken "in its own note ... prunes experts, heal fine-tune, prompt coaching, human-play recordings"
    (lines 307, 397). Not in the saved copies; it rests on a 10-23 repo record of an arc-explainer note.
    "So at least one team near the top does fine-tune" is an inference; a Turducken member (dzjiann) called
    Flash-Next SFT and RL "unstable" [repo record: docs/research/2026-10-19-top/leaders.md:108].
11. "A short prompt cost them about 12 points" (line 397). Source is a 10-08 commit in Son Pham's repo,
    before the 10-18 merger, on their own 25-game bench, not the board. Attribute to Son Pham's team.
    "Temperature 0.6 and noborder" are from their 10-11 repo stack, not from a Turducken statement.
12. "No top-30 team has published" (line 305) and "No team in the top 12 has published what it changed"
    (line 394). Inconsistent with the digest's own use of Turducken's note, which names its changes (vaguely).
13. xz "runs the same public Intel W4A16 weights" (line 398). Not in the saved copies (repo record only).
    "ported their own harness on 10-25": xz wrote "I've ported my solution to Kaggle", which is a port of
    the solution to the Kaggle accelerator, not a statement about a harness. "Specialise in throughput" is
    inference from the team name and posts. Rank, score, entries and the copycat-30 validation are verified.
14. "Stall cutoff (-9.1) and more streams (29.90 vs 47.30)" listed as ruled out (line 312). Numbers are
    right, but both are single offline runs; Francise Liang's own bar is about 10 points, so -9.1 (paired
    10/11/4) sits under it, and the 14-stream run also had a slower machine.
15. "persistgym shows truncation costs almost nothing when facts can be looked up again" used to rule out
    summary-on-trim (line 313). True for the synthetic probe, but the same paper's ARC-AGI-3 result
    (one model, four games, one run each) found summarizing old turns recovered 86% of a 63-point gap.
    The citation is selective.
16. "Structured notes: Franzen says disabling them was the biggest gain in that area" (line 310). Not
    verified: Franzen's write-up is not among the saved copies.
17. "Rodionov: a flexible interface lost to plain text" (line 56). Textual beat flexible-interface
    executable in 3 of 4 settings (both gpt-5.5 settings and gpt-5.4 high by 0.62); it lost at gpt-5.4 xhigh
    (40.67 vs 44.72).
18. Naming: "Just Explore" and "graph-explorer" (lines 313 to 321) are the same work (Rudakov et al.
    2512.24156; "Just Explore" is its repo name).

Verified as stated: Twin 93.3 vs 61.1 on the same model and agent (with the caveats above); Twin
"certified negatives" and goal-discovery signals; Tycho's typed evidence contract; Kepler's arXiv id;
Rodionov's "ranks first in every setting" wording; graph explorer near-zero action efficiency (Twin); SWE-agent
informative feedback and the linting ablation (18.0 vs 15.0); Tufa's 40-minute late start [claim];
Francise Liang's numbers; xz rank 3 at 58.59 from 15 entries and public-25 plus copycat-30 validation; our
rank 18 at 40.98 with ranks 13 to 17 at 41.0 to 42.3 and ranks 3 to 12 at 45.0 to 58.6.
