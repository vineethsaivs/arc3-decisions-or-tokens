# Harness research pass, 2026-11-01 (workflow wf_27428c3d-32e)

Ten agents read our transcripts, the patched harness code, public ARC-AGI-3 harnesses, the agent literature and the
competition forum, proposed harness changes, and a skeptic tried to refute each top candidate. It motivated the replay
check (`franzen_overlays/verify.py`, arm `l1a1y8sv2r3lf15ustv`), which passed its gate and made the final draw (36.38).

- `DIGEST.txt`: the full digest as the workflow returned it (model, candidates with skeptic verdicts, research summaries
  per angle). Numbers in it are the agents' own counts; the paper reproduces the transcript ones with
  `paper/v3/analysis/discovery_read.py`.
- `scripts/`: the agents' scratch scripts, kept as run. They read a pickle built by `scripts/parse.py OUT.pkl DIR...` from
  run transcripts under `solution/kaggle/generated/kernel-outputs/<run>/`; `common.py` holds the pickle path (set `PKL`).
  Not maintained; the paper's script supersedes them.
- `LITERATURE.md`: every outside source checked against saved copies (2026-11-02), with BibTeX and the statements in the
  digest that did not hold up. `LEDGER.md`: candidates, skeptic verdicts, the built candidate predicted against measured.
- Not kept: the 288 MB turn pickle, third-party pages and PDFs (public harness write-ups, forum topics); their sources
  are cited in the paper's bibliography.

Headline: levels per game = window / clock per level; throughput is at its floor and execution is cheap, so the clock per
level is set by discovery (how fast wrong rules and wrong goals die). The recorded history gives counterexamples for zero
actions, yet the model replayed its simulator against it on about 6% of cleared levels. Twin, the public harness with the
largest fixed-weight gain, enforces that replay; the others prompt or instruct it. Skeptics refuted all four top candidates as level levers a three-run gate could
see; verify was built anyway and gated on use plus no harm. Details: `docs/PROGRESS.md` 2026-11-01 19:40Z, session log
section 3.5, claim V3-W29.

Corrections found 2026-11-02 (all reflected in the paper, Appendix `app:research`): rank 3 is xz, not Turducken; the
digest's 81% opener reuse did not reproduce (73%); only Twin has all five 'converged' mechanisms; Twin's 61.1 ablation
removes the whole harness, not replay alone; EMPA and AutumnBench specifics are unverified; the start-time figure is a
720 s release deadline in Franzen's notebook (6 of 23 lf15 runs started 11 to 21 min early, none of the ustv runs).
