# Build report, v3 PDF, final-version refresh (2026-11-01)

Build: `cd paper/v3/tex && TECTONIC=/opt/homebrew/bin/tectonic sh build.sh` builds `main.pdf` (public) and `main_anon.pdf`
(anonymised; `main_anon.tex` sets `\buildanon`). Tectonic 0.15.0, packages cached, about 9 s. Figures are read from `../figs`
through `\figfile`; all 14 figures and the cover were present at build time (no placeholder boxes). `main_owner.tex` (owner notes,
never to be posted) is unchanged and not in `build.sh`.

## Numbers

| Item | Public | Anonymised |
| --- | ---: | ---: |
| Main text | pages 1 to 17 (about 16.7 pages; was about 12.2 on 10-21) | same |
| References | pages 18 to 21 | same |
| Appendices | pages 22 to 69 | 22 to 67 (rank history hidden) |
| Total pages | 69 | 67 |
| Errors, overfull boxes, underfull hboxes | 0, 0, 0 | 0, 0, 0 |
| Underfull vboxes (page-level, appendix floats) | 6 | 7 |
| Undefined references or citations, `??`, `[?]`, BibTeX warnings | 0 | 0 |
| Dashes (en, em, LaTeX `--`) in the tex tree and refs.bib | 0 | 0 |

Tectonic still prints the known "internal consistency problem" and "rerun seems needed" notes; the log has no undefined reference.
Both copies have one LaTeX warning, "Text page N contains only floats" (an appendix page).

The main text grew by about 4.5 pages, almost all of it figures: the new Sec. 6 carries four (clock, gates, glyph, scroll;
about 2.6 pages with their captions) and Sec. 8 one (board history). To pay for them, Sec. 5 lost its stale "interpretation" and
"test that would resolve it" paragraphs (the test ran), Sec. 4 and Sec. 3 lost about 150 words, Sec. 7.1 "A reading" was
shortened, Table 25 (what the adapter changed) moved to Appendix I, and Sec. 8 dropped two paragraphs. The figures are scaled
to 3.7 to 4.5 inches wide, so the small labels inside fig_clock, fig_glyph and fig_scroll are at the edge of legible; their
full-size PDFs are in `paper/v3/figs/`. Moving fig_glyph and fig_scroll to the appendix (where their scripts' README put them)
would return about 1.5 pages if the length matters more than the placement.

## Words per body section (`python3 -I -B paper/v3/analysis/v3_style_check.py`)

| Section | Prose | Captions | Cells | Boxes |
| --- | ---: | ---: | ---: | ---: |
| Abstract | 294 | 0 | 0 | 0 |
| 1 Introduction | 927 | 0 | 0 | 90 |
| 2 Setting and system | 528 | 84 | 0 | 0 |
| 3 Measuring a noisy agent | 668 | 254 | 142 | 0 |
| 4 Token-budget curve | 659 | 105 | 0 | 0 |
| 5 A context lever under test | 517 | 121 | 0 | 4 |
| 6 Fixing what the model is shown (new) | 1,046 | 440 | 0 | 0 |
| 7 Fine-tuning | 880 | 95 | 0 | 3 |
| 8 Leaderboard | 508 | 107 | 0 | 0 |
| 9 What to copy, transfer | 220 | 0 | 0 | 218 |
| 10 Related work and credit | 526 | 0 | 0 | 0 |
| 11 Limitations and conclusion | 400 | 0 | 0 | 0 |
| Total | 7,173 | 1,206 | 142 | 315 |

(Sec. 6 prose counts include run-in headings. The Sec. 5 and Sec. 7 boxes live in `sections/placeholders.tex`.)

## Number density and style

79 prose paragraphs; 0 over 5 numbers; 0 sentences over 3 numbers. "we do not know why" once (Sec. 5); "coincides with" once.
No record codes, "replica", "novel" or "significant" in body prose, captions or cells. The Writeup body was checked with the same
counter (Accuracy line, headings, captions and the What-to-copy box excluded): 0 paragraphs over 5, 0 sentences over 3. It is
1,490 words by `wc -w` with HTML comments removed, title and subtitle included.

## What changed in this build

- New Sec. 6 `sections/06_harness.tex` (diagnosis, clock, three gated changes, the first build killed by a naming bug, the scroll
  null, what three runs can say); later sections renumbered (fine-tuning 7, board 8, copy 9, related 10, limits 11). New Appendix J
  `sections/i2_harness.tex`: the three changes as built, the predeclared gate items, every run (play and counters), pooled readings,
  leads with intervals, earlier arms. Reader script: `paper/v3/analysis/harness_read.py` (writes `figs/data/harness_runs.csv`).
- Abstract, intro (five contributions, stale "confounded with tokens" sentence fixed, what we do not claim, Result box), Sec. 8
  (26 live draws, rank 17 of 5,012, Final Submissions verified), Sec. 9 (item 6, a fifth transferable object), Sec. 10 (Tufa's
  step-verification credit), Sec. 11 (harness limits, the interleaving result in the conclusion).
- Appendix C: four draw rows (38.17, 29.63, 35.25 and the harness build's 33.56 on 2026-11-01), rank rows for 10-29 to 11-01 plus a live read, Final Submissions paragraph.
- Scroll arm final (V3-W27): three runs (111, 122, 130 levels, mean 121.0), gate PASS by 0.1 level, not promoted, no gain in bp35 or lf52; every earlier "two runs", INCOMPLETE and "killed unless" statement is rewritten (Sec. 6, Sec. 11, Appendix J, Writeup). fig_gates re-run with the third run.
- Board refresh to 2026-11-01 18:35Z: rank 18 of 5,030, 27 live draws, 16 valid draws (mean 34.48); one draw of the harness build (33.56) is stated to say nothing about its mean. `fig_draws` rebuilt from `board_history_build.py` (now reads `kapi_subs_2026-11-01.txt`; new family "harness build"; labels moved apart). No Final Submission is marked by hand (counter 0 of 2; Kaggle auto-picks the best two).
- Appendix J ends with one neutral paragraph that a fourth change, tool2, was predeclared and its result is added when all three runs are read. No numbers and no claims about it anywhere in the paper.
- Gate readers copied to `paper/v3/analysis/harness_gates/` (from `solution/analysis/harness_2026-10/gates/`).
- Claims V3-W18 to V3-W27 added in `paper/v3/evidence/claims_addendum.md`; claims table rebuilt (182 rows); `--check`: every cited
  id has a row.
- Easy-to-find refresh spots are marked `BOARD-REFRESH` (grep the tex tree and the Writeup).

## Numbers I could not trace, or that differ from what I was told

- The seg8 gate record says 71 of 184 human level starts in 13 of 21 games hold an edge-split object; the figure agent's recount
  with the real patched module under a stated rule gives 76 in 13 of 22 (73 if sparse dot lattices are dropped), and the 71 is not
  reproducible. The paper uses 76 and says so in Appendix J.
- The clock numbers differ from the 10-30 notes: the six runs give 76% of the clock on cleared levels (13.4 minutes per cleared
  level) and 24% on the level open at the end, and 27% of the clock in turns with no executed action (not about two thirds and 28%).
  The paper uses the fig_clock numbers.
- The scroll gate printed PASS on three runs (levels 121.0 against a bar of 120.9; score 54.10 against 53.35) and the change was not promoted by judgement (record `promote: false`): mean below the control and the build it extends, no gain in bp35 or lf52. The paper says so.
- Family statistics (twelve valid draws, mean 34.59; Figures 2b and 8) close on 2026-10-28 and are not redrawn; the later draws
  (three of the 15-stream build, one of the harness build) enter only the 16-draw mean 34.48 and the 30-draw counts. Their submission-file sizes were not read.
- No Final Submission is marked by hand (counter 0 of 2, Kaggle auto-picks the two best public draws): owner-reported from `docs/PROGRESS.md` (V3-W25), not re-read here. The tool2 gate has no result in the paper.
- Carried over from the 10-21 build: the 128K history (V3-W4), the RE-ARC-3 column of the adapter table, the 130-level stop rule
  (V3-L11), field early-ending counts, the 3.5% leaner-history estimate, Vertex list prices, figure-only labels.
- The cover image was not changed (it names the yield, serving and imitation arrows only); it does not mention the harness work.
