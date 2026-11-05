# Skeptic lens, verified (2026-11-04)

Numbers from `paper/v3/analysis/v3_skeptic_checks.py` (computed, json beside this file) or read from the cited files.

Key facts
- Pair gaps 16, 6, 4, 22, 3, 4: mean shift 9.17, SD incl. shift 8.25, SD about shift 5.59 (computed). Paired t p 0.036, sign p 0.031.
- 3 of 6 first runs were the highest run to date when run (computed): selection of a high first run is an untested candidate (inference).
- Same day, rq5 and rq6 (2.5 h later) read 130 and 128, +5.8 over the other six (computed): no monotone later-is-lower drift there.
- Across days: 9 runs mean 124.0 vs 21 controls 111.48 (excl. franzen-start-v1) = +12.52; z 3.8 at SD 8.25, 5.3 at 5.95. Across days minus interleaved: 15.0, z 2.6 / 3.6. So run noise does not explain the failure to reproduce; the gate file itself says 7 of 9 treated runs carried other serving changes and 17% more tokens.
- Two-run MDE (2.8 x SD): 23.1 at 8.25, 16.7 at 5.95, 15.7 at 5.6, 13.2 at 4.7.
- Rank p adapters vs 18: 0.0038 only if ties are counted as extreme (5/1330). V3-W9 says "ties counted half": inconsistent, fix the hedge text (with ties half the value is 0.0023). vs 2 same-card: min p 0.10.
- Board (computed): 31 completed since fork, 28 live; 17 valid mean 34.59; with 10-18 zero 18 draws 32.67; adapter draws 35.90, 39.79, 35.98 mean 37.22; 17 valid plus adapter 34.99.
- Notebook URL returned HTTP 404 logged-out on 2026-11-04 (curl). Repo 200. D3 dataset page 200.
- Tufa quote confirmed on arcprize.org milestone-1 page: "Tufa Labs noted that, counterintuitively, hand-crafted tools actually hurt the model" (Qwen 3.6 27B entry).
- 1.9 source: docs/PROGRESS.md 2026-10-21 15:20Z (WA wave 1 Delta +1.88 on 17 bench games, selection-biased), 10-22 14:40Z (wave 2 only -2.83, pooled +0.25 vs +1.75 bar). Not in claims_addendum; V3-G13/G14 (status 10-21 01:00Z, "none run") cannot support it.

Word budget: body is 1,481 words (computed, comments, title and subtitle removed). Full compact fix set is +60 words (1,541). Needs about 41 words cut elsewhere, or drop the optional items.
