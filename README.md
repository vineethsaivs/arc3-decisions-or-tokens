# Decisions or Tokens? Release kit

Code, data and sources behind the paper "Decisions or Tokens? What Moves an Offline ARC-AGI-3 Agent, and How to Tell".
Licence: see `LICENSE` (paper, figures and data CC BY 4.0; our code under CC BY 4.0 or MIT-0, your choice). Third-party files and what is NOT included: `THIRD_PARTY.md`.

## Reproduce the headline figures (CPU, no network, about a minute)

```sh
sh reproduce.sh              # figures, plus the MDE at the pairs' SD and two runs per arm
sh reproduce.sh 6.5 12       # your run-to-run SD in levels, runs per arm
python3 notebook_cpu.py 6.5  # same data in plain Python: pair spreads, eight-run SD, MDE table
```

`reproduce.sh` regenerates the noise figure (Fig 2, "Identical code, different scores", `fig_noise_draws.py`) and the
discovery figure (`fig_discovery.py`) from the CSV and JSON files here, in a scratch copy, and writes them to
`./reproduced_figs`. It needs `uv` (matplotlib and numpy come from uv's cache, or one online install) or a `python3` that has
matplotlib. The minimum detectable effect is `MDE(n) = 2.80 * SD * sqrt(2/n)`: n runs per arm, 80% power, two-sided 5%
(Eq. `mde`, `paper/v3/tex/sections/03_measure.tex`). At the pairs' SD of 8.25 and two runs per arm it prints 23.1 levels. <!-- B1 B4 -->

## What is included

| Path | What |
| --- | --- |
| `paper/v3/tex` | LaTeX sources (`build.sh` uses tectonic), `main.pdf`, `main_anon.pdf` |
| `paper/v3/figs`, `paper/v3/analysis` | built figures; figure scripts (`analysis/figs`, header of each says how to run), their data, the style and claims-table checkers |
| `paper/v3/evidence` | `claims_addendum.md`, `dossier_1013_1021.md` (claim ids), `data/` CSVs |
| `paper/v3/writeup` | Writeup text and alt text |
| `paper/v2/analysis/out` | data files from the earlier analysis that the v3 figure scripts read |
| `solution/kaggle/franzen_overlays`, `build_franzen_arms.py` | the overlays on Franzen's notebook and the script that builds each arm |
| `solution/sim` | tests for the overlays; `fixtures/` holds the vendored third-party trees the tests build against |
| `solution/analysis/harness_2026-10`, `docs/research/2026-11-01-harness` | gate readers and the harness ledger, literature notes and scripts |
| `solution/kaggle/generated/gate-predeclared-*.json`, `gate-results-*.json` | gate rules written before each run, and the results |

## Figure scripts that run from this export

Run (14 of 16 scripts, tested in a scratch copy of this export): `fig_budget`, `fig_discovery`, `fig_draws`, `fig_finetune`,
`fig_glyph`, `fig_loop`, `fig_noise_draws`, `fig_power`, `fig_scroll`, `fig_selection`, `fig_trainnoise`, `fig_tried`,
`fig_yield`, `cover` (`fig_glyph` and `fig_scroll` need `uv run --python 3.12` and shiiin9's file, listed below). Do not run
here: `fig_clock` and `fig_gates`, which read per-run `benchmark.json` kernel outputs that are not included.

Overlay tests: run `uv run --python 3.12 python -B solution/sim/test_franzen_<name>.py` from the repo root. Several need
the ARC-AGI-3 environment files, kernel outputs or a venv with the framework (flask, imageio) that is not included.

## What is NOT included

- Model weights, Kaggle datasets, the large pickles and every kernel-output transcript.
- Secrets: `.env`, Kaggle tokens, anything credential-like (see `SCAN.md` in the kit; the export was scanned).
- The notebook as submitted to Kaggle (a built copy of the fork is in `solution/kaggle/generated/candidates-fork-franzen`).
- Board scores and ranks cannot be regenerated; they are verified by submission id on Kaggle.

## Third-party material and credit

- Franzen's Milestone 2 notebook and patch bundle (`solution/sim/fixtures/franzen-bundle`, `pennyroyal-v253`): his work, under
  his terms. Our overlays patch it. Tufa Labs' Duck harness is inside that bundle: theirs, no licence file at the repo root.
- `docs/research/2026-10-04-climb/evidence/kernels/shiiin9_form_priority_dprime.py`: shiiin9's D-prime priority file, licence not read.
- `solution/kaggle/franzen_overlays/nvfp4kv`: Lesilva's NVFP4 KV patches, Apache-2.0, with his LICENSE and NOTICE.
- Qwen3.8-Flash-Next, the Intel W4A16 checkpoint and the albucino drafter are under the Qwen Community License; none is included.
- Full list, including REAP keep-448, the Olympie and lovedheart forks: `THIRD_PARTY.md`. See also the paper's Appendix "Licences, release and AI use".

## Checks

`python3 -I -B paper/v3/analysis/v3_style_check.py` and `python3 -I -B paper/v3/analysis/build_claims_appendix.py --check`
(from this root) run the paper's style and claims-table checks. `sh ../verify_manifest.sh` (in the release kit) confirms
every file against `MANIFEST.sha256`.
