"""CPU notebook (plain Python, one cell per '# %%' block, so Jupytext or VS Code can turn it into a notebook later).

What it does: loads the identical-run pairs and the eight concurrent runs from the data files of this export, prints how far
byte-identical runs differ, and prints the minimum detectable effect (MDE) for those SDs and for one you supply.
No network, no GPU, standard library only.

  python3 notebook_cpu.py [SD [N]]     SD: your run-to-run SD in levels (default: the pairs' SD); N: runs per arm (default 2)

Claim ids (paper/v3/evidence/claims_addendum.md, dossier_1013_1021.md): B1 pair SD 8.25; V3-X3 eight-run SD 5.95; B4 MDE
23.1 at two runs; V3-X12 MDE 9.4 at twelve.
"""
# %% Setup: find the data (export root, release/, or DATA_ROOT)
import csv
import math
import statistics as st
import sys
from pathlib import Path

# DATA_ROOT: the folder that holds paper/ (the export root). Leave None to look next to this file, or in ./repo of the
# current folder when __file__ is undefined (Jupyter, Kaggle, exec). On Kaggle set it to the attached dataset folder.
DATA_ROOT = None
try:
    HERE = Path(__file__).resolve().parent
except NameError:  # run as a notebook cell or through exec(): no __file__
    HERE = Path.cwd()
ROOT = Path(DATA_ROOT) if DATA_ROOT else HERE if (HERE / "paper/v3/evidence/data").is_dir() else HERE / "repo"
assert (ROOT / "paper/v3/evidence/data").is_dir(), f"data not found under {ROOT}; set DATA_ROOT at the top of this cell"
PAIRS_CSV = ROOT / "paper/v2/analysis/out/e4_fig_data.csv"                 # six identical-code pairs (panel a), claim B1
RUNS_CSV = ROOT / "paper/v3/evidence/data/vertex_vs_kaggle_runs.csv"       # eight identical runs on one day, claim V3-X3
Z = 2.80   # (z of 0.975 + z of 0.80) rounded as in the paper's Eq. (mde); two-sided 5%, 80% power


def mde(sd, n):
    """Smallest true level shift a two-arm test with n runs per arm detects (claim B4)."""
    return Z * sd * math.sqrt(2 / n)


# %% Pairs: two runs of byte-identical code each; the spread is pure run-to-run noise
# Columns: panel, label, run 1 levels, run 2 levels, spread (the header text of the 4th column is "v2_or_more...", so read by position)
pairs = [(r[1], int(r[2]), int(r[3])) for r in list(csv.reader(open(PAIRS_CSV)))[1:] if r[0] == "a"]
spreads = sorted(abs(a - b) for _, a, b in pairs)
sd_pairs = math.sqrt(sum(d * d for d in spreads) / (2 * len(spreads)))   # pooled SD from pair differences, 6 df
assert len(pairs) == 6 and spreads == [3, 4, 4, 6, 16, 22], spreads      # B1; fails loudly if the data file changes
assert round(sd_pairs, 2) == 8.25
print("Identical-run pairs (levels of 183):")
for name, a, b in sorted(pairs, key=lambda p: -abs(p[1] - p[2])):
    print(f"  {name:9s} {a:4d} {b:4d}  spread {abs(a - b):2d}")
print(f"Spreads {spreads}; pooled SD {sd_pairs:.2f} (6 df)")

# %% Runs: eight identical runs started together on one day
runs = [int(r["levels_25"]) for r in csv.DictReader(open(RUNS_CSV)) if r["run"].startswith("vertex-rq1020-")]
sd_runs = st.stdev(runs)
assert len(runs) == 8 and round(sd_runs, 2) == 5.95                      # V3-X3
print(f"\nEight concurrent identical runs: {sorted(runs)}; range {max(runs) - min(runs)}, SD {sd_runs:.2f} (7 df)")

# %% MDE: what a test of two arms can resolve at each SD
def _arg(i, default):  # a notebook kernel puts its own flags in sys.argv; ignore anything that is not a number
    try:
        return float(sys.argv[i])
    except (IndexError, ValueError):
        return default


sd_user = _arg(1, None)
n_user = _arg(2, 2)
print("\nMinimum detectable effect, levels (80% power, two-sided 5%):")
print(f"  {'SD':>6s} {'n=2':>7s} {'n=6':>7s} {'n=12':>7s}")
for label, sd in [("pairs", sd_pairs), ("runs", sd_runs)] + ([("yours", sd_user)] if sd_user else []):
    print(f"  {sd:6.2f} " + " ".join(f"{mde(sd, n):7.1f}" for n in (2, 6, 12)) + f"   ({label})")
assert round(mde(sd_pairs, 2), 1) == 23.1 and round(mde(sd_pairs, 12), 1) == 9.4   # B4, V3-X12
if sd_user:
    print(f"\nYour SD {sd_user:g}, {n_user:g} runs per arm: MDE {mde(sd_user, n_user):.1f} levels")
print("A level change smaller than the MDE cannot be told from run-to-run noise by this test.")
