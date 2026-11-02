#!/bin/sh
# One command: regenerate the noise ruler figure (paper Fig 2a/2b, fig_noise_draws.py) and the discovery figure
# (fig_discovery.py) from the data files in this export, then print the minimum detectable effect (MDE) for your SD.
# No network is used by the scripts. uv installs matplotlib and numpy from its local cache (--offline); if the cache is
# empty it falls back to one online install, then to a system python3 that already has matplotlib.
#
#   sh reproduce.sh [SD [N]]      SD: run-to-run SD in levels (default 8.25, the six identical pairs); N: runs per arm (default 2)
#   REPRO_OUT=dir sh reproduce.sh  where the figures go (default ./reproduced_figs)
#
# The figure scripts write into paper/v3/figs, so they run in a throw-away copy; the export itself is never modified.
set -eu
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$HERE"; [ -d "$ROOT/paper/v3/analysis/figs" ] || ROOT="$HERE/repo"
[ -d "$ROOT/paper/v3/analysis/figs" ] || { echo "reproduce.sh: cannot find paper/v3/analysis/figs next to this script or in ./repo" >&2; exit 1; }
SD="${1:-8.25}"   # 8.25: SD of six identical-code pairs, claim B1 (paper/v3/tex/sections/03_measure.tex)
N="${2:-2}"
OUT="${REPRO_OUT:-$PWD/reproduced_figs}"

# ---- MDE: Eq. (mde) in 03_measure.tex, MDE(n) = 2.80 * sigma * sqrt(2/n); 80% power, two-sided 5%, n runs per arm (claim B4)
python3 -I -c '
import math, sys
sd, n = float(sys.argv[1]), float(sys.argv[2])
assert sd > 0 and n >= 1, "need SD > 0 and N >= 1"
print("MDE at SD %.2f levels, %g runs per arm: %.1f levels  (2.80 * SD * sqrt(2/n))" % (sd, n, 2.80 * sd * math.sqrt(2 / n)))
# Paper values to compare: SD 8.25 and n=2 gives 23.1, n=12 gives 9.4 (B4, V3-X12); SD 5.95 and n=2 gives 16.7 (V3-X3).
' "$SD" "$N"

# ---- figures, in a scratch copy so that paper/v3/figs of the export stays byte-identical to the manifest
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/paper/v3/evidence" "$TMP/paper/v3/figs" "$TMP/paper/v2/analysis"
cp -R "$ROOT/paper/v3/analysis" "$TMP/paper/v3/"
cp -R "$ROOT/paper/v3/evidence/data" "$TMP/paper/v3/evidence/"
cp -R "$ROOT/paper/v2/analysis/out" "$TMP/paper/v2/analysis/"
cd "$TMP"
run() {  # run <script>
  if command -v uv >/dev/null 2>&1; then
    uv run --offline --no-project --with matplotlib --with numpy python -I -B "$1" 2>/dev/null \
      || uv run --no-project --with matplotlib --with numpy python -I -B "$1"
  else
    python3 -I -B "$1"
  fi
}
run paper/v3/analysis/figs/fig_noise_draws.py
run paper/v3/analysis/figs/fig_discovery.py
mkdir -p "$OUT"
for f in fig_noise_draws fig_discovery; do cp paper/v3/figs/$f.pdf paper/v3/figs/$f.png "$OUT/"; done
for f in fig_noise_draws fig_discovery; do
  if cmp -s "$OUT/$f.png" "$ROOT/paper/v3/figs/$f.png"; then echo "$f.png: pixel-identical to the shipped figure"
  else echo "$f.png: regenerated; differs from the shipped file (matplotlib or font version), compare by eye"; fi
done
echo "figures written to $OUT"
