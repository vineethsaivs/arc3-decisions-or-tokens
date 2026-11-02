#!/bin/sh
# Build both copies with tectonic (a single binary; no TeX Live needed).
#   TECTONIC=/path/to/tectonic sh build.sh        (default: tectonic on PATH)
# Output: main.pdf (public) and main_anon.pdf (anonymised). Figures are read from ../figs.
cd "$(dirname "$0")" || exit 1
for f in main main_anon; do
  ${TECTONIC:-tectonic} -X compile --keep-logs "$f.tex" 2>&1 | grep -v "^note: downloading" | tail -25
done
