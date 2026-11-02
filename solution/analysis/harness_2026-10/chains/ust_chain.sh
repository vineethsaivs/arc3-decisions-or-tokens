#!/bin/bash
# tool2 (ust) gate wave, predeclared in gate-predeclared-2026-10-31-franzen-l1a1y8sv2r3lf15ust.json. At most 2 GPU
# sessions: W1 now (usc W3 holds the other slot); W2 when usc W3 is fetched; W3 when W1 is fetched; probe when W2 is.
cd <repo>
export KAGGLE_CONFIG_DIR="$PWD/ARC-AGI-3-Kaggle-Starter/.kaggle"
K=solution/kaggle/generated/kernel-outputs
F="benchmark.json,serve.log,kernel.log,_p0.txt"
fetch() { rm -rf "$K/$2"; solution/kaggle/kapi.sh output "arc3-franzen-$1-a" "$K/$2" "$F" >/dev/null 2>&1; echo "$(date -u +%T) fetched $1 -> $2: $(ls $K/$2 2>/dev/null | wc -l) files"; }
(
  echo "$(date -u +%T) lane A: push ust W1"
  solution/kaggle/run_franzen_batch.sh l1a1y8sv2r3lf15ust
  fetch l1a1y8sv2r3lf15ust kaggle-lf15ust-1031-1
  echo "$(date -u +%T) lane A: push ust W3"
  solution/kaggle/run_franzen_batch.sh l1a1y8sv2r3lf15ustc
  fetch l1a1y8sv2r3lf15ustc kaggle-lf15ust-1031-3
  echo "$(date -u +%T) lane A done"
) &
(
  until grep -q "fetched l1a1y8sv2r3lf15uscc" ~/.cache/arc3/usc_chain.log 2>/dev/null; do sleep 60; done
  echo "$(date -u +%T) lane B: push ust W2"
  solution/kaggle/run_franzen_batch.sh l1a1y8sv2r3lf15ustb
  fetch l1a1y8sv2r3lf15ustb kaggle-lf15ust-1031-2
  echo "$(date -u +%T) lane B: push ust probe"
  solution/kaggle/run_franzen_batch.sh l1a1y8sv2r3kp12lf15ust
  fetch l1a1y8sv2r3kp12lf15ust franzen-l1a1y8sv2r3kp12lf15ust-v1
  echo "$(date -u +%T) lane B done"
) &
wait
echo "ust wave done"
