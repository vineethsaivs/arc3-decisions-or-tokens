#!/bin/bash
# verify (ustv) gate wave, predeclared in gate-predeclared-2026-11-01-franzen-l1a1y8sv2r3lf15ustv.json. At most 2 GPU
# sessions: lane A: W1 now, then the probe, then W3 only if quota shows >= 2.6 h free (decided without reading any run);
# lane B: W2 when tool2's W3 (ustc) is complete.
cd <repo>
export KAGGLE_CONFIG_DIR="$PWD/ARC-AGI-3-Kaggle-Starter/.kaggle"
K=solution/kaggle/generated/kernel-outputs
F="benchmark.json,serve.log,kernel.log,_p0.txt"
fetch() { solution/kaggle/kapi.sh output "arc3-franzen-$1-a" "$K/$2" "$F" >/dev/null 2>&1; echo "$(date -u +%T) fetched $1 -> $2: $(ls $K/$2 2>/dev/null | wc -l) files"; }
st() { solution/kaggle/kapi.sh status "arc3-franzen-$1-a" 2>/dev/null | python3 -c "import json,sys;print(json.load(sys.stdin).get('status') or 'unknown')" 2>/dev/null || echo unknown; }
(
  echo "$(date -u +%T) lane A: push ustv W1"
  solution/kaggle/run_franzen_batch.sh l1a1y8sv2r3lf15ustv
  fetch l1a1y8sv2r3lf15ustv kaggle-lf15ustv-1101-1
  echo "$(date -u +%T) lane A: push ustv probe"
  solution/kaggle/run_franzen_batch.sh l1a1y8sv2r3kp12lf15ustv
  fetch l1a1y8sv2r3kp12lf15ustv franzen-l1a1y8sv2r3kp12lf15ustv-v1
  free=$(solution/kaggle/kapi.sh quota | head -1 | sed -E 's/.*free (-?[0-9.]+)h.*/\1/')
  echo "$(date -u +%T) lane A: quota free ${free} h"
  if python3 -c "import sys; sys.exit(0 if float('$free') >= 2.6 else 1)" 2>/dev/null; then
    echo "$(date -u +%T) lane A: push ustv W3"
    solution/kaggle/run_franzen_batch.sh l1a1y8sv2r3lf15ustvc
    fetch l1a1y8sv2r3lf15ustvc kaggle-lf15ustv-1101-3
  else
    echo "$(date -u +%T) lane A: W3 skipped (quota)"
  fi
  echo "$(date -u +%T) lane A done"
) &
(
  while :; do s=$(st l1a1y8sv2r3lf15ustc); case "$s" in running|queued|unknown) sleep 120 ;; *) break ;; esac; done
  echo "$(date -u +%T) lane B: ustc is $s; push ustv W2"
  solution/kaggle/run_franzen_batch.sh l1a1y8sv2r3lf15ustvb
  fetch l1a1y8sv2r3lf15ustvb kaggle-lf15ustv-1101-2
  echo "$(date -u +%T) lane B done"
) &
wait
echo "ustv wave done"
