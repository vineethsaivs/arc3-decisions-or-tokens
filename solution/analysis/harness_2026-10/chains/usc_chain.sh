#!/bin/bash
# scroll (usc) gate wave, predeclared in gate-predeclared-2026-10-31-franzen-l1a1y8sv2r3lf15usc.json. Two lanes keep at
# most 2 GPU sessions: lane A starts W1 once us8's W2/W3 are done (beside us8's probe), then W3; lane B starts W2 once
# us8's probe is done, then the probe.
cd <repo>
K=solution/kaggle/generated/kernel-outputs
F="benchmark.json,serve.log,kernel.log,_p0.txt"
L=~/.cache/arc3/us8w_chain.log
fetch() { rm -rf "$K/$2"; solution/kaggle/kapi.sh output "arc3-franzen-$1-a" "$K/$2" "$F" >/dev/null 2>&1; echo "$(date -u +%T) fetched $1 -> $2: $(ls $K/$2 2>/dev/null | wc -l) files"; }
(
  until grep -q "push us8 probe" $L; do sleep 60; done
  echo "$(date -u +%T) lane A: push usc W1"
  solution/kaggle/run_franzen_batch.sh l1a1y8sv2r3lf15usc
  fetch l1a1y8sv2r3lf15usc kaggle-lf15usc-1031-1
  echo "$(date -u +%T) lane A: push usc W3"
  solution/kaggle/run_franzen_batch.sh l1a1y8sv2r3lf15uscc
  fetch l1a1y8sv2r3lf15uscc kaggle-lf15usc-1031-3
  echo "$(date -u +%T) lane A done"
) &
(
  until grep -q "us8 wave done" $L; do sleep 60; done
  echo "$(date -u +%T) lane B: push usc W2"
  solution/kaggle/run_franzen_batch.sh l1a1y8sv2r3lf15uscb
  fetch l1a1y8sv2r3lf15uscb kaggle-lf15usc-1031-2
  echo "$(date -u +%T) lane B: push usc probe"
  solution/kaggle/run_franzen_batch.sh l1a1y8sv2r3kp12lf15usc
  fetch l1a1y8sv2r3kp12lf15usc franzen-l1a1y8sv2r3kp12lf15usc-v1
  echo "$(date -u +%T) lane B done"
) &
wait
echo "usc wave done"
