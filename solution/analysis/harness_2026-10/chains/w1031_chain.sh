#!/bin/bash
# Same predeclared runs, earlier: unstick2's probe now (fits the 1.5 h left), then at the 10-31 refresh unstick2 W3
# with seg8 (us8) W1; us8 W2 + W3 at 05:15Z (after the scored run's first hours), then the us8 probe.
cd <repo>
K=solution/kaggle/generated/kernel-outputs
F="benchmark.json,serve.log,kernel.log,_p0.txt"
S=<scratch>/fb519d0b-5ca4-49bf-b4d4-4ffb21661429/scratchpad/unst
fetch() { solution/kaggle/kapi.sh output "arc3-franzen-$1-a" "$K/$2" "$F" >/dev/null 2>&1; echo "$(date -u +%T) fetched $1 -> $2: $(ls $K/$2 2>/dev/null | wc -l) files"; }
echo "$(date -u +%T) push unstick2 probe"
solution/kaggle/run_franzen_batch.sh l1a1y8sv2r3kp12lf15uns2
fetch l1a1y8sv2r3kp12lf15uns2 franzen-l1a1y8sv2r3kp12lf15uns2-v1
echo "probe fetched"
until [ "$(date -u +%Y%m%d%H%M)" -ge 202610310001 ]; do sleep 60; done
echo "$(date -u +%T) push unstick2 W3 and us8 W1"
solution/kaggle/run_franzen_batch.sh l1a1y8sv2r3lf15uns2c l1a1y8sv2r3lf15us8
fetch l1a1y8sv2r3lf15uns2c kaggle-lf15uns2-1030-3
fetch l1a1y8sv2r3lf15us8 kaggle-lf15us8-1031-1
echo "fetched W3"
v=$(python3 $S/uns2_gate.py 2>&1 | grep "^VERDICT" | awk '{print $2}')
echo "$(date -u +%T) unstick2 verdict $v"
[ "$v" = PASS ] || { echo "unstick2 not PASS: stopping before us8 W2 for a manual decision"; exit 0; }
until [ "$(date -u +%Y%m%d%H%M)" -ge 202610310515 ]; do sleep 60; done
echo "$(date -u +%T) push us8 W2 W3"
solution/kaggle/run_franzen_batch.sh l1a1y8sv2r3lf15us8b l1a1y8sv2r3lf15us8c
fetch l1a1y8sv2r3lf15us8b kaggle-lf15us8-1031-2
fetch l1a1y8sv2r3lf15us8c kaggle-lf15us8-1031-3
echo "$(date -u +%T) push us8 probe"
solution/kaggle/run_franzen_batch.sh l1a1y8sv2r3kp12lf15us8
fetch l1a1y8sv2r3kp12lf15us8 franzen-l1a1y8sv2r3kp12lf15us8-v1
echo "us8 wave done"
