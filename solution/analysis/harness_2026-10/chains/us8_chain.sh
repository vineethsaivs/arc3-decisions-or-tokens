#!/bin/bash
# seg8 wave on 10-31: after unstick2's W3 and probe are fetched, pick us8 (unstick2 passed) or s8 (killed),
# W1+W2 at 05:15Z, then W3 + probe; fetch each. Gate: gate-predeclared-2026-10-31-franzen-l1a1y8sv2r3lf15us8.json
cd <repo>
K=solution/kaggle/generated/kernel-outputs
F="benchmark.json,serve.log,kernel.log,_p0.txt"
S=<scratch>/fb519d0b-5ca4-49bf-b4d4-4ffb21661429/scratchpad/unst
until grep -q "fetched W3" ~/.cache/arc3/uns2_chain.log 2>/dev/null; do sleep 60; done
v=$(python3 $S/uns2_gate.py 2>&1 | grep "^VERDICT" | awk '{print $2}')
P=$K/franzen-l1a1y8sv2r3kp12lf15uns2-v1
tb=$(cat $P/kernel.log $P/serve.log 2>/dev/null | grep -c Traceback)
[ -s $P/kernel.log ] || tb=missing
if [ "$v" = PASS ] && [ "$tb" = 0 ]; then arm=us8; else arm=s8; fi
echo "$(date -u +%T) unstick2 verdict $v, probe tracebacks $tb: arm $arm"
until [ "$(date -u +%Y%m%d%H%M)" -ge 202610310515 ]; do sleep 60; done
echo "$(date -u +%T) push W1 W2 ($arm)"
solution/kaggle/run_franzen_batch.sh l1a1y8sv2r3lf15$arm l1a1y8sv2r3lf15${arm}b
solution/kaggle/kapi.sh output arc3-franzen-l1a1y8sv2r3lf15$arm-a $K/kaggle-lf15$arm-1031-1 "$F" >/dev/null 2>&1
solution/kaggle/kapi.sh output arc3-franzen-l1a1y8sv2r3lf15${arm}b-a $K/kaggle-lf15$arm-1031-2 "$F" >/dev/null 2>&1
echo "$(date -u +%T) wave1 fetched $(ls $K/kaggle-lf15$arm-1031-1 | wc -l) and $(ls $K/kaggle-lf15$arm-1031-2 | wc -l) files"
echo "$(date -u +%T) push W3 probe ($arm)"
solution/kaggle/run_franzen_batch.sh l1a1y8sv2r3lf15${arm}c l1a1y8sv2r3kp12lf15$arm
solution/kaggle/kapi.sh output arc3-franzen-l1a1y8sv2r3lf15${arm}c-a $K/kaggle-lf15$arm-1031-3 "$F" >/dev/null 2>&1
solution/kaggle/kapi.sh output arc3-franzen-l1a1y8sv2r3kp12lf15$arm-a $K/franzen-l1a1y8sv2r3kp12lf15$arm-v1 "$F" >/dev/null 2>&1
echo "$(date -u +%T) fetched W3 $(ls $K/kaggle-lf15$arm-1031-3 | wc -l) files, probe $(ls $K/franzen-l1a1y8sv2r3kp12lf15$arm-v1 | wc -l) files"
