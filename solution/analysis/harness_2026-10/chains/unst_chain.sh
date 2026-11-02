#!/bin/bash
# after wave 1: fetch both W, push the 3rd W and the probe, fetch them when done
cd <repo>
K=solution/kaggle/generated/kernel-outputs
st() { solution/kaggle/kapi.sh status "$1" 2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin).get("status","?"))' 2>/dev/null; }
until s1=$(st arc3-franzen-l1a1y8sv2r3lf15unst-a); s2=$(st arc3-franzen-l1a1y8sv2r3lf15unstb-a); case "$s1$s2" in *running*|*queued*|*\?*) false;; *) true;; esac; do sleep 240; done
echo "$(date -u +%T) wave1 $s1 $s2"
solution/kaggle/kapi.sh output arc3-franzen-l1a1y8sv2r3lf15unst-a $K/kaggle-lf15unst-1030-1 "benchmark.json,serve.log,kernel.log,_p0.txt" >/dev/null 2>&1
solution/kaggle/kapi.sh output arc3-franzen-l1a1y8sv2r3lf15unstb-a $K/kaggle-lf15unst-1030-2 "benchmark.json,serve.log,kernel.log,_p0.txt" >/dev/null 2>&1
echo "$(date -u +%T) fetched $(ls $K/kaggle-lf15unst-1030-1 | wc -l) and $(ls $K/kaggle-lf15unst-1030-2 | wc -l) files"
solution/kaggle/run_franzen_batch.sh l1a1y8sv2r3lf15unstc l1a1y8sv2r3kp12lf15unst
solution/kaggle/kapi.sh output arc3-franzen-l1a1y8sv2r3lf15unstc-a $K/kaggle-lf15unst-1030-3 "benchmark.json,serve.log,kernel.log,_p0.txt" >/dev/null 2>&1
solution/kaggle/kapi.sh output arc3-franzen-l1a1y8sv2r3kp12lf15unst-a $K/franzen-l1a1y8sv2r3kp12lf15unst-v1 "benchmark.json,serve.log,kernel.log,_p0.txt" >/dev/null 2>&1
echo "$(date -u +%T) fetched W3 $(ls $K/kaggle-lf15unst-1030-3 | wc -l) files, probe $(ls $K/franzen-l1a1y8sv2r3kp12lf15unst-v1 | wc -l) files"
