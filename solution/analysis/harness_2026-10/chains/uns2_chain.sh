#!/bin/bash
# unstick2 (fixed) wave: W1+W2 at 06:00Z 10-30; W3 + probe after the 10-31 00:00Z quota refresh; fetch each
cd <repo>
K=solution/kaggle/generated/kernel-outputs
F="benchmark.json,serve.log,kernel.log,_p0.txt"
until [ "$(date -u +%Y%m%d%H%M)" -ge 202610300500 ]; do sleep 60; done
echo "$(date -u +%T) push W1 W2"
solution/kaggle/run_franzen_batch.sh l1a1y8sv2r3lf15uns2 l1a1y8sv2r3lf15uns2b
solution/kaggle/kapi.sh output arc3-franzen-l1a1y8sv2r3lf15uns2-a $K/kaggle-lf15uns2-1030-1 "$F" >/dev/null 2>&1
solution/kaggle/kapi.sh output arc3-franzen-l1a1y8sv2r3lf15uns2b-a $K/kaggle-lf15uns2-1030-2 "$F" >/dev/null 2>&1
echo "$(date -u +%T) wave1 fetched $(ls $K/kaggle-lf15uns2-1030-1 | wc -l) and $(ls $K/kaggle-lf15uns2-1030-2 | wc -l) files"
until [ "$(date -u +%Y%m%d%H%M)" -ge 202610310001 ]; do sleep 60; done
echo "$(date -u +%T) push W3 probe"
solution/kaggle/run_franzen_batch.sh l1a1y8sv2r3lf15uns2c l1a1y8sv2r3kp12lf15uns2
solution/kaggle/kapi.sh output arc3-franzen-l1a1y8sv2r3lf15uns2c-a $K/kaggle-lf15uns2-1030-3 "$F" >/dev/null 2>&1
solution/kaggle/kapi.sh output arc3-franzen-l1a1y8sv2r3kp12lf15uns2-a $K/franzen-l1a1y8sv2r3kp12lf15uns2-v1 "$F" >/dev/null 2>&1
echo "$(date -u +%T) fetched W3 $(ls $K/kaggle-lf15uns2-1030-3 | wc -l) files, probe $(ls $K/franzen-l1a1y8sv2r3kp12lf15uns2-v1 | wc -l) files"
