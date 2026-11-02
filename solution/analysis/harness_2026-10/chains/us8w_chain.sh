#!/bin/bash
# After the 13:01Z network blip: wait for us8b/us8c (already running), fetch, then the us8 probe. Writes the markers
# usc_chain.sh waits on ("push us8 probe", "us8 wave done") to its own log, which usc_chain.sh reads.
cd <repo>
export KAGGLE_CONFIG_DIR="$PWD/ARC-AGI-3-Kaggle-Starter/.kaggle"
K=solution/kaggle/generated/kernel-outputs
F="benchmark.json,serve.log,kernel.log,_p0.txt"
fetch() { rm -rf "$K/$2"; solution/kaggle/kapi.sh output "arc3-franzen-$1-a" "$K/$2" "$F" >/dev/null 2>&1; echo "$(date -u +%T) fetched $1 -> $2: $(ls $K/$2 2>/dev/null | wc -l) files"; }
st() { solution/kaggle/kapi.sh status "arc3-franzen-$1-a" 2>/dev/null | python3 -c "import json,sys;print(json.load(sys.stdin).get('status') or 'unknown')" 2>/dev/null || echo unknown; }
while :; do s="$(st l1a1y8sv2r3lf15us8b) $(st l1a1y8sv2r3lf15us8c)"; case "$s" in *running*|*queued*|*unknown*) sleep 300 ;; *) echo "$(date -u +%T) us8b/us8c: $s"; break ;; esac; done
fetch l1a1y8sv2r3lf15us8b kaggle-lf15us8-1031-2
fetch l1a1y8sv2r3lf15us8c kaggle-lf15us8-1031-3
echo "$(date -u +%T) us8 W done"
echo "$(date -u +%T) push us8 probe"
solution/kaggle/run_franzen_batch.sh l1a1y8sv2r3kp12lf15us8
fetch l1a1y8sv2r3kp12lf15us8 franzen-l1a1y8sv2r3kp12lf15us8-v1
echo "us8 wave done"
