#!/bin/bash
# tool2 gate, resumed 11-01 after the session restart: W3 (ustc) and the probe together, then fetch both.
cd <repo>
export KAGGLE_CONFIG_DIR="$PWD/ARC-AGI-3-Kaggle-Starter/.kaggle"
K=solution/kaggle/generated/kernel-outputs
F="benchmark.json,serve.log,kernel.log,_p0.txt"
fetch() { solution/kaggle/kapi.sh output "arc3-franzen-$1-a" "$K/$2" "$F" >/dev/null 2>&1; echo "$(date -u +%T) fetched $1 -> $2: $(ls $K/$2 2>/dev/null | wc -l) files"; }
echo "$(date -u +%T) push ust W3 and probe"
solution/kaggle/run_franzen_batch.sh l1a1y8sv2r3lf15ustc l1a1y8sv2r3kp12lf15ust
fetch l1a1y8sv2r3lf15ustc kaggle-lf15ust-1031-3
fetch l1a1y8sv2r3kp12lf15ust franzen-l1a1y8sv2r3kp12lf15ust-v1
echo "ust wave done"
