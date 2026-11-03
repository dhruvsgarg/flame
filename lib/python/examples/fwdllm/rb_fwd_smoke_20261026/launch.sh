#!/usr/bin/env bash
# RoBERTa-L · Yelp-P `fwdllm` lr / var smoke: 4 legs, 1 GPU each, n 20, C = K = 10, 120 commits or 2.5 h.
set -euo pipefail
cd /home/dgarg39/aish_test/flame/lib/python/examples
source /coc/scratch/dgarg/miniconda3/etc/profile.d/conda.sh && conda activate test_fwdllm
export FWDLLM_FD_SCALE_INVARIANT=1
GPUS=${GPUS:-4,5,6,7}
D=fwdllm/rb_fwd_smoke_20261026
setsid nohup python scripts/bench_pool.py --jobs $D/jobs.tsv --out $D/POOL --gpus "$GPUS" --gpus-per-leg 1 --per-gpu 1 \
  --timeout-s 10800 > $D/bench.log 2>&1 < /dev/null &
echo "launched pgid=$(ps -o pgid= -p $! | tr -d ' ') gpus=$GPUS log=$D/bench.log"
