#!/usr/bin/env bash
# RoBERTa-L · Yelp-P `fluxtune` paper leg, no momentum (RUN_PLAN decision 15): bench, 4 GPUs, 30 workers.
# Stops at acc ≥ 0.82 ×2; the 30.5 h timeout kills (no final eval). Stop: kill -TERM -- -<pgid>
set -euo pipefail
cd /home/dgarg39/aish_test/flame/lib/python/examples
source /coc/scratch/dgarg/miniconda3/etc/profile.d/conda.sh && conda activate test_fwdllm
export FWDLLM_FD_SCALE_INVARIANT=1
GPUS=${GPUS:-4,5,6,7}
D=fwdllm/paper_rb_20261026b
setsid nohup python scripts/bench_pool.py --jobs $D/jobs.tsv --out $D/POOL --gpus "$GPUS" --gpus-per-leg 4 --per-gpu 1 \
  --timeout-s 109800 > $D/bench.log 2>&1 < /dev/null &
echo "launched pgid=$(ps -o pgid= -p $! | tr -d ' ') gpus=$GPUS log=$D/bench.log"
