#!/usr/bin/env bash
# RoBERTa-L · Yelp-P `fwdllm` lr test on the rb60 config: 2 legs (lr 3e-3, 1e-3), 2 GPUs each, n 100, C = K = 10, stop at acc >= 0.83 x2 evals (no commit cap).
set -euo pipefail
cd /home/dgarg39/aish_test/flame/lib/python/examples
source /coc/scratch/dgarg/miniconda3/etc/profile.d/conda.sh && conda activate test_fwdllm
export FWDLLM_FD_SCALE_INVARIANT=1
GPUS=${GPUS:-4,5,6,7}
D=fwdllm/rb_lr_20261027
setsid nohup python scripts/bench_pool.py --jobs $D/jobs.tsv --out $D/POOL --gpus "$GPUS" --gpus-per-leg 2 --per-gpu 1 \
  --timeout-s 432000 > $D/bench.log 2>&1 < /dev/null &
echo "launched pgid=$(ps -o pgid= -p $! | tr -d ' ') gpus=$GPUS log=$D/bench.log"
