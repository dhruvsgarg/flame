#!/usr/bin/env bash
# SmolLM2-1.7B · Yahoo, near-IID niid_label_clients=100_alpha=100 (RUN_PLAN decision 18): fluxtune + fwdllm, LoRA r 2, bench, 4 GPUs per leg (0-3, 4-7), 24 workers each.
# Stops at acc >= 0.65 x2 (final full eval); the 15 h timeout kills (no final eval). Stop: kill -TERM -- -<pgid>
set -euo pipefail
cd /home/dgarg39/aish_test/flame/lib/python/examples
source /coc/scratch/dgarg/miniconda3/etc/profile.d/conda.sh && conda activate test_fwdllm
export FWDLLM_FD_SCALE_INVARIANT=1
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
D=fwdllm/paper_su_20261029
setsid nohup python scripts/bench_pool.py --jobs $D/jobs.tsv --out $D/POOL --gpus "$GPUS" --gpus-per-leg 4 --per-gpu 1 \
  --timeout-s 54000 > $D/bench.log 2>&1 < /dev/null &
echo "launched pgid=$(ps -o pgid= -p $! | tr -d ' ') gpus=$GPUS log=$D/bench.log"
