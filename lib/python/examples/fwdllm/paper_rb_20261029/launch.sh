#!/usr/bin/env bash
# RoBERTa-L · Yelp-P `fwdllm` (RUN_PLAN run 3): config = rb60 run `experiments/run_20261023_160644_rb60_*` (LoRA r 6, lr 1e-2, var 0.3, 380 commits / 480 vclock h; decision 19).
# Run 3 reuses rb60 (harness, 8 GPUs, `expt_scripts/nodes/run_node_roberta_yelp_to60_queue.sh`); this bench launcher is only for a rerun. Stop: kill -TERM -- -<pgid>
set -euo pipefail
cd /home/dgarg39/aish_test/flame/lib/python/examples
source /coc/scratch/dgarg/miniconda3/etc/profile.d/conda.sh && conda activate test_fwdllm
export FWDLLM_FD_SCALE_INVARIANT=1
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
D=fwdllm/paper_rb_20261029
setsid nohup python scripts/bench_pool.py --jobs $D/jobs.tsv --out $D/POOL --gpus "$GPUS" --gpus-per-leg 8 --per-gpu 1 \
  --timeout-s 109800 > $D/bench.log 2>&1 < /dev/null &
echo "launched pgid=$(ps -o pgid= -p $! | tr -d ' ') gpus=$GPUS log=$D/bench.log"
