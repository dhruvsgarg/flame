#!/usr/bin/env bash
# Lane A (GPUs 0-3): V1/V2 bench legs + harness twins. Lane B (GPUs 4-7): W sweep on the bench.
#   tmux new-window -t dg_flame -n fwdsm "bash fwdllm/fwd_smoke_20261025/launch.sh 2>&1 | tee fwdllm/fwd_smoke_20261025/launch.log"
set -u
B="$(cd "$(dirname "$0")" && pwd)"
source /coc/scratch/dgarg/miniconda3/etc/profile.d/conda.sh && conda activate test_fwdllm
export FWDLLM_FD_SCALE_INVARIANT=1 EXPT_GPU_ALLOW_PEER=1
POOL="python $B/../../scripts/bench_pool.py"

lane_a () {
  $POOL --jobs "$B/jobs_identity.tsv" --out "$B/bench_identity" --seeds 1 --gpus 2,3 --per-gpu 1 \
    --timeout-s 5400 > "$B/pool_identity.log" 2>&1 &
  bash "$B/run_identity_harness.sh" V1 0,1 0-31,64-95                  # RoBERTa: 20 trainers on 2 GPUs
  wait
  bash "$B/run_identity_harness.sh" V2 0,1,2,3 0-63,64-127             # SmolLM: 20 trainers + aggregator on 4 GPUs
}
lane_b () {
  $POOL --jobs "$B/jobs_sweep.tsv" --out "$B/bench_sweep" --seeds 1 --gpus 4,5,6,7 --per-gpu 2 \
    --timeout-s 10800 > "$B/pool_sweep.log" 2>&1
}
echo "=== start $(date -Is) ==="
lane_a & lane_b & wait
echo "=== done $(date -Is) ==="
