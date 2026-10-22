#!/usr/bin/env bash
# G1: grad-aware weights. One arm = P0 `exp` (run_20261018_044202, the `uniform` arm) with --agg-rate-type grad_aware; 2 h cap.
# Every process (trainers and aggregator) stays on GPUS; an arm starts only when all of GPUS are free.
# Matches the P0 `exp` config (paper_expts_fluxtune/configs/p0_exp_distilbert_agnews.yaml) except the rate type.
#   tmux new-window -t dg_flame -n g1q "bash nodes/run_node_g1_queue.sh 2>&1 | tee ../g1_queue_\$(date +%Y%m%d_%H%M).log"
# NODE_DRY_RUN=1 checks every arm's flags in seconds without waiting or launching.
set -u
. "$(dirname "$0")/_node_lib.sh"
export EXPT_GPU_ALLOW_PEER=1  # the launcher's check reads every GPU; a peer's jobs sit outside GPUS
# Starvation predicates target controller arms; a fixed pool and FwdLLM's var gate can trip them falsely.
export NODE_WATCH_ARGS="--trips-floor 0 --i-floor-frac 1.01"

GPUS=0,1,2,3
export FLAME_SLOT_GPUS=$GPUS  # else the aggregator takes an idle-looking GPU outside --gpu-ids (runner.py:322)
COMMON=(--mode sim --yes --clean --force --gpu-ids "$GPUS" --num-trainers 100
        --dataset agnews --partition-method niid_label_clients=100_alpha=1 --adapter-reduction-factor 16
        --server-update-audit --no-cos-ground-truth-audit --eval-every-n-commits 4 --max-runtime-s 86400)

FLUX=(--only fluxtune "${COMMON[@]}" --agg-goal 10 --c 30
      --probe-combine mean --perturbation-count 10 --server-step-rule trust_ratio --gate-safety-s 1.5
      --commit-gate fixed --pool-target 90 --agg-rate-type grad_aware
      --var-stopping-policy off --budget-stop-frac 0 --b-max-probe-every 0 --phi-stop halt
      --sim-wall-ceiling-h 2)


wait_gpus_free () {  # no compute process and < 1 GB used on every GPU in GPUS
  [ "${NODE_DRY_RUN:-0}" = "1" ] && return 0
  echo "=== waiting for GPUs $GPUS to be free  $(date -Is) ==="
  until [ -z "$(nvidia-smi --query-compute-apps=gpu_bus_id --format=csv,noheader -i "$GPUS")" ] \
        && nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPUS" \
           | awk '$1 > 1000 {busy=1} END {exit busy}'; do sleep 30; done
}


wait_gpus_free; node_run g1q "G1 grad_aware (2 h)" "${FLUX[@]}" --rho-schedule exp --t-res 300

echo "=== [g1q] done $(date -Is) ==="
