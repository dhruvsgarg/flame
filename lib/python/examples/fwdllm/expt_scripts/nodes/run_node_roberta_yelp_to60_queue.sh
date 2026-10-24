#!/usr/bin/env bash
# RoBERTa-L · Yelp-P (RUN_PLAN §2 roberta_large_yelp cell), FwdLLM then FluxTune, each stopped at TARGET acc (0.60).
# One lane on all 8 GPUs. n starts at NUM_TRAINERS (100); an OOM / no-commit arm reruns at FALLBACK_TRAINERS (75),
# and an arm already done at the larger n is rerun so both arms share n. Plots render at the end (plot_roberta_yelp60.py).
#   tmux new-window -t dg_flame -n rb60 "bash nodes/run_node_roberta_yelp_to60_queue.sh 2>&1 | tee ../rb60_queue_\$(date +%Y%m%d_%H%M).log"
# Env: TARGET (0.60), HOURS (12, wall cap per arm), MAX_COMMITS (380), STRIDE (4), T_RES (71 = Φ* 2 at ρ0 0.14),
#   FWD_LR (0.01), FLUX_D / FWD_D (delay divisors, measured: paper_expts_fluxtune/D_CALIBRATION.md), ARMS ("fwdllm fluxtune"), NODE_DRY_RUN=1 (flags only).
#   STOP_AT_TARGET (1): 0 drops the target-acc stop, so a paper / S run ends on the §2 stop rule (stall, GL, commit cap, wall).
set -u
. "$(dirname "$0")/_node_lib.sh"
source /coc/scratch/dgarg/miniconda3/etc/profile.d/conda.sh && conda activate test_fwdllm
export FWDLLM_FD_SCALE_INVARIANT=1 EXPT_GPU_ALLOW_PEER=1
export NODE_WATCH_ARGS="--trips-floor 0 --i-floor-frac 1.01"
export NODE_ABORT_ON_FAULT=0  # a failed arm is retried at the fallback n, not a node abort

GPUS=0,1,2,3,4,5,6,7
export FLAME_SLOT_GPUS=$GPUS
TARGET=${TARGET:-0.60}
STOP_AT_TARGET=${STOP_AT_TARGET:-1}
HOURS=${HOURS:-12}
MAX_COMMITS=${MAX_COMMITS:-380}
STRIDE=${STRIDE:-4}
T_RES=${T_RES:-71}
FWD_LR=${FWD_LR:-0.01}
FLUX_D=${FLUX_D:-0.0169}
FWD_D=${FWD_D:-0.169}
NUM_TRAINERS=${NUM_TRAINERS:-100}
FALLBACK_TRAINERS=${FALLBACK_TRAINERS:-75}
ARMS=${ARMS:-fwdllm fluxtune}
# Final full eval (38,000 rows) runs after the wall cap: ~5 min est.; 1 h margin before the hard kill.
export NODE_TIMEOUT_S=$(( HOURS * 3600 + 3600 ))

common () {  # common <n>: cell flags shared by both arms
  local n=$1
  printf '%s\n' --mode sim --yes --clean --force --allow-stale-profile --gpu-ids "$GPUS" --num-trainers "$n" \
    --dataset yelp-p --partition-method niid_label_clients=100_alpha=1 \
    --model-type roberta-large --peft-method lora --lora-r 6 --fd-scale-invariant \
    --train-batch-size 32 --min-initial-frac 0.9 --send-timeout-s 3600 \
    --server-update-audit --no-cos-ground-truth-audit \
    --eval-max-samples 3800 --eval-resample --eval-every-n-commits "$STRIDE" --final-full-eval \
    --saturation-stop --sat-stall-frac 0.003 --max-commits "$MAX_COMMITS" \
    --max-runtime-s 1000000000 --sim-wall-ceiling-s $(( HOURS * 3600 ))
  [ "$STOP_AT_TARGET" = 1 ] && printf '%s\n' --target-acc "$TARGET" --converge-window 1
  return 0
}

arm_args () {  # arm_args <arm> <n>
  local arm=$1 n=$2 c=$(( (3 * $2 + 5) / 10 ))  # C = 0.3·n, K = C/3 (30/10 at n 100, 23/8 at n 75)
  mapfile -t base < <(common "$n")
  case "$arm" in
    fwdllm)   printf '%s\n' --only fwdllm "${base[@]}" --learning-rate "$FWD_LR" --delay-divisor "$FWD_D" --delay-floor 4.0 \
                --phi-stop halt --phi-stop-threshold 1e9 ;;  # phi_stop off disables the stall/GL detector
    fluxtune) printf '%s\n' --only fluxtune "${base[@]}" --agg-goal $(( (c + 1) / 3 )) --c "$c" \
                --delay-divisor "$FLUX_D" --delay-floor 4.0 \
                --probe-combine mean --perturbation-count 10 --server-step-rule trust_ratio --rho0 0.14 \
                --commit-gate fixed --pool-target 50 --rho-schedule exp --t-res "$T_RES" --agg-rate-type uniform \
                --var-stopping-policy off --budget-stop-frac 0 --b-max-probe-every 0 --phi-stop halt ;;
  esac
}

arm_status () {  # failed | oom | reached | timeout | stopped
  local run="$1"
  grep -qs 'CUDA out of memory\|OutOfMemoryError' "$run"/*trainers.log "$run"/*aggregator.log && { echo oom; return; }
  [ "${2:-0}" != "0" ] && { echo failed; return; }
  grep -qs '\[SIM_WALL_CEILING\]' "$run"/*aggregator.log && { echo timeout; return; }
  grep -hos '"test-accuracy": [0-9.]*' "$run"/telemetry/aggregator_*.jsonl \
    | awk -v t="$TARGET" '$2 >= t {r = 1} END {print r ? "reached" : "stopped"}'
}

wait_gpus_free () {
  [ "${NODE_DRY_RUN:-0}" = "1" ] && return 0
  echo "=== waiting for GPUs $GPUS to be free  $(date -Is) ==="
  until [ -z "$(nvidia-smi --query-compute-apps=gpu_bus_id --format=csv,noheader -i "$GPUS")" ] \
        && nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPUS" \
           | awk '$1 > 1000 {busy=1} END {exit busy}'; do sleep 30; done
}

QDIR="$EXPT_DIR/rb60_$(date +%Y%m%d_%H%M%S)"; mkdir -p "$QDIR"
echo "=== [rb60] arms '$ARMS', n $NUM_TRAINERS (fallback $FALLBACK_TRAINERS), target $TARGET, cap ${HOURS} h; queue dir $QDIR  $(date -Is) ==="

run_arm () {  # run_arm <arm> <n>; sets ST, appends runs.tsv (no $(...): node_run needs its own process group)
  local arm=$1 n=$2 args rc run
  mapfile -t args < <(arm_args "$arm" "$n")
  export FLAME_RUN_LABEL="rb60_${arm}_n${n}"
  wait_gpus_free
  node_run "rb60" "$arm · n $n" "${args[@]}"; rc=$?
  [ "${NODE_DRY_RUN:-0}" = "1" ] && { ST=dry; return; }
  run="$(ls -1dt "$RUNS"/run_*_"$FLAME_RUN_LABEL"_* 2>/dev/null | head -1)"
  ST=$( [ -n "$run" ] && arm_status "$run" "$rc" || echo failed )
  printf '%s\t%s\t%s\t%s\n' "$arm" "$n" "$(basename "${run:-none}")" "$ST" >> "$QDIR/runs.tsv"
}

n=$NUM_TRAINERS
done_arms=()
for arm in $ARMS; do
  run_arm "$arm" "$n"
  if { [ "$ST" = oom ] || [ "$ST" = failed ]; } && [ "$n" != "$FALLBACK_TRAINERS" ]; then
    echo "!!! [rb60] $arm at n $n: $ST -- dropping to n $FALLBACK_TRAINERS"
    n=$FALLBACK_TRAINERS
    for prev in "${done_arms[@]}"; do run_arm "$prev" "$n"; done  # keep n equal across arms
    run_arm "$arm" "$n"
  fi
  echo "--- [rb60] $arm n $n: $ST"
  done_arms+=("$arm")
done

[ "${NODE_DRY_RUN:-0}" = "1" ] && exit 0
echo "=== [rb60] arms done $(date -Is); rendering plots ==="
python "$EXPT_DIR/expt_scripts/plot_roberta_yelp60.py" "$QDIR" --target "$TARGET" --n "$n"
echo "=== [rb60] done $(date -Is); arm -> run dir: $QDIR/runs.tsv ==="
