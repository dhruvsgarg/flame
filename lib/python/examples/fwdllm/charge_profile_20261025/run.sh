#!/usr/bin/env bash
# G0 charge-profile real legs, 30 min each, side by side on jayne: DistilBERT · AG News (GPUs 0-3), SmolLM2-1.7B · Yahoo (GPUs 4-7).
# §2 recipe (pool 50, agg goal 10, exp, uniform); SmolLM at s 2.5 / Φ_budget 4 (T_res 715). Fewer trainers than the paper.
# SmolLM D shortened (divisor 0.3, mean ~42 s) so commits happen in 30 min; the profiled ops don't depend on D.
# Usage: [WALL_S=1800] [MODELS="distil smollm"] run.sh [--dry-run]; then profile_sim_charges.py on each real run dir.
set -u
B="$(cd "$(dirname "$0")" && pwd)"
EX="$(cd "$B/.." && pwd)"
source /coc/scratch/dgarg/miniconda3/etc/profile.d/conda.sh && conda activate test_fwdllm
export FWDLLM_FD_SCALE_INVARIANT=1 FLAME_CONDA_ENV=test_fwdllm EXPT_GPU_ALLOW_PEER=1  # the GPU check is node-wide; lanes share the node
DRY=${1:-}
WALL_S=${WALL_S:-1800}
MODELS=${MODELS:-distil smollm}

FLUX=(--only fluxtune --agg-goal 10 --fd-scale-invariant --send-timeout-s 3600 --delay-floor 4.0
      --probe-combine mean --perturbation-count 10 --server-step-rule trust_ratio
      --commit-gate fixed --pool-target 50 --rho-schedule exp --agg-rate-type uniform --var-stopping-policy off
      --budget-stop-frac 0 --b-max-probe-every 0 --phi-stop halt --no-cos-ground-truth-audit
      --eval-max-samples 1000 --eval-resample --eval-every-n-commits 4 --max-runtime-s "$WALL_S")
DISTIL=(--dataset agnews --partition-method niid_label_clients=100_alpha=1 --num-trainers 30 --c 20 --c-async 20
        --train-batch-size 8 --delay-divisor 0.478 --rho0 0.07 --t-res 283)
SMOLLM=(--dataset yahoo --partition-method niid_label_clients=100_alpha=1 --num-trainers 16 --c 12 --c-async 12
        --model-type llama --model-name HuggingFaceTB/SmolLM2-1.7B --peft-method lora --lora-r 4 --adapter-reduction-factor 16
        --train-batch-size 32 --delay-divisor 0.3 --gate-safety-s 2.5 --t-res 715)

lane () {  # lane <model> <gpus> <cpus> <model flags...>
  local model=$1 gpus=$2 cpus=$3 port; shift 3
  port="$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1])')"
  printf 'listener %s 127.0.0.1\nallow_anonymous true\npersistence false\nlog_dest file %s\n' \
    "$port" "$B/mosquitto_${model}.log" > "$B/mosquitto_${model}.conf"
  /usr/sbin/mosquitto -c "$B/mosquitto_${model}.conf" >/dev/null 2>&1 &
  local broker=$!
  for _ in $(seq 50); do (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null && break; sleep 0.1; done
  echo "=== $model real gpus=$gpus cpus=$cpus broker=$port $(date -Is) ==="
  FLAME_MQTT_BROKER="localhost:$port" FLAME_SLOT_GPUS="$gpus" FLAME_RUN_TAG="g0_${model}_real" \
    FLAME_RUN_LABEL="g0_${model}" taskset -c "$cpus" bash "$EX/expt_scripts/run_sequential.sh" \
    --mode real "${FLUX[@]}" "$@" --gpu-ids "$gpus" --yes $DRY < /dev/null > "$B/${model}_real.log" 2>&1
  echo "=== $model exit=$? $(date -Is) ==="
  kill "$broker" 2>/dev/null
}

stop () {  # trainers run in their own sessions, so also stop every process carrying a lane tag
  trap - INT TERM; echo "stopping"
  local p tagged=""
  for p in $(pgrep -u "$(id -u)"); do
    tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -q '^FLAME_RUN_TAG=g0_' && tagged="$tagged $p"
  done
  kill -TERM $tagged $(jobs -p) 2>/dev/null; sleep 10; kill -KILL $tagged 2>/dev/null; exit 130
}
trap stop INT TERM

cd "$EX"
for m in $MODELS; do
  case $m in
    distil) lane distil 0,1,2,3 0-63 "${DISTIL[@]}" & ;;
    smollm) lane smollm 4,5,6,7 64-127 "${SMOLLM[@]}" & ;;
  esac
  [ -n "$DRY" ] || sleep 60  # lanes launched in the same second share one smoke_logs/<ts> dir
done
wait
echo "=== all done $(date -Is) ==="
