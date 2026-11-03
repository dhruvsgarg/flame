#!/usr/bin/env bash
# Per-task A40 GPU cost for the D divisor (EXPERIMENTS §9): one fluxtune trainer alone on one GPU, sim mode (no sleeps),
# trainer-side recipe settings (P 10, mean, bs, model, LoRA), eval off. Lanes: DistilBERT bs 8 / bs 32, RoBERTa-L bs 32,
# SmolLM2-1.7B bs 32. Each task's GPU time is the trainer log's `gpu=` field. One Ctrl+C stops all.
# Usage: [GPUS="4 5 6 7"] [WALL_S=600] run.sh [--dry-run]
set -u
B="$(cd "$(dirname "$0")" && pwd)"
EX="$(cd "$B/.." && pwd)"
source /coc/scratch/dgarg/miniconda3/etc/profile.d/conda.sh && conda activate test_fwdllm
export FWDLLM_FD_SCALE_INVARIANT=1 FLAME_CONDA_ENV=test_fwdllm
DRY=${1:-}
WALL_S=${WALL_S:-600}
read -r -a GPUS <<< "${GPUS:-4 5 6 7}"

ONE=(--only fluxtune --mode sim --num-trainers 1 --c 1 --c-async 1 --agg-goal 1 --min-initial-trainers 1
     --fd-scale-invariant --probe-combine mean --perturbation-count 10 --eval-every-n-commits 1000000
     --max-runtime-s $(( WALL_S * 40 )) --sim-wall-ceiling-s "$WALL_S" --allow-stale-profile --no-cos-ground-truth-audit)
AGNEWS=(--dataset agnews --partition-method niid_label_clients=100_alpha=1)
YELP=(--dataset yelp-p --partition-method niid_label_clients=100_alpha=1 --model-type roberta-large --peft-method lora --lora-r 6)
YAHOO=(--dataset yahoo --partition-method niid_label_clients=100_alpha=1 --model-type llama
       --model-name HuggingFaceTB/SmolLM2-1.7B --peft-method lora --lora-r 4 --adapter-reduction-factor 16)

lane () {  # lane <name> <gpu> <cpus> <flags...>
  local name=$1 gpu=$2 cpus=$3 port; shift 3
  port="$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1])')"
  printf 'listener %s 127.0.0.1\nallow_anonymous true\npersistence false\nlog_dest file %s\n' \
    "$port" "$B/mosquitto_$name.log" > "$B/mosquitto_$name.conf"
  /usr/sbin/mosquitto -c "$B/mosquitto_$name.conf" >/dev/null 2>&1 &
  local broker=$!
  for _ in $(seq 50); do (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null && break; sleep 0.1; done
  echo "=== $name gpu=$gpu cpus=$cpus broker=$port $(date -Is) ==="
  FLAME_MQTT_BROKER="localhost:$port" FLAME_SLOT_GPUS="$gpu" FLAME_RUN_TAG="ftp_$name" FLAME_RUN_LABEL="ftp_$name" \
    taskset -c "$cpus" bash "$EX/expt_scripts/run_sequential.sh" "${ONE[@]}" "$@" --gpu-ids "$gpu" --yes $DRY \
    < /dev/null > "$B/$name.log" 2>&1
  echo "=== $name exit=$? $(date -Is) ==="
  kill "$broker" 2>/dev/null
}

stop () {  # trainers run in their own sessions, so also stop every process carrying a lane tag
  trap - INT TERM; echo "stopping"
  local p tagged=""
  for p in $(pgrep -u "$(id -u)"); do
    tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -q '^FLAME_RUN_TAG=ftp_' && tagged="$tagged $p"
  done
  kill -TERM $tagged $(jobs -p) 2>/dev/null; sleep 10; kill -KILL $tagged 2>/dev/null; exit 130
}
trap stop INT TERM

cd "$EX"
lane distil_bs8   "${GPUS[0]}" 64-79   "${AGNEWS[@]}" --train-batch-size 8 --force &  # wall-budget check: the ceiling ends these legs by design
[ -n "$DRY" ] || sleep 3  # run_sequential names its smoke_logs dir by the second
lane distil_bs32  "${GPUS[1]}" 80-95   "${AGNEWS[@]}" --train-batch-size 32 --force &
[ -n "$DRY" ] || sleep 3
lane roberta_bs32 "${GPUS[2]}" 96-111  "${YELP[@]}" --train-batch-size 32 &
[ -n "$DRY" ] || sleep 3
lane smollm_bs32  "${GPUS[3]}" 112-127 "${YAHOO[@]}" --train-batch-size 32 &
wait
echo "=== all done $(date -Is) ==="
