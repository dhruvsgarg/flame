#!/usr/bin/env bash
# fwdllm setup smokes (RUN_PLAN §3.5): one run_sequential.sh --dry-run per leg -> leg YAMLs + bench jobs files.
# n 20 (shards 0-19 of niid_label_clients=100_alpha=1), c 10 / K 10 as at n 100, eval 1,000 rows stride 4, no final eval.
#   jobs_identity.tsv: V legs (25 commits), bench vs harness at the same seed (1234)
#   jobs_sweep.tsv:    W legs (100 commits, 2 h wall), lr x var screen
set -u
B="$(cd "$(dirname "$0")" && pwd)"
cd "$B/../expt_scripts"
source /coc/scratch/dgarg/miniconda3/etc/profile.d/conda.sh && conda activate test_fwdllm
export FWDLLM_FD_SCALE_INVARIANT=1 EXPT_GPU_ALLOW_PEER=1

common=(--dry-run --yes --force --allow-stale-profile --only fwdllm --mode sim --gpu-ids 0 --num-trainers 20
  --partition-method niid_label_clients=100_alpha=1 --fd-scale-invariant --train-batch-size 32 --min-initial-frac 0.9
  --send-timeout-s 3600 --server-update-audit --no-cos-ground-truth-audit --eval-max-samples 1000 --eval-resample
  --eval-every-n-commits 4 --delay-floor 4.0 --phi-stop halt --phi-stop-threshold 1e9 --max-runtime-s 1000000000)
roberta=(--dataset yelp-p --model-type roberta-large --peft-method bitfit --delay-divisor 0.169)
smollm=(--dataset yahoo --model-type llama --model-name HuggingFaceTB/SmolLM2-1.7B --peft-method lora --lora-r 4
  --adapter-reduction-factor 16 --delay-divisor 0.047)

leg () {  # leg <jobs file> <phase> <gpu_gb> <workers> <commits> <wall_s> <model> <lr> <var>
  local jobs=$1 phase=$2 gb=$3 w=$4 mc=$5 wall=$6 model=$7 lr=$8 var=$9 out y
  local -n m=$model
  out=$(./run_sequential.sh "${common[@]}" "${m[@]}" --learning-rate "$lr" --var-threshold "$var" \
        --max-commits "$mc" --sim-wall-ceiling-s "$wall" 2>&1)
  y=$(echo "$out" | sed -n 's/^--dry-run: generated cfgs in \([^ ]*\) .*/\1/p')
  y=$(ls "$y"/*.yaml | head -1)
  [ -f "$y" ] || { echo "!!! $phase: no yaml"; echo "$out" | tail -5; exit 1; }
  printf '%s\tfwdllm\tsyn_0\t%s\tgpu_gb=%s\tram_gb=24\tworkers=%s\n' "$phase" "$y" "$gb" "$w" >> "$jobs"
  echo "$phase -> $y"
  sleep 1  # smoke_logs dirs are per second
}

: > "$B/jobs_identity.tsv"; : > "$B/jobs_sweep.tsv"
leg "$B/jobs_identity.tsv" V1_rb_lr1e-2_v0.2 20 4 25 3600 roberta 0.01 0.2
leg "$B/jobs_identity.tsv" V2_sm_lr1e-3_v0.2 34 4 25 3600 smollm 0.001 0.2
for lr in 0.0001 0.0003 0.001 0.003; do leg "$B/jobs_sweep.tsv" "W_sm_lr${lr}_v0.2" 34 4 100 7200 smollm "$lr" 0.2; done
for v in 0.1 0.4; do leg "$B/jobs_sweep.tsv" "W_sm_lr0.001_v${v}" 34 4 100 7200 smollm 0.001 "$v"; done
for lr in 0.001 0.003 0.01 0.03; do leg "$B/jobs_sweep.tsv" "W_rb_lr${lr}_v0.2" 20 4 100 7200 roberta "$lr" 0.2; done
for v in 0.1 0.4; do leg "$B/jobs_sweep.tsv" "W_rb_lr0.01_v${v}" 20 4 100 7200 roberta 0.01 "$v"; done
wc -l "$B"/jobs_*.tsv
