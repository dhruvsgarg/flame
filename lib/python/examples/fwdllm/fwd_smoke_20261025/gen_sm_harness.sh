#!/usr/bin/env bash
# W_sm legs on the harness (bench 1.7B legs crash at the first dispatch, §3.5): n 10, c 10 / K 10, else as gen.sh.
set -u
B="$(cd "$(dirname "$0")" && pwd)"
cd "$B/../expt_scripts"
source /coc/scratch/dgarg/miniconda3/etc/profile.d/conda.sh && conda activate test_fwdllm
export FWDLLM_FD_SCALE_INVARIANT=1 EXPT_GPU_ALLOW_PEER=1
args=(--dry-run --yes --force --allow-stale-profile --only fwdllm --mode sim --gpu-ids 0 --num-trainers 10
  --partition-method niid_label_clients=100_alpha=1 --fd-scale-invariant --train-batch-size 32 --min-initial-frac 1.0
  --send-timeout-s 3600 --server-update-audit --no-cos-ground-truth-audit --eval-max-samples 1000 --eval-resample
  --eval-every-n-commits 4 --delay-floor 4.0 --phi-stop halt --phi-stop-threshold 1e9 --max-runtime-s 1000000000
  --dataset yahoo --model-type llama --model-name HuggingFaceTB/SmolLM2-1.7B --peft-method lora --lora-r 4
  --adapter-reduction-factor 16 --delay-divisor 0.047 --max-commits 100 --sim-wall-ceiling-s 7200)
: > "$B/jobs_sm_harness.tsv"
for lv in 0.0001:0.2 0.0003:0.2 0.001:0.2 0.003:0.2 0.001:0.1 0.001:0.4; do
  lr=${lv%:*} v=${lv#*:}
  out=$(./run_sequential.sh "${args[@]}" --learning-rate "$lr" --var-threshold "$v" 2>&1)
  d=$(echo "$out" | sed -n 's/^--dry-run: generated cfgs in \([^ ]*\) .*/\1/p')
  y=$(ls "$d"/*.yaml | head -1)
  [ -f "$y" ] || { echo "!!! lr $lr var $v: no yaml"; echo "$out" | tail -5; exit 1; }
  printf 'H_sm_lr%s_v%s\t%s\n' "$lr" "$v" "$y" >> "$B/jobs_sm_harness.tsv"
  echo "H_sm_lr${lr}_v${v} -> $y"
  sleep 1
done
