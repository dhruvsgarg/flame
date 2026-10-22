#!/usr/bin/env bash
# SmolLM2-1.7B Yahoo N=50 run: the 20261020_0125 screen config, Phi halt, HOURS wall ceiling, watchdog + acc tracker.
# Eval stride 7: 6,000-row eval ~220 s vs ~43 s/commit (stride 5 blocked commits).
# Vclock cap = 40x wall (screen ran ~14x), so the wall ceiling ends the run.
# Usage: HOURS=3 ./launch_smollm17b_yahoo_n50.sh   Stop: kill -TERM -- -<pgid printed below>
set -euo pipefail
cd "$(dirname "$0")"
source /coc/scratch/dgarg/miniconda3/etc/profile.d/conda.sh && conda activate test_fwdllm
export FWDLLM_FD_SCALE_INVARIANT=1 EXPT_GPU_ALLOW_PEER=1
HOURS=${HOURS:-24}
VCLOCK=$(python -c "print(int($HOURS*3600*40))")
TS=$(date +%Y%m%d_%H%M)
LOG=../smollm17b_yahoo_n50_${TS}.log
SINCE=$(date +%s)
setsid nohup ./run_sequential.sh --yes --clean --only fluxtune --mode sim --allow-stale-profile \
  --dataset yahoo --partition-method "niid_label_clients=100_alpha=1" --num-trainers 50 --num-gpus 8 \
  --agg-goal 10 --c 30 --min-initial-frac 0.9 \
  --model-type llama --model-name HuggingFaceTB/SmolLM2-1.7B --peft-method lora --lora-r 4 \
  --adapter-reduction-factor 16 --fd-scale-invariant --delay-divisor 0.0192 --delay-floor 4.0 --send-timeout-s 3600 \
  --probe-combine mean --server-step-rule trust_ratio --gate-safety-s 1.5 \
  --commit-gate fixed --pool-target 50 --rho-schedule pool --agg-rate-type uniform \
  --var-stopping-policy off --max-iter-per-data-id 20 \
  --budget-stop-frac 0 --b-max-probe-every 0 --phi-stop halt \
  --server-update-audit --no-cos-ground-truth-audit \
  --eval-max-samples 6000 --eval-every-n-commits 7 \
  --max-runtime-s "$VCLOCK" --sim-wall-ceiling-h "$HOURS" \
  > "$LOG" 2>&1 < /dev/null &
sleep 1
PGID=$(ps -o pgid= -p $! | tr -d ' ')
setsid nohup python watch_arm.py --exp-dir ../experiments --since "$SINCE" --pgid "$PGID" --kill \
  --max-hours "$(python -c "print($HOURS+0.5)")" \
  > "../smollm17b_yahoo_n50_${TS}_watch.log" 2>&1 < /dev/null &
setsid nohup python track_acc.py --exp-dir ../experiments --since "$SINCE" --pgid "$PGID" \
  --out "../smollm17b_yahoo_n50_${TS}_acc.log" > /dev/null 2>&1 < /dev/null &
echo "launched pgid=$PGID hours=$HOURS vclock_cap=$VCLOCK log=$LOG"
echo "watch=../smollm17b_yahoo_n50_${TS}_watch.log acc=../smollm17b_yahoo_n50_${TS}_acc.log"
