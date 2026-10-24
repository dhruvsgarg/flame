#!/usr/bin/env bash
# SmolLM2-1.7B · Yahoo, RUN_PLAN §2 cell: N 50, bs 32, --rho0 0.14 (`exp`, T_res 71 = Φ* 2; implied s 5.62), watchdog + acc tracker.
# Stop rule, both arms: stall + GL + commit cap + wall cap. The Phi rail is lifted for `fwdllm` (not part of that rule).
# Eval: 6,000 rows, fresh draw per eval; one full-test eval at the end (~40 min, est.), so the watchdog gets FINAL_EVAL_H.
# Vclock cap 1e9 s, so the wall ceiling, not the vclock, ends the run.
# Usage: ARM=fluxtune|fwdllm HOURS=12 MAX_COMMITS=380 STRIDE=4 ./launch_smollm17b_yahoo_n50.sh [extra run_sequential flags]
#   MAX_COMMITS = HOURS*3600 / measured s/commit (S sets it; 380 is the bs 32 estimate). Same value in both arms.
#   STRIDE = ceil(t_eval / s_commit) + 1.   Stop: kill -TERM -- -<pgid printed below>
set -euo pipefail
cd "$(dirname "$0")"
source /coc/scratch/dgarg/miniconda3/etc/profile.d/conda.sh && conda activate test_fwdllm
export FWDLLM_FD_SCALE_INVARIANT=1 EXPT_GPU_ALLOW_PEER=1
ARM=${ARM:-fluxtune}
HOURS=${HOURS:-12}
MAX_COMMITS=${MAX_COMMITS:-380}
STRIDE=${STRIDE:-4}
NUM_TRAINERS=${NUM_TRAINERS:-75}
FINAL_EVAL_H=${FINAL_EVAL_H:-1}
VCLOCK=1000000000
case "$ARM" in
  fluxtune) ARM_FLAGS=(--probe-combine mean --perturbation-count 10 --server-step-rule trust_ratio --rho0 0.14
                       --commit-gate fixed --pool-target 50 --rho-schedule exp --t-res 71 --agg-rate-type uniform
                       --var-stopping-policy off --force); DELAY_DIV=0.0047 ;;  # --force: wall-budget check wants the vclock cap to bind
  # --force: fwdllm has only an agnews charge profile; the 1.7B vclock is mispriced either way until G0.
  fwdllm)   ARM_FLAGS=(--phi-stop-threshold 1e9 --force); DELAY_DIV=0.047 ;;  # D: paper_expts_fluxtune/D_CALIBRATION.md
  *) echo "ARM must be fluxtune or fwdllm" >&2; exit 2 ;;
esac
TS=$(date +%Y%m%d_%H%M)
TAG=smollm17b_yahoo_n50_${ARM}_${TS}
LOG=../${TAG}.log
SINCE=$(date +%s)
# --allow-stale-profile: no 1.7B charge profile yet (G0), so the DistilBERT-derived one stands in.
setsid nohup ./run_sequential.sh --yes --clean --only "$ARM" --mode sim --allow-stale-profile \
  --dataset yahoo --partition-method "niid_label_clients=100_alpha=1" --num-trainers "$NUM_TRAINERS" --num-gpus 8 \
  --agg-goal 10 --c 30 --min-initial-frac 0.9 \
  --model-type llama --model-name HuggingFaceTB/SmolLM2-1.7B --peft-method lora --lora-r 4 \
  --adapter-reduction-factor 16 --fd-scale-invariant --delay-divisor "$DELAY_DIV" --delay-floor 4.0 --send-timeout-s 3600 \
  --train-batch-size 32 "${ARM_FLAGS[@]}" \
  --budget-stop-frac 0 --b-max-probe-every 0 --phi-stop halt \
  --saturation-stop --sat-stall-frac 0.003 --max-commits "$MAX_COMMITS" \
  --server-update-audit --no-cos-ground-truth-audit \
  --eval-max-samples 6000 --eval-resample --eval-every-n-commits "$STRIDE" --final-full-eval \
  --max-runtime-s "$VCLOCK" --sim-wall-ceiling-h "$HOURS" "$@" \
  > "$LOG" 2>&1 < /dev/null &
sleep 1
PGID=$(ps -o pgid= -p $! | tr -d ' ')
setsid nohup python watch_arm.py --exp-dir ../experiments --since "$SINCE" --pgid "$PGID" --kill \
  --max-hours "$(python -c "print($HOURS+0.5+$FINAL_EVAL_H)")" \
  > "../${TAG}_watch.log" 2>&1 < /dev/null &
setsid nohup python track_acc.py --exp-dir ../experiments --since "$SINCE" --pgid "$PGID" \
  --out "../${TAG}_acc.log" > /dev/null 2>&1 < /dev/null &
echo "launched arm=$ARM pgid=$PGID hours=$HOURS max_commits=$MAX_COMMITS stride=$STRIDE vclock_cap=$VCLOCK log=$LOG"
echo "watch=../${TAG}_watch.log acc=../${TAG}_acc.log"
