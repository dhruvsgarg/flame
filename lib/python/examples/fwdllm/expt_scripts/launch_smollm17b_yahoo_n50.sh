#!/usr/bin/env bash
# SmolLM2-1.7B · Yahoo, RUN_PLAN §2 cell: N 50, bs 32, s 2.0, Φ_budget 4 (`exp`, ρ0 0.0498, T_res 1,119; RUN_PLAN decision 12), watchdog + acc tracker.
# Stop rule, both arms (operator 10-25): stop once TARGET is held for CONVERGE_WINDOW evals, else at HOURS wall (20 h).
# No commit cap, saturation stop off (SAT_STOP=1 re-enables). The Phi rail is lifted for `fwdllm`.
# Eval: 6,000 rows, fresh draw per eval; every stop above exits cleanly, so one full-test eval follows (FINAL_EVAL_H).
# Usage: ARM=fluxtune|fwdllm [TARGET=0.65] [HOURS=20] [MAX_COMMITS=0] [STRIDE=3] ./launch_smollm17b_yahoo_n50.sh [extra flags]
#   MAX_COMMITS 0 = no cap (a cap also scales the saturation warm-up). STRIDE = ceil(t_eval / s_commit) + 1 (104 s / 99 s, 10-23).
#   Stop: kill -TERM -- -<pgid printed below>
set -euo pipefail
cd "$(dirname "$0")"
source /coc/scratch/dgarg/miniconda3/etc/profile.d/conda.sh && conda activate test_fwdllm
export FWDLLM_FD_SCALE_INVARIANT=1 EXPT_GPU_ALLOW_PEER=1
ARM=${ARM:-fluxtune}
HOURS=${HOURS:-20}
MAX_COMMITS=${MAX_COMMITS:-0}
STRIDE=${STRIDE:-3}
NUM_TRAINERS=${NUM_TRAINERS:-60}  # 60 ran 14.6 h on 7 GPUs, 0 OOM (10-23); 75 untested
TARGET=${TARGET:-0.65}
CONVERGE_WINDOW=${CONVERGE_WINDOW:-2}  # = tta_consecutive
SAT_STOP=${SAT_STOP:-0}
STOP_FLAGS=(--target-stop "$TARGET" --target-stop-window "$CONVERGE_WINDOW")  # in-process, unlike --target-acc (kills, no final eval)
[ "$MAX_COMMITS" != 0 ] && STOP_FLAGS+=(--max-commits "$MAX_COMMITS")
[ "$SAT_STOP" = 1 ] && STOP_FLAGS+=(--saturation-stop --sat-stall-frac 0.003)
FINAL_EVAL_H=${FINAL_EVAL_H:-1}
VCLOCK=1000000000
case "$ARM" in
  fluxtune) ARM_FLAGS=(--probe-combine mean --perturbation-count 10 --server-step-rule trust_ratio --gate-safety-s 2
                       --commit-gate fixed --pool-target 50 --rho-schedule exp --t-res 1119 --agg-rate-type uniform
                       --var-stopping-policy off --phi-stop-threshold 4 --force); DELAY_DIV=0.0047 ;;  # --force: wall-budget check wants the vclock cap to bind
  # --force: fwdllm has only an agnews charge profile; the 1.7B vclock is mispriced either way until G0.
  fwdllm)   ARM_FLAGS=(--phi-stop-threshold 1e9 --force); DELAY_DIV=0.047 ;;  # D: paper_expts_fluxtune/D_CALIBRATION.md
  *) echo "ARM must be fluxtune or fwdllm" >&2; exit 2 ;;
esac
TS=$(date +%Y%m%d_%H%M)
TAG=smollm17b_yahoo_n50_${ARM}_${TS}
LOG=../${TAG}.log
SINCE=$(date +%s)
# --allow-stale-profile: kept as a guard; fluxtune_yahoo.yaml was re-profiled 10-25 (RUN_PLAN decision 12).
setsid nohup ./run_sequential.sh --yes --clean --only "$ARM" --mode sim --allow-stale-profile \
  --dataset yahoo --partition-method "niid_label_clients=100_alpha=1" --num-trainers "$NUM_TRAINERS" --num-gpus 8 \
  --agg-goal 10 --c 30 --min-initial-frac 0.9 \
  --model-type llama --model-name HuggingFaceTB/SmolLM2-1.7B --peft-method lora --lora-r 4 \
  --adapter-reduction-factor 16 --fd-scale-invariant --delay-divisor "$DELAY_DIV" --delay-floor 4.0 --send-timeout-s 3600 \
  --train-batch-size 32 "${ARM_FLAGS[@]}" \
  --budget-stop-frac 0 --b-max-probe-every 0 --phi-stop halt \
  "${STOP_FLAGS[@]}" \
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
echo "launched arm=$ARM pgid=$PGID target=$TARGET/$CONVERGE_WINDOW sat_stop=$SAT_STOP hours=$HOURS max_commits=$MAX_COMMITS stride=$STRIDE vclock_cap=$VCLOCK log=$LOG"
echo "watch=../${TAG}_watch.log acc=../${TAG}_acc.log"
