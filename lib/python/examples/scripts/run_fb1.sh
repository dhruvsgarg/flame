#!/usr/bin/env bash
# FB1 paper runs (FELIX FX-N131 / N150): every stack + * arms + the ladder (FedBuff, b1, uo_o, fn, fx), cifar n 300 and speech n 100
#   x s0 / s0u / s50u / smu, streams from +24 h of the 6-day trace (<= 120 h), each leg until its 5-eval mean hits targets.json.
#   Two bench pools share $GPUS (queued by free GPU memory + RAM); legs that may never hit start first.
# Usage (tmux, jayne or shepherd): TAG=<mmdd_hhmm> [GPUS=0,2-7 on jayne] [SMOKE=1] [DS=cifar|speech] bash .../scripts/run_fb1.sh
#   SMOKE=1: tier FB1S on `smu` only (2 h vclock, low target): config audit, stop at target, RAM flat.
# One Ctrl+C stops all (R20). The script refuses a dirty tree.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
[ -n "${TAG:-}" ] || { echo "[FB1] set TAG"; exit 1; }
[ -z "$(git status --porcelain -uno)" ] || { echo "[FB1] tree dirty"; exit 1; }
if [ -n "${SMOKE:-}" ]; then T=FB1S; ONLY="_smu_"; LEG_S=3600; else T=FB1; ONLY="_"; LEG_S=86400; fi
OUT=experiments/${T}_$TAG
PY="conda run --no-capture-output -n dg_flame python"
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
DS=${DS:-cifar speech}
BLS="felix,fedbuff,refl,feddance,fedavg,oort,oort_star"
FIRST="_(ctl_srv|sync_srv|rstar_srv|fdstar_srv|b1|uo_o) "  # stacks that may run the whole 120 h start first
mkdir -p "$OUT"
git log -1 --format='%h %s' > "$OUT/code_state.txt"
export FLAME_TELEMETRY_LEAN=1

pids=()
stop() { echo "[FB1] stopping"; for p in "${pids[@]}"; do kill -TERM -- -"$p" 2>/dev/null; done; sleep 10
         for p in "${pids[@]}"; do kill -KILL -- -"$p" 2>/dev/null; done; exit 130; }
trap stop INT TERM HUP

[[ $DS == *speech* ]] && { setsid $PY scripts/bench_pool.py --harness-args "--tier $T --datasets google_speech --baselines $BLS" \
    --only "gs_${T}$ONLY" --first "$FIRST" --out "$OUT/speech" --gpus "$GPUS" --per-gpu 1 --leg-gpu-gb 10 --leg-ram-gb 30 --workers 4 \
    --timeout-s $LEG_S > "$OUT/speech.log" 2>&1 & pids+=($!); }
[[ $DS == *cifar* ]] && { setsid $PY scripts/bench_pool.py --harness-args "--tier $T --datasets cifar10 --baselines $BLS" \
    --only "^${T}$ONLY" --first "$FIRST" --out "$OUT/cifar" --gpus "$GPUS" --per-gpu 4 --timeout-s $LEG_S > "$OUT/cifar.log" 2>&1 & pids+=($!); }
echo "[FB1] running $T; logs $OUT/{cifar,speech}.log"
wait

for ds in $DS; do
    $PY scripts/fl1_verdict.py "$OUT/$ds" --json-out "$OUT/${ds}_verdict.json" > "$OUT/${ds}_verdict.md" 2>&1
    $PY scripts/config_audit.py "$OUT/$ds" > "$OUT/${ds}_config_audit.md" 2>&1 || echo "[FB1] $ds config_audit: mismatches"
    $PY scripts/selection_audit.py "$OUT/$ds" > "$OUT/${ds}_selection_audit.md" 2>&1 || echo "[FB1] $ds selection_audit failed"
done
[ -n "${SMOKE:-}" ] && TGT="--target cifar10=0.35 google_speech=0.25" || TGT=""
$PY scripts/target_report.py $(for ds in $DS; do echo "$OUT/$ds"; done) $TGT --out "$OUT/target" > "$OUT/target_report.md" 2>&1
echo "[FB1] done: $OUT/target_report.md, {cifar,speech}_verdict.md"
