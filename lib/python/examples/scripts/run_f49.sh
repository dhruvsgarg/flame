#!/usr/bin/env bash
# F49 decider (FELIX FX-N141 S6, ~2-3 h on jayne): Felix new (fn_chr) vs FedBuff, REFL, FedDance, FedAvg, Oort + * arms,
#   cifar n 300 and speech n 100 x syn_0 / syn_50_3st / MobiPerf streams, 12 h, augmentation on, all on the bench.
#   Two bench pools share GPUs 0,2-7 (queued by free GPU memory + RAM). Then run_f48.sh (sanity: 24 h ramp / eval, sweep, speech harness).
# Usage (tmux on jayne): TAG=<mmdd_hhmm> bash /home/dgarg39/flame/lib/python/examples/scripts/run_f49.sh
# One Ctrl+C stops all (R20). The script refuses a dirty tree.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
[ -n "${TAG:-}" ] || { echo "[F49] set TAG"; exit 1; }
[ -z "$(git status --porcelain -uno)" ] || { echo "[F49] tree dirty"; exit 1; }
OUT=experiments/F49_$TAG
PY="conda run --no-capture-output -n dg_flame python"
GPUS=${GPUS:-0,2,3,4,5,6,7}
BLS="felix,fedbuff,refl,feddance,fedavg,oort,oort_star"
FIRST="_(fn_chr|ctl_srv|smu_fn) "  # long legs (Felix, FedBuff) start first
mkdir -p "$OUT"
git log -1 --format='%h %s' > "$OUT/code_state.txt"
export FLAME_TELEMETRY_LEAN=1

pids=()
stop() { echo "[F49] stopping"; for p in "${pids[@]}"; do kill -TERM -- -"$p" 2>/dev/null; done; sleep 10
         for p in "${pids[@]}"; do kill -KILL -- -"$p" 2>/dev/null; done; exit 130; }
trap stop INT TERM HUP

setsid timeout --foreground 16200 $PY scripts/bench_pool.py --harness-args "--tier F49 --datasets google_speech --baselines $BLS" \
    --only gs_F49_ --first "$FIRST" --out "$OUT/speech" --gpus "$GPUS" --per-gpu 1 --leg-gpu-gb 10 --leg-ram-gb 30 --workers 4 \
    --timeout-s 14400 > "$OUT/speech.log" 2>&1 & pids+=($!)
setsid timeout --foreground 14400 $PY scripts/bench_pool.py --harness-args "--tier F49 --datasets cifar10 --baselines $BLS" \
    --only "^F49_" --first "$FIRST" --out "$OUT/cifar" --gpus "$GPUS" --per-gpu 4 --timeout-s 12600 > "$OUT/cifar.log" 2>&1 & pids+=($!)
echo "[F49] running; logs $OUT/{cifar,speech}.log"
wait

for ds in cifar speech; do
    $PY scripts/fl1_verdict.py "$OUT/$ds" --json-out "$OUT/${ds}_verdict.json" > "$OUT/${ds}_verdict.md" 2>&1
    $PY scripts/config_audit.py "$OUT/$ds" > "$OUT/${ds}_config_audit.md" 2>&1 || echo "[F49] $ds config_audit: mismatches"
    $PY scripts/selection_audit.py "$OUT/$ds" > "$OUT/${ds}_selection_audit.md" 2>&1 || echo "[F49] $ds selection_audit failed"
done
$PY scripts/dynamic_c_report.py "$OUT/cifar" "$OUT/speech" > "$OUT/dynamic_c.md" 2>&1
echo "[F49] done: $OUT/{cifar,speech}_verdict.md, dynamic_c.md"
