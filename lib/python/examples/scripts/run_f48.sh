#!/usr/bin/env bash
# F48 (FELIX FX-N141 S6) on jayne, three pools at once (~10-11 h, speech-bound):
#   GPU 5     cifar bench F48L: FedBuff, fn, fn_chr, fn_ev x MobiPerf, 24 h (c ramp + eval-task confirm)
#   GPUs 6-7  cifar bench F48R: rate `new` sweep on fn_chr + FedBuff x s0u / smu, 12 h
#   GPUs 0,2-4 speech harness F48: Felix new / old / FedBuff x s0u / s50u / smu, 12 h, one 4-GPU leg at a time
# Usage (tmux on jayne): TAG=<mmdd_hhmm> bash /home/dgarg39/flame/lib/python/examples/scripts/run_f48.sh
# One Ctrl+C stops all (R20). The script refuses a dirty tree.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
[ -n "${TAG:-}" ] || { echo "[F48] set TAG"; exit 1; }
[ -z "$(git status --porcelain -uno)" ] || { echo "[F48] tree dirty"; exit 1; }
OUT=experiments/F48_$TAG
PY="conda run --no-capture-output -n dg_flame python"
mkdir -p "$OUT"
git log -1 --format='%h %s' > "$OUT/code_state.txt"
export FLAME_TELEMETRY_LEAN=1

pids=()
stop() { echo "[F48] stopping"; for p in "${pids[@]}"; do kill -TERM -- -"$p" 2>/dev/null; done; sleep 10
         for p in "${pids[@]}"; do kill -KILL -- -"$p" 2>/dev/null; done; exit 130; }
trap stop INT TERM HUP

setsid timeout --foreground 25200 $PY scripts/bench_pool.py --harness-args "--tier F48L --datasets cifar10" --only F48L_ \
    --out "$OUT/cifar_L" --gpus 5 --per-gpu 4 --timeout-s 21600 > "$OUT/cifar_L.log" 2>&1 & pids+=($!)
setsid timeout --foreground 18000 $PY scripts/bench_pool.py --harness-args "--tier F48R --datasets cifar10" --only F48R_ \
    --out "$OUT/cifar_R" --gpus 6,7 --per-gpu 8 --timeout-s 14400 > "$OUT/cifar_R.log" 2>&1 & pids+=($!)
setsid timeout --foreground 64800 $PY scripts/harness_pool.py --tier F48 --datasets google_speech --baselines felix,fedbuff \
    --gpu-ids 0,2,3,4 --output-dir "$OUT/speech" > "$OUT/speech.log" 2>&1 & pids+=($!)
echo "[F48] running; logs $OUT/{cifar_L,cifar_R,speech}.log"
wait

for x in L R; do
    $PY scripts/fl1_verdict.py "$OUT/cifar_$x" --json-out "$OUT/cifar_${x}_verdict.json" > "$OUT/cifar_${x}_verdict.md" 2>&1
    $PY scripts/config_audit.py "$OUT/cifar_$x" > "$OUT/cifar_${x}_config_audit.md" 2>&1 || echo "[F48] cifar_$x config_audit: mismatches"
    $PY scripts/selection_audit.py "$OUT/cifar_$x" > "$OUT/cifar_${x}_selection_audit.md" 2>&1 || echo "[F48] cifar_$x selection_audit failed"
done
$PY scripts/fl1_verdict.py "$OUT/speech" --json-out "$OUT/speech_verdict.json" > "$OUT/speech_verdict.md" 2>&1
$PY scripts/selection_audit.py "$OUT/speech" > "$OUT/speech_selection_audit.md" 2>&1 || echo "[F48] speech selection_audit failed"
$PY scripts/dynamic_c_report.py "$OUT/cifar_L" "$OUT/cifar_R" "$OUT/speech" > "$OUT/dynamic_c.md" 2>&1
echo "[F48] done: $OUT/{cifar_L,cifar_R,speech}_verdict.md, dynamic_c.md"
