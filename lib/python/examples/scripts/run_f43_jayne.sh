#!/usr/bin/env bash
# F43 (FX-N139): baselines + Felix contenders at the reference cohort, toward the 70% goal. One Ctrl+C stops all (R20).
#   cifar  : bench, n 300 c 60 K 15, 120 h vclock (stop at 70%, 5-eval mean), both cells, 1 seed, lean telemetry, 26 legs at once
#   speech : harness, n 100 c 30 K 10, 24 h vclock, syn_0, Felix l3b_e3 then l3b (~3.8 h each; FedBuff / FedBuff*: run_f43_shepherd.sh)
# Usage (tmux on jayne): bash lib/python/examples/scripts/run_f43_jayne.sh
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
TS=$(date +%m%d_%H%M)
OUT=experiments/F43_jayne_$TS
mkdir -p "$OUT"
PY="conda run --no-capture-output -n dg_flame python"
CIFAR_GPUS=${CIFAR_GPUS:-0,5,6}
SPEECH_GPUS=${SPEECH_GPUS:-2,3,4}
SPEECH_PHASES=${SPEECH_PHASES:-"gs_F43_s0u_l3b_e3_srv gs_F43_s0u_l3b_srv"}
SPEECH_BLS=${SPEECH_BLS:-felix}

pids=()
stop() { echo "[F43] stopping"; for p in "${pids[@]}"; do kill -TERM -- -"$p" 2>/dev/null; done; sleep 10
         for p in "${pids[@]}"; do kill -KILL -- -"$p" 2>/dev/null; done; exit 130; }
trap stop INT TERM

git log -1 --format='%h %s' > "$OUT/code_state.txt"
FLAME_TELEMETRY_LEAN=1 setsid timeout --foreground 30600 $PY scripts/bench_pool.py --harness-args "--tier F43 --datasets cifar10" \
    --out "$OUT/cifar" --gpus "$CIFAR_GPUS" --per-gpu 9 --timeout-s 30000 > "$OUT/cifar.log" 2>&1 &
pids+=($!)
setsid timeout --foreground 30600 $PY scripts/harness_pool.py --tier F43 --datasets google_speech --baselines "$SPEECH_BLS" \
    --phases "$SPEECH_PHASES" --gpu-ids "$SPEECH_GPUS" --gpu-cpus-per-trainer 0.1 --gate --deadline-h 8 \
    --output-dir "$OUT/speech" > "$OUT/speech.log" 2>&1 &
pids+=($!)
echo "[F43] cifar pid ${pids[0]}, speech pid ${pids[1]}; logs $OUT/{cifar,speech}.log"
wait

T_CIFAR=0.4,0.45,0.5,0.55,0.6,0.65,0.7
$PY scripts/bench_report.py "$OUT/cifar" --targets "$T_CIFAR" --json-out "$OUT/cifar_report.json" \
    --ceiling experiments/ceiling_1022/b64_cifar10_n300_syn_0.json experiments/ceiling_1022/b64_cifar10_n300_syn_50_3st.json \
    > "$OUT/cifar_report.md" 2>&1
$PY scripts/bench_report.py "$OUT/speech" --targets 0.5,0.6,0.65,0.7 --json-out "$OUT/speech_report.json" \
    --ceiling experiments/ceiling_1022/google_speech_n100_syn_0.json > "$OUT/speech_report.md" 2>&1
echo "[F43] done: $OUT/{cifar,speech}_report.md"
