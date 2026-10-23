#!/usr/bin/env bash
# FL1 (FX-N141): one loop step = tier F44 over jayne (all GPUs) + shepherd GPUs 0-3, judged vs FedBuff (fl1_verdict.py).
#   jayne    : cifar bench n 300 (s0 static + s0u + s50u, 12 h, 18 legs + FedBuff) on CIFAR_GPUS; speech anchors on SPEECH_GPUS
#   shepherd : speech S1 / S2 arms (n 50, s0u, 12 h) on SHEP_GPUS via ssh; its run dirs come back to the same paths here
# Both trees must sit on the same commit (/home/dgarg39/flame on each node). One Ctrl+C stops all (R20).
# Usage (tmux on jayne): bash /home/dgarg39/flame/lib/python/examples/scripts/run_fl1.sh   [env: CIFAR_GPUS SPEECH_GPUS SHEP_GPUS TAG]
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
EX=$PWD
OUT=experiments/FL1_${TAG:-$(date +%m%d_%H%M)}
PY="conda run --no-capture-output -n dg_flame python"
CIFAR_GPUS=${CIFAR_GPUS:-0,1,2,3}
SPEECH_GPUS=${SPEECH_GPUS:-4,5,6,7}
SHEP_GPUS=${SHEP_GPUS:-0,1,2,3}
SHEP=${SHEP:-shepherd}
CEIL=experiments/ceiling_1022
SPEECH_HERE="gs_F44_s0u_ctl_srv gs_F44_s0u_l3b_srv"
SPEECH_SHEP="gs_F44_s0u_b1 gs_F44_s0u_b1gw gs_F44_s0u_b1gw_e3"

here=$(git rev-parse HEAD)
there=$(ssh -o BatchMode=yes "$SHEP" "git -C $EX rev-parse HEAD" 2>/dev/null | tail -1)
[ "$here" = "$there" ] || { echo "[FL1] commit mismatch: jayne $here vs $SHEP $there"; exit 1; }
[ -z "$(git status --porcelain -uno)" ] || { echo "[FL1] jayne tree dirty"; exit 1; }
mkdir -p "$OUT"
git log -1 --format='%h %s' > "$OUT/code_state.txt"

speech() {  # $1 phases, $2 GPUs, $3 out dir
    echo "$PY scripts/harness_pool.py --tier F44 --datasets google_speech --baselines felix,fedbuff --phases '$1' --gpu-ids $2" \
         "--gpu-cpus-per-trainer 0.1 --gate --deadline-h 5.5 --output-dir $3"; }
pids=()
stop() { echo "[FL1] stopping"; ssh -o BatchMode=yes "$SHEP" "pkill -TERM -f 'harness_pool.py --tier F44'" 2>/dev/null
         for p in "${pids[@]}"; do kill -TERM -- -"$p" 2>/dev/null; done; sleep 10
         for p in "${pids[@]}"; do kill -KILL -- -"$p" 2>/dev/null; done; exit 130; }
trap stop INT TERM

FLAME_TELEMETRY_LEAN=1 setsid timeout --foreground 21600 $PY scripts/bench_pool.py --harness-args "--tier F44 --datasets cifar10" \
    --out "$OUT/cifar" --gpus "$CIFAR_GPUS" --per-gpu 6 --timeout-s 20000 > "$OUT/cifar.log" 2>&1 & pids+=($!)
setsid timeout --foreground 21600 bash -c "$(speech "$SPEECH_HERE" "$SPEECH_GPUS" "$OUT/speech")" > "$OUT/speech.log" 2>&1 & pids+=($!)
setsid ssh -tt -o BatchMode=yes "$SHEP" "cd $EX && mkdir -p $OUT && timeout --foreground 21600 $(speech "$SPEECH_SHEP" "$SHEP_GPUS" "$OUT/speech_shep")" \
    > "$OUT/speech_shep.log" 2>&1 & pids+=($!)
echo "[FL1] running; logs $OUT/{cifar,speech,speech_shep}.log"
wait

rsync -a "$SHEP:$EX/$OUT/speech_shep/" "$OUT/speech_shep/"
for d in $(cut -f12 "$OUT"/speech_shep/*/summary.tsv | grep '^/'); do rsync -a "$SHEP:$d/" "$d/"; done  # sim_dir: same path here
$PY scripts/fl1_verdict.py "$OUT/cifar" --ceiling $CEIL/b64_cifar10_n300_syn_0.json $CEIL/b64_cifar10_n300_syn_50_3st.json \
    --json-out "$OUT/cifar_verdict.json" > "$OUT/cifar_verdict.md" 2>&1
$PY scripts/fl1_verdict.py "$OUT/speech" "$OUT/speech_shep" --ceiling $CEIL/google_speech_n50_syn_0.json \
    --json-out "$OUT/speech_verdict.json" > "$OUT/speech_verdict.md" 2>&1
$PY scripts/config_audit.py "$OUT/cifar" "$OUT/speech" "$OUT/speech_shep" > "$OUT/config_audit.md" 2>&1 \
    || echo "[FL1] config_audit: mismatches (see config_audit.md)"
echo "[FL1] done: $OUT/{cifar,speech}_verdict.md, config_audit.md"
