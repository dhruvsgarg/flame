#!/usr/bin/env bash
# F43 (FX-N139) speech baselines on shepherd GPUs 4-7 (~6 h; stage B keeps 0-3), from the worktree ~/flame_f43 (T15: never the live tree).
#   FedBuff, then FedBuff* if FedBuff ends within 3 h (--deadline-h 3). One Ctrl+C stops all (harness_pool, R20).
# Usage (tmux on shepherd): cd ~/flame_f43 && git fetch && git checkout --detach origin/dg/pending_fl_baselines && bash lib/python/examples/scripts/run_f43_shepherd.sh
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
OUT=experiments/F43_shepherd_$(date +%m%d_%H%M)
PY="conda run --no-capture-output -n dg_flame python"
$PY scripts/harness_pool.py --tier F43 --datasets google_speech --baselines fedbuff \
    --phases "gs_F43_s0u_ctl_srv gs_F43_s0u_fbstar_srv" --gpu-ids "${SPEECH_GPUS:-4,5,6,7}" --gpu-cpus-per-trainer 0.1 \
    --gate --deadline-h 3 --output-dir "$OUT/speech"
$PY scripts/bench_report.py "$OUT/speech" --targets 0.5,0.6,0.65,0.7 --json-out "$OUT/speech_report.json" \
    --ceiling experiments/ceiling_1022/google_speech_n100_syn_0.json > "$OUT/speech_report.md" 2>&1
echo "[F43] done: $OUT/speech_report.md (copy $OUT and its async_cifar10/experiments/run_*gs_F43_* dirs to jayne)"
