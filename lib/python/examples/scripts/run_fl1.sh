#!/usr/bin/env bash
# FL1 (FX-N141): one loop step = tier F44 on cifar only (bench n 300: s0 static + s0u + s50u, 12 h), judged vs FedBuff (fl1_verdict.py).
#   Runs on shepherd GPUs 0-3 (~6 legs / GPU, beside other users' jobs); results rsync to jayne at the end. One Ctrl+C stops all (R20).
# Usage (tmux on shepherd): bash /home/dgarg39/flame/lib/python/examples/scripts/run_fl1.sh   [env: GPUS PER_GPU TAG]
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
EX=$PWD
OUT=experiments/FL1_${TAG:-$(date +%m%d_%H%M)}
PY="conda run --no-capture-output -n dg_flame python"
GPUS=${GPUS:-0,1,2,3}
CEIL=experiments/ceiling_1022

[ -z "$(git status --porcelain -uno)" ] || { echo "[FL1] tree dirty"; exit 1; }
mkdir -p "$OUT"
git log -1 --format='%h %s' > "$OUT/code_state.txt"

pid=
stop() { echo "[FL1] stopping"; kill -TERM -- -"$pid" 2>/dev/null; sleep 10; kill -KILL -- -"$pid" 2>/dev/null; exit 130; }
trap stop INT TERM HUP

FLAME_TELEMETRY_LEAN=1 setsid timeout --foreground 21600 $PY scripts/bench_pool.py --harness-args "--tier F44 --datasets cifar10" \
    --out "$OUT/cifar" --gpus "$GPUS" --per-gpu "${PER_GPU:-6}" --timeout-s 20000 > "$OUT/cifar.log" 2>&1 & pid=$!
echo "[FL1] running; log $OUT/cifar.log"
wait

$PY scripts/fl1_verdict.py "$OUT/cifar" --ceiling $CEIL/b64_cifar10_n300_syn_0.json $CEIL/b64_cifar10_n300_syn_50_3st.json \
    --json-out "$OUT/cifar_verdict.json" > "$OUT/cifar_verdict.md" 2>&1
$PY scripts/config_audit.py "$OUT/cifar" > "$OUT/config_audit.md" 2>&1 || echo "[FL1] config_audit: mismatches (see config_audit.md)"
[ "$(hostname -s)" = jayne ] || rsync -a "$OUT" "jayne:$EX/experiments/" || echo "[FL1] rsync to jayne failed; copy $OUT by hand"
echo "[FL1] done: $OUT/cifar_verdict.md, config_audit.md"
