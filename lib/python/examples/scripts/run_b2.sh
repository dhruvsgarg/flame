#!/usr/bin/env bash
# FX-N151 / N129 B2: the same F49 legs on the harness (reference) and on the bench, in parallel on disjoint GPUs, then bench_validate.
# Usage (tmux, jayne or shepherd): TAG=<mmdd_hhmm> [DS=google_speech|cifar10] [CELL=smu] [ARMS="sync_srv ctl_srv"]
#   [BLS=refl,feddance,fedavg,oort,fedbuff] [HGPUS=0,1,2,3] [BGPUS=4,5,6,7] [UPD=0.05] [RT=14400] bash .../scripts/run_b2.sh
# RT: vclock seconds on both sides (default the tier's 12 h); bench_validate compares at the matched horizon.
# Speech: harness legs take 4 GPUs (~20-25 min per 12 h sync leg, ~35 min async), bench 1 GPU each. One Ctrl+C stops all (R20).
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
[ -n "${TAG:-}" ] || { echo "[B2] set TAG"; exit 1; }
[ -z "$(git status --porcelain -uno)" ] || { echo "[B2] tree dirty"; exit 1; }
DS=${DS:-google_speech}; CELL=${CELL:-smu}; ARMS=${ARMS:-sync_srv ctl_srv}; BLS=${BLS:-refl,feddance,fedavg,oort,fedbuff}
HGPUS=${HGPUS:-0,1,2,3}; BGPUS=${BGPUS:-4,5,6,7}; UPD=${UPD:-0.05}
P=$([ "$DS" = google_speech ] && echo gs_ || echo "")
PHASES=$(for a in $ARMS; do echo -n "${P}F49_${CELL}_$a "; done)
OUT=experiments/B2_${DS}_$TAG
PY="conda run --no-capture-output -n dg_flame python"
mkdir -p "$OUT"
git log -1 --format='%h %s' > "$OUT/code_state.txt"
CAP=$([ -n "${RT:-}" ] && echo "--scale-smoke $RT" || echo ""); BRT=$([ -n "${RT:-}" ] && echo "--runtime-s $RT" || echo "")
BENCH=$([ "$DS" = google_speech ] && echo "--per-gpu 1 --leg-gpu-gb 22 --leg-ram-gb 30 --workers 4" || echo "--per-gpu 4")

pids=()
stop() { echo "[B2] stopping"; for p in "${pids[@]}"; do kill -TERM -- -"$p" 2>/dev/null; done; sleep 10
         for p in "${pids[@]}"; do kill -KILL -- -"$p" 2>/dev/null; done; exit 130; }
trap stop INT TERM HUP
setsid $PY scripts/harness_pool.py --tier F49 --datasets "$DS" --baselines "$BLS" --phases "$PHASES" --gpu-ids "$HGPUS" $CAP \
    --deadline-h 12 --output-dir "$OUT/h" > "$OUT/h.log" 2>&1 & pids+=($!)
setsid $PY scripts/bench_pool.py --harness-args "--tier F49 --datasets $DS --baselines $BLS --phases '$PHASES'" --out "$OUT/b" \
    --gpus "$BGPUS" $BENCH $BRT --timeout-s 14400 > "$OUT/b.log" 2>&1 & pids+=($!)
echo "[B2] running $PHASES ($BLS) on $DS; logs $OUT/{h,b}.log"
wait
$PY scripts/bench_validate.py --bench "$OUT/b" --harness "$OUT/h" --upd "$UPD" --json-out "$OUT/bench_validate.json" \
    > "$OUT/bench_validate.md" 2>&1 || echo "[B2] bench_validate: FAIL rows"
echo "[B2] done: $OUT/bench_validate.md"
