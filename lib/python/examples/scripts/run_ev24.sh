#!/usr/bin/env bash
# FX-N157 exit: EV24 (dispatch to a client still computing) must be 0 on 12 h MobiPerf `fx` + `fn` (bench, both datasets) and the
# harness speech `smu` `fx`. Usage (tmux, shepherd): TAG=<mmdd_hhmm> [HGPUS=0,1,2,3] [BGPUS=4,5,6,7] bash .../scripts/run_ev24.sh
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
[ -n "${TAG:-}" ] || { echo "[EV24] set TAG"; exit 1; }
[ -z "$(git status --porcelain -uno)" ] || { echo "[EV24] tree dirty"; exit 1; }
[ "$(hostname -s)" = shepherd ] || { [ -n "${HGPUS:-}" ] && [ -n "${BGPUS:-}" ]; } || { echo "[EV24] off shepherd: ask the operator, then set HGPUS and BGPUS"; exit 1; }
HGPUS=${HGPUS:-0,1,2,3}; BGPUS=${BGPUS:-4,5,6,7}
OUT=experiments/EV24_$TAG
PY="conda run --no-capture-output -n dg_flame python"
mkdir -p "$OUT"
git log -1 --format='%h %s' > "$OUT/code_state.txt"

pids=()
stop() { echo "[EV24] stopping"; for p in "${pids[@]}"; do kill -TERM -- -"$p" 2>/dev/null; done; sleep 10
         for p in "${pids[@]}"; do kill -KILL -- -"$p" 2>/dev/null; done; exit 130; }
trap stop INT TERM HUP
setsid $PY scripts/harness_pool.py --tier F48 --datasets google_speech --baselines felix --phases gs_F48_smu_fx --gpu-ids "$HGPUS" \
    --deadline-h 12 --output-dir "$OUT/h" > "$OUT/h.log" 2>&1 & pids+=($!)
for ds in cifar10 google_speech; do
    P=$([ "$ds" = google_speech ] && echo gs_ || echo "")
    BENCH=$([ "$ds" = google_speech ] && echo "--per-gpu 1 --leg-gpu-gb 22 --leg-ram-gb 30 --workers 4" || echo "--per-gpu 4")
    setsid $PY scripts/bench_pool.py --harness-args "--tier FB2E2E --datasets $ds --baselines felix --phases '${P}FB2E2E_smu_fx'" \
        --harness-args "--tier FB2LAD --datasets $ds --baselines felix --phases '${P}FB2LAD_smu_fn'" --out "$OUT/b_$ds" \
        --gpus "$BGPUS" $BENCH --runtime-s 43200 --timeout-s 14400 > "$OUT/b_$ds.log" 2>&1 & pids+=($!)
done
echo "[EV24] running; logs $OUT/{h,b_cifar10,b_google_speech}.log"
wait
for d in "$OUT"/h "$OUT"/b_*/; do
    d=${d%/}
    # harness run dirs live in async_cifar10/experiments, bench ones under the pool
    runs=$([ "$d" = "$OUT/h" ] && find async_cifar10/experiments -maxdepth 1 -name 'run_*gs_F48_smu_fx_*' -newer "$OUT/code_state.txt" \
           || find "$d" -type d -name 'run_*')
    $PY async_cifar10/scripts/parity/event_invariants.py $runs --json-out "$d/ev.json" > "$d/ev.md" 2>&1
    echo "[EV24] $d: $(grep -h 'EV24' "$d/ev.md" | head -3)"
done
