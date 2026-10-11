#!/usr/bin/env bash
# Run 18 (~9h on jayne, 8 GPUs): GPU unavailability screen on all six after FX-D50/D55/D60/D64 (FX-N9), then the long legs.
# Phase 0 (~16 min): PL3 scale smoke of the new shapes: speech G0U n=50 (2 GPUs), speech G1A oort + felix (SGD, n=100).
#   cifar G1A n=300 was smoked in run 17 (felix, same density).
# Phase A (~100 min): cifar G0U + G0UC all six (GPU) | T3 sync oort/oort_star/feddance x syn_50 + mobiperf, both datasets (CPU).
# Phase B (~140 min): speech G0U + G0UC syn_50, all six (first speech floors).
# Phase C (rest; no leg starts past T0+9.3h): G1A speech felix + fedbuff on SGD 0.04 b16 (FX-N74), then oort + feddance (FX-N5).
# After: scripts/accuracy_table.py <pool dirs>  and  parity_ladder.py --grade <pool> --regrade --max-stage 9  (C5).
# Usage: run_block_20261008.sh [--detach] [OUT_DIR]; BLOCK_PHASES="0" runs phase 0 only.
# --detach survives a lost terminal; stop with: kill -INT -- -$(cat OUT/PGID). Ctrl+C stops a foreground run.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ "${1:-}" = "--detach" ]; then
  shift
  OUT="${1:-$HERE/../experiments/block_$(date +%Y%m%d_%H%M)}"
  mkdir -p "$OUT"
  setsid nohup bash "${BASH_SOURCE[0]}" "$OUT" > "$OUT/block.out" 2>&1 < /dev/null &
  echo "$!" > "$OUT/PGID"
  echo "detached: $OUT  (log: $OUT/BLOCK.log; stop: kill -INT -- -$!)"
  exit 0
fi
OUT="${1:-$HERE/../experiments/block_$(date +%Y%m%d_%H%M)}"
mkdir -p "$OUT"
PHASES="${BLOCK_PHASES:-0 A B C}"
POOL=(conda run --no-capture-output -n dg_flame python "$HERE/harness_pool.py")
KIDS=()
interrupt() {
  for p in "${KIDS[@]}"; do kill -INT "$p" 2>/dev/null; done
  for p in "${KIDS[@]}"; do wait "$p" 2>/dev/null; done
  echo "interrupted" >> "$OUT/BLOCK.log"; exit 130
}
trap interrupt INT TERM
say() { echo "[$(date '+%F %T')] $*" | tee -a "$OUT/BLOCK.log"; }
pool() {  # pool LABEL ARGS... ; a failure never stops the block
  local label="$1"; shift
  say "START $label"
  "${POOL[@]}" "$@" --output-dir "$OUT/$label" > "$OUT/$label.out" 2>&1
  say "DONE  $label rc=$?"
}
T0=$(date +%s)
left_h() { python3 -c "print(round(max(0.05, ($T0 + 9.3 * 3600 - $(date +%s)) / 3600), 2))"; }
GPUS=(--gpu-ids 0,1,2,3,4,5,6,7 --no-gate)
CIFAR=(--datasets cifar10 --gpu-cpus-per-trainer 0.2 "${GPUS[@]}")
SPEECH=(--datasets google_speech "${GPUS[@]}")
has() { [[ " $PHASES " == *" $1 "* ]]; }

if has 0; then
  pool s0_gs_g0u "${SPEECH[@]}" --tier G0U,G0UC --baselines felix --traces syn_50 --scale-smoke 300
  pool s0_gs_g1a "${SPEECH[@]}" --tier G1A --baselines "oort felix" --scale-smoke 300
  bad=$(grep "DONE  s0_.*rc=[1-9]" "$OUT/BLOCK.log")
  bad+=$(ls "$OUT"/s0_*/ABORT.txt "$OUT"/s0_*/*/*/STALLED.txt 2>/dev/null)
  bad+=$(grep -h "GPU_TIGHT\|RAM_TIGHT\|SKIP " "$OUT"/s0_*/pool.log 2>/dev/null)
  bad+=$(grep -hE "MISSING|CHECKER_ERROR" "$OUT"/s0_*/SUMMARY.txt 2>/dev/null)
  if [ -n "$bad" ]; then say "ABORT phase 0: $bad"; exit 4; fi
  say "PHASE 0 OK"
fi

if has A; then
  KIDS=()
  ( trap 'exit 130' INT TERM; pool a_cifar_g0u "${CIFAR[@]}" --tier G0U,G0UC ) &
  KIDS+=($!)
  ( trap 'exit 130' INT TERM; sleep 120
    pool a_t3_sync --datasets all --tier T3 --baselines "oort oort_star feddance" --traces "syn_50 mobiperf_3st" --no-gate ) &
  KIDS+=($!)
  for p in "${KIDS[@]}"; do wait "$p"; done
  KIDS=()
  say "PHASE A DONE"
fi

if has B; then
  pool b_gs_g0u "${SPEECH[@]}" --tier G0U,G0UC --traces syn_50 --deadline-h "$(left_h)"
fi

if has C; then
  pool c_gs_g1a_async "${SPEECH[@]}" --tier G1A --baselines "felix fedbuff" --deadline-h "$(left_h)"
  pool c_gs_g1a_sync "${SPEECH[@]}" --tier G1A --baselines "oort feddance" --deadline-h "$(left_h)"
fi
say "BLOCK DONE"
