#!/usr/bin/env bash
# Run 17 (~6h on jayne, 8 GPUs): confirm FX-D60-D63 and finish the G1A accuracy rows (FX-N74).
# Host RAM fits two speech n=100 legs or one cifar n=300 leg (FX-D59), so speech runs first, then cifar, 4 GPUs per leg.
# Phase 0 (~40 min): G1A felix scale smoke (PL3), speech then cifar; any OOM, abort, stall, MISSING, GPU_TIGHT or RAM_TIGHT aborts.
# Phase A (~25 min, CPU): T3 + T3C (real replicate) refl syn_50, and T3 felix+fedbuff syn_0, both datasets.
# Phase B (~4.6h, no leg starts past T0+5.8h): G1A 90 min, full data, eval every 20 rounds.
#   speech felix + fedbuff pairs (~3.4h) -> cifar refl + fedbuff sim legs (~45 min; graded against run 16's real legs).
# After: scripts/accuracy_table.py --runs <run dirs>  and  parity_ladder.py --grade <pool> --regrade --max-stage 9.
# Usage: run_block_20261007.sh [--detach] [OUT_DIR]; BLOCK_PHASES="0" runs phase 0 only.
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
PHASES="${BLOCK_PHASES:-0 A B}"
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
left_h() { python3 -c "print(round(max(0.05, ($T0 + 5.8 * 3600 - $(date +%s)) / 3600), 2))"; }
GPUS=(--gpu-ids 0,1,2,3,4,5,6,7 --no-gate --tier G1A)
CIFAR=(--datasets cifar10 --gpu-cpus-per-trainer 0.2 "${GPUS[@]}")
SPEECH=(--datasets google_speech "${GPUS[@]}")
has() { [[ " $PHASES " == *" $1 "* ]]; }

if has 0; then
  pool s0_speech "${SPEECH[@]}" --baselines felix --scale-smoke 300
  pool s0_cifar "${CIFAR[@]}" --baselines felix --scale-smoke 300
  bad=$(grep "DONE  s0_.*rc=[1-9]" "$OUT/BLOCK.log")
  bad+=$(ls "$OUT"/s0_*/ABORT.txt "$OUT"/s0_*/*/*/STALLED.txt 2>/dev/null)
  bad+=$(grep -h "GPU_TIGHT\|RAM_TIGHT\|SKIP " "$OUT"/s0_*/pool.log 2>/dev/null)
  bad+=$(grep -hE "MISSING|CHECKER_ERROR" "$OUT"/s0_*/SUMMARY.txt 2>/dev/null)
  if [ -n "$bad" ]; then say "ABORT phase 0: $bad"; exit 4; fi
  say "PHASE 0 OK"
fi

if has A; then
  KIDS=()
  ( trap 'exit 130' INT TERM; pool a_refl --datasets all --tier T3,T3C --baselines refl --traces syn_50 ) &
  KIDS+=($!)
  ( trap 'exit 130' INT TERM; sleep 240; pool a_async --datasets all --tier T3 --baselines "felix fedbuff" --traces syn_0 --no-gate ) &
  KIDS+=($!)
  for p in "${KIDS[@]}"; do wait "$p"; done
  say "PHASE A DONE"
fi

if has B; then
  pool b_gs "${SPEECH[@]}" --baselines "felix fedbuff" --deadline-h "$(left_h)"
  pool b_cifar "${CIFAR[@]}" --baselines "refl fedbuff" --tier G1AS --deadline-h "$(left_h)"
fi
say "BLOCK DONE"
