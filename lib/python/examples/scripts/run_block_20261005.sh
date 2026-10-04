#!/usr/bin/env bash
# Run 12 (~3h on jayne, 7 GPUs): short/small/parallel first (operator). Two pools side by side (Leases share the node):
#   A. cifar G0U: all six x {syn_50, mobiperf_3st}, n=50, 15 min, 1 GPU per leg (FX-N9 screen)
#   B. speech G1S (felix + fedbuff, n=100, 45 min, 3 GPUs: FX-D42/D43/D44) + G0U felix/fedbuff (n=50, 30 min, 2 GPUs); speech D x5, 450s timeout (FX-D46)
# Stall rules (FX-D45) kill a leg alone after 15 min with no committed round. Usage: run_block_20261005.sh [OUT_DIR]
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:-$HERE/../experiments/block_$(date +%Y%m%d_%H%M)}"
mkdir -p "$OUT"
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
  "${POOL[@]}" "$@" ${BLOCK_EXTRA:-} --output-dir "$OUT/$label" > "$OUT/$label.out" 2>&1
  say "DONE  $label rc=$?"
}

# B first: it runs the gate (collect + one smoke pair per dataset) before any GPU leg starts.
( trap 'exit 130' INT TERM; pool gs google_speech --tier G1S,G0U --baselines "felix fedbuff" ) &
KIDS+=($!)
( trap 'exit 130' INT TERM; sleep 240; pool cifar cifar10 --tier G0U --no-gate ) &
KIDS+=($!)
for p in "${KIDS[@]}"; do wait "$p"; done
say "BLOCK DONE"
