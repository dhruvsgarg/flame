#!/usr/bin/env bash
# Run 15 (~6h on jayne, 8 GPUs): correctness + accuracy first (FX-N74), then the FX-D50/D53 confirms.
# Phase A (~45 min, CPU, R14): T3 felix+fedbuff on syn_50 + mobiperf_3st (FX-D50) || T3 refl on syn_0 + syn_50 (FX-D53), both datasets.
# Phase B (~5h, GPU, no leg starts past T0+4.4h): G1A = reference n, syn_0, full data (no streaming), eval every 20 rounds, 90 min.
#   cifar  (GPUs 0-3, 0.2 CPU/trainer): felix -> refl
#   speech (GPUs 4-7, 3 per leg):       felix (server lr 1.0) -> refl -> fedbuff (server lr 1.0)
# After: scripts/accuracy_table.py <OUT>/cifar_* <OUT>/gs_*  and  parity_ladder.py --grade <pool> --regrade --max-stage 9.
# Usage: run_block_20261006.sh [OUT_DIR]; BLOCK_EXTRA="--smoke" smokes every pool. Ctrl+C stops it.
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
T0=$(date +%s)
left_h() { python3 -c "print(round(max(0.05, ($T0 + 4.4 * 3600 - $(date +%s)) / 3600), 2))"; }
CIFAR=(--datasets cifar10 --gpu-cpus-per-trainer 0.2 --gpu-ids 0,1,2,3 --no-gate --tier G1A)
SPEECH=(--datasets google_speech --gpu-ids 4,5,6,7 --gpus-per-job 3 --no-gate --tier G1A)

KIDS=()
( trap 'exit 130' INT TERM; pool cpu_async --datasets all --tier T3 --baselines "felix fedbuff" --traces "syn_50 mobiperf_3st" ) &
KIDS+=($!)
( trap 'exit 130' INT TERM; sleep 240; pool cpu_refl --datasets all --tier T3 --baselines refl --traces "syn_0 syn_50" --no-gate ) &
KIDS+=($!)
for p in "${KIDS[@]}"; do wait "$p"; done
say "PHASE A DONE"

KIDS=()
( trap 'exit 130' INT TERM; pool cifar_felix "${CIFAR[@]}" --baselines felix --deadline-h "$(left_h)";
  pool cifar_refl "${CIFAR[@]}" --baselines refl --deadline-h "$(left_h)" ) &
KIDS+=($!)
( trap 'exit 130' INT TERM; pool gs_felix "${SPEECH[@]}" --baselines felix --deadline-h "$(left_h)";
  pool gs_refl "${SPEECH[@]}" --baselines refl --deadline-h "$(left_h)";
  pool gs_fedbuff "${SPEECH[@]}" --baselines fedbuff --deadline-h "$(left_h)" ) &
KIDS+=($!)
for p in "${KIDS[@]}"; do wait "$p"; done
say "BLOCK DONE"
