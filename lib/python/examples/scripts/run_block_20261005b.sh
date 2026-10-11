#!/usr/bin/env bash
# Runs 12-14 (~13h on jayne, 7 GPUs); pools share the node (Leases), each chain in priority order.
# Phase A (~7h) screens (C11):
#   speech: G1S felix+fedbuff (n=100, 45 min, FX-D42/43/44/46) + G0U felix+fedbuff (n=50, 30 min, FX-N9) -> G0T felix syn_0 (FX-N13)
#   cifar:  G0U all six (FX-N9) -> G0T all six x {linear, events} syn_0 -> same at syn_50 -> G0To oracle arms syn_0
# Phase B (auto after A, operator 2026-10-05; no starts past T0+11.4h): speech G2 oort+refl at D x5 (3 GPUs, FX-N5);
#   cifar G0UC real replicates (floors, Q2) -> G1U felix+fedbuff syn_50 at n=300, 90 min (FX-N9 long confirm).
# cifar GPU legs take 0.2 CPU/trainer, raised per baseline to 1.5x its measured cores p95; legs throttled > 5% are flagged
# CPU_SAT in SUMMARY. Speech keeps 0.4 (aggregator-heavy, FX-N70).
# Usage: run_block_20261005b.sh [OUT_DIR]; BLOCK_EXTRA="--smoke" smokes every pool. Ctrl+C stops it.
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
CIFAR=(--datasets cifar10 --gpu-cpus-per-trainer 0.2 --no-gate)
T0=$(date +%s)
left_h() { python3 -c "print(round(max(0.05, ($T0 + 11.4 * 3600 - $(date +%s)) / 3600), 2))"; }

# speech first: it runs the gate (collect + one smoke pair per dataset) before any GPU leg starts.
( trap 'exit 130' INT TERM; pool gs --datasets google_speech --tier G1S,G0U --baselines "felix fedbuff";
  pool gs_t --datasets google_speech --tier G0T --baselines felix --traces syn_0 --no-gate ) &
KIDS+=($!)
( trap 'exit 130' INT TERM; sleep 240; pool cifar "${CIFAR[@]}" --tier G0U;
  pool cifar_t0 "${CIFAR[@]}" --tier G0T --traces syn_0;
  pool cifar_t50 "${CIFAR[@]}" --tier G0T --traces syn_50;
  pool cifar_o "${CIFAR[@]}" --tier G0To --baselines "felix oort refl fedbuff" --traces syn_0 ) &
KIDS+=($!)
for p in "${KIDS[@]}"; do wait "$p"; done
say "PHASE A DONE"

KIDS=()
( trap 'exit 130' INT TERM; pool gs_g2 --datasets google_speech --tier G2 --baselines "oort refl" --gpus-per-job 3 --no-gate \
    --deadline-h "$(left_h)" ) &
KIDS+=($!)
( trap 'exit 130' INT TERM; pool cifar_uc "${CIFAR[@]}" --tier G0UC --deadline-h "$(left_h)";
  pool cifar_g1u "${CIFAR[@]}" --tier G1U --deadline-h "$(left_h)" ) &
KIDS+=($!)
for p in "${KIDS[@]}"; do wait "$p"; done
say "BLOCK DONE"
