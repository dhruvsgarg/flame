#!/usr/bin/env bash
# ~10.5h block on jayne: speech G1 + G2 (FX-N4/N5/N10) beside the 3h oort syn_50 CPU pair (FX-N62), then felix cifar 7500s x2 real + sim (FX-N65).
# Usage: run_block_20261004.sh [OUT_DIR]   Ctrl+C tears down every running pool and stops the block (R20).
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:-$HERE/../experiments/block_$(date +%Y%m%d_%H%M)}"
EXP="$HERE/../async_cifar10/experiments"
mkdir -p "$OUT"
PY=(conda run --no-capture-output -n dg_flame python)
POOL=("${PY[@]}" "$HERE/harness_pool.py" --datasets)
KIDS=()
interrupt() {
  for p in "${KIDS[@]}"; do kill -INT "$p" 2>/dev/null; done
  for p in "${KIDS[@]}"; do wait "$p" 2>/dev/null; done
  echo "interrupted" >> "$OUT/BLOCK.log"; exit 130
}
trap interrupt INT TERM
say() { echo "[$(date '+%F %T')] $*" | tee -a "$OUT/BLOCK.log"; }
pool() {  # pool LABEL ARGS... : one pool in the foreground of its caller; a failure never stops the block
  local label="$1"; shift
  say "START $label"
  "${POOL[@]}" "$@" ${BLOCK_EXTRA:-} --output-dir "$OUT/$label" > "$OUT/$label.out" 2>&1
  say "DONE  $label rc=$?"
}

# A. speech G1 then G2, 4 GPUs per leg (two legs side by side); the first pool runs the gate.
( trap 'exit 130' INT TERM
  pool gs_g1 google_speech --tier G1 --gpus-per-job 4
  pool gs_g2 google_speech --tier G2 --baselines "oort refl feddance" --gpus-per-job 4 --no-gate ) &
KIDS+=($!)
# A'. beside it on spare cores: unaware oort syn_50 for 3h (FX-N62)
( trap 'exit 130' INT TERM; sleep 300; pool n62 cifar10 --tier N62 --no-gate ) &
KIDS+=($!)
for p in "${KIDS[@]}"; do wait "$p"; done
KIDS=()

# B. felix cifar n=300 syn_0 7500s: pair + a real replicate (whole node per leg; FX-N65)
( trap 'exit 130' INT TERM; pool g1l cifar10 --tier G1L --no-gate ) &
KIDS+=($!); wait "${KIDS[0]}"; KIDS=()
REALS=($(ls -dt "$EXP"/run_*_G1L*_felix_*_real 2>/dev/null | head -n 2))
SIM=$(ls -dt "$EXP"/run_*_G1L_*_felix_*_sim 2>/dev/null | head -n 1)
if [ "${#REALS[@]}" -eq 2 ] && [ -n "$SIM" ]; then
  CHK="$HERE/../async_cifar10/scripts/parity_check.py"
  "${PY[@]}" "$CHK" --real "${REALS[0]}" --sim "${REALS[1]}" --control --json-out "$OUT/g1l_control.json" > "$OUT/g1l_control.txt" 2>&1
  "${PY[@]}" -c "import json,sys; sys.path.insert(0,'$HERE/../async_cifar10/scripts'); from parity.checks import control_floors; json.dump(control_floors(json.load(open('$OUT/g1l_control.json'))), open('$OUT/g1l_floors.json','w'))"
  for R in "${REALS[@]}"; do
    "${PY[@]}" "$CHK" --real "$R" --sim "$SIM" --floors "$OUT/g1l_floors.json" --json-out "$OUT/g1l_$(basename "$R").json" > "$OUT/g1l_$(basename "$R").txt" 2>&1
  done
  say "G1L control + floored grades written"
fi
say "BLOCK DONE"
