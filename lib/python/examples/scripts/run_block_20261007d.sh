#!/usr/bin/env bash
# Run 21 accuracy screens (~30 min, sim-only G1AS): cifar fedbuff on SGD 0.04 x 1.0 (FX-N74) | speech refl at the REFL repo lr 0.05.
# No leg starts past T0+1h. After: parity_ladder.py --grade <pool> --regrade --max-stage 9; accuracy_table.py on lane A (C5).
# Usage: run_block_20261007b.sh [--detach] [OUT_DIR]. Stop: kill -INT -- -$(cat OUT/PGID).
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ "${1:-}" = "--detach" ]; then
  shift
  OUT="${1:-$HERE/../experiments/block_$(date +%Y%m%d_%H%M)_run21}"
  mkdir -p "$OUT"
  setsid nohup bash "${BASH_SOURCE[0]}" "$OUT" > "$OUT/block.out" 2>&1 < /dev/null &
  echo "$!" > "$OUT/PGID"
  echo "detached: $OUT  (log: $OUT/BLOCK.log; stop: kill -INT -- -$!)"
  exit 0
fi
OUT="${1:-$HERE/../experiments/block_$(date +%Y%m%d_%H%M)_run21}"
mkdir -p "$OUT"
PY=(conda run --no-capture-output -n dg_flame python)
POOL=("${PY[@]}" "$HERE/harness_pool.py")
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
  "${POOL[@]}" "$@" --deadline-h "$(left_h)" --output-dir "$OUT/$label" > "$OUT/$label.out" 2>&1
  say "DONE  $label rc=$?"
}
T0=$(date +%s)
left_h() { python3 -c "print(round(max(0.05, ($T0 + 1.0 * 3600 - $(date +%s)) / 3600), 2))"; }
GPUS=(--gpu-ids 0,1,2,3,4,5,6,7 --no-gate)
CIFAR=(--datasets cifar10 --gpu-cpus-per-trainer 0.2 "${GPUS[@]}")
SPEECH=(--datasets google_speech "${GPUS[@]}")
ln -sfn "$HERE/../experiments/block_20261005_0401/cifar_uc" "$OUT/floors_cifar_uc_1005"  # G0UC floors (parity_ladder sibling glob)
ln -sfn "$HERE/../experiments/block_20261007_1456_run19/s_gs_g0u" "$OUT/floors_gs_uc_run19"

lane() { ( trap 'exit 130' INT TERM; "$@" ) & KIDS+=($!); }
lane_c() { pool c_g1as_fedbuff "${CIFAR[@]}" --tier G1AS --baselines fedbuff; }
lane_s() { sleep 30; pool s_gs_g1as_refl "${SPEECH[@]}" --tier G1AS --baselines refl --trainer-hp "learningRate=0.05"; }
lane lane_c; lane lane_s
for p in "${KIDS[@]}"; do wait "$p"; done
KIDS=()
say "BLOCK DONE"
