#!/usr/bin/env bash
# Run 19 screen (3h on jayne, 8 GPUs): short legs to close roots before the one overnight block (C0, C11).
# Lane S (GPU): speech G0U + G0UC syn_50 for fedbuff/refl/oort_star/feddance (first speech floors; FX-D50/D55/D66 on GPU), felix G0U.
#   oort syn_50 confirmed 10-07 (FX-D66); felix/oort_star floors skipped (cifar spread 0.02/0.06).
# Lane C (GPU): cifar G0U syn_50 + mobiperf, all but oort (confirmed 10-07), graded on the runs 12-14 G0UC floors (linked).
# Lane A (accuracy): in-process cifar fedbuff lr pairs (fl_lr_check), then speech refl G1AS on its paper SGD 0.005 (never run).
# Lane T (CPU, after lane C): T3C oort/oort_star/feddance syn_50 floors for run 18's 7 T3 syn_50 reds.
# No leg starts past T0+3h. After: parity_ladder.py --grade <pool> --regrade --max-stage 9; accuracy_table.py on lane A (C5).
# Usage: run_block_20261007b.sh [--detach] [OUT_DIR]. Stop: kill -INT -- -$(cat OUT/PGID).
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ "${1:-}" = "--detach" ]; then
  shift
  OUT="${1:-$HERE/../experiments/block_$(date +%Y%m%d_%H%M)_run19}"
  mkdir -p "$OUT"
  setsid nohup bash "${BASH_SOURCE[0]}" "$OUT" > "$OUT/block.out" 2>&1 < /dev/null &
  echo "$!" > "$OUT/PGID"
  echo "detached: $OUT  (log: $OUT/BLOCK.log; stop: kill -INT -- -$!)"
  exit 0
fi
OUT="${1:-$HERE/../experiments/block_$(date +%Y%m%d_%H%M)_run19}"
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
left_h() { python3 -c "print(round(max(0.05, ($T0 + 3.0 * 3600 - $(date +%s)) / 3600), 2))"; }
GPUS=(--gpu-ids 0,1,2,3,4,5,6,7 --no-gate)
CIFAR=(--datasets cifar10 --gpu-cpus-per-trainer 0.2 "${GPUS[@]}")
SPEECH=(--datasets google_speech "${GPUS[@]}")
ln -sfn "$HERE/../experiments/block_20261005_0401/cifar_uc" "$OUT/floors_cifar_uc_1005"  # G0UC floors (parity_ladder sibling glob)

lane() { ( trap 'exit 130' INT TERM; "$@" ) & KIDS+=($!); }
lane_s() {
  pool s_gs_g0u "${SPEECH[@]}" --tier G0U,G0UC --traces syn_50 --baselines "fedbuff refl oort_star feddance"
}
lane_s2() {
  sleep 60
  pool s2_gs_g0u_felix "${SPEECH[@]}" --tier G0U --traces syn_50 --baselines felix
}
lane_c() {
  sleep 30
  pool c_cifar_g0u "${CIFAR[@]}" --tier G0U --baselines "felix fedbuff refl oort_star feddance"
  pool t_t3c --datasets all --tier T3C --baselines "oort oort_star feddance" --traces syn_50 --no-gate
}
lane_a() {
  say "START a_lr_cifar_fedbuff"
  "${PY[@]}" "$HERE/fl_lr_check.py" --dataset cifar10 --k 10 --rate 1.0 --batch 32 --rounds 120 --eval-every 20 \
    --staleness 5 --flame-opt fedbuff --gpus 7,6,5,4 --pairs 0.000195:40.9 0.04:1.0 0.04:0.3 0.01:1.0 > "$OUT/a_lr_cifar_fedbuff.out" 2>&1
  say "DONE  a_lr_cifar_fedbuff rc=$?"
  pool a_gs_g1as_refl "${SPEECH[@]}" --tier G1AS --baselines refl
}
lane lane_s; lane lane_s2; lane lane_c; lane lane_a
for p in "${KIDS[@]}"; do wait "$p"; done
KIDS=()
say "BLOCK DONE"
