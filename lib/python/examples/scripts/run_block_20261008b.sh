#!/usr/bin/env bash
# Run 25 (~5h on jayne, 8 GPUs): FX-N77 transport fixes (zero-copy MQTT, pipelined chunks, EOT before LEAVE) under unavailability.
# Phase 0 (~8 min): PL3 scale smoke, speech + cifar G0U felix syn_50 together at production density; aborts the block on failure.
# Phase A (~35 min): CHG real syn_0 legs (G0U cohort, all six, both datasets) -> re-derive sim_charge_profiles (L17).
# Phase B (rest; no leg starts past T0+5h): G0U + G0UC all six: cifar x {syn_50, mobiperf_3st} | speech x syn_50 (PR20 b, PR21 part).
# After: parity_ladder.py --grade <pool> --regrade --max-stage 9 per pool; accuracy_table.py (C5).
# Usage: run_block_20261008b.sh [--detach] [OUT_DIR]; BLOCK_PHASES="0" runs phase 0 only. Stop: kill -INT -- -$(cat OUT/PGID).
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ "${1:-}" = "--detach" ]; then
  shift
  OUT="${1:-$HERE/../experiments/block_$(date +%Y%m%d_%H%M)_run25}"
  mkdir -p "$OUT"
  setsid nohup bash "${BASH_SOURCE[0]}" "$OUT" > "$OUT/block.out" 2>&1 < /dev/null &
  echo "$!" > "$OUT/PGID"
  echo "detached: $OUT  (log: $OUT/BLOCK.log; stop: kill -INT -- -$!)"
  exit 0
fi
OUT="${1:-$HERE/../experiments/block_$(date +%Y%m%d_%H%M)_run25}"
mkdir -p "$OUT"
PHASES="${BLOCK_PHASES:-0 A B}"
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
left_h() { python3 -c "print(round(max(0.05, ($T0 + 5.0 * 3600 - $(date +%s)) / 3600), 2))"; }
GPUS=(--gpu-ids 0,1,2,3,4,5,6,7 --no-gate)
CIFAR=(--datasets cifar10 --gpu-cpus-per-trainer 0.2 "${GPUS[@]}")
SPEECH=(--datasets google_speech "${GPUS[@]}")
lane() { ( trap 'exit 130' INT TERM; "$@" ) & KIDS+=($!); }
later() { sleep "$1"; shift; "$@"; }  # stagger a second lane's lease scan
join_lanes() { for p in "${KIDS[@]}"; do wait "$p"; done; KIDS=(); }
has() { [[ " $PHASES " == *" $1 "* ]]; }

if has 0; then
  lane pool s0_gs_g0u "${SPEECH[@]}" --tier G0U --baselines felix --traces syn_50 --scale-smoke 300
  lane later 20 pool s0_cifar_g0u "${CIFAR[@]}" --tier G0U --baselines felix --traces syn_50 --scale-smoke 300
  join_lanes
  bad=$(grep "DONE  s0_.*rc=[1-9]" "$OUT/BLOCK.log")
  bad+=$(ls "$OUT"/s0_*/ABORT.txt "$OUT"/s0_*/*/*/STALLED.txt 2>/dev/null)
  bad+=$(grep -h "GPU_TIGHT\|RAM_TIGHT\|SKIP " "$OUT"/s0_*/pool.log 2>/dev/null)
  bad+=$(grep -hE "MISSING|CHECKER_ERROR" "$OUT"/s0_*/SUMMARY.txt 2>/dev/null)
  if [ -n "$bad" ]; then say "ABORT phase 0: $bad"; exit 4; fi
  say "PHASE 0 OK"
fi

if has A; then
  lane pool a_gs_chg "${SPEECH[@]}" --tier CHG
  lane later 20 pool a_cifar_chg "${CIFAR[@]}" --tier CHG
  join_lanes
  reals=$(awk -F'\t' 'FNR > 1 && $11 != "" {print $11}' "$OUT"/a_*_chg/*/summary.tsv 2>/dev/null | sort -u)
  for ds in cifar10 google_speech; do
    n=$(echo "$reals" | grep -c "$([ $ds = google_speech ] && echo _gs_ || echo /run_[0-9_]*_CHG_)")
    if [ "$n" -ge 4 ]; then
      "${PY[@]}" "$HERE/profile_felix_charges.py" --dataset $ds --harness gpu \
        --out "$HERE/../async_cifar10/sim_charge_profiles/gpu_$ds.yaml" $reals >> "$OUT/BLOCK.log" 2>&1
      say "PROFILE $ds rc=$? from $n real legs"
    else
      say "PROFILE $ds skipped: $n real legs (kept the old profile)"
    fi
  done
  say "PHASE A DONE"
fi

if has B; then
  lane pool b_gs_g0u "${SPEECH[@]}" --tier G0U,G0UC --traces syn_50
  lane later 30 pool b_cifar_g0u "${CIFAR[@]}" --tier G0U,G0UC
  join_lanes
fi
say "BLOCK DONE"
