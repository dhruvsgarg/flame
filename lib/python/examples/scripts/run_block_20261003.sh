#!/usr/bin/env bash
# One 8h block on jayne (FX-N68, FX-N5, FX-N4): feddance replicates (3 sim + 1 real), G2 oort, speech G1 felix + fedbuff.
# Usage: run_block_20261003.sh [OUT_DIR]   Ctrl+C tears down the running pool and stops the block (R20).
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:-$HERE/../experiments/block_$(date +%Y%m%d_%H%M)}"
mkdir -p "$OUT"
PY=(conda run --no-capture-output -n dg_flame python)
POOL=("${PY[@]}" "$HERE/harness_pool.py" --datasets)
CHILD=
interrupt() { [ -n "$CHILD" ] && kill -INT "$CHILD" 2>/dev/null && wait "$CHILD"; echo "interrupted" >> "$OUT/BLOCK.log"; exit 130; }
trap interrupt INT TERM
say() { echo "[$(date '+%F %T')] $*" | tee -a "$OUT/BLOCK.log"; }
run() {  # run LABEL ARGS... : one pool, a failure never stops the block
  local label="$1"; shift
  say "START $label"
  "${POOL[@]}" "$@" ${BLOCK_EXTRA:-} --output-dir "$OUT/$label" > "$OUT/$label.out" 2>&1 &
  CHILD=$!; wait "$CHILD"; say "DONE  $label rc=$?"; CHILD=
}

# 1. feddance sim replicates: sim<->sim spread of the lock-in draw (~27 min). The first pool runs the gate.
for i in 1 2 3; do run "g2s_$i" cifar10 --tier G2S --baselines feddance $([ "$i" -gt 1 ] && echo --no-gate); done
SIMS=$(for i in 1 2 3; do awk 'NR==2{for(j=1;j<=NF;j++) if($j ~ /_sim$/) print $j}' "$OUT"/g2s_$i/G2S/*/summary.txt 2>/dev/null; done)
# the sim leg's own dir is named in its pool's legs list; fall back to newest G2S sims
[ -z "$SIMS" ] && SIMS=$(ls -dt "$HERE"/../async_cifar10/experiments/run_*_G2S_dbg_feddance_*_sim | head -n 3)
"${PY[@]}" "$HERE/replicate_spread.py" $SIMS | tee "$OUT/g2s_spread.txt"
# 2. feddance real replicate (~1.7h): real<->real spread vs the overnight real leg (FX-N68 floor)
run g2c cifar10 --tier G2C --baselines feddance --no-gate
REALS="$(ls -d "$HERE"/../async_cifar10/experiments/run_20261003_032639_G2_dbg_feddance_*_real) $(ls -dt "$HERE"/../async_cifar10/experiments/run_*_G2C_dbg_feddance_*_real | head -n 1)"
"${PY[@]}" "$HERE/replicate_spread.py" $REALS | tee "$OUT/g2c_spread.txt"
# 3. G2 oort at n=300 (FX-D39 verify, FX-N5; ~2.2h). oort_star is graded with unavailability, not syn_0.
run g2_oort cifar10 --tier G2 --baselines oort --no-gate
# 4. speech G1 felix + fedbuff, two legs side by side on 4 GPUs each (FX-N4; ~2.5h)
run gs_g1 google_speech --tier G1 --gpus-per-job 4 --no-gate
say "BLOCK DONE"
