#!/usr/bin/env bash
# PR27: confirm FX-D121-D124 (jayne only). Stages run IN PARALLEL: the pools lease disjoint cores/ports/GPUs node-wide (L28), so a
# stage that doesn't fit waits for leases. Early probes (early_probe.py) print PASS/FAIL per claim while the legs run; the end check
# flags interference (throttled legs or dur > 1.25 x est) -> rerun that stage solo (ROBUST R24).
# Usage: scripts/run_pr27.sh [stage...]   (default: A B C D; A feddance G0U, B feddance G0UC, C T3 Oort, D cifar Oort G0U)
set -u
cd "$(dirname "$0")/.."
ts=$(date +%Y%m%d_%H%M); since=$(date +%s); out=experiments/pr27_$ts; mkdir -p "$out"
stages=("$@"); [ $# -eq 0 ] && stages=(A B C D)
declare -A ARGS=(
  [A]="--tier G0U --datasets google_speech --baselines feddance"
  [B]="--tier G0UC --datasets google_speech --baselines feddance"
  [C]="--tier T3 --datasets all --baselines oort,oort_star"
  [D]="--tier G0U --datasets cifar10 --baselines oort,oort_star")
pids=()
trap 'kill -INT "${pids[@]}" 2>/dev/null; sleep 5; kill 0; exit 130' INT TERM
for s in "${stages[@]}"; do
  conda run --no-capture-output -n dg_flame python scripts/harness_pool.py --output-dir "experiments/pool_${ts}_PR27_$s" ${ARGS[$s]} \
    > "$out/$s.out" 2>&1 &
  pids+=($!); sleep 90   # stagger so each pool's leases settle before the next sizes itself
done
python3 -I scripts/early_probe.py --since "$since" --watch 7200 > "$out/probes.txt" 2>&1 &
wait "${pids[@]}"
for s in "${stages[@]}"; do
  awk -F'\t' -v s="$s" 'NR>1 && ($8>0 || $3>1.25*$4) {printf "INTERFERENCE stage %s: %s dur=%ss est=%ss sat=%s -> rerun solo\n", s, $1, $3, $4, $8}' \
    "experiments/pool_${ts}_PR27_$s/jobs.tsv" 2>/dev/null
done > "$out/interference.txt"
python3 -I scripts/early_probe.py --since "$since" > "$out/probes_final.txt" 2>&1
echo "done: $out (probes.txt, probes_final.txt, interference.txt)"
