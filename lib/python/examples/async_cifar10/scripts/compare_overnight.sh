#!/bin/bash
# Post-run comparison for the overnight n300 runs. Two views:
#   1. PER-BASELINE sim-vs-real tracking (the only apples-to-apples pairing):
#      compare_parity.py for each baseline's sim run dir vs its real run dir.
#   2. CROSS-BASELINE, separately for sim and for real (one plot set each):
#      analyze_run.py --compare-streaming over the 4 runs of a mode.
#
#   compare_overnight.sh            # auto-discovers latest run dir per (baseline,mode)
# Output: /tmp/overnight_compare/{parity_<baseline>.txt, sim_cross/, real_cross/}
set -u
source ~/miniconda3/etc/profile.d/conda.sh && conda activate dg_flame
EX=/home/dgarg39/flame/lib/python/examples/async_cifar10
cd "$EX" || exit 1
ROOT=/home/dgarg39/flame
OUT=/tmp/overnight_compare; mkdir -p "$OUT"
BASELINES="felix oort refl feddance"

# latest run dir for a (baseline, mode) — names end in _stream_<mode>
rundir() { ls -dt experiments/run_*_${1}_n300_*_stream_${2}* 2>/dev/null | head -1; }

echo "### 1. PER-BASELINE sim-vs-real parity ###"
for b in $BASELINES; do
  dr=$(rundir "$b" real); ds=$(rundir "$b" sim)
  if [ -z "$dr" ] || [ -z "$ds" ]; then
    echo "  $b: missing run dir (real='$dr' sim='$ds')"; continue
  fi
  echo "  $b: real=$(basename "$dr") sim=$(basename "$ds")"
  python scripts/compare_parity.py \
    --real "$dr"/telemetry/aggregator_*.jsonl --sim "$ds"/telemetry/aggregator_*.jsonl \
    --real-trainer-dir "$dr"/telemetry/ --sim-trainer-dir "$ds"/telemetry/ \
    > "$OUT/parity_${b}.txt" 2>&1
  grep -E "\[OK\]|\[!!\]|\[XX\]|CHECKS" "$OUT/parity_${b}.txt" | sed 's/^/      /'
done

echo "### 2. CROSS-BASELINE (sim plots, then real plots) ###"
for mode in sim real; do
  dirs=(); labels=()
  for b in $BASELINES; do
    d=$(rundir "$b" "$mode"); [ -z "$d" ] && continue
    dirs+=("$d/telemetry"); labels+=("$b")
  done
  if [ ${#dirs[@]} -ge 2 ]; then
    echo "  $mode: ${labels[*]}"
    python "$ROOT"/scripts/analysis/analyze_run.py \
      --compare-streaming "${dirs[@]}" --labels "${labels[@]}" \
      --out "$OUT/${mode}_cross" > "$OUT/${mode}_cross.log" 2>&1
    echo "    -> $OUT/${mode}_cross/ ($(find "$OUT/${mode}_cross" -name '*.pdf' 2>/dev/null | wc -l) plots)"
  else
    echo "  $mode: <2 run dirs, skipping cross-baseline"
  fi
done
echo "DONE -> $OUT"
