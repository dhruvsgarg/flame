#!/usr/bin/env bash
# ============================================================================
# harness_overnight.sh — the unattended no-GPU harness campaign (ROBUST_FL_READINESS S1).
#
# Runs pytest, then a fixed list of harness_suite.sh phases (baselines x traces x
# harness modes), each step under a hard timeout; a failed or hung step is recorded
# and the campaign moves on. Stops launching new phases past --deadline-h.
#
# Needs a node with NO other FL workers of yours (the suite's timeout path kills
# stray trainers/aggregators by name). CPU only; GPUs untouched.
#
#   bash lib/python/examples/async_cifar10/scripts/harness_overnight.sh [--deadline-h 5.5] [--phases 'P0 P1 ...'] [--dry-run]
#
# Read afterwards: experiments/overnight_<ts>/SUMMARY.txt (+ per-phase dirs).
# ============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EX_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LIB_DIR="$(cd "$EX_DIR/../.." && pwd)"
export FLAME_CONDA_ENV="${FLAME_CONDA_ENV:-dg_flame}"
export EXPT_AUTOCLEAN=1   # a timed-out leg's leftovers must not block the next launch

DEADLINE_H=5.5
PHASES="P0 P1 P2 P3 P4 P5 P6"
DRY=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --deadline-h) DEADLINE_H="$2"; shift 2 ;;
    --phases)     PHASES="$2"; shift 2 ;;
    --dry-run)    DRY="--dry-run"; shift ;;
    -h|--help)    sed -n 2,15p "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

ROOT="$EX_DIR/experiments/overnight_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$ROOT"
exec > >(tee -a "$ROOT/overnight.log") 2>&1
T0=$(date +%s)
DEADLINE_S=$(python3 -c "print(int($DEADLINE_H * 3600))")

# Refuse to start next to someone else's FL run: autoclean would kill it.
STRAYS="$(pgrep -u "$(id -u)" -af 'trainer/pytorch/main.py|trainer/forward_training|aggregator/pytorch/main_|flame.launch.run_experiment' | grep -v pgrep || true)"
if [ -n "$STRAYS" ] && [ -z "$DRY" ]; then
  echo "ABORT: FL processes of yours are already running on $(hostname); the harness would kill them:"
  echo "$STRAYS" | head -20
  exit 3
fi

B6="felix fedbuff oort oort_star refl feddance"
# id | est min | harness_suite args
PHASE_TABLE=(
  "P1|50|--baselines '$B6' --traces syn_0 --runtime-s 300"
  "P2|75|--baselines '$B6' --traces syn_50 --runtime-s 450 --trace-scale 4"
  "P3|75|--baselines '$B6' --traces mobiperf_3st --runtime-s 450 --trace-scale 4"
  "P4|17|--baselines 'felix fedbuff' --traces syn_0 --runtime-s 300 --agg-hp 'simColdStartGate=true'"
  "P5|25|--baselines 'felix refl' --traces syn_20 --runtime-s 450 --trace-scale 4"
  "P6|17|--harness tiny_cpu --baselines 'felix oort' --traces syn_0 --runtime-s 300"
)
echo "overnight root: $ROOT  host=$(hostname)  deadline=${DEADLINE_H}h  phases='$PHASES'  commit=$(git -C "$LIB_DIR" rev-parse --short HEAD)$(git -C "$LIB_DIR" diff --quiet || echo '+dirty')"
echo "P0 pytest ~15m | P1 syn_0 x6 ~50m | P2 syn_50 x6 ~75m | P3 mobiperf_3st x6 ~75m | P4 cold-start A/B ~17m | P5 syn_20 ~25m | P6 tiny_cpu ~17m"

_elapsed() { echo $(( $(date +%s) - T0 )); }
_want() { [[ " $PHASES " == *" $1 "* ]]; }

if _want P0; then
  echo "=== [$(date '+%F %T')] P0 pytest"
  if [ -z "$DRY" ]; then
    ( cd "$LIB_DIR" && timeout 2400 conda run --no-capture-output -n "$FLAME_CONDA_ENV" python -m pytest -q \
        -p no:cacheprovider tests examples/async_cifar10/scripts/parity examples/async_cifar10/trainer/pytorch \
        examples/fwdllm/expt_scripts ) > "$ROOT/P0_pytest.txt" 2>&1
    echo "  P0 rc=$? :: $(tail -1 "$ROOT/P0_pytest.txt")"
  fi
fi

for row in "${PHASE_TABLE[@]}"; do
  IFS='|' read -r pid est args <<< "$row"
  _want "$pid" || continue
  if [ "$(_elapsed)" -ge "$DEADLINE_S" ]; then
    echo "=== [$(date '+%F %T')] $pid SKIPPED: past the ${DEADLINE_H}h deadline"; continue
  fi
  echo "=== [$(date '+%F %T')] $pid (~${est}m): harness_suite.sh $args"
  # Hard ceiling per phase: 2x its estimate, so one wedged phase can't eat the night.
  eval timeout --kill-after=60 $(( est * 120 )) bash "$SCRIPT_DIR/harness_suite.sh" $args \
      --output-dir "$ROOT/$pid" $DRY > "$ROOT/${pid}.log" 2>&1
  rc=$?
  echo "  $pid rc=$rc elapsed=$(( $(_elapsed) / 60 ))m"
  [ "$rc" = 124 ] && echo "  $pid HIT ITS PHASE CEILING — later rows of this phase are missing"
done

# One table across phases.
{
  echo "overnight $ROOT  total=$(( $(_elapsed) / 60 ))m"
  [ -f "$ROOT/P0_pytest.txt" ] && echo "P0 pytest: $(tail -1 "$ROOT/P0_pytest.txt")"
  echo
  printf 'phase\t'; head -1 "$(ls "$ROOT"/P*/summary.tsv 2>/dev/null | head -1)" 2>/dev/null | cut -f1-10
  for f in "$ROOT"/P*/summary.tsv; do
    [ -f "$f" ] || continue
    p="$(basename "$(dirname "$f")")"
    tail -n +2 "$f" | cut -f1-10 | sed "s/^/$p\t/"
  done
} | column -t -s $'\t' > "$ROOT/SUMMARY.txt"
echo; cat "$ROOT/SUMMARY.txt"
echo; echo "done: $ROOT/SUMMARY.txt"
