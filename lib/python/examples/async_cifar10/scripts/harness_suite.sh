#!/usr/bin/env bash
# ============================================================================
# harness_suite.sh — no-GPU local harness (ROBUST_FL_READINESS S1 / FELIX FX-N1).
#
# For every (trace, baseline): one real + one sim leg via debug_run.sh on CPU
# (--harness stub|tiny_cpu); then per leg the ground-truth EVENT invariants
# (scripts/parity/event_invariants.py) and per pair the real<->sim parity battery.
# Every step has a hard timeout, so a hung leg or checker never stalls the suite.
# Operator-launched: killing this script kills every process it started.
#
# Output (one dir, read it afterwards):
#   <out>/summary.tsv|txt   trace, baseline, event verdict real/sim (+failing checks),
#                           parity verdict/score/roots, crash lines, timeout, run dirs
#   <out>/events/<trace>_<baseline>.{json,txt}   event-invariant report, both legs
#   <out>/parity/<trace>_<baseline>.{json,txt}   parity report
#   <out>/runs/<trace>_<baseline>/               debug_run.sh logs
#
# Usage:
#   harness_suite.sh [--harness stub] [--baselines 'felix oort ...'] [--traces 'syn_0 syn_50']
#                    [--runtime-s 300] [--num-trainers 60] [--delay-factor 4] [--trace-scale 4]
#                    [--timeout-buffer-s 300] [--output-dir DIR] [--agg-hp 'k=v ...'] [--dry-run]
# ============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EX_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LIB_DIR="$(cd "$EX_DIR/../.." && pwd)"
# shellcheck source=../../scripts/expt_runner.sh
source "$LIB_DIR/examples/scripts/expt_runner.sh"
export FLAME_CONDA_ENV="${FLAME_CONDA_ENV:-dg_flame}"   # an active `base` shell must not win
export CUDA_VISIBLE_DEVICES=""   # CPU only, even on a node whose driver/torch mismatch (S0)

HARNESS=stub
BASELINES="felix fedbuff oort oort_star refl feddance"
TRACES="syn_0"
RUNTIME_S=300
NUM_TRAINERS=60
DELAY_FACTOR=4
TRACE_SCALE=""
TIMEOUT_BUFFER_S=300
CHECK_TIMEOUT_S=900
OUT="$EX_DIR/experiments/harness_$(date +%Y%m%d_%H%M%S)"
DRY_RUN=0
AGG_HP=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --harness)          HARNESS="$2"; shift 2 ;;
    --baselines)        BASELINES="$2"; shift 2 ;;
    --traces)           TRACES="$2"; shift 2 ;;
    --runtime-s)        RUNTIME_S="$2"; shift 2 ;;
    --num-trainers)     NUM_TRAINERS="$2"; shift 2 ;;
    --delay-factor)     DELAY_FACTOR="$2"; shift 2 ;;
    --trace-scale)      TRACE_SCALE="$2"; shift 2 ;;
    --timeout-buffer-s) TIMEOUT_BUFFER_S="$2"; shift 2 ;;
    --output-dir)       OUT="$2"; shift 2 ;;
    --agg-hp)           AGG_HP="$2"; shift 2 ;;
    --dry-run)          DRY_RUN=1; shift ;;
    -h|--help)          sed -n 2,23p "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
# Compress availability time with the delays so a short leg still crosses trace transitions.
if [ -n "$TRACE_SCALE" ]; then export FLAME_TRACE_TIME_SCALE="$TRACE_SCALE"; else unset FLAME_TRACE_TIME_SCALE; fi

PY="$(conda run -n "$FLAME_CONDA_ENV" which python 2>/dev/null | tail -1)"
[ -x "$PY" ] || { echo "cannot resolve python for env $FLAME_CONDA_ENV" >&2; exit 2; }

mkdir -p "$OUT/parity" "$OUT/events" "$OUT/runs"
SUMMARY_TSV="$OUT/summary.tsv"
printf 'trace\tbaseline\tev_real\tev_sim\tparity\tscore\tn_fail\troots\tcrash_lines\ttimeout\treal_dir\tsim_dir\n' > "$SUMMARY_TSV"
echo "harness=$HARNESS baselines='$BASELINES' traces='$TRACES' runtime_s=$RUNTIME_S n=$NUM_TRAINERS" \
     "delay_factor=$DELAY_FACTOR trace_scale=${TRACE_SCALE:-1} agg_hp='$AGG_HP' env=$FLAME_CONDA_ENV" \
  | tee "$OUT/suite.cfg"

_find_dir() {  # $1 marker, $2 baseline, $3 real|sim -- newest matching run dir after the marker
  find "$EX_DIR/experiments" -maxdepth 1 -type d -newer "$1" -name "run_*dbg_${2}_n*_${3}" 2>/dev/null | sort | tail -1
}

_crash_lines() {
  local n=0 d
  for d in "$@"; do
    [ -d "$d" ] || continue
    n=$(( n + $(grep -rhE "Traceback \(most recent call last\)|CRITICAL|Segmentation fault" "$d" --include='*.log' 2>/dev/null | wc -l) ))
  done
  echo "$n"
}

_event_verdict() {  # $1 event json, $2 leg index -> PASS | FAIL:check,check | MISSING
  python3 - "$1" "$2" <<'PY'
import json, sys
try:
    r = json.load(open(sys.argv[1]))[int(sys.argv[2])]
except Exception:
    print("MISSING"); raise SystemExit
bad = [k.split("_")[0] for k, v in r["checks"].items() if v["status"] in ("FAIL", "ERROR")]
print("PASS" if not bad else "FAIL:" + ",".join(bad))
PY
}

for trace in $TRACES; do
  for b in $BASELINES; do
    label="${trace}_${b}"
    run_dir="$OUT/runs/$label"; mkdir -p "$run_dir"
    marker="$run_dir/.ts_start"; touch "$marker"
    args=(--baselines "$b" --mode both --runtime-s "$RUNTIME_S" --num-trainers "$NUM_TRAINERS"
          --harness "$HARNESS" --delay-factor "$DELAY_FACTOR" --trace "$trace")
    [ -n "$AGG_HP" ] && args+=(--agg-hp "$AGG_HP")
    echo "=== [$(date '+%F %T')] [$label] debug_run.sh ${args[*]}"
    if [ "$DRY_RUN" = 1 ]; then
      env FLAME_LOGDIR="$run_dir" bash "$SCRIPT_DIR/debug_run.sh" "${args[@]}" --dry-run > "$run_dir/shell.log" 2>&1
      tail -3 "$run_dir/shell.log"; continue
    fi
    # real + sim run back to back: budget both legs
    expt_timed_run "$label" $(( 2 * RUNTIME_S )) "$TIMEOUT_BUFFER_S" "$run_dir/shell.log" -- \
      env FLAME_LOGDIR="$run_dir" bash "$SCRIPT_DIR/debug_run.sh" "${args[@]}"
    rc=$?; timed_out=0; [ "$rc" = 124 ] && timed_out=1

    real_dir="$(_find_dir "$marker" "$b" real)"; sim_dir="$(_find_dir "$marker" "$b" sim)"
    crashes="$(_crash_lines "$real_dir" "$sim_dir")"

    ev_real=MISSING; ev_sim=MISSING
    ev_json="$OUT/events/${label}.json"
    legs=(); [ -n "$real_dir" ] && legs+=("$real_dir"); [ -n "$sim_dir" ] && legs+=("$sim_dir")
    if [ "${#legs[@]}" -gt 0 ]; then
      ( cd "$EX_DIR" && timeout "$CHECK_TIMEOUT_S" "$PY" -u scripts/parity/event_invariants.py "${legs[@]}" \
          --json-out "$ev_json" ) > "$OUT/events/${label}.txt" 2>&1 || true
      i=0
      if [ -n "$real_dir" ]; then ev_real="$(_event_verdict "$ev_json" $i)"; i=$((i + 1)); fi
      if [ -n "$sim_dir" ]; then ev_sim="$(_event_verdict "$ev_json" $i)"; fi
    fi

    verdict=MISSING; score=""; nfail=""; roots=""
    if [ -n "$real_dir" ] && [ -n "$sim_dir" ]; then
      json="$OUT/parity/${label}.json"
      ( cd "$EX_DIR" && PYTHONIOENCODING=utf-8 timeout "$CHECK_TIMEOUT_S" "$PY" -u scripts/parity_check.py \
          --real "$real_dir" --sim "$sim_dir" --agg-goal 10 --budget-s "$RUNTIME_S" --json-out "$json" ) \
          > "$OUT/parity/${label}.txt" 2>&1 || true
      if [ -f "$json" ]; then
        read -r verdict score nfail roots < <(python3 - "$json" <<'PY'
import json, sys
s = json.load(open(sys.argv[1])).get("summary", {})
print("PASS" if s.get("passed") else "FAIL", s.get("score"), s.get("n_fail"),
      ",".join(map(str, s.get("roots") or [])) or "-")
PY
)
      else
        verdict=CHECKER_ERROR
      fi
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$trace" "$b" "$ev_real" "$ev_sim" "$verdict" \
      "$score" "$nfail" "$roots" "$crashes" "$timed_out" "$real_dir" "$sim_dir" >> "$SUMMARY_TSV"
    echo "  [$label] events real=$ev_real sim=$ev_sim | parity=$verdict score=$score roots=$roots | crashes=$crashes timeout=$timed_out"
  done
done

column -t -s $'\t' "$SUMMARY_TSV" > "$OUT/summary.txt"
echo; cat "$OUT/summary.txt"
echo; echo "results: $OUT"
