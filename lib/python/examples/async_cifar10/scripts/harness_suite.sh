#!/usr/bin/env bash
# ============================================================================
# harness_suite.sh — no-GPU local harness (ROBUST_FL_READINESS S1 / FELIX FX-N22).
#
# For every (trace, baseline): the requested legs via debug_run.sh (--harness stub|tiny_cpu on CPU, or
# none = the production GPU path); then per leg the ground-truth EVENT invariants
# (scripts/parity/event_invariants.py) and per pair the real<->sim parity battery. A sim-only run grades
# against a stored real leg (--real-from). Every step has a hard timeout.
# Operator-launched: killing this script kills every process it started.
#
# Output (one dir, read it afterwards):
#   <out>/summary.tsv|txt   trace, baseline, event verdict real/sim (+failing checks), parity
#                           verdict/score/roots, crash lines, timeout, run dirs, real-leg source
#   <out>/events/<trace>_<baseline>.{json,txt}   event-invariant report, own legs
#   <out>/parity/<trace>_<baseline>.{json,txt}   parity report
#   <out>/runs/<trace>_<baseline>/               debug_run.sh logs (+ legs.txt: this pair's run dirs)
#
# Usage:
#   harness_suite.sh [--harness stub|tiny_cpu|none] [--baselines 'felix oort ...'] [--traces 'syn_0 syn_50']
#                    [--mode both|real|sim] [--real-from auto|<real run dir>|<suite/campaign dir>]
#                    [--runtime-s 300] [--num-trainers 60] [--delay-factor 4] [--trace-scale 4]
#                    [--timeout-buffer-s 300] [--output-dir DIR] [--agg-hp 'k=v ...'] [--trainer-hp 'k=v ...']
#                    [--inject-bug no_busy_hold|order_by_sct|freeze_trainer_clock]
#                    [--isolate [--broker-port P] [--run-tag T]] [--gpu-ids 0,1,..] [--dry-run]
#                    [--dataset cifar10|google_speech] [--agg-goal N] [--concurrency C] [--sim-ceiling-x X]
#   harness_suite.sh --grade-only --real-dir R --sim-dir S --baselines B --traces T --runtime-s N --output-dir DIR
#
# --isolate (FX-N22 slot): private mosquitto on its own port + a run tag scoping every sweep, so this
# suite can run beside others. Cores come from the caller's affinity (examples/scripts/harness_pool.py uses taskset).
# ============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EX_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LIB_DIR="$(cd "$EX_DIR/../.." && pwd)"
# shellcheck source=../../scripts/expt_runner.sh
source "$LIB_DIR/examples/scripts/expt_runner.sh"
export FLAME_CONDA_ENV="${FLAME_CONDA_ENV:-dg_flame}"   # an active `base` shell must not win

HARNESS=stub
BASELINES="felix fedbuff oort oort_star refl feddance"
TRACES="syn_0"
MODE=both
REAL_FROM=""
RUNTIME_S=300
NUM_TRAINERS=60
DELAY_FACTOR=""   # default 4 on the CPU harness, unscaled on the GPU path (--harness none)
TRACE_SCALE=""
TIMEOUT_BUFFER_S=300
CHECK_TIMEOUT_S=900
OUT="$EX_DIR/experiments/harness_$(date +%Y%m%d_%H%M%S)"
DRY_RUN=0
AGG_HP=""
TRAINER_HP=""
INJECT_BUG=""
ISOLATE=0
BROKER_PORT=""
RUN_TAG=""
GPU_IDS=""
GRADE_ONLY=0
DATASET=cifar10
AGG_GOAL=""; CONC=""; SIM_CEIL_X=1
GIVEN_REAL=""
GIVEN_SIM=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --harness)          HARNESS="$2"; shift 2 ;;
    --baselines)        BASELINES="$2"; shift 2 ;;
    --traces)           TRACES="$2"; shift 2 ;;
    --mode)             MODE="$2"; shift 2 ;;
    --real-from)        REAL_FROM="$2"; shift 2 ;;
    --runtime-s)        RUNTIME_S="$2"; shift 2 ;;
    --num-trainers)     NUM_TRAINERS="$2"; shift 2 ;;
    --delay-factor)     DELAY_FACTOR="$2"; shift 2 ;;
    --trace-scale)      TRACE_SCALE="$2"; shift 2 ;;
    --timeout-buffer-s) TIMEOUT_BUFFER_S="$2"; shift 2 ;;
    --output-dir)       OUT="$2"; shift 2 ;;
    --agg-hp)           AGG_HP="$2"; shift 2 ;;
    --trainer-hp)       TRAINER_HP="$2"; shift 2 ;;
    --inject-bug)       INJECT_BUG="$2"; shift 2 ;;
    --isolate)          ISOLATE=1; shift ;;
    --broker-port)      BROKER_PORT="$2"; shift 2 ;;
    --run-tag)          RUN_TAG="$2"; shift 2 ;;
    --gpu-ids)          GPU_IDS="$2"; shift 2 ;;
    --grade-only)       GRADE_ONLY=1; shift ;;
    --dataset)          DATASET="$2"; shift 2 ;;
    --agg-goal)         AGG_GOAL="$2"; shift 2 ;;
    --concurrency)      CONC="$2"; shift 2 ;;
    --sim-ceiling-x)    SIM_CEIL_X="$2"; shift 2 ;;
    --real-dir)         GIVEN_REAL="$2"; shift 2 ;;
    --sim-dir)          GIVEN_SIM="$2"; shift 2 ;;
    --dry-run)          DRY_RUN=1; shift ;;
    -h|--help)          sed -n 2,30p "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
case "$MODE" in both|real|sim) ;; *) echo "--mode must be both|real|sim" >&2; exit 2 ;; esac
case "$HARNESS" in stub|tiny_cpu|none) ;; *) echo "--harness must be stub|tiny_cpu|none" >&2; exit 2 ;; esac
[ -z "$DELAY_FACTOR" ] && [ "$HARNESS" != none ] && DELAY_FACTOR=4
if [ "$HARNESS" = none ]; then
  if [ -n "$GPU_IDS" ]; then export FLAME_GPU_IDS="$GPU_IDS" FLAME_SLOT_GPUS="$GPU_IDS"; fi
else
  export CUDA_VISIBLE_DEVICES=""   # CPU only, even on a node whose driver/torch mismatch (S0)
fi
# Compress availability time with the delays so a short leg still crosses trace transitions.
if [ -n "$TRACE_SCALE" ]; then export FLAME_TRACE_TIME_SCALE="$TRACE_SCALE"; else unset FLAME_TRACE_TIME_SCALE; fi
# S1: re-inject a known sim bug (flame.harness.injected) so the checker must catch it.
if [ -n "$INJECT_BUG" ]; then export FLAME_INJECT_BUG="$INJECT_BUG"; else unset FLAME_INJECT_BUG; fi

PY="$(conda run -n "$FLAME_CONDA_ENV" which python 2>/dev/null | tail -1)"
[ -x "$PY" ] || { echo "cannot resolve python for env $FLAME_CONDA_ENV" >&2; exit 2; }
mkdir -p "$OUT/parity" "$OUT/events" "$OUT/runs"

# FX-N22 slot isolation: private broker + run tag; torn down on any exit.
BROKER_PID=""
_stop_broker() { [ -n "$BROKER_PID" ] && kill "$BROKER_PID" 2>/dev/null; BROKER_PID=""; }
trap _stop_broker EXIT
# Ctrl+C/SIGTERM stops the suite; a running leg is torn down by expt_timed_run's own trap.
trap 'echo "[$(date "+%F %T")] INTERRUPT — harness_suite stopped" >&2; exit 130' INT TERM
if [ "$ISOLATE" = 1 ] && [ "$GRADE_ONLY" = 0 ] && [ "$DRY_RUN" = 0 ]; then
  export FLAME_RUN_TAG="${RUN_TAG:-slot_$(basename "$OUT")_$$}"
  [ -n "$BROKER_PORT" ] || BROKER_PORT="$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1])')"
  printf 'listener %s 127.0.0.1\nallow_anonymous true\npersistence false\nlog_dest file %s\n' \
    "$BROKER_PORT" "$OUT/mosquitto.log" > "$OUT/mosquitto.conf"
  "${MOSQUITTO:-$(command -v mosquitto || echo /usr/sbin/mosquitto)}" -c "$OUT/mosquitto.conf" > /dev/null 2>&1 &
  BROKER_PID=$!
  for _ in $(seq 50); do (exec 3<>"/dev/tcp/127.0.0.1/$BROKER_PORT") 2>/dev/null && break; sleep 0.1; done
  (exec 3<>"/dev/tcp/127.0.0.1/$BROKER_PORT") 2>/dev/null || { echo "private broker failed on port $BROKER_PORT" >&2; exit 5; }
  export FLAME_MQTT_BROKER="localhost:$BROKER_PORT"
fi

SUMMARY_TSV="$OUT/summary.tsv"
printf 'trace\tbaseline\tev_real\tev_sim\tparity\tscore\tn_fail\troots\tcrash_lines\ttimeout\treal_dir\tsim_dir\treal_src\n' > "$SUMMARY_TSV"
echo "dataset=$DATASET agg_goal=${AGG_GOAL:-cfg} c=${CONC:-cfg} sim_ceiling_x=$SIM_CEIL_X harness=$HARNESS baselines='$BASELINES' traces='$TRACES' mode=$MODE real_from='$REAL_FROM' runtime_s=$RUNTIME_S n=$NUM_TRAINERS" \
     "delay_factor=$DELAY_FACTOR trace_scale=${TRACE_SCALE:-1} agg_hp='$AGG_HP' trainer_hp='$TRAINER_HP' inject_bug='$INJECT_BUG'" \
     "run_tag=${FLAME_RUN_TAG:-} broker=${FLAME_MQTT_BROKER:-system} cpus=$(taskset -cp $$ 2>/dev/null | awk '{print $NF}') gpus=${GPU_IDS:-} env=$FLAME_CONDA_ENV" \
  | tee "$OUT/suite.cfg"

_bank_key() {  # $1 trace, $2 baseline -> the real-bank key options (harness_bank.KEY_FIELDS)
  printf '%s\0' --baseline "$2" --trace "$1" --n "$NUM_TRAINERS" --runtime-s "$RUNTIME_S" \
    --dataset "$DATASET" --agg-goal "$AGG_GOAL" --concurrency "$CONC" \
    --delay-factor "$DELAY_FACTOR" --trace-scale "${TRACE_SCALE:-1}" --harness "$HARNESS" \
    --agg-hp "$AGG_HP" --trainer-hp "$TRAINER_HP" --inject-bug "$INJECT_BUG"
}

_crash_lines() {
  local n=0 d
  for d in "$@"; do
    [ -d "$d" ] || continue
    n=$(( n + $(grep -rhE "Traceback \(most recent call last\)|CRITICAL|Segmentation fault|Fatal Python error" "$d" --include='*.log' --exclude='*_resources.log' 2>/dev/null | wc -l) ))  # node-wide monitor, not the run
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
    real_dir=""; sim_dir=""; timed_out=0; real_src="-"
    if [ "$GRADE_ONLY" = 1 ]; then
      real_dir="$GIVEN_REAL"; sim_dir="$GIVEN_SIM"; real_src="given"
    else
      args=(--baselines "$b" --mode "$MODE" --runtime-s "$RUNTIME_S" --trace "$trace")
      # GPU path at the config's n=300 keeps its own join barrier (290), as the stored GPU legs did.
      [ "$HARNESS" != none ] || [ "$NUM_TRAINERS" != 300 ] && args+=(--num-trainers "$NUM_TRAINERS")
      [ -n "$DELAY_FACTOR" ] && args+=(--delay-factor "$DELAY_FACTOR")
      [ "$DATASET" != cifar10 ] && args+=(--dataset "$DATASET")
      [ -n "$AGG_GOAL" ] && args+=(--agg-goal "$AGG_GOAL")
      [ -n "$CONC" ] && args+=(--concurrency "$CONC")
      [ "$SIM_CEIL_X" != 1 ] && args+=(--sim-wall-ceiling-s "$(( RUNTIME_S * SIM_CEIL_X ))")
      [ "$HARNESS" != none ] && args+=(--harness "$HARNESS")
      [ -n "$AGG_HP" ] && args+=(--agg-hp "$AGG_HP")
      [ -n "$TRAINER_HP" ] && args+=(--trainer-hp "$TRAINER_HP")
      echo "=== [$(date '+%F %T')] [$label] debug_run.sh ${args[*]}"
      if [ "$DRY_RUN" = 1 ]; then
        env FLAME_LOGDIR="$run_dir" bash "$SCRIPT_DIR/debug_run.sh" "${args[@]}" --dry-run > "$run_dir/shell.log" 2>&1
        tail -3 "$run_dir/shell.log"; continue
      fi
      legs=$(( 1 + SIM_CEIL_X )); [ "$MODE" = real ] && legs=1; [ "$MODE" = sim ] && legs=$SIM_CEIL_X
      : > "$run_dir/legs.txt"
      expt_timed_run "$label" $(( legs * (RUNTIME_S + NUM_TRAINERS / 4 + 30) )) "$TIMEOUT_BUFFER_S" "$run_dir/shell.log" -- \
        env FLAME_LOGDIR="$run_dir" FLAME_RUN_DIR_FILE="$run_dir/legs.txt" bash "$SCRIPT_DIR/debug_run.sh" "${args[@]}"
      [ "$?" = 124 ] && timed_out=1
      real_dir="$(grep -E '_real$' "$run_dir/legs.txt" | tail -1)"; sim_dir="$(grep -E '_sim$' "$run_dir/legs.txt" | tail -1)"
      [ -n "$real_dir" ] && real_src="own"
    fi
    crashes="$(_crash_lines "$real_dir" "$sim_dir")"

    # Events on this run's own legs only (a banked real leg was checked when it ran).
    ev_real=MISSING; ev_sim=MISSING
    ev_json="$OUT/events/${label}.json"
    own=(); [ -n "$real_dir" ] && own+=("$real_dir"); [ -n "$sim_dir" ] && own+=("$sim_dir")
    if [ "${#own[@]}" -gt 0 ]; then
      ( cd "$EX_DIR" && timeout --foreground "$CHECK_TIMEOUT_S" "$PY" -u scripts/parity/event_invariants.py "${own[@]}" \
          --json-out "$ev_json" ) > "$OUT/events/${label}.txt" 2>&1 || true
      i=0
      if [ -n "$real_dir" ]; then ev_real="$(_event_verdict "$ev_json" $i)"; i=$((i + 1)); fi
      if [ -n "$sim_dir" ]; then ev_sim="$(_event_verdict "$ev_json" $i)"; fi
    fi
    [ "$MODE" = sim ] && [ "$GRADE_ONLY" = 0 ] && ev_real="-"
    # FX-L24: a healthy own real leg joins the bank; a sim-only leg borrows the matching one.
    if [ "$real_src" = own ] && [[ "$ev_real" != MISSING ]] && [ "$timed_out" = 0 ]; then
      mapfile -d '' key < <(_bank_key "$trace" "$b")
      "$PY" "$LIB_DIR/examples/scripts/harness_bank.py" register --real-dir "$real_dir" "${key[@]}" || true
    fi
    if [ -z "$real_dir" ] && [ -n "$sim_dir" ] && [ -n "$REAL_FROM" ]; then
      if [ "$REAL_FROM" = auto ]; then
        mapfile -d '' key < <(_bank_key "$trace" "$b")
        IFS=$'\t' read -r real_dir status < <("$PY" "$LIB_DIR/examples/scripts/harness_bank.py" lookup "${key[@]}")
        real_src="bank:${status:-NONE}"
      elif [ -d "$REAL_FROM/telemetry" ]; then
        real_dir="$REAL_FROM"; real_src="given"
      else
        real_dir="$("$PY" "$LIB_DIR/examples/scripts/harness_bank.py" from-dir "$REAL_FROM" --trace "$trace" --baseline "$b")"
        real_src="dir:$( [ -n "$real_dir" ] && echo unchecked || echo NONE)"
      fi
    fi

    verdict=MISSING; score=""; nfail=""; roots=""
    if [ -n "$real_dir" ] && [ -n "$sim_dir" ]; then
      json="$OUT/parity/${label}.json"
      ( cd "$EX_DIR" && PYTHONIOENCODING=utf-8 timeout --foreground "$CHECK_TIMEOUT_S" "$PY" -u scripts/parity_check.py \
          --real "$real_dir" --sim "$sim_dir" --agg-goal "${AGG_GOAL:-10}" --budget-s "$RUNTIME_S" --json-out "$json" ) \
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
    elif [ "$MODE" != both ] && [ "$GRADE_ONLY" = 0 ]; then
      verdict="-"   # single leg, no pair to grade
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$trace" "$b" "$ev_real" "$ev_sim" "$verdict" \
      "$score" "$nfail" "$roots" "$crashes" "$timed_out" "$real_dir" "$sim_dir" "$real_src" >> "$SUMMARY_TSV"
    echo "  [$label] events real=$ev_real sim=$ev_sim | parity=$verdict score=$score roots=$roots | crashes=$crashes timeout=$timed_out real_src=$real_src"
  done
done

column -t -s $'\t' "$SUMMARY_TSV" > "$OUT/summary.txt"
echo; cat "$OUT/summary.txt"
echo; echo "results: $OUT"
