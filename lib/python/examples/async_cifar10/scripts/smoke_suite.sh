#!/usr/bin/env bash
# ============================================================================
# smoke_suite.sh  —  Sequential smoke campaign with per-run timeout guard.
#
# Runs five ordered steps.  Each individual (baseline, mode) invocation of
# debug_run.sh is wrapped in a wall-clock timeout: if the run does not
# self-terminate within (runtime_s + timeout_buffer_s), it is killed via
# SIGTERM → SIGKILL on the whole process group so no orphan trainers remain.
# The suite then continues with the next run.
#
# Steps
#   1  pytest          — unit tests (flame/tests/), expected: 536p/7s
#   2  syn_0  sim      — byte-identity regression, all 6 baselines
#   3  syn_20 sim      — all 6 baselines (availability active, fast mode)
#   4  syn_20 both     — all 6 baselines × sim + real (parity gate)
#   5  syn_50 starvation — feddance + oort, both modes (B2.0.2 regression)
#      Expected: [SIM_STARVATION] events present, no [SIM_WALL_CEILING],
#                self-stops via "stopping run" in both modes.
#
# Per-run checks (applied to the Python agg output log):
#   stopping_run  count of "stopping run" lines — must be > 0
#   wall_ceiling  count of [SIM_WALL_CEILING] lines — must be 0
#   starvation    count of [SIM_STARVATION] lines — informational
#
# Report is written to <output-dir>/report.txt and printed at the end.
#
# Usage:
#   smoke_suite.sh [OPTIONS]
#
# Options:
#   --runtime-syn0-s  N    Budget (wall/vclock) for syn_0 runs     [default: 900]
#   --runtime-syn20-s N    Budget for syn_20 runs                  [default: 1800]
#   --runtime-syn50-s N    Budget for syn_50 starvation runs       [default: 3600]
#   --timeout-buffer-s N   Extra wall-sec before force-kill        [default: 600]
#   --steps LIST           Comma-separated steps to run (1–5)      [default: 1,2,3,4,5]
#   --baselines NAMES      Space-separated baseline list (steps 2–4) [default: all 6]
#   --starvation-baselines NAMES  Baselines for step 5            [default: feddance oort]
#   --output-dir DIR       Log + report directory                  [default: /tmp/smoke_suite_<ts>]
#   --dry-run              Print commands without running them
#   --help
# ============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EX_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"          # async_cifar10/
LIB_DIR="$(cd "$EX_DIR/../.." && pwd)"           # lib/python/
DEBUG_RUN="$SCRIPT_DIR/debug_run.sh"

# ── Defaults ─────────────────────────────────────────────────────────────────
RUNTIME_SYN0_S=900
RUNTIME_SYN20_S=1800
RUNTIME_SYN50_S=3600
TIMEOUT_BUFFER_S=600
ALL_BASELINES="felix oort oort_star refl feddance fedbuff"
STARV_BASELINES="feddance oort"
STEPS="1,2,3,4,5"
OUTPUT_DIR="/tmp/smoke_suite_$(date +%Y%m%d_%H%M%S)"
DRY_RUN=0

# ── Arg parsing ──────────────────────────────────────────────────────────────
usage() {
  grep '^#' "$0" | grep -v '^#!/' | sed 's/^# \{0,1\}//'
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --runtime-syn0-s)       RUNTIME_SYN0_S="$2";    shift 2 ;;
    --runtime-syn20-s)      RUNTIME_SYN20_S="$2";   shift 2 ;;
    --runtime-syn50-s)      RUNTIME_SYN50_S="$2";   shift 2 ;;
    --timeout-buffer-s)     TIMEOUT_BUFFER_S="$2";  shift 2 ;;
    --steps)                STEPS="$2";             shift 2 ;;
    --baselines)            ALL_BASELINES="$2";     shift 2 ;;
    --starvation-baselines) STARV_BASELINES="$2";   shift 2 ;;
    --output-dir)           OUTPUT_DIR="$2";        shift 2 ;;
    --dry-run)              DRY_RUN=1;              shift   ;;
    --help|-h)              usage ;;
    *) echo "Unknown arg: $1" >&2; usage ;;
  esac
done

# ── Setup ────────────────────────────────────────────────────────────────────
mkdir -p "$OUTPUT_DIR/runs"
SUITE_LOG="$OUTPUT_DIR/suite.log"
REPORT="$OUTPUT_DIR/report.txt"
SUITE_START=$(date +%s)

_log()  { echo "[$(date '+%F %T')] $*" | tee -a "$SUITE_LOG"; }
_step() { _log ""; _log "══════ STEP $* ══════"; }

# ── Result tracking ──────────────────────────────────────────────────────────
# Each entry: "label|status|stopping_run|wall_ceiling|starvation"
declare -a RUN_RESULTS=()

_record() {
  RUN_RESULTS+=("${1}|${2}|${3}|${4}|${5}")
}

# ── Per-run timeout wrapper ───────────────────────────────────────────────────
# _run_baseline <label> <runtime_s> [debug_run_args...]
#
# Launches debug_run.sh in a new session (setsid) so the entire process tree
# (launcher + 300 trainers) is in one killable group.  Kills on timeout via
# SIGTERM → 20s grace → SIGKILL.  Isolates each run's Python output into
# <output-dir>/runs/<label>/debug_run.out via FLAME_LOGDIR.
#
# Returns 0 (PASS), 1 (FAIL/ERROR), 124 (TIMEOUT).
_run_baseline() {
  local label="$1" runtime_s="$2"; shift 2
  local run_dir="$OUTPUT_DIR/runs/$label"
  mkdir -p "$run_dir"
  local wall_timeout=$(( runtime_s + TIMEOUT_BUFFER_S ))

  _log "  [$label] start (runtime=${runtime_s}s  wall_timeout=${wall_timeout}s)"

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "  DRY: FLAME_LOGDIR=$run_dir bash $DEBUG_RUN $*" | tee -a "$SUITE_LOG"
    _record "$label" SKIP 0 0 0
    return 0
  fi

  local ts_start; ts_start=$(date +%s)

  # setsid gives the child a new session whose SID=PGID=pid, so
  # kill -TERM/-KILL on -$runner_pid reaches the whole process tree.
  env FLAME_LOGDIR="$run_dir" \
    setsid bash "$DEBUG_RUN" "$@" \
    >"$run_dir/shell.log" 2>&1 &
  local runner_pid=$!

  local deadline=$(( ts_start + wall_timeout ))
  local timed_out=0
  while kill -0 "$runner_pid" 2>/dev/null; do
    sleep 5
    if [[ "$(date +%s)" -ge "$deadline" ]]; then
      _log "  [$label] TIMEOUT after ${wall_timeout}s — killing process group $runner_pid"
      kill -TERM -"$runner_pid" 2>/dev/null || true
      sleep 20
      kill -KILL -"$runner_pid" 2>/dev/null || true
      timed_out=1
      break
    fi
  done
  wait "$runner_pid" 2>/dev/null
  local run_rc=$?
  local elapsed=$(( $(date +%s) - ts_start ))

  # ── Grep checks on the Python aggregator output ───────────────────────────
  # debug_run.sh writes Python output to $FLAME_LOGDIR/debug_run.out via
  # run_node(). All logger.info messages (stopping run, SIM_WALL_CEILING,
  # SIM_STARVATION) appear in that file.
  local agg_log="$run_dir/debug_run.out"
  local stopping=0 ceiling=0 starv=0
  if [[ -f "$agg_log" ]]; then
    stopping=$(grep -ic "stopping run"    "$agg_log" 2>/dev/null || echo 0)
    ceiling=$( grep -c  "SIM_WALL_CEILING" "$agg_log" 2>/dev/null || echo 0)
    starv=$(   grep -c  "\[SIM_STARVATION\]" "$agg_log" 2>/dev/null || echo 0)
  fi

  # ── Classify ─────────────────────────────────────────────────────────────
  local status
  if   [[ "$timed_out" == "1" ]];    then status="TIMEOUT"
  elif [[ "$run_rc"    != "0"  ]];   then status="ERROR(rc=$run_rc)"
  elif [[ "$stopping"  -eq 0   ]];   then status="FAIL(no_stop)"
  elif [[ "$ceiling"   -gt 0   ]];   then status="FAIL(wall_ceil)"
  else                                    status="PASS"
  fi

  _log "  [$label] $status  elapsed=${elapsed}s  stopping=${stopping}  wall_ceil=${ceiling}  starvation=${starv}"
  _record "$label" "$status" "$stopping" "$ceiling" "$starv"

  [[ "$status" == "PASS" ]]
}

# Convenience: run one baseline × one mode.
_run_one() {
  local baseline="$1" mode="$2" trace="$3" runtime_s="$4" step_pfx="$5"
  local label="${step_pfx}_${baseline}_${trace}_${mode}"
  _run_baseline "$label" "$runtime_s" \
    --baselines "$baseline" --mode "$mode" --trace "$trace" --runtime-s "$runtime_s" \
    || true   # never abort the suite on a single run failure
}

# ── Step 1: Unit tests ───────────────────────────────────────────────────────
step1_pytest() {
  _step "1  Unit tests"
  local conda_env="${FLAME_CONDA_ENV:-dg_flame}"

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "  DRY: conda run -n $conda_env python -m pytest $LIB_DIR/tests/ -q --tb=short" \
      | tee -a "$SUITE_LOG"
    _record "s1_pytest" SKIP 0 0 0
    return
  fi

  local pytest_log="$OUTPUT_DIR/runs/s1_pytest.log"
  mkdir -p "$OUTPUT_DIR/runs"
  conda run -n "$conda_env" \
    python -m pytest "$LIB_DIR/tests/" -q --tb=short \
    >"$pytest_log" 2>&1
  local rc=$?
  local summary; summary=$(tail -3 "$pytest_log")
  if [[ "$rc" == "0" ]]; then
    _log "  PASS  $summary"
    _record "s1_pytest" PASS 0 0 0
  else
    _log "  FAIL  $summary"
    _log "  Full log: $pytest_log"
    _record "s1_pytest" "FAIL(rc=$rc)" 0 0 0
  fi
}

# ── Step 2: syn_0 regression (sim only) ──────────────────────────────────────
step2_syn0_sim() {
  _step "2  syn_0 regression — sim, all 6 baselines"
  _log "  (gate-ON with always-available trace; sim completes in << runtime_s wall-seconds)"
  for bl in $ALL_BASELINES; do
    _run_one "$bl" sim syn_0 "$RUNTIME_SYN0_S" s2
  done
}

# ── Step 3: syn_20 sim ───────────────────────────────────────────────────────
step3_syn20_sim() {
  _step "3  syn_20 — sim, all 6 baselines"
  for bl in $ALL_BASELINES; do
    _run_one "$bl" sim syn_20 "$RUNTIME_SYN20_S" s3
  done
}

# ── Step 4: syn_20 both modes ────────────────────────────────────────────────
step4_syn20_both() {
  _step "4  syn_20 — sim then real, all 6 baselines"
  _log "  (real runs take ~${RUNTIME_SYN20_S}s wall each; total step ~$(( ${#ALL_BASELINES//[! ]/} * RUNTIME_SYN20_S / 60 ))+ min)"
  for bl in $ALL_BASELINES; do
    _run_one "$bl" sim  syn_20 "$RUNTIME_SYN20_S" s4
    _run_one "$bl" real syn_20 "$RUNTIME_SYN20_S" s4
  done
}

# ── Step 5: syn_50 starvation regression ─────────────────────────────────────
step5_syn50_starvation() {
  _step "5  syn_50 starvation — $STARV_BASELINES, both modes (B2.0.2 regression)"
  _log "  Expected: [SIM_STARVATION] present, no [SIM_WALL_CEILING], self-stops cleanly."
  for bl in $STARV_BASELINES; do
    _run_one "$bl" sim  syn_50 "$RUNTIME_SYN50_S" s5
    _run_one "$bl" real syn_50 "$RUNTIME_SYN50_S" s5
  done
}

# ── Final report ─────────────────────────────────────────────────────────────
_final_report() {
  local total=${#RUN_RESULTS[@]}
  local pass=0 fail=0 timeout=0 skip=0 error=0

  {
    local divider; divider=$(printf '═%.0s' {1..68})
    echo "$divider"
    echo "  SMOKE SUITE REPORT"
    printf "  Generated  : %s\n"  "$(date '+%F %T')"
    printf "  Duration   : %ds\n" "$(( $(date +%s) - SUITE_START ))"
    printf "  Output     : %s\n"  "$OUTPUT_DIR"
    printf "  Runtimes   : syn_0=%ss  syn_20=%ss  syn_50=%ss  buffer=%ss\n" \
      "$RUNTIME_SYN0_S" "$RUNTIME_SYN20_S" "$RUNTIME_SYN50_S" "$TIMEOUT_BUFFER_S"
    echo "$divider"
    echo ""
    printf "%-48s  %-20s  %8s  %9s  %11s\n" \
      "RUN" "STATUS" "stop_run" "wall_ceil" "starvation"
    printf "%-48s  %-20s  %8s  %9s  %11s\n" \
      "$(printf '%0.s-' {1..48})" "$(printf '%0.s-' {1..20})" "--------" "---------" "-----------"

    for entry in "${RUN_RESULTS[@]}"; do
      IFS='|' read -r label status stopping ceiling starv <<< "$entry"
      case "$status" in
        PASS)    (( pass++    )) ;;
        SKIP)    (( skip++    )) ;;
        TIMEOUT) (( timeout++ )) ;;
        ERROR*)  (( error++   )) ;;
        FAIL*)   (( fail++    )) ;;
      esac
      printf "%-48s  %-20s  %8s  %9s  %11s\n" \
        "$label" "$status" "$stopping" "$ceiling" "$starv"
    done

    echo ""
    echo "TOTAL $total runs:  PASS=$pass  FAIL=$fail  ERROR=$error  TIMEOUT=$timeout  SKIP=$skip"
    echo ""

    # Attention list
    local bad=()
    for entry in "${RUN_RESULTS[@]}"; do
      IFS='|' read -r label status _ _ _ <<< "$entry"
      case "$status" in PASS|SKIP) ;; *) bad+=("  $label  →  $status") ;; esac
    done
    if [[ ${#bad[@]} -gt 0 ]]; then
      echo "Runs needing investigation:"
      printf '%s\n' "${bad[@]}"
      echo ""
      echo "Logs:  $OUTPUT_DIR/runs/<label>/debug_run.out  (Python agg output)"
      echo "       $OUTPUT_DIR/runs/<label>/shell.log       (debug_run.sh output)"
    else
      echo "All runs clean."
    fi
    echo "$divider"
  } | tee "$REPORT" | tee -a "$SUITE_LOG"

  # Return non-zero if anything went wrong
  (( fail + error + timeout == 0 ))
}

# ── Main ─────────────────────────────────────────────────────────────────────
_log "Smoke suite started"
_log "  output      : $OUTPUT_DIR"
_log "  steps       : $STEPS"
_log "  baselines   : $ALL_BASELINES"
_log "  starvation  : $STARV_BASELINES"
_log "  runtimes    : syn_0=${RUNTIME_SYN0_S}s  syn_20=${RUNTIME_SYN20_S}s  syn_50=${RUNTIME_SYN50_S}s  buffer=${TIMEOUT_BUFFER_S}s"

IFS=',' read -ra _steps_arr <<< "$STEPS"
for s in "${_steps_arr[@]}"; do
  s="${s// /}"   # strip any accidental spaces
  case "$s" in
    1) step1_pytest ;;
    2) step2_syn0_sim ;;
    3) step3_syn20_sim ;;
    4) step4_syn20_both ;;
    5) step5_syn50_starvation ;;
    *) _log "Unknown step '$s' (valid: 1–5)" ;;
  esac
done

_final_report
