#!/bin/bash
# Shared experiment-launch harness for the flame examples.
#
# Sourced by each example's driver (async_cifar10 scripts/debug_run.sh,
# fwdllm expt_scripts/run_sequential.sh) so the genuinely-identical mechanics
# live in ONE place: robust conda activation, PYTHONPATH pinning, the
# launch+progress-ticker loop, and post-run log-health assertions. The
# example-specific parts (config discovery, YAML patching, the per-baseline
# knob set, and the tier/check spec fed to expt_runner.py) stay in each driver.
#
# Usage (from a driver):
#   source "<repo>/lib/python/examples/scripts/expt_runner.sh"
#   expt_activate_conda [default_env]        # activate + report python
#   expt_pin_pythonpath "$REPO_ROOT"         # this checkout's flame ahead of any editable install
#   expt_launch label cfg example_dir budget_s n_exps logdir [experiments_dir]
#   expt_assert_log "$logdir/label.out" label
#
# Pre-flight display + gate is done by the Python side (expt_runner.py): a driver
# builds a `spec` dict of tiers+checks and calls render_and_gate() -- see that
# file. $EXPT_RUNNER_DIR / $EXPT_RUNNER_PY are exported here so a driver's own
# python heredoc can `sys.path.insert(0, EXPT_RUNNER_DIR); import expt_runner`.

# Absolute dir of THIS harness (works when sourced), so drivers/python can find
# the Python renderer next to it regardless of the caller's cwd.
EXPT_RUNNER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPT_RUNNER_PY="$EXPT_RUNNER_DIR/expt_runner.py"
export EXPT_RUNNER_DIR EXPT_RUNNER_PY

# --- robust conda activation ------------------------------------------------
# Env choice: FLAME_CONDA_ENV overrides; else the shell's already-active env
# (CONDA_DEFAULT_ENV); else the caller-supplied default (arg $1). No hardcoded
# fallback beyond what the caller passes -- cifar passes dg_flame, fwdllm passes
# nothing (so it requires an active env, matching its original behavior).
expt_activate_conda() {
  local default_env="${1:-}"
  local envname="${FLAME_CONDA_ENV:-${CONDA_DEFAULT_ENV:-$default_env}}"
  if [ -z "$envname" ]; then
    echo "ERROR: no conda env active and FLAME_CONDA_ENV not set." >&2
    echo "       Activate an env first (conda activate <name>), set FLAME_CONDA_ENV=<name>," >&2
    echo "       or have the driver pass a default to expt_activate_conda." >&2
    return 1
  fi
  local cb=""
  if command -v conda >/dev/null 2>&1; then
    cb="$(conda info --base 2>/dev/null)"
  elif [ -n "${CONDA_EXE:-}" ]; then
    cb="$(dirname "$(dirname "$CONDA_EXE")")"
  fi
  if [ -z "$cb" ] || [ ! -f "$cb/etc/profile.d/conda.sh" ]; then
    local c
    for c in "$HOME/miniconda3" "/coc/scratch/${USER%??}/miniconda3" \
             "/coc/scratch/$USER/miniconda3" "$HOME/anaconda3" /opt/conda; do
      [ -f "$c/etc/profile.d/conda.sh" ] && cb="$c" && break
    done
  fi
  if [ -z "$cb" ] || [ ! -f "$cb/etc/profile.d/conda.sh" ]; then
    echo "ERROR: conda not found. Activate '$envname' yourself or set CONDA_EXE." >&2
    return 1
  fi
  # shellcheck disable=SC1091
  source "$cb/etc/profile.d/conda.sh"
  conda activate "$envname" || { echo "ERROR: 'conda activate $envname' failed" >&2; return 1; }
  echo "conda: base=$cb env=$envname python=$(which python)"
}

# Force this checkout's flame package ahead of anything already on sys.path
# (e.g. a stale `pip install -e` editable pointing at a different clone).
expt_pin_pythonpath() {
  local repo_root="$1"
  export PYTHONPATH="$repo_root/lib/python${PYTHONPATH:+:$PYTHONPATH}"
}

# expt_launch label cfg example_dir budget_s n_exps logdir [experiments_dir]
# Runs one flame experiment-config file in the foreground with a 30s progress
# ticker (elapsed/remaining/percent + how many run_* dirs have appeared). Sets
# EXPT_LAST_RC and returns the launcher's exit code.
expt_launch() {
  local label="$1" cfg="$2" example_dir="$3" budget_s="${4:-0}" n_exps="${5:-1}" logdir="$6"
  local exp_dir="${7:-$example_dir/experiments}"
  mkdir -p "$logdir"
  # Marker touched BEFORE launch so expt_assert_run can find exactly the run
  # dir(s)/logs this launch produced (find -newer "$marker"). Exported for the
  # caller's health check.
  EXPT_LAST_MARKER="$logdir/.marker_${label}"; : > "$EXPT_LAST_MARKER"; export EXPT_LAST_MARKER
  local start_ts; start_ts=$(date +%s)
  local initial_runs; initial_runs=$(find "$exp_dir" -maxdepth 1 -name "run_*" -type d 2>/dev/null | wc -l)

  echo "[$(date '+%F %T')] START $label ($n_exps exp(s), ~${budget_s}s budget)" | tee -a "$logdir/expt_runner.log"

  # Background progress ticker.
  (
    while true; do
      sleep 30
      local now elapsed pct=0 remaining=0
      now=$(date +%s); elapsed=$(( now - start_ts ))
      if [ "$budget_s" -gt 0 ]; then
        pct=$(( elapsed * 100 / budget_s )); remaining=$(( budget_s - elapsed ))
        [ "$pct" -gt 100 ] && pct=100; [ "$remaining" -lt 0 ] && remaining=0
      fi
      local curr started
      curr=$(find "$exp_dir" -maxdepth 1 -name "run_*" -type d 2>/dev/null | wc -l)
      started=$(( curr - initial_runs )); [ "$started" -lt 0 ] && started=0
      printf "  [%s] %s | %ds elapsed / ~%ds (%d%%) | exp started: %d/%d\n" \
        "$(date '+%T')" "$label" "$elapsed" "$budget_s" "$pct" "$started" "$n_exps"
    done
  ) &
  local ticker_pid=$!

  python -m flame.launch.run_experiment "$cfg" --example-dir "$example_dir" \
      < /dev/null >> "$logdir/${label}.out" 2>&1
  local rc=$?

  kill "$ticker_pid" 2>/dev/null
  wait "$ticker_pid" 2>/dev/null

  local elapsed=$(( $(date +%s) - start_ts ))
  echo "[$(date '+%F %T')] DONE  $label exit=$rc (took ${elapsed}s / ~${budget_s}s budget)" | tee -a "$logdir/expt_runner.log"
  EXPT_LAST_RC=$rc
  return $rc
}

# expt_assert_run example_dir marker [label] -- post-run health check that scans
# the RIGHT files (the FL signals do NOT land in the runner .out): agg_round
# events live in experiments/run_*/telemetry/aggregator_*.jsonl; "stopping run" /
# SIM_WALL_CEILING / SIM_STARVATION / tracebacks live in the dedicated
# experiments/run_*/*_aggregator.log. Only files newer than `marker` (touched by
# expt_launch pre-launch) are scanned, so it attributes signals to THIS run.
# Prints a one-line verdict and exports EXPT_LAST_HEALTH with the verdict word
# (COMPLETED / CRASH / NO_AGG_ROUNDS / WALL_CEILING / NO_MARKER) so a driver can
# fold it into its own summary; returns 0 only if there were agg_rounds, no
# [SIM_WALL_CEILING], and no crash marker.
#
# Vocabulary note: a healthy run is "COMPLETED", NOT "PASS". Completing is a
# process outcome, not a check result -- PASS/FAIL is reserved for actual
# checks (e.g. the real<->sim parity battery in scripts.parity.cli).
expt_assert_run() {
  local example_dir="$1" marker="$2" label="${3:-run}"
  local expdir="$example_dir/experiments"
  if [ ! -e "$marker" ]; then
    EXPT_LAST_HEALTH="NO_MARKER"; export EXPT_LAST_HEALTH
    echo "  [$label] NO_MARKER (cannot scope health check)"; return 1
  fi
  local agg=0 stop=0 wall=0 starv=0 crash=0 nlogs=0 f n
  while IFS= read -r f; do
    n=$(grep -c "agg_round" "$f" 2>/dev/null); agg=$(( agg + ${n:-0} ))
  done < <(find "$expdir" -name "aggregator_*.jsonl" -newer "$marker" 2>/dev/null)
  while IFS= read -r f; do
    nlogs=$(( nlogs + 1 ))
    n=$(grep -ic "stopping run" "$f" 2>/dev/null);          stop=$((  stop  + ${n:-0} ))
    n=$(grep -c  "SIM_WALL_CEILING" "$f" 2>/dev/null);       wall=$((  wall  + ${n:-0} ))
    n=$(grep -c  "\[SIM_STARVATION\]" "$f" 2>/dev/null);     starv=$(( starv + ${n:-0} ))
    n=$(grep -cE "Traceback \(most recent call last\)|CUDA error|Segmentation fault" "$f" 2>/dev/null); crash=$(( crash + ${n:-0} ))
  done < <(find "$expdir" -name "*_aggregator.log" -newer "$marker" 2>/dev/null)
  # A healthy, fully-run experiment is COMPLETED (a process outcome), not PASS.
  local status="COMPLETED"
  [ "$agg" -eq 0 ]   && status="NO_AGG_ROUNDS"
  [ "$wall" -gt 0 ]  && status="WALL_CEILING"
  [ "$crash" -gt 0 ] && status="CRASH"
  EXPT_LAST_HEALTH="$status"; export EXPT_LAST_HEALTH
  printf "  [%s] %-14s agg_round=%s stopping_run=%s wall_ceiling=%s starvation=%s crash=%s (%s agg log(s))\n" \
    "$label" "$status" "$agg" "$stop" "$wall" "$starv" "$crash" "$nlogs"
  [ "$agg" -gt 0 ] && [ "$wall" -eq 0 ] && [ "$crash" -eq 0 ]
}

# expt_timed_run label runtime_s buffer_s shell_log -- cmd... -- run a command in
# its OWN process group under a wall-clock timeout, killing the whole tree
# (SIGTERM grace -> SIGKILL) if it overruns, then sweeping stragglers so no
# orphan trainers/aggregator survive and GPU memory drains. This is the campaign
# primitive shared by smoke_suite.sh (and any future fwdllm suite). Emits a 5s
# elapsed / kill-in ticker to stderr. Returns the cmd's rc, or 124 on timeout.
expt_timed_run() {
  local label="$1" runtime_s="$2" buffer_s="$3" shell_log="$4"; shift 4
  [ "${1:-}" = "--" ] && shift
  local settle_s="${EXPT_KILL_SETTLE_S:-20}" grace_s="${EXPT_KILL_GRACE_S:-20}"
  local wall_timeout=$(( runtime_s + buffer_s ))
  local ts_start; ts_start=$(date +%s)
  local deadline=$(( ts_start + wall_timeout )) timed_out=0
  # set -m: assign the background job its own PGID so kill -<pid> hits the tree
  # (flame's spawner uses plain Popen with no setsid, so descendants inherit it).
  set -m
  "$@" > "$shell_log" 2>&1 &
  local pid=$!
  set +m
  local ela kin
  while kill -0 "$pid" 2>/dev/null; do
    sleep 5
    # Re-test liveness: if the child finished DURING the sleep, exit normally
    # rather than force-killing a dead pid and misreporting TIMEOUT (this bites
    # whenever a run completes near the deadline).
    kill -0 "$pid" 2>/dev/null || break
    ela=$(( $(date +%s) - ts_start )); kin=$(( deadline - $(date +%s) )); [ "$kin" -lt 0 ] && kin=0
    printf '\r  [%s] %ds/%ds  kill in %ds   ' "$label" "$ela" "$runtime_s" "$kin" >&2
    if [ "$(date +%s)" -ge "$deadline" ]; then
      printf '\n' >&2
      echo "  [$label] TIMEOUT after ${wall_timeout}s — killing process group $pid" >&2
      kill -TERM -"$pid" 2>/dev/null || true
      sleep "$grace_s"
      kill -KILL -"$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null
      timed_out=1; break
    fi
  done
  printf '\n' >&2
  [ "$timed_out" = "0" ] && wait "$pid" 2>/dev/null
  local rc=$?
  if [ "$timed_out" = "1" ]; then
    pkill -9 -f "trainer/pytorch/main.py"    2>/dev/null || true
    pkill -9 -f "trainer/forward_training"   2>/dev/null || true
    pkill -9 -f "aggregator/pytorch/main_"   2>/dev/null || true
    sleep "$settle_s"
    EXPT_LAST_RC=124; return 124
  fi
  EXPT_LAST_RC=$rc; return "$rc"
}

# expt_dispatch_after "hook1,hook2" -- run the requested post-launch hooks. Each
# hook `foo` is delegated to a shell function `after_foo` the DRIVER defines
# (e.g. after_parity, after_plot, after_sanity) -- so the mechanism is shared but
# each example supplies its own parity CLI / plotter / analysis command. Unknown
# or unsupported hooks warn and are skipped (not fatal).
expt_dispatch_after() {
  local csv="${1:-}"; [ -n "$csv" ] || return 0
  local hook
  IFS=',' read -ra _hooks <<< "$csv"
  for hook in "${_hooks[@]}"; do
    hook="${hook// /}"; [ -n "$hook" ] || continue
    if declare -F "after_${hook}" >/dev/null 2>&1; then
      echo "=== after: ${hook} ==="
      "after_${hook}" || echo "  [after:${hook}] returned non-zero (continuing)"
    else
      echo "  [after:${hook}] not supported for this example — skipping" >&2
    fi
  done
}
