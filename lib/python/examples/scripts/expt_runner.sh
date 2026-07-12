#!/bin/bash
# Shared experiment-launch harness for the flame examples: robust conda activation,
# PYTHONPATH pinning, the launch+progress-ticker loop, and post-run log-health
# assertions. Sourced by each driver (async_cifar10 debug_run.sh, fwdllm
# run_sequential.sh); config discovery / YAML patching / knob sets stay per-driver.
#
# Usage (from a driver):
#   source "<repo>/lib/python/examples/scripts/expt_runner.sh"
#   expt_activate_conda [default_env]        # activate + report python
#   expt_pin_pythonpath "$REPO_ROOT"         # this checkout's flame ahead of any editable install
#   expt_launch label cfg example_dir budget_s n_exps logdir [experiments_dir]
#   expt_assert_log "$logdir/label.out" label
#
# Pre-flight display + gate lives on the Python side (expt_runner.py): a driver
# builds a `spec` of tiers+checks and calls render_and_gate(). $EXPT_RUNNER_DIR /
# $EXPT_RUNNER_PY are exported so a driver's python heredoc can import expt_runner.

# Absolute dir of THIS harness (works when sourced), so drivers/python can find
# the Python renderer next to it regardless of the caller's cwd.
EXPT_RUNNER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPT_RUNNER_PY="$EXPT_RUNNER_DIR/expt_runner.py"
export EXPT_RUNNER_DIR EXPT_RUNNER_PY

# --- robust conda activation ------------------------------------------------
# Env choice: FLAME_CONDA_ENV overrides; else the shell's already-active env
# (CONDA_DEFAULT_ENV); else the caller-supplied default (arg $1). No hardcoded
# fallback -- cifar passes dg_flame, fwdllm passes nothing (requires an active env).
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

# FL worker process patterns — single source of truth for interrupt teardown and
# the clean-slate preflight (run orchestrator, trainers, aggregator, watcher).
EXPT_WORKER_PATS=(
  'flame.launch.run_experiment'
  'trainer/forward_training'
  'trainer/pytorch/main.py'
  'aggregator/pytorch/main_'
  'converge_watch.py'
)

# expt_assert_clean_slate [label] -- refuse to launch on top of stray FL workers
# from a prior/crashed/Ctrl+C'd run (own procs only). EXPT_AUTOCLEAN=1 kills them
# and re-checks; default ABORTS and prints the kill command. Returns 1 if dirty.
# EXPT_GPU_FREE_MB (default 500) warns on residual GPU memory; EXPT_GPU_STRICT=1 aborts.
expt_assert_clean_slate() {
  local label="${1:-preflight}" uid; uid="$(id -u)"
  _ecs_scan() {
    local p pids out=""
    for p in "${EXPT_WORKER_PATS[@]}"; do
      pids="$(pgrep -u "$uid" -f "$p" 2>/dev/null | tr '\n' ' ')"
      [ -n "$pids" ] && out+="    ${p} -> ${pids}\n"
    done
    printf '%b' "$out"
  }
  local dirty; dirty="$(_ecs_scan)"
  if [ -n "$dirty" ]; then
    echo "  [$label] NOT CLEAN — stray FL workers from a previous run:" >&2
    printf '%b' "$dirty" >&2
    if [ "${EXPT_AUTOCLEAN:-0}" = "1" ]; then
      echo "  [$label] EXPT_AUTOCLEAN=1 → killing and re-checking ..." >&2
      # TERM the orchestrator first (lets it tear down its own group), then KILL all.
      pkill -TERM -u "$uid" -f 'flame.launch.run_experiment' 2>/dev/null || true
      sleep "${EXPT_CLEAN_GRACE_S:-3}"
      local p
      for p in "${EXPT_WORKER_PATS[@]}"; do
        pkill -9 -u "$uid" -f "$p" 2>/dev/null || true
      done
      sleep 2
      dirty="$(_ecs_scan)"
      if [ -n "$dirty" ]; then
        echo "  [$label] STILL NOT CLEAN after autoclean — aborting:" >&2
        printf '%b' "$dirty" >&2; return 1
      fi
      echo "  [$label] cleaned." >&2
    else
      echo "  [$label] refusing to launch. Clean it, or re-run with EXPT_AUTOCLEAN=1 (or --clean):" >&2
      echo "      pkill -TERM -f 'flame.launch.run_experiment'; sleep 3; pkill -9 -f 'trainer/forward_training'; pkill -9 -f 'trainer/pytorch/main.py'; pkill -9 -f 'aggregator/pytorch/main_'; pkill -9 -f converge_watch.py" >&2
      return 1
    fi
  fi
  # Residual GPU memory here is a peer's job (your workers are gone) -> warn only.
  if command -v nvidia-smi >/dev/null 2>&1; then
    local maxused; maxused="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1)"
    local thresh="${EXPT_GPU_FREE_MB:-500}"
    if [ -n "$maxused" ] && [ "$maxused" -gt "$thresh" ] 2>/dev/null; then
      echo "  [$label] WARNING: a GPU shows ${maxused}MB used (> ${thresh}MB) — a peer job may be resident." >&2
      if [ "${EXPT_GPU_STRICT:-0}" = "1" ]; then
        echo "  [$label] EXPT_GPU_STRICT=1 → aborting." >&2; return 1
      fi
    fi
  fi
  echo "  [$label] clean slate verified." >&2
  return 0
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

  # Launch the run in its OWN process group (set -m -> bg job's pgid == its pid;
  # run_experiment's Popen children inherit it, no setsid — same pattern as
  # expt_timed_run) so the convergence watcher can signal the whole tree.
  set -m
  python -m flame.launch.run_experiment "$cfg" --example-dir "$example_dir" \
      < /dev/null >> "$logdir/${label}.out" 2>&1 &
  local run_pid=$!
  set +m

  # Progress ticker. SELF-TERMINATES the instant the run process is gone, so it
  # never outlives the run (watcher kill, crash, Ctrl+C). Torn down below BY PID
  # (+ its in-flight `sleep` child), NEVER by process group: job control doesn't
  # reliably place a backgrounded subshell in a fresh group here, so a group
  # signal would miss it and a following `wait` would hang the harness.
  (
    while kill -0 "$run_pid" 2>/dev/null; do
      sleep 30
      kill -0 "$run_pid" 2>/dev/null || break   # run gone -> stop; never outlive it
      local now elapsed pct=0
      now=$(date +%s); elapsed=$(( now - start_ts ))
      if [ "$budget_s" -gt 0 ]; then
        pct=$(( elapsed * 100 / budget_s )); [ "$pct" -gt 100 ] && pct=100
      fi
      local curr started
      curr=$(find "$exp_dir" -maxdepth 1 -name "run_*" -type d 2>/dev/null | wc -l)
      started=$(( curr - initial_runs )); [ "$started" -lt 0 ] && started=0
      printf "  [%s] %s | %ds elapsed / ~%ds (%d%%) | exp started: %d/%d\n" \
        "$(date '+%T')" "$label" "$elapsed" "$budget_s" "$pct" "$started" "$n_exps"
    done
  ) &
  local ticker_pid=$!

  # PID-based ticker teardown (NOT group-based — see above). Idempotent: the
  # ticker may already have self-exited when run_pid died. Kills the in-flight
  # `sleep` child first, then the subshell, then reaps so no `wait` can hang.
  _expt_stop_ticker() {
    [ -n "$ticker_pid" ] || return 0
    pkill -P "$ticker_pid" 2>/dev/null || true   # its in-flight `sleep`
    kill "$ticker_pid" 2>/dev/null || true
    wait "$ticker_pid" 2>/dev/null || true
  }

  # Optional convergence-stop watcher (EXPERIMENTS.md WS2) -- ONLY when the driver
  # set EXPT_TARGET_ACC. Side-car: on "EXPT_CONVERGE_WINDOW consecutive data bins
  # all >= EXPT_TARGET_ACC" it writes converge.json + kills the run's process
  # group. Absent -> the run is governed only by its max_runtime_s/max_data_id caps.
  local watcher_pid="" cj="" sj=""
  EXPT_LAST_CONVERGE_JSON=""; export EXPT_LAST_CONVERGE_JSON
  if [ -n "${EXPT_TARGET_ACC:-}" ]; then
    cj="$logdir/converge_${label}.json"
    sj="$logdir/stall_${label}.json"
    # EXPT_STALL_WINDOW_S=0 (default) disables the stall guard -> pure wall/window run.
    python "$EXPT_RUNNER_DIR/converge_watch.py" \
        --exp-dir "$exp_dir" --marker "$EXPT_LAST_MARKER" \
        --target-acc "$EXPT_TARGET_ACC" --window "${EXPT_CONVERGE_WINDOW:-20}" \
        --pgid "$run_pid" --converge-json "$cj" --stall-json "$sj" \
        --stall-window-s "${EXPT_STALL_WINDOW_S:-0}" --stall-min-delta "${EXPT_STALL_MIN_DELTA:-0.01}" \
        --stall-on "${EXPT_STALL_ON:-either}" --loss-min-rel-delta "${EXPT_LOSS_MIN_REL_DELTA:-0.01}" \
        --poll "${EXPT_CONVERGE_POLL_S:-15}" 2>&1 | tee -a "$logdir/expt_runner.log" &
    watcher_pid=$!
    EXPT_LAST_CONVERGE_JSON="$cj"; export EXPT_LAST_CONVERGE_JSON
  fi

  # Unconditional wall-clock BACKSTOP watchdog. budget_s (--max-runtime-s) is
  # ALSO enforced INSIDE the aggregator (self-stop via _check_early_stop_conditions),
  # but that self-stop has DOCUMENTED failure modes (fwdllm real-mode MQTT hangs,
  # simulate_fwdllm.md §A -- a hang stuck at MQTT join or gone silent post-dispatch
  # never reaches the training loop where the self-stop check would fire at all).
  # Without an EXTERNAL timeout, a hung run blocks `wait "$run_pid"` below FOREVER,
  # which blocks every subsequent run in a run_sequential.sh sequence -- exactly
  # what this backstop exists to prevent. Fires only when budget_s>0; waits
  # EXPT_BUDGET_GRACE_S past budget_s so a clean self-stop (or the convergence
  # watcher above, if active) wins the race under normal operation -- this is a
  # backstop for "never self-stopped", not a replacement for the aggregator's cap.
  local watchdog_pid="" timeout_marker="$logdir/.timeout_${label}"
  rm -f "$timeout_marker"
  if [ "$budget_s" -gt 0 ]; then
    (
      local grace="${EXPT_BUDGET_GRACE_S:-120}"
      local deadline=$(( start_ts + budget_s + grace ))
      while kill -0 "$run_pid" 2>/dev/null; do
        if [ "$(date +%s)" -ge "$deadline" ]; then
          kill -0 "$run_pid" 2>/dev/null || break   # re-check: avoid a late race
          echo "[$(date '+%F %T')] TIMEOUT $label: still alive ${grace}s past its ${budget_s}s budget -- force-killing so the run sequence isn't blocked." \
            | tee -a "$logdir/expt_runner.log" >&2
          : > "$timeout_marker"
          kill -TERM -"$run_pid" 2>/dev/null || true
          kill -TERM  "$run_pid" 2>/dev/null || true
          [ -n "$watcher_pid" ] && kill -TERM "$watcher_pid" 2>/dev/null
          pkill -TERM -f converge_watch.py 2>/dev/null || true
          sleep "${EXPT_KILL_GRACE_S:-20}"
          kill -KILL -"$run_pid" 2>/dev/null || true
          kill -KILL  "$run_pid" 2>/dev/null || true
          local p
          for p in "${EXPT_WORKER_PATS[@]}"; do
            pkill -9 -f "$p" 2>/dev/null || true
          done
          break
        fi
        sleep 5
      done
    ) &
    watchdog_pid=$!
  fi
  _expt_stop_watchdog() {
    [ -n "$watchdog_pid" ] || return 0
    kill "$watchdog_pid" 2>/dev/null || true
    wait "$watchdog_pid" 2>/dev/null || true
  }

  # Ctrl+C/SIGTERM teardown: the run is in its OWN process group (set -m) so a
  # terminal SIGINT never reaches it -- without this trap it (and the watcher and
  # ticker) would orphan and keep the GPU pinned. TERM run group + watcher + ticker,
  # escalate to KILL after a grace, sweep stragglers, exit 130.
  _expt_interrupt() {
    trap - INT TERM
    echo "" >&2
    echo "[$(date '+%F %T')] INTERRUPT — tearing down '$label' (run pgid=$run_pid) ..." >&2
    kill -TERM -"$run_pid" 2>/dev/null || true   # run's process group ...
    kill -TERM  "$run_pid" 2>/dev/null || true   # ... and its leader (group may not have formed)
    _expt_stop_ticker                            # PID-based; never a group signal
    _expt_stop_watchdog                          # PID-based; never a group signal
    # $watcher_pid is the tee of the `converge_watch.py | tee` pipeline, so also
    # kill the poller by name.
    [ -n "$watcher_pid" ] && kill -TERM "$watcher_pid" 2>/dev/null
    pkill -TERM -f converge_watch.py 2>/dev/null || true
    sleep "${EXPT_INT_GRACE_S:-5}"
    kill -KILL -"$run_pid" 2>/dev/null || true
    kill -KILL  "$run_pid" 2>/dev/null || true
    pkill -9 -f 'flame.launch.run_experiment' 2>/dev/null || true
    pkill -9 -f 'trainer/forward_training'  2>/dev/null || true
    pkill -9 -f 'trainer/pytorch/main.py'   2>/dev/null || true
    pkill -9 -f 'aggregator/pytorch/main_'  2>/dev/null || true
    pkill -9 -f converge_watch.py           2>/dev/null || true
    echo "[$(date '+%F %T')] INTERRUPT — teardown complete for '$label'. GPU/RAM freed." >&2
    exit 130
  }
  trap _expt_interrupt INT TERM

  wait "$run_pid"
  local rc=$?
  trap - INT TERM

  _expt_stop_ticker   # PID-based (+ its sleep child); may already have self-exited
  _expt_stop_watchdog # PID-based; may already have self-exited (or just fired -- either way idempotent)
  if [ -n "$watcher_pid" ]; then kill "$watcher_pid" 2>/dev/null; wait "$watcher_pid" 2>/dev/null; fi

  # Watcher verdict: converge.json = CONVERGED, stall.json = STALLED. On a
  # watcher-driven kill (either), sweep any orphaned workers so GPU memory frees.
  EXPT_LAST_CONVERGED=0; EXPT_LAST_STALLED=0; export EXPT_LAST_CONVERGED EXPT_LAST_STALLED
  [ -n "$cj" ] && [ -f "$cj" ] && EXPT_LAST_CONVERGED=1
  [ -n "$sj" ] && [ -f "$sj" ] && EXPT_LAST_STALLED=1
  if [ "$EXPT_LAST_CONVERGED" = "1" ] || [ "$EXPT_LAST_STALLED" = "1" ]; then
    pkill -9 -f "trainer/forward_training" 2>/dev/null || true
    pkill -9 -f "aggregator/pytorch/main_" 2>/dev/null || true
  fi

  # Backstop-watchdog verdict: it already swept EXPT_WORKER_PATS itself, but sweep
  # once more here in case anything spawned in the gap between its sweep and this
  # process actually exiting -- so a dirty node never blocks the NEXT run's launch.
  EXPT_LAST_TIMED_OUT=0; export EXPT_LAST_TIMED_OUT
  if [ -f "$timeout_marker" ]; then
    EXPT_LAST_TIMED_OUT=1
    local p
    for p in "${EXPT_WORKER_PATS[@]}"; do
      pkill -9 -f "$p" 2>/dev/null || true
    done
    rm -f "$timeout_marker"
  fi

  local elapsed=$(( $(date +%s) - start_ts ))
  echo "[$(date '+%F %T')] DONE  $label exit=$rc converged=${EXPT_LAST_CONVERGED} stalled=${EXPT_LAST_STALLED} timed_out=${EXPT_LAST_TIMED_OUT} (took ${elapsed}s / ~${budget_s}s budget)" | tee -a "$logdir/expt_runner.log"
  EXPT_LAST_RC=$rc
  return $rc
}

# expt_assert_run example_dir marker [label] -- post-run health check. FL signals
# do NOT land in the runner .out: agg_round events live in
# experiments/run_*/telemetry/aggregator_*.jsonl; "stopping run" / SIM_WALL_CEILING
# / SIM_STARVATION / tracebacks live in experiments/run_*/*_aggregator.log. Only
# files newer than `marker` are scanned, so signals attribute to THIS run. Prints a
# one-line verdict and exports EXPT_LAST_HEALTH (COMPLETED / CRASH / NO_AGG_ROUNDS /
# WALL_CEILING / NO_MARKER); returns 0 only for agg_rounds && no wall-ceiling && no
# crash. A healthy run is "COMPLETED", not "PASS" (PASS/FAIL is for actual checks).
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
  # Convergence-mode verdict (EXPERIMENTS.md WS2): only with a target accuracy. A
  # watcher-driven stop is CONVERGED; running out the safety caps is DID_NOT_CONVERGE.
  # A crash/wall-ceiling still wins (a failure, not a verdict).
  if [ -n "${EXPT_TARGET_ACC:-}" ] && [ "$crash" -eq 0 ] && [ "$wall" -eq 0 ] && [ "$agg" -gt 0 ]; then
    if [ "${EXPT_LAST_CONVERGED:-0}" = "1" ]; then status="CONVERGED"
    elif [ "${EXPT_LAST_STALLED:-0}" = "1" ]; then status="STALLED"
    else status="DID_NOT_CONVERGE"; fi
  fi
  # Backstop watchdog fired (expt_launch): the run never self-stopped at its
  # budget (a hang, e.g. the documented fwdllm real-mode MQTT hangs) and had to
  # be force-killed so the sequence could continue. Takes priority -- "we had to
  # force-kill this" is the operationally load-bearing fact; agg/crash counts
  # above still print for diagnosis.
  [ "${EXPT_LAST_TIMED_OUT:-0}" = "1" ] && status="TIMEOUT_KILLED"
  EXPT_LAST_HEALTH="$status"; export EXPT_LAST_HEALTH
  printf "  [%s] %-14s agg_round=%s stopping_run=%s wall_ceiling=%s starvation=%s crash=%s (%s agg log(s))\n" \
    "$label" "$status" "$agg" "$stop" "$wall" "$starv" "$crash" "$nlogs"
  [ "$agg" -gt 0 ] && [ "$wall" -eq 0 ] && [ "$crash" -eq 0 ]
}

# expt_timed_run label runtime_s buffer_s shell_log -- cmd... -- run a command in
# its OWN process group under a wall-clock timeout, killing the whole tree
# (SIGTERM grace -> SIGKILL) on overrun, then sweeping stragglers so no orphan
# trainers/aggregator survive and GPU memory drains. The campaign primitive shared
# by smoke_suite.sh. Emits a 5s elapsed/kill-in ticker. Returns cmd's rc, or 124.
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
# hook `foo` is delegated to a shell function `after_foo` the DRIVER defines (e.g.
# after_parity, after_plot, after_sanity), so the mechanism is shared but each
# example supplies its own command. Unknown hooks warn and are skipped (not fatal).
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
