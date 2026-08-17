#!/usr/bin/env bash
# Shared harness for the node scripts: run an arm, then PROVE it ran.
#
# 2026-08-08 cost a full night on three nodes because 12 arms died at aggregator
# startup and the chain kept going (handoff §22.1). An arm that produces zero
# commits is a systemic fault, not a bad config, so the node aborts on it; an arm
# that commits and then fails is just that arm, so the chain continues.
set -u

EXPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # .../examples/fwdllm
RUNS="$EXPT_DIR/experiments"

node_run () {  # node_run <node-label> <arm-label> [launcher flags...]
  local node="$1" label="$2"; shift 2
  echo "=== [$node] $label  $(date -Is) ==="

  # NODE_DRY_RUN=1: generate every arm's configs and stop. Checks flag parsing
  # and knob placement for all arms in seconds, without burning a node.
  if [ "${NODE_DRY_RUN:-0}" = "1" ]; then
    "$EXPT_DIR/expt_scripts/run_sequential.sh" "$@" --dry-run >/tmp/node_dry.$$ 2>&1 || {
      echo "!!! [$node] $label DRY-RUN FAILED:" >&2; tail -5 /tmp/node_dry.$$ >&2
      rm -f /tmp/node_dry.$$; exit 1; }
    grep -o 'generated cfgs in [^ ]*' /tmp/node_dry.$$ | sed "s|^|--- [$node] $label |"
    rm -f /tmp/node_dry.$$
    return 0
  fi

  local before since
  before="$(ls -1dt "$RUNS"/run_* 2>/dev/null | head -1)"
  since="$(date +%s)"

  # Job control so the launcher gets its OWN process group: the watchdog kills by
  # group, and without this that group is the node script itself.
  set -m
  "$EXPT_DIR/expt_scripts/run_sequential.sh" "$@" &
  local run_pid=$!
  set +m

  # Side-car: kill an arm that has stopped progressing rather than let it burn the
  # slot. Its predicates are mechanical (hang / gate starvation / zero steps), NOT
  # accuracy -- holding a plateau is what a controller arm is supposed to do.
  # NODE_WATCH=0 disables; NODE_WATCH_ARGS passes through extra flags.
  local watch_pid=""
  if [ "${NODE_WATCH:-1}" = "1" ]; then
    python "$EXPT_DIR/expt_scripts/watch_arm.py" --exp-dir "$RUNS" --since "$since" \
      --pgid "$run_pid" --kill ${NODE_WATCH_ARGS:-} &
    watch_pid=$!
  fi

  wait "$run_pid" || echo "!!! [$node] $label launcher rc=$? (continuing if it committed)"
  if [ -n "$watch_pid" ]; then
    kill "$watch_pid" 2>/dev/null; wait "$watch_pid" 2>/dev/null
  fi

  local run commits
  run="$(ls -1dt "$RUNS"/run_* 2>/dev/null | head -1)"
  if [ -z "$run" ] || [ "$run" = "$before" ]; then
    echo "!!! [$node] $label produced NO RUN DIRECTORY -- aborting node." >&2
    exit 1
  fi
  commits="$(grep -c '"event": "server_update"' "$run"/telemetry/aggregator_*.jsonl 2>/dev/null || echo 0)"
  echo "--- [$node] $label -> $(basename "$run"), commits=$commits"
  if [ -f "$run/arm_stall.json" ]; then
    echo "!!! [$node] $label was KILLED BY THE WATCHDOG:" >&2
    cat "$run/arm_stall.json" >&2
  fi
  # Enactment lines: cheap, and the only way to catch a knob that did not take.
  grep -m1 -h '\[ProbeDim\]'      "$run"/*trainers.log   2>/dev/null
  grep -m1 -h '\[FD\] spacing'    "$run"/*trainers.log   2>/dev/null
  grep -m1 -h '\[probe_combine'   "$run"/*trainers.log   2>/dev/null
  grep -m1 -h '\[TrainableScope\]' "$run"/*trainers.log  2>/dev/null
  grep -m1 -h '\[ServerStep\]'    "$run"/*aggregator.log 2>/dev/null
  grep -m1 -h '\[CommitGate\]'    "$run"/*aggregator.log 2>/dev/null
  grep -m1 -h '\[CosProbe\]'      "$run"/*aggregator.log 2>/dev/null
  if [ "$commits" -lt 5 ]; then
    echo "!!! [$node] $label committed $commits times -- systemic fault, aborting node." >&2
    grep -m1 -A12 Traceback "$run"/*aggregator.log >&2 2>/dev/null
    exit 1
  fi
  # The four in-flight gates of buildplan §6, at the end rather than never. A
  # breach does NOT abort: the arm is already spent, and a controller arm that
  # breaches is void as ACCEPTANCE, which is a reading, not a launcher decision.
  python "$EXPT_DIR/expt_scripts/check_arm_health.py" "$run" ${NODE_ARM_KIND:+--expect-controller} \
    || echo "!!! [$node] $label BREACHED a §6 gate above -- read before scoring it" >&2
}
