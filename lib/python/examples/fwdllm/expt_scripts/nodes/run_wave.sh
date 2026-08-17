#!/usr/bin/env bash
# Phase 4 across four nodes, as two waves. One command per node, and the two
# waves chain with `&&` -- no manual step in between.
#
#   S=<1|2|3|4>
#   tmux new -s p4 "run_wave.sh 1 $S && run_wave.sh 2 $S 2>&1 | tee ~/p4_slot$S.log"
#
# ALWAYS IN TMUX: a slot outlives a login, and that is how the 2026-08-16
# backprop-ceiling number was lost.
#
# WHAT EACH WAVE IS FOR
#   wave 1  the instruments, plus the one dataset that already has a calibration.
#           Slots 1-2 each run a REAL-mode arm and turn it into that dataset's sim
#           charge profile -- profile_sim_charges.py pools `vclock_charge` events
#           with time_mode == "real", so only a real run can produce one, and
#           without one a seq-256 sim arm is priced off agnews numbers and its
#           vclock is wrong. Slots 3-4 run the agnews controller/control pair,
#           which needs no profile and is the arm that scores the controller.
#   wave 2  the same pair on yahoo and yelp-p, priced by wave 1's profiles.
#
# THE ONLY CROSS-NODE DEPENDENCY, and this script handles it. `/home/.../flame` is
# LOCAL disk per node; `/coc/scratch` is shared. So a profile written on node 1
# would not exist on node 2. Wave 1 PUBLISHES each profile to the shared dir
# below, and wave 2 FETCHES both and WAITS for them (--wait-profiles-min, default
# 180) before launching. The tokenizer cache needs none of this -- it is already
# on /coc/scratch, 101/101 for all three datasets (buildplan §10).
set -u
. "$(dirname "$0")/_node_lib.sh"

WAVE="${1:?usage: run_wave.sh <1|2> <1|2|3|4>}"
SLOT="${2:?usage: run_wave.sh <1|2> <1|2|3|4>}"

FW="$EXPT_DIR"
P4="$FW/expt_scripts/nodes/run_node_p4.sh"
PROBE_LOGS="$FW/experiments/_probe_logs"
SHARED_PROFILES="${FWDLLM_SHARED_PROFILES:-/coc/scratch/dgarg/fl_datasets/fwdllm/sim_charge_profiles}"
WAIT_MIN="${WAVE2_WAIT_PROFILES_MIN:-180}"
export FWDLLM_FD_SCALE_INVARIANT=1
DRY="${NODE_DRY_RUN:-0}"

# SMOKE=1: same code path, ~15 min per slot. Its profiles are built from a
# 10-minute real run and are NOT fit to price a real arm, so they go to their own
# shared dir and never overwrite the production ones. Also tightens the watchdog
# so it is actually armed inside a smoke's horizon.
REAL_BUDGET=3000
if [ "${SMOKE:-0}" = "1" ]; then
  export SMOKE=1
  REAL_BUDGET=600
  SHARED_PROFILES="$SHARED_PROFILES/smoke"
  WAIT_MIN="${WAVE2_WAIT_PROFILES_MIN:-30}"
  export NODE_WATCH_ARGS="${NODE_WATCH_ARGS:---stall-window-s 420 --grace-commits 25}"
fi
PROF_DIR="$FW/sim_charge_profiles"
[ "${SMOKE:-0}" = "1" ] && PROF_DIR="$FW/sim_charge_profiles/smoke"
mkdir -p "$PROBE_LOGS" "$SHARED_PROFILES" "$PROF_DIR"

# The probe and the profiler run OUTSIDE run_sequential.sh and so miss its conda
# activation -- and the base env has no h5py. Same helper the launcher uses.
# shellcheck disable=SC1091
. "$FW/../scripts/expt_runner.sh"
expt_activate_conda || exit 2
expt_pin_pythonpath "$(cd "$FW/../../../.." && pwd)"

# A real-mode arm, then its own sim charge profile, then publish it.
profile_dataset () {
  local ds="$1" budget="$2"
  local out="$PROF_DIR/fluxtune_$ds.yaml"
  if [ -f "$out" ]; then
    echo "--- [wave1-$SLOT] $(basename "$out") already exists -- skipping the real run"
    cp -f "$out" "$SHARED_PROFILES/" && echo "--- published to $SHARED_PROFILES"
    return 0
  fi
  echo "=== [wave1-$SLOT] real-mode $ds for its sim charge profile  $(date -Is) ==="
  local args=(--only fluxtune --mode real --dataset "$ds"
              --num-trainers 100 --num-gpus 8 --agg-goal 10 --c 30
              --adapter-reduction-factor 16 --max-runtime-s "$budget")
  if [ "$DRY" = "1" ]; then
    "$FW/expt_scripts/run_sequential.sh" "${args[@]}" --dry-run \
      | grep -E 'generated cfgs|✗|ERROR' || true
    echo "--- [wave1-$SLOT] NODE_DRY_RUN: would profile into $(basename "$out")"
    return 0
  fi
  node_run "wave1-$SLOT" "real-mode $ds (for its sim charge profile)" \
    "${args[@]}" --yes --clean

  local run
  run="$(ls -1dt "$FW"/experiments/run_*"$ds"*real* 2>/dev/null | head -1)"
  [ -n "$run" ] || { echo "!!! [wave1-$SLOT] no real $ds run dir." >&2; return 1; }
  echo "--- [wave1-$SLOT] profiling from $(basename "$run")"
  ( cd "$FW/expt_scripts" && python profile_sim_charges.py --real-run "$run" \
      --out "$out" --only-observed ) || return 1
  # The tag the launcher gates on. If it does not name this dataset, the profile
  # is refused for it and the arm silently needs --force again.
  grep -A3 '^_meta' "$out"
  cp -f "$out" "$SHARED_PROFILES/" && echo "--- published to $SHARED_PROFILES"
}

# Wave 2's barrier: pull both profiles out of the shared dir, waiting for whichever
# node is still producing one. Without this the arm launches on an agnews-priced
# profile (needing --force) and its vclock is wrong.
fetch_profiles () {
  local deadline=$(( $(date +%s) + WAIT_MIN * 60 )) missing
  while :; do
    missing=""
    for ds in yahoo yelp-p; do
      local src="$SHARED_PROFILES/fluxtune_$ds.yaml"
      if [ -f "$src" ]; then
        cp -f "$src" "$PROF_DIR/"
      else
        missing="$missing $ds"
      fi
    done
    [ -z "$missing" ] && { echo "--- [wave2-$SLOT] both profiles in place"; return 0; }
    if [ "$DRY" = "1" ]; then
      echo "--- [wave2-$SLOT] NODE_DRY_RUN: would wait for$missing (up to ${WAIT_MIN}m)"
      return 0
    fi
    if [ "$(date +%s)" -ge "$deadline" ]; then
      echo "!!! [wave2-$SLOT] waited ${WAIT_MIN}m and still missing:$missing" >&2
      echo "!!! Wave 1 slot 1 (yahoo) / slot 2 (yelp-p) must finish first." >&2
      return 1
    fi
    echo "--- [wave2-$SLOT] waiting for$missing in $SHARED_PROFILES ($(date +%H:%M))"
    sleep 300
  done
}

case "$WAVE/$SLOT" in
  # ---- wave 1 --------------------------------------------------------------
  1/1)
    # §9 rung 1 runs FIRST because it is ~10 min and it gates wave 2's yahoo
    # slots: ~0.70 clears the whole data path and makes the yahoo gap purely
    # optimization; ~0.30 indicts the path and those arms measure nothing.
    echo "=== [wave1-1] backprop ceiling, yahoo (task A)  $(date -Is) ==="
    CFG="$(ls -1dt "$FW"/experiments/run_*yahoo*/aggregator_config.json 2>/dev/null | head -1)"
    [ -n "$CFG" ] || { echo "!!! [wave1-1] no yahoo run config to read." >&2; exit 1; }
    if [ "$DRY" = "1" ]; then
      echo "--- [wave1-1] NODE_DRY_RUN: would probe off $CFG"
    else
      python "$FW/expt_scripts/probe_backprop_ceiling.py" --config "$CFG" \
        --dataset yahoo --clients 10 --epochs 3 2>&1 \
        | tee "$PROBE_LOGS/backprop_ceiling_yahoo_$(date +%Y%m%d_%H%M).log"
    fi
    profile_dataset yahoo "$REAL_BUDGET"
    ;;
  1/2) profile_dataset yelp-p "$REAL_BUDGET" ;;
  1/3) "$P4" agnews controller ;;
  1/4) "$P4" agnews control ;;

  # ---- wave 2 --------------------------------------------------------------
  2/1) fetch_profiles && "$P4" yahoo  controller ;;
  2/2) fetch_profiles && "$P4" yahoo  control ;;
  2/3) fetch_profiles && "$P4" yelp-p controller ;;
  2/4) fetch_profiles && "$P4" yelp-p control ;;

  *) echo "unknown wave/slot '$WAVE/$SLOT' (want wave 1|2, slot 1|2|3|4)" >&2; exit 2 ;;
esac

echo "=== [wave$WAVE-$SLOT] done $(date -Is) ==="
