#!/usr/bin/env bash
# Phase 4 on four nodes. ONE command per node, no waves, NO CROSS-NODE DEPENDENCY.
#
#   tmux new -s p4 'run_node.sh <1|2|3|4> 2>&1 | tee ~/p4_node<N>.log'
#   SMOKE=1 run_node.sh <N>          # same path, ~20-30 min, sizes the real run
#
# ALWAYS IN TMUX: a node outlives a login, and that is how the 2026-08-16
# backprop-ceiling number was lost.
#
# EACH NODE OWNS ONE DATASET, END TO END:
#   1  agnews controller -> agnews control                    (no profile needed)
#   2  real yahoo  -> fluxtune_yahoo.yaml  -> controller -> control
#   3  real yelp-p -> fluxtune_yelp-p.yaml -> controller -> control
#   4  backprop ceilings (yahoo, then yelp-p) -- §9 rung 1, ~10 min each, then free
#
# WHY THIS SHAPE AND NOT A SPLIT-BY-ARM ONE. `/home/.../flame` is LOCAL disk per
# node (only `/coc/scratch` is shared), so putting a pair's two arms on two nodes
# means either a cross-node handoff of the sim charge profile, or each node
# profiling its own real run. The second is worse than it looks: the profile is
# what converts vclock to real work, so a controller and a control charged from
# DIFFERENT profiles are no longer compared at equal vclock -- and this project
# already rejected a shared profile over a 0.6-3.4% clock effect. Keeping a pair
# on one node makes it identically priced BY CONSTRUCTION, and removes the
# barrier entirely. The cost is wall clock on the seq-256 nodes; see SMOKE.
#
# The tokenizer cache is shared and already 101/101 for all three datasets
# (buildplan §10), so nothing here waits on anything.
set -u
. "$(dirname "$0")/_node_lib.sh"

NODE="${1:?usage: run_node.sh <1|2|3|4>}"

FW="$EXPT_DIR"
P4="$FW/expt_scripts/nodes/run_node_p4.sh"
PROBE_LOGS="$FW/experiments/_probe_logs"
mkdir -p "$PROBE_LOGS"
export FWDLLM_FD_SCALE_INVARIANT=1
DRY="${NODE_DRY_RUN:-0}"

REAL_BUDGET=3000
if [ "${SMOKE:-0}" = "1" ]; then
  export SMOKE=1
  REAL_BUDGET=600
  # Armed inside a smoke's horizon; the production defaults are 20 min / 200.
  export NODE_WATCH_ARGS="${NODE_WATCH_ARGS:---stall-window-s 420 --grace-commits 25}"
fi

# shellcheck disable=SC1091
. "$FW/../scripts/expt_runner.sh"
expt_activate_conda || exit 2
expt_pin_pythonpath "$(cd "$FW/../../../.." && pwd)"

# §9 rung 1: what this data path reaches with an EXACT gradient. ~0.70 clears
# tokenization/label vocab/seq length/h5 ranges/partition/loader and makes the
# gap purely optimization; ~0.30 indicts the path and that dataset's arms are
# measuring nothing. agnews calibrated at 0.850.
ceiling () {
  local ds="$1"
  local cfg; cfg="$(ls -1dt "$FW"/experiments/run_*"$ds"*/aggregator_config.json 2>/dev/null | head -1)"
  [ -n "$cfg" ] || cfg="$(ls -1dt "$FW"/experiments/run_*/aggregator_config.json 2>/dev/null | head -1)"
  [ -n "$cfg" ] || { echo "!!! [node$NODE] no run config to read hyperparameters from." >&2; return 1; }
  echo "=== [node$NODE] backprop ceiling, $ds (§9 rung 1)  $(date -Is) ==="
  if [ "$DRY" = "1" ]; then
    echo "--- [node$NODE] NODE_DRY_RUN: would probe $ds off $cfg"; return 0
  fi
  python "$FW/expt_scripts/probe_backprop_ceiling.py" --config "$cfg" \
    --dataset "$ds" --clients 10 --epochs 3 2>&1 \
    | tee "$PROBE_LOGS/backprop_ceiling_${ds}_$(date +%Y%m%d_%H%M).log"
}

# A real-mode arm, then this dataset's own sim charge profile. Real mode is the
# only source: profile_sim_charges.py pools `vclock_charge` events with
# time_mode == "real" and finds nothing in a sim run. Both arms below then charge
# from this one file, so the pair is priced identically.
profile_dataset () {
  local ds="$1"
  local out="$FW/sim_charge_profiles/fluxtune_$ds.yaml"
  # A smoke's REAL_BUDGET=600 cannot price a vclock -- it charged fedavg 50x high off
  # one warmup stall (P4.9 defect 4) -- and a profile that LOOKS present skips the real
  # run forever after. Exercise the path, write to the gitignored smoke/ dir, price
  # nothing: "smoke profiles must never price a scored arm" (d5b10a2).
  if [ "${SMOKE:-0}" = "1" ]; then
    mkdir -p "$FW/sim_charge_profiles/smoke"
    out="$FW/sim_charge_profiles/smoke/fluxtune_$ds.yaml"
  fi
  if [ -f "$out" ]; then
    echo "--- [node$NODE] ${out#$FW/} exists -- skipping the real run"; return 0
  fi
  echo "=== [node$NODE] real-mode $ds for its sim charge profile  $(date -Is) ==="
  local args=(--only fluxtune --mode real --dataset "$ds"
              --num-trainers 100 --num-gpus 8 --agg-goal 10 --c 30
              --adapter-reduction-factor 16 --max-runtime-s "$REAL_BUDGET")
  if [ "$DRY" = "1" ]; then
    "$FW/expt_scripts/run_sequential.sh" "${args[@]}" --dry-run \
      | grep -E 'generated cfgs|✗|ERROR' || true
    echo "--- [node$NODE] NODE_DRY_RUN: would profile into ${out#$FW/}"
    return 0
  fi
  node_run "node$NODE" "real-mode $ds (for its sim charge profile)" \
    "${args[@]}" --yes --clean

  local run
  run="$(ls -1dt "$FW"/experiments/run_*"$ds"*real* 2>/dev/null | head -1)"
  [ -n "$run" ] || { echo "!!! [node$NODE] no real $ds run dir -- cannot profile." >&2; return 1; }

  # REFUSE to profile a truncated run. A killed real arm still has telemetry, and
  # profiling it wrote a yelp-p file off n=7 samples against a healthy n=3601
  # (2026-08-17) -- a garbage profile that then LOOKS present and skips the real
  # run on every retry. Better no profile than a plausible wrong one.
  if [ -f "$run/arm_stall.json" ]; then
    echo "!!! [node$NODE] the real $ds arm was killed by the watchdog -- NOT profiling." >&2
    cat "$run/arm_stall.json" >&2; return 1
  fi
  local _n
  _n="$(grep -ho '"event": "vclock_charge"' "$run"/telemetry/aggregator_*.jsonl 2>/dev/null | wc -l)"
  if [ "${_n:-0}" -lt "${MIN_CHARGE_SAMPLES:-200}" ]; then
    echo "!!! [node$NODE] only $_n vclock_charge samples in $(basename "$run") " >&2
    echo "!!!   (want >= ${MIN_CHARGE_SAMPLES:-200}) -- too short to price a vclock. NOT profiling." >&2
    return 1
  fi

  # Seed from the baseline profile so the `charge:` flags and rationales carry
  # over. profile_sim_charges.py writes a BRAND-NEW entry with `charge: false`
  # ("review before enabling"), so a from-scratch per-dataset profile charges
  # NOTHING -- which would silently un-price the very arms task B exists to price.
  [ -f "$out" ] || cp -f "$FW/sim_charge_profiles/fluxtune.yaml" "$out"
  echo "--- [node$NODE] profiling from $(basename "$run") ($_n charge samples)"
  ( cd "$FW/expt_scripts" && python profile_sim_charges.py --real-run "$run" \
      --out "$out" --only-observed ) || return 1
  # Prove the charges survived the reseed -- an all-false profile is inert.
  python - "$out" <<'PYCHK'
import sys, yaml
d = yaml.safe_load(open(sys.argv[1])) or {}
on = [f"{l}.{k}" for l, e in d.items() if l != "_meta"
      for k, v in (e or {}).items() if (v or {}).get("charge")]
print("  [profile] charging %d entries: %s" % (len(on), ", ".join(on) or "NONE -- inert!"))
sys.exit(0 if on else 1)
PYCHK
  # The tag the launcher gates on: if it does not name this dataset the profile is
  # refused for it and the arm silently falls back to needing --force.
  grep -A3 '^_meta' "$out"
}

# A dataset's whole pair, in order, on this node. The controller runs first: it is
# the arm that scores the controller, and if it breaches a §6 gate there is no
# point spending the slot on its control.
pair () {
  local ds="$1"
  "$P4" "$ds" controller || echo "!!! [node$NODE] $ds controller returned $? -- continuing to control" >&2
  "$P4" "$ds" control
}

case "$NODE" in
  1) pair agnews ;;
  2) profile_dataset yahoo  && pair yahoo ;;
  3) profile_dataset yelp-p && pair yelp-p ;;
  4) ceiling yahoo; ceiling yelp-p ;;
  *) echo "unknown node '$NODE' (want 1|2|3|4)" >&2; exit 2 ;;
esac

echo "=== [node$NODE] done $(date -Is) ==="
echo "Read every arm this node produced:"
echo "  for r in \$(ls -1dt $FW/experiments/run_* | head -3); do \\"
echo "    python $FW/expt_scripts/check_arm_health.py \$r; done"
