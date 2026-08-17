#!/usr/bin/env bash
# Everything worth reading off a node, in ~40 lines instead of a whole tee log.
#
#   summarize_node.sh [n_runs]        # default: the 3 newest runs
#
# `experiments/` is machine-local, so this is what travels between nodes. It
# reads only persisted state -- nothing here needs the run's stdout to have been
# captured, which is why the launch command does not need `tee`.
set -u
FW="$(cd "$(dirname "$0")/.." && pwd)"
N="${1:-3}"

echo "===== $(hostname)  $(date -Is) ====="

for r in $(ls -1dt "$FW"/experiments/run_* 2>/dev/null | head -"$N"); do
  echo
  python "$FW/expt_scripts/check_arm_health.py" "$r" 2>&1 || true
  # The watchdog's verdict, if it fired.
  [ -f "$r/arm_stall.json" ] && { echo "  !! WATCHDOG KILLED THIS ARM:"; sed 's/^/     /' "$r/arm_stall.json"; }
  # A miss here means the vclock ran with NO profiled charges and only warned.
  _sc=$(grep -ho "SIM_CHARGE_PROFILE.*failed to load[^|]*" "$r"/*aggregator.log 2>/dev/null | head -1)
  [ -n "$_sc" ] && echo "  !! UNCHARGED VCLOCK: $_sc"
done

echo
echo "----- newest preflight -----"
_L=$(ls -1dt "$FW"/expt_scripts/smoke_logs/* 2>/dev/null | head -1)
if [ -n "$_L" ] && [ -f "$_L/manifest.tsv.spec.json" ]; then
  python - "$_L/manifest.tsv.spec.json" <<'PY'
import json, sys
checks = json.load(open(sys.argv[1]))["checks"]
bad = [c for c in checks if c["level"] != "ok"]
print(f"  {len(checks)} checks, {len(bad)} not ok")
for c in bad:
    print(f"    {c['level'].upper()}: {c['name']} -- {c['detail'][:150]}")
PY
fi

echo
echo "----- backprop ceilings (§9 rung 1) -----"
for f in "$FW"/experiments/_probe_logs/backprop_ceiling_*.log; do
  [ -f "$f" ] || continue
  echo "  $(basename "$f")"
  grep -E "^\[rig\]|epoch +[0-9]+:|subsample acc" "$f" 2>/dev/null | sed 's/^/    /'
done

echo
echo "----- sim charge profiles present -----"
ls -1 "$FW"/sim_charge_profiles/fluxtune*.yaml 2>/dev/null | sed 's|.*/|  |'
