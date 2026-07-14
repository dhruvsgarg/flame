#!/usr/bin/env bash
# Node A of the S1 server-momentum A/B (simulate_fwdllm.md §A "Short-run
# iteration loop"): the WITHOUT-momentum leg, all 3 baselines, real only (sim
# isn't needed here -- it answers a real<->sim parity question, not "does
# momentum help training stability"). Sequential, 8min/run each (~24min total).
#
# Run from the repo root:
#   bash lib/python/examples/fwdllm/expt_scripts/run_momentum_ab_without.sh
#
# Prerequisites: conda env `dg_flame` active, an MQTT broker reachable (real
# mode needs one; this script does not start one for you).

set -uo pipefail

RUNS=(
  "lib/python/examples/fwdllm/expt_scripts/fluxtune_n10_smoke_short.yaml"
  "lib/python/examples/fwdllm/expt_scripts/fwdllm_n10_smoke_short.yaml"
  "lib/python/examples/fwdllm/expt_scripts/fwdllm_plus_n10_smoke_short.yaml"
)

for yaml in "${RUNS[@]}"; do
  echo "=== $(date '+%Y-%m-%d %H:%M:%S') START (without momentum): $yaml ==="
  python -m flame.launch.run_experiment "$yaml"
  status=$?
  echo "=== $(date '+%Y-%m-%d %H:%M:%S') END (exit=$status): $yaml ==="
  sleep 5
done
