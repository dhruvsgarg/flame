#!/usr/bin/env bash
# Node B of the S1 server-momentum A/B (simulate_fwdllm.md §A "Short-run
# iteration loop"): the WITH-momentum leg (server_momentum=0.9), all 3
# baselines, real only -- pairs with run_momentum_ab_without.sh (node A).
# Sequential, 8min/run each (~24min total).
#
# Run from the repo root:
#   bash lib/python/examples/fwdllm/expt_scripts/run_momentum_ab_with.sh
#
# Prerequisites: conda env `dg_flame` active, an MQTT broker reachable (real
# mode needs one; this script does not start one for you).

set -uo pipefail

RUNS=(
  "lib/python/examples/fwdllm/expt_scripts/fluxtune_n10_smoke_short_momentum.yaml"
  "lib/python/examples/fwdllm/expt_scripts/fwdllm_n10_smoke_short_momentum.yaml"
  "lib/python/examples/fwdllm/expt_scripts/fwdllm_plus_n10_smoke_short_momentum.yaml"
)

for yaml in "${RUNS[@]}"; do
  echo "=== $(date '+%Y-%m-%d %H:%M:%S') START (with momentum=0.9): $yaml ==="
  python -m flame.launch.run_experiment "$yaml"
  status=$?
  echo "=== $(date '+%Y-%m-%d %H:%M:%S') END (exit=$status): $yaml ==="
  sleep 5
done
