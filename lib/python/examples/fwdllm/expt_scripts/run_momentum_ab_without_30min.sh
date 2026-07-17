#!/usr/bin/env bash
# Node A of the S1 server-momentum A/B, 30min (1800s) variant (simulate_fwdllm.md
# §A): the WITHOUT-momentum leg, all 3 baselines, real only. The 8min
# `run_momentum_ab_without.sh` pair only reached the early-cold-start window
# (data_id ~0-15) and showed momentum making the exact acc=0.25/mcc=0 collapse
# MORE frequent (fluxtune) or EARLIER (fwdllm/fwdllm_plus), not less -- this
# 30min pair reaches deeper into the run to see if that direction holds.
# Sequential, ~30min/run each (~90min total).
#
# Run from the repo root:
#   bash lib/python/examples/fwdllm/expt_scripts/run_momentum_ab_without_30min.sh
#
# Prerequisites: conda env `dg_flame` active, an MQTT broker reachable (real
# mode needs one; this script does not start one for you).

set -uo pipefail

RUNS=(
  "lib/python/examples/fwdllm/expt_scripts/fluxtune_n10_smoke.yaml"
  "lib/python/examples/fwdllm/expt_scripts/fwdllm_n100_smoke.yaml"
  "lib/python/examples/fwdllm/expt_scripts/fwdllm_plus_n100_smoke.yaml"
)

for yaml in "${RUNS[@]}"; do
  echo "=== $(date '+%Y-%m-%d %H:%M:%S') START (without momentum): $yaml ==="
  python -m flame.launch.run_experiment "$yaml"
  status=$?
  echo "=== $(date '+%Y-%m-%d %H:%M:%S') END (exit=$status): $yaml ==="
  sleep 5
done
