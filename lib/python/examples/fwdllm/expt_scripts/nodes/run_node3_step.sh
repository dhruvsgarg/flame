#!/usr/bin/env bash
# Node 3 -- S-A vs S-B, isolated (handoff §15.4, §15.6)
#
# All three arms hold the pool fixed at N = 200 (cap-bound, threshold unreachable)
# so the ONLY difference is the step rule. That matters: under a free gate the
# pool moves with the step and neither effect is attributable.
#
# PREDICTIONS at commit ~40:
#   arm 1 (raw_sgd, control): rho ~ 0.12, ||dtheta|| GROWING, ||theta_tr||^2
#         super-linear, single-class collapse by ~commit 150 (the anchor)
#   arm 2 (const rho*=0.01): rho pinned at 0.01 exactly. ||theta|| still grows as
#         (1+rho*^2)^(T/2) -- geometric, ~50x slower. **S-A alone is not enough**,
#         and that is the point of running it.
#   arm 3 (rm rho*=0.02 t^-0.55): rho decays exactly on schedule, ||theta_tr||^2
#         SUB-linear, no collapse at any horizon.
#   Cost side, and the honest risk: rho*cos is the progress rate, so arms 2-3 buy
#   stability with speed. §22.2 saw 0.33-0.36 accuracy at commit 40 where the
#   anchor had 0.58-0.66. If arm 3 is still flat by commit 150 while the anchor
#   peaks at 0.85, the setpoint is too small and s (not the rule) is what to move.
set -u
. "$(dirname "$0")/_node_lib.sh"

A=(--only fluxtune --mode sim --yes --clean --force
   --server-update-audit --pool-split-half-audit
   --num-trainers 100 --num-gpus 8 --max-runtime-s 14400 --agg-goal 10 --c 30
   --var-threshold 0 --max-iter-per-data-id 20 --var-stopping-policy off)

node_run node3 "raw_sgd eta=0.01 (control)" "${A[@]}"
node_run node3 "S-A only: const rho*=0.01"  "${A[@]}" \
    --server-step-rule trust_ratio --rho-star 0.01 --rho-schedule const
node_run node3 "S-A+S-B: rm rho*=0.02"      "${A[@]}" \
    --server-step-rule trust_ratio --rho-star 0.02 --rho-schedule rm --rho-exp 0.55

echo "=== [node3] done $(date -Is) ==="
