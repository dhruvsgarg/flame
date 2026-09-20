#!/usr/bin/env bash
# Node 1 -- S-C: a commit rule with no configured threshold (handoff §15.7, §22.3)
#
# The design requirement this tests: the aggregator should decide when it has
# pooled enough from a quantity that does NOT depend on alpha, K, C or the round
# index. `var < 0.3` fails that on every count -- its floor moves 4.7x with alpha
# and 36x with ||theta|| (§9.7c). n_target replaces it with the criterion itself,
#
#     commit when   n_eff >= N_req = p * (rho_t / s)^2 / G_rule
#
# where p is read off the model, G_rule is closed form, rho_t is the step the
# server is about to take, and s ~ 0.4 is the O(1) safety factor. Nothing in it
# carries units, so the SETPOINT is fixed while the pool it demands moves freely
# with the anneal, the data, and the cohort.
#
# PREDICTIONS at commit ~10 (rho* = 0.01, mean, p = 450,340 => N_req = 28):
#   arm 1 (var):      I ~ 1-2 -- under `mean` var collapses below 0.3 immediately,
#                     so the pool is set by an accident of units (§22.2)
#   arm 2 (n_target): I ~ 3, and I FALLS as rho anneals -- the gate spends less
#                     pooling on smaller steps, which is the criterion working
#   arm 3 (alpha=0.1): I within 10% of arm 2 at the SAME s. The var gate cannot
#                     do this -- its floor at alpha=0.1 is 3.5x higher (§9.7c).
#                     A miss here is the one result that would sink S-C.
set -u
. "$(dirname "$0")/_node_lib.sh"

A=(--only fluxtune --mode sim --yes --clean --force
   --server-update-audit --pool-split-half-audit
   --num-trainers 100 --num-gpus 8 --max-runtime-s 14400 --agg-goal 10 --c 30
   --probe-combine mean
   --server-step-rule trust_ratio --rho-star 0.01 --rho-schedule rm --rho-exp 0.55)

node_run node1 "var gate (control)"     "${A[@]}" --commit-gate var
node_run node1 "n_target s=0.4"         "${A[@]}" --commit-gate n_target --gate-safety-s 0.4
node_run node1 "n_target s=0.4 a=0.1"   "${A[@]}" --commit-gate n_target --gate-safety-s 0.4 \
    --partition-method 'niid_label_clients=100_alpha=0.1'

echo "=== [node1] done $(date -Is) ==="
