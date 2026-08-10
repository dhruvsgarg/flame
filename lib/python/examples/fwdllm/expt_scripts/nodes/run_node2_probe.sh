#!/usr/bin/env bash
# Node 2 -- S-H at MATCHED N, the A/B the 08-08 attempt could not have made
#   (handoff §11.3, §22.2)
#
# Averaging the P probes cuts b^2 by E*P = 29.9 -- confirmed in-run: var fell
# 26.6-28.4x (§22.2). But the var gate then commits at I~1 instead of 18.5, so a
# free-gate `select` vs `mean` comparison moves the pool by ~10x at the same time
# and measures nothing. Both arms here PIN the pool: var_threshold=0 is never
# satisfied, so every commit fires at the cap, giving N = K*20 = 200 exactly.
#
# PREDICTIONS at commit ~10:
#   arm 2 vs arm 1, at identical N: rho drops by sqrt(E*P) = 5.45x, and so does
#         ||dtheta||. This is the clean form of the check -- rho alone is
#         gate-confounded, rho*sqrt(N) is not.
#   ||theta_tr||^2 growth rate falls ~30x (it goes as rho^2)
#   arm 3 (free gate): same aim, I collapses 20 -> 1-2, i.e. ~10x fewer SERIAL
#         round trips per commit. That is what S-H actually buys under the
#         shipped gate, and it is the S-D result for free.
set -u
. "$(dirname "$0")/_node_lib.sh"

A=(--only fluxtune --mode sim --yes --clean --force
   --server-update-audit --pool-split-half-audit
   --num-trainers 100 --num-gpus 8 --max-runtime-s 14400 --agg-goal 10 --c 30)
# var_threshold=0 is unreachable => the max_iter cap sets I, identically in both.
PIN=(--var-threshold 0 --max-iter-per-data-id 20 --var-stopping-policy off)

node_run node2 "select, N pinned at 200" "${A[@]}" "${PIN[@]}" --probe-combine select
node_run node2 "mean,   N pinned at 200" "${A[@]}" "${PIN[@]}" --probe-combine mean
node_run node2 "mean,   free var gate"   "${A[@]}" --probe-combine mean

echo "=== [node2] done $(date -Is) ==="
