#!/usr/bin/env bash
# Node 2 -- B1: measure cos(G,g). The number the whole document is written in
#   (handoff §15.1, §22.3e)
#
# Every rho/cos figure here is still a prediction from (a/b)*sqrt(N/p), and the
# only estimator we had -- the pool split-half cosine -- is dead: over 179
# commits at MATCHED N=200 it returned -1.7e-05 +- 1.2e-04 on select and
# -3.1e-05 +- 1.1e-04 on mean. Both negative, both inside one SE of zero. So
# manufacture g instead: one backward pass, server-side, on a fixed held-out
# batch, once per commit. No protocol change and no client cost.
#
# Arms 1-2 REPLICATE node2's pinned pair exactly (raw_sgd, N=200, select/mean),
# so the probe lands on a pool whose rho, N and var are already known to 1% and
# the only new quantity is cos itself.
#
# PREDICTIONS at commit ~10:
#   arm 1 (select, N=200): cos_pred = sqrt(2.988*200/450340) = 0.0364.
#         If measured cos ~ 0.036, the closed form is right and the §9.5
#         shortfall was an artefact of the split-half estimator.
#         If measured cos ~ 0.008-0.015, effective n is >=5.5x below N and EVERY
#         sizing table in §3.5 is optimistic by that factor.
#   arm 2 (mean, N=200):   cos must rise by sqrt(10/2.988) = 1.83x over arm 1.
#         This is G_rule measured directly for the first time; a miss here means
#         b^2/a is not what §3.1 says and S-H's whole case is arithmetic, not fact.
#   arm 3 (mean, free gate): the §22.3d operating point (rho ~ 0.086, peak 0.864).
#         Gives rho/cos where the system actually WORKS -- which is what `s`
#         should have been read off all along (§3.3).
#   All arms: ||G||/||g|| tests the independent-pooling assumption (§12), and
#         ||g|| vs rms|d| resolves H-B.
#
# This node has no failure mode -- any cos is progress -- which is why it is first.
#
# 8h vclock (~2.4h wall x 3 arms), double the 08-09 portfolio: those arms reached
# only ~190 commits, which is roughly where the raw-SGD anchor turns over -- every
# question here is about what happens PAST that point. --sim-wall-ceiling-h 2.5 is
# the unattended failsafe, not a target.
set -u
. "$(dirname "$0")/_node_lib.sh"
A=(--only fluxtune --mode sim --yes --clean --force
   --server-update-audit --pool-split-half-audit --cos-ground-truth-audit
   --num-trainers 100 --num-gpus 8 --max-runtime-s 28800 --sim-wall-ceiling-h 2.5 --agg-goal 10 --c 30)
PIN=(--var-threshold 0 --max-iter-per-data-id 20 --var-stopping-policy off)
node_run node2 "cos: select, N=200"    "${A[@]}" "${PIN[@]}" --probe-combine select
node_run node2 "cos: mean,   N=200"    "${A[@]}" "${PIN[@]}" --probe-combine mean
node_run node2 "cos: mean,   free gate" "${A[@]}" --probe-combine mean
echo "=== [node2] done $(date -Is) ==="
