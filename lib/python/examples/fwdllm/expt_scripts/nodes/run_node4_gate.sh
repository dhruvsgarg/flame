#!/usr/bin/env bash
# Node 4 -- S-C, properly this time (handoff §15.7, §22.3a)
#
# The 08-09 gate node could not answer B4: under the anneal, N_req = p(rho_t/s)^2
# /G_rule went 28.1 -> 0.0 by commit ~20, so `n_target` committed at the I=1 floor
# on 1,271 of 1,273 commits -- and the `var` control floored identically at 1.14.
# Two gates, no contrast, both arms peaking below 0.40 and DECAYING to ~0.28.
#
# The cause is a composition, not a gate bug: S-C sizes the pool from the step,
# S-B shrinks the step, so cos = sqrt(G_rule*N/p) falls too and progress rho*cos
# decays as rho^2 instead of rho. Robbins-Monro wants a fixed-quality gradient
# with a shrinking step -- not both shrinking together.
#
# `gate_rho_ref=setpoint` sizes N_req from rho*_0 instead. Under `rm` rho*=0.02
# mean, N_req is then flat at 112.6 where `annealed` decays it 2441x by commit
# 1200 (expt_scripts/test_commit_gate.py). No-op under const and under raw_sgd.
#
# rho* = 0.06 here, not 0.01: §22.3d says 0.01 does not learn at any pool, so
# testing the gate at 0.01 would again measure a stalled arm rather than the gate.
#
# PREDICTIONS at commit ~50:
#   arm 1 (annealed, control): N_req collapses, I floors at 1, N ~ 10. Reproduces
#         the 08-09 failure ON PURPOSE -- it is the within-node control.
#   arm 2 (setpoint): N_req flat at p(0.06/0.4)^2/10 = 1013 -- ABOVE what K=10
#         can pool in 20 iterations, so expect I to sit at the cap and N at 200.
#         That is itself the finding: at rho*=0.06 the criterion demands 5x the
#         pool the cohort can supply, which is an argument for K, not for s.
#   arm 3 (setpoint @ alpha=0.1): I within 10% of arm 2 at the SAME s. This is
#         the whole scale-free claim -- the var gate cannot do it, its floor at
#         alpha=0.1 is 3.5x higher (§9.7c).
#
# WHAT WOULD SINK S-C: arm 3 needing a different s than arm 2. That breaks C2's
# reframing (§25) -- the controller would be as unportable as the threshold it
# replaced. Note arm 2's prediction already implies s and K are coupled; if I is
# cap-bound in both arms the alpha comparison is uninformative and this node has
# to be re-run at K=30.
#
# 8h vclock (~2.4h wall x 3 arms), double the 08-09 portfolio: those arms reached
# only ~190 commits, which is roughly where the raw-SGD anchor turns over -- every
# question here is about what happens PAST that point. --sim-wall-ceiling-h 2.5 is
# the unattended failsafe, not a target.
set -u
. "$(dirname "$0")/_node_lib.sh"
A=(--only fluxtune --mode sim --yes --clean --force
   --server-update-audit --pool-split-half-audit
   --num-trainers 100 --num-gpus 8 --max-runtime-s 28800 --sim-wall-ceiling-h 2.5 --agg-goal 10 --c 30
   --probe-combine mean --commit-gate n_target --gate-safety-s 0.4
   --server-step-rule trust_ratio --rho-star 0.06 --rho-schedule rm --rho-exp 0.25)
node_run node4 "n_target rho_ref=annealed" "${A[@]}" --gate-rho-ref annealed
node_run node4 "n_target rho_ref=setpoint" "${A[@]}" --gate-rho-ref setpoint
node_run node4 "n_target setpoint a=0.1"   "${A[@]}" --gate-rho-ref setpoint \
    --partition-method 'niid_label_clients=100_alpha=0.1'
echo "=== [node4] done $(date -Is) ==="
