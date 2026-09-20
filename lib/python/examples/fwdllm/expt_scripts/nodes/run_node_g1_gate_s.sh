#!/usr/bin/env bash
# G-1 -- the commit gate at the safety factor the model actually derived.
#
# The gate has come back inconclusive three times. Every one of those runs used
# gate_safety_s = 0.4, which fl_fwd_ft_solution.md 4.4 superseded to 2.6-4.3 and
# which was never propagated out of the yaml. At s = 0.4 the gate demands
# N_req = p(rho*/s)^2/G_rule = 1,013 at rho*=0.06 -- 5x what K=10 can pool, so it
# pinned at the I cap and every A/B was byte-equivalent to a hard pin. That is the
# whole of the invented "K >= 30, then K >= 51" cohort-width requirement.
#
# 17 arms now put the boundary sharply: rho/cos <= 2.67 holds its peak (12/12),
# rho/cos >= 3.11 degrades or collapses (4/4). So s ~= 2.9 -- and at s = 2.9 the
# same setpoint needs N_req = 19, i.e. I = 2 at K = 10. The gate has never once
# been given a reachable target.
#
# Two arms, because 2.9 is the COLLAPSE BOUNDARY and an operating point normally
# wants margin. Control is run_20260810_035045 (mean, rho*=0.06, N pinned 200,
# I=20, rho/cos=0.90, peak 0.801 @ 317 commits) -- already on disk, not re-run.
#
# PREDICTIONS, registered before launch:
#   arm 1  s=2.9: N_req 19.3 -> I=2,  N=20.  rho/cos = 2.85 (at the boundary)
#   arm 2  s=1.5: N_req 72.1 -> I=8,  N=80.  rho/cos = 1.42 (2x margin)
#   Commit rate ~ 1/I, discounted ~70% for the dearer round trip (P3 K-sweep):
#     expect ~7x and ~2.2x the control's commits per vclock-hour.
#   Progress per commit ~ sqrt(N): 0.32x and 0.63x of control.
#   Net progress per vclock-hour: ~2.2x and ~1.4x the control. BOTH should pass
#   the control's 0.801 inside the same wall clock. Score A (= sum rho*cos*||theta||,
#   the coordinate H-R settled), never Lambda across these -- p is fixed here, so
#   they agree, but A is what the ledger now carries.
#
# WHAT WOULD SINK THIS: arm 1 falling >0.015 from its peak. Then s=2.9 is a
# boundary and not an operating point, the gate ships at s~1.5, and C2's claim is
# "a correctly-sized gate", not "a gate at the criterion". If BOTH arms hold and
# beat the control per wall clock, G-1 is settled and gate_safety_s changes
# default from 0.4 to the surviving value.
#
# EARLY KILL: [CommitGate] must show n_req ~19 (arm 1) / ~72 (arm 2) and n_have
# rising OFF the cap. If I sits at 20 again, the flag did not land -- kill it.
set -u
. "$(dirname "$0")/_node_lib.sh"
A=(--only fluxtune --mode sim --yes --clean --force
   --server-update-audit --cos-ground-truth-audit
   --num-trainers 100 --num-gpus 8 --max-runtime-s 14400 --sim-wall-ceiling-h 2.0
   --agg-goal 10 --c 30
   --probe-combine mean --commit-gate n_target --gate-rho-ref setpoint
   --server-step-rule trust_ratio --rho-star 0.06 --rho-schedule rm --rho-exp 0.25)
node_run g1 "n_target s=2.9 (at the criterion)" "${A[@]}" --gate-safety-s 2.9
node_run g1 "n_target s=1.5 (2x margin)"        "${A[@]}" --gate-safety-s 1.5
echo "=== [g1] done $(date -Is) ==="
