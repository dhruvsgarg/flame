#!/usr/bin/env bash
# G-1b -- is s ~= 2.9 an OPERATING POINT, or only a boundary?
#
# G-1 (002208 / 022448) settled the gate's throughput claim and left its
# setpoint claim untouched. It enacted exactly -- n_req 19.3/72.1, I=2/8 off
# the cap -- and netted 2.13x/1.33x the control's progress per vclock-hour
# against a registered 2.2x/1.4x. What it never did was VISIT the boundary:
# gate_rho_ref=setpoint sizes the pool from rho*_0 while rho_schedule=rm
# anneals the step, so realised rho/cos fell 2.84 -> 0.95 over 79 commits and
# Phi extrapolates to 1.24, nowhere near B_max ~ 3.6-4.2.
#
# Three things change here, and none of them is the gate:
#   1. rho_schedule=const  -- hold rho/cos AT 2.85 for the whole run, which is
#      the only way the criterion is under test at all.
#   2. cos_probe_every=25  -- the audit fwd+bwds its 1024-sample reference every
#      commit at ~80 ms/sample = 83 s, against 1.6 s for the rest of the commit
#      path. Cost is LINEAR in the reference, so chunking buys nothing; a stride
#      does. D is read in 50-commit blocks, so k=25 costs no resolution.
#   3. horizon from the prediction, not the node -- ~3,600 commits at I=2.
#
# PREDICTIONS, registered before launch:
#   arm 1  s=2.9 const: rho/cos pinned at 2.85. B ~ 0.5*T*rho^2 = 6.5 at
#     T=3600, so Phi is unbounded -- this arm is EXPECTED to cross B_max and
#     degrade. The question is WHERE: if the criterion is right, it holds its
#     peak until Phi ~ 3.6-4.2, i.e. around commit 700-800, and degrades after.
#   arm 2  s=1.5 const: rho/cos = 1.42, half the budget rate; the same Phi
#     arrives at ~4x the commits, i.e. beyond this horizon. Should hold.
#   Both should pass the control's 0.801 well before they turn.
#
# WHAT WOULD SINK THIS: arm 1 degrading EARLY (Phi < 2.5 at the turn) means
# s=2.9 is not the boundary and the criterion is calibrated wrong. Arm 1 holding
# to Phi > 5 means s=2.9 is conservative and the portfolio's 16/16 was luck.
# Either way, score the turn against Phi -- never against commit count.
#
# READ THE PRECONDITION FIRST. "Falls >0.015 from its peak" is only meaningful
# once an arm has PEAKED >= 0.80 (model 4.2). G-1's condition fired on both arms
# at Phi=1.03 and Lambda<=0.083 and meant nothing. Score peak retention only
# after the arm has actually learned.
#
# EARLY KILL: [CommitGate] n_req ~19 (arm 1) / ~72 (arm 2), n_have off the cap;
# [ServerStep] rho must stay AT 0.06 and not decay -- if it anneals, const did
# not land and this is G-1 again. [COS_GROUND_TRUTH_AUDIT] must say "every 25
# commit(s)"; if it says 1, expect ~85 s/commit and a dead run.
set -u
. "$(dirname "$0")/_node_lib.sh"
A=(--only fluxtune --mode sim --yes --clean --force
   --server-update-audit --cos-ground-truth-audit --cos-probe-every 25
   --num-trainers 100 --num-gpus 8 --max-runtime-s 28800 --sim-wall-ceiling-h 6.0
   --agg-goal 10 --c 30
   --probe-combine mean --commit-gate n_target --gate-rho-ref setpoint
   --server-step-rule trust_ratio --rho-star 0.06 --rho-schedule const)
node_run g1b "n_target s=2.9 const (at the criterion)" "${A[@]}" --gate-safety-s 2.9
node_run g1b "n_target s=1.5 const (2x margin)"        "${A[@]}" --gate-safety-s 1.5
echo "=== [g1b] done $(date -Is) ==="
