#!/usr/bin/env bash
# P-1 -- tau(P): a build prerequisite for 3.4, not an ablation.
#
# 3.4 needs a mid-run P change, which needs to know what P costs in wall clock
# first. P is a TRAINER-side override (perturbation_count) -- verify via
# "[probe_combine=mean] P=..." in the trainer log, never via aggregator_config.json
# alone (S-C's gate needs its own copy too, so both must agree; run_sequential.sh
# writes both, test_model_args_parity.py enforces it).
#
# Pool is PINNED (--var-threshold 0 --max-iter-per-data-id 20 --var-stopping-policy
# off) so every commit costs EXACTLY I=20 rounds regardless of the gate -- this
# isolates tau(P) (round-trip time) from N_req (which the gate would otherwise
# change nothing about here, since P doesn't enter N_req, but pinning keeps I
# identical across both P legs so the ONLY thing that can differ is per-round
# compute time). expts/wall_clock_preflight.py does NOT know about this pinning
# override (it always derives I from the gate's N_req/K, not from
# --max-iter-per-data-id) -- ignore its I/commits_projected numbers for this
# node; the budget below is hand-derived for the pinned I=20 case:
#   tau(K=10) ~= 4.45s (task 0.7's model) * I=20 = 89.1s vclock/commit
#   vclock=10800s (3h) -> ~121 commits -> 2,420 pinned rounds, real wall ~611s (10 min)
# Deliberately generous relative to that estimate -- this is cheap either way.
#
# KILL: report and STOP (do not let 3.4 proceed) if tau(30)/tau(10) >= 2.5 --
# compute-bound at that ratio, the pooling gain from higher P cancels.
set -u
. "$(dirname "$0")/_node_lib.sh"

A=(--only fluxtune --mode sim --yes --clean --force
   --server-update-audit --cos-ground-truth-audit --cos-probe-every 25
   --num-trainers 100 --num-gpus 8 --agg-goal 10 --c 30
   --probe-combine mean --commit-gate n_target --gate-rho-ref setpoint
   --server-step-rule trust_ratio --rho-star 0.06 --rho-schedule const --gate-safety-s 1.5
   --var-threshold 0 --max-iter-per-data-id 20 --var-stopping-policy off
   --max-runtime-s 10800 --sim-wall-ceiling-h 4.0)

node_run p1 "P=10 (mean, pinned pool)" "${A[@]}" --perturbation-count 10
node_run p1 "P=30 (mean, pinned pool)" "${A[@]}" --perturbation-count 30

echo "=== [p1] done $(date -Is) -- compare tau(P) from [ServerStep]/agg_round timing, not accuracy ==="
