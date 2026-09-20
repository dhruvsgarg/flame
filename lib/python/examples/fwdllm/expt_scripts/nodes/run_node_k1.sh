#!/usr/bin/env bash
# K-1 -- is K a time lever at all, or does C carry the wall clock? (P5.1, P5.3 K-C)
#
# Every K/C measurement to date moved C WITH K (30/60/100, C/K = 3/2/2 -- task
# 0.6's replay), so it could only falsify arithmetic, never decide. K-1 is the
# first arm to move K while holding C FIXED at 30.
#
# TWO RIVAL PREDICTIONS, and this arm separates them:
#   Pooling model (P2 as written): tau ~ K^0.63 -> commits/vclock-h improves as
#     K^0.37 (x1.5 over 10->30), D falls with K -> D(30)/D(10) ~= 0.63.
#   Throughput model (K-C):        commit rate set by C/n_req -> FLAT in K
#     within +-15%, D flat too (staleness = C/n_req = 0.42 on all three).
#
# K=10/C=30 leg ALREADY EXISTS -- 145729 (G-1b arm 2) is exactly
# mean/trust_ratio/n_target/s=1.5/rho*=.06/stride=25/K=10/C=30 (this session's
# fl_fwd_ft_practice.md P4.2). Reuse it; only K=20 and K=30 need launching.
#
# s=1.5, NOT 013917's s=0.4 -- that value is superseded (P6: "the K>=30/>=51
# cohort requirement" was an artifact of it). n_req = p*(rho*/s)^2/G_rule =
# 450340*(0.06/1.5)^2/10 = 72.1 -- fixed across all three legs, so I = ceil(72.1/K)
# is the only thing that changes: 8 (K=10, already have it) / 4 (K=20) / 3 (K=30).
#
# Budgets sized with expts/wall_clock_preflight.py (task 0.7), then padded ~1.5x
# against the K=10 empirical shortfall (145729 landed at 65% of the model's
# projected commit count) so both legs clear the >=1000-commit / ~40-cos-fire
# floor with margin even if the model over-predicts again:
#   K=20  vclock=70000s (19.4h)  ceiling=10h  -> projects 2,818 commits (~1,832 @ 0.65x), real wall ~14,205s
#   K=30  vclock=45000s (12.5h)  ceiling=7h   -> projects 2,105 commits (~1,368 @ 0.65x), real wall ~10,610s
# Both comfortably inside their ceilings, so --dry-run's own preflight (0.7)
# should pass; it is still the FIRST thing this node does.
#
# KILL: n_req != 72 (+-1), rho != 0.06, or C != 30 on either arm. If commit
# rate is flat in K within +-15% -> K is not a time lever, the controller
# hill-climbs C (model 4.6c/5.2/5.5e get rewritten). If D also flat within
# +-20% -> staleness is free at these levels and dynamic_kc's k_max=15 is far
# too conservative.
set -u
. "$(dirname "$0")/_node_lib.sh"

echo "=== [k1] K=10/C=30 leg: REUSING 145729, not relaunching -- see header ==="

A=(--only fluxtune --mode sim --yes --clean --force
   --server-update-audit --cos-ground-truth-audit --cos-probe-every 25
   --num-trainers 100 --num-gpus 8 --c 30
   --probe-combine mean --commit-gate n_target --gate-rho-ref setpoint
   --server-step-rule trust_ratio --rho-star 0.06 --rho-schedule const
   --gate-safety-s 1.5)

node_run k1 "K=20 C=30 (fixed C, moved K)" "${A[@]}" \
  --agg-goal 20 --max-runtime-s 70000 --sim-wall-ceiling-h 10.0

node_run k1 "K=30 C=30 (fixed C, moved K)" "${A[@]}" \
  --agg-goal 30 --max-runtime-s 45000 --sim-wall-ceiling-h 7.0

echo "=== [k1] done $(date -Is) ==="
