#!/usr/bin/env bash
# G-2 -- annealed vs setpoint, re-tested as an A/B (P5.1, P3 "annealed vs setpoint").
#
# gate_rho_ref=annealed was shipped on a re-read of two OLD arms (013917/035045),
# not an A/B, and those arms used s=0.4 -- SUPERSEDED (P6: "the K>=30/>=51 cohort
# requirement" and "evaluate N_req at s=0.4" are both dead-end entries traced to
# this exact value). This node reruns the comparison at the CURRENT shipped
# s=1.5, matched vclock (not matched commits -- annealed and setpoint order
# differently under the two clocks, which is the whole question, P4.3).
#
# n_req = 450340*(0.06/1.5)^2/10 = 72.1 (both legs, s fixed) -> I=8, matching
# 145729's enactment exactly. Both legs get the SAME --max-runtime-s so the
# vclock match is exact by construction, not approximate.
#
# Budget (task 0.7's model): vclock=40000s (11.1h) -> projects ~1,246 commits,
# real wall ~6,281s vs an 8h ceiling -- large margin either side.
#
# NEVER run annealed at rho* <= 0.01 -- 220627's dead zone (N_req -> 0 by c20,
# I floored at 1, peak decayed 0.394 -> 0.274). rho*=0.06 only.
#
# READ: at matched vclock, does annealed still bank more commits and end higher
# (013917: 0.821 vs 035045's 0.801, at 2.3x the commits) -- or does setpoint at
# s=1.5 close that gap now that its pool is 14x smaller (n_req 72 vs the old
# 1013)? Score peak accuracy and B/Bmax, never a commit-indexed comparison.
set -u
. "$(dirname "$0")/_node_lib.sh"

A=(--only fluxtune --mode sim --yes --clean --force
   --server-update-audit --cos-ground-truth-audit --cos-probe-every 25
   --num-trainers 100 --num-gpus 8 --agg-goal 10 --c 30
   --probe-combine mean --commit-gate n_target
   --server-step-rule trust_ratio --rho-star 0.06 --rho-schedule rm --rho-exp 0.25
   --gate-safety-s 1.5
   --max-runtime-s 40000 --sim-wall-ceiling-h 8.0)

node_run g2 "gate_rho_ref=annealed (rm, s=1.5)" "${A[@]}" --gate-rho-ref annealed
node_run g2 "gate_rho_ref=setpoint (rm, s=1.5)" "${A[@]}" --gate-rho-ref setpoint

echo "=== [g2] done $(date -Is) ==="
