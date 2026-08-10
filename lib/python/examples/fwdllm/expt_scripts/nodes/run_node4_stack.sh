#!/usr/bin/env bash
# SUPERSEDED 2026-08-09 by run_node3_step.sh -- same arms with the pool pinned, so
# the step rule is attributable. Its rho* sizing predates the p correction (§7).
# Node 4 -- S-A + S-B, and the portability test (handoff §15.4, §15.6, H-K)
#
# S-A makes rho an OPERATOR constant instead of an emergent one; S-B supplies the
# boundedness S-A alone does not (a constant rho* still grows ||theta|| as
# (1+rho*^2)^(T/2)). They ship together, and arm 1 vs 2 is what separates them.
#
# Sizing: measured cos <= 0.015 with s ~ 0.3-0.5 puts rho* at 0.005-0.0075 under
# `select`; under `mean` cos ~ 0.027 so rho* ~ 0.01. Arms straddle that boundary.
#
# PREDICTIONS at commit ~10:
#   EVERY arm: logged rho EQUALS rho*_t to 4 digits, every commit. That is the
#              enactment check -- if it does not, the step rule did not take.
#   arm 1 (const):  rho flat at 0.01; ||theta_tr||^2 still super-linear => S-A
#                   alone is NOT sufficient, which is the point of running it
#   arm 2 (rm):     rho decays as t^-0.55; ||theta_tr||^2 grows SUB-linearly
#   arm 3 (rm, low rho*): under budget -- stable but visibly slower to climb
#   arm 4: full stack (mean + trust_ratio + rm) at alpha=0.1 with the SAME rho*
#          as arm 2. THE money arm: if it holds accuracy at 1000x the
#          heterogeneity with no knob changed, the setpoint is portable (H-K).
#
# 4 arms x 4h vclock ~= 1.25h wall each ~= 5h total.
set -u

cd "$(dirname "$0")/.." || exit 1

A=(--only fluxtune --mode sim --yes --clean --force
   --server-update-audit --pool-split-half-audit
   --num-trainers 100 --num-gpus 8 --max-runtime-s 14400
   --agg-goal 10 --c 30)

run () {  # label, then flags
  local label="$1"; shift
  echo "=== [node4] $label  $(date -Is) ==="
  ./run_sequential.sh "${A[@]}" "$@" \
    || echo "!!! [node4] $label FAILED (rc=$?); continuing"
}

run "S-A only: trust_ratio const rho*=0.01" \
    --server-step-rule trust_ratio --rho-star 0.01 --rho-schedule const

run "S-A+S-B: rho*=0.02 t^-0.55" \
    --server-step-rule trust_ratio --rho-star 0.02 --rho-schedule rm --rho-exp 0.55

run "S-A+S-B: rho*=0.005 t^-0.55 (under budget)" \
    --server-step-rule trust_ratio --rho-star 0.005 --rho-schedule rm --rho-exp 0.55

run "FULL STACK @ alpha=0.1, same rho* as arm 2" \
    --probe-combine mean \
    --server-step-rule trust_ratio --rho-star 0.02 --rho-schedule rm --rho-exp 0.55 \
    --partition-method 'niid_label_clients=100_alpha=0.1'

echo "=== [node4] done $(date -Is) ==="
