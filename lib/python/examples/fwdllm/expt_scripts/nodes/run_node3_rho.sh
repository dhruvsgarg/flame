#!/usr/bin/env bash
# Node 3 -- S-A+S-B at the MEASURED setpoint (handoff §22.3b, §22.3d, §3.3)
#
# The step rule is not in question: rho tracked rho*_t to 8.7e-5 over 186 commits
# and the geometric prediction for a constant rho* landed to 0.013%. What is
# wrong is the number it was set to. At s=0.4 the stack is bounded and does not
# learn -- peak 0.488 at rho*=0.01, 0.379 at rho*=0.02 -- against 0.855 for the
# anchor and 0.864 for a `mean` arm sitting at rho = 0.086.
#
# Sorted by realised rho, the eleven 08-09 arms are monotone in both accuracy and
# ||theta||^2 growth shape, and the crossover from sub-linear to super-linear is
# at rho ~ 0.09-0.12. So walk the band the curve points at, not the one the
# derivation predicted.
#
# exp=0.25, not 0.55: at 0.55 the anneal took rho 17x below its own setpoint by
# commit 186, i.e. the arm spent its whole budget in the dead zone. Robbins-Monro
# needs exp > 0.5 ASYMPTOTICALLY; at T ~ 200 what matters is that sum(rho_t) is
# large enough to arrive. Trading formal convergence for finite-horizon progress
# is the deliberate choice here, and arm 3 is where it is riskiest.
#
# All arms: `mean` + N pinned at 200, so the ONLY difference is rho*.
#
# PREDICTIONS at commit ~150:
#   arm 1 (rho*=0.03): stable, ||theta_tr|| < 15, peak 0.80-0.84 (bracketing the
#         n2a2 pinned-mean arm, which reached 0.804 still climbing at rho=0.032)
#   arm 2 (rho*=0.06): THE ARM. peak >= 0.86, matching the free-gate n2a3 result,
#         with ||theta_tr|| under 20 instead of 48.6 and a sub-linear log-log slope.
#   arm 3 (rho*=0.09): at or just past the crossover. Expect the best early
#         accuracy and a log-log slope near 1.0 -- if it exceeds 1.2, 0.09 is over
#         the boundary under `mean` and the band closes at 0.06.
#
# WHAT WOULD SINK S-A+S-B: arm 3 diverging AND arm 1 staying under 0.6 peak. That
# would mean no setpoint both learns and holds, and the step rule is not a fix on
# its own -- it would need S-I (which is measured free, §22.3c) to widen the band.
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
   --var-threshold 0 --max-iter-per-data-id 20 --var-stopping-policy off
   --probe-combine mean
   --server-step-rule trust_ratio --rho-schedule rm --rho-exp 0.25)
node_run node3 "rm rho*=0.03 exp=0.25" "${A[@]}" --rho-star 0.03
node_run node3 "rm rho*=0.06 exp=0.25" "${A[@]}" --rho-star 0.06
node_run node3 "rm rho*=0.09 exp=0.25" "${A[@]}" --rho-star 0.09
echo "=== [node3] done $(date -Is) ==="
