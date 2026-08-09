#!/usr/bin/env bash
# Node 1 -- H-K: does data heterogeneity enter only through n_eff? (handoff sec 9.3, 15.13)
#
# No code needed. With the 08-07 alpha=1 arms this completes a 3x2 grid
# (alpha 1/10/100 x K 10/20). n_eff is replayable from default logs
# (`self.var`, `chosen jvp`), so these arms score S-K's sensor too.
#
# PREDICTIONS at commit ~10:
#   K=10: gate is cap-bound, so alpha moves nothing -- I ~ 18.5, N ~ 185, rho ~ 0.16
#   K=20: gate is live, so alpha moves the var floor -- alpha=100 commits soonest
#         (smallest I/N), alpha=1 latest
#   n_eff/N falls monotonically as alpha falls, at both K
#   rho*sqrt(n_eff) invariant across alpha  <- load-bearing; a miss falsifies H-K
#
# 4 arms x 4h vclock ~= 1.25h wall each ~= 5h total.
set -u

cd "$(dirname "$0")/.." || exit 1

A=(--only fluxtune --mode sim --yes --clean --force
   --server-update-audit --pool-split-half-audit
   --num-trainers 100 --num-gpus 8 --max-runtime-s 14400)

# alpha:K:c -- alpha=1 already covered by the 08-07 portfolio at both K.
for ARM in 10:10:30 100:10:30 10:20:40 100:20:40; do
  IFS=: read -r ALPHA K C <<< "$ARM"
  echo "=== [node1] alpha=$ALPHA K=$K c=$C  $(date -Is) ==="
  # Deliberately not `&&`: each arm is an independent probe, so one failure
  # must not cancel the rest (the 08-07 node-3 chain lost fwdllm_plus this way).
  ./run_sequential.sh "${A[@]}" \
      --partition-method "niid_label_clients=100_alpha=${ALPHA}" \
      --agg-goal "$K" --c "$C" \
    || echo "!!! [node1] arm alpha=$ALPHA K=$K FAILED (rc=$?); continuing"
done

echo "=== [node1] done $(date -Is) ==="
