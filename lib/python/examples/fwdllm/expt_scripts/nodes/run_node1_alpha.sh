#!/usr/bin/env bash
# SUPERSEDED 2026-08-09 -- ran 08-08, scored in handoff §9.7. Kept as the record.
# Node 1 -- H-K: does data heterogeneity enter only through n_eff? (handoff sec 9.3, 15.13)
#
# No code needed. With the 08-07 alpha=1 arms this gives a 3x2 grid spanning
# 1000x in alpha (0.1 / 1 / 100 x K 10/20). alpha=100 is the homogeneous
# reference that explains the historical var_threshold 0.1 -> 0.3 move;
# alpha=0.1 is the maximum-heterogeneity stress point. n_eff is replayable from
# default logs (`self.var`, `chosen jvp`), so these arms score S-K's sensor too.
#
# PREDICTIONS at commit ~10:
#   K=10: gate is cap-bound, so alpha moves nothing -- I ~ 18.5, N ~ 185, rho ~ 0.16
#   K=20: gate is live, so alpha moves the var floor -- alpha=100 commits soonest
#         (smallest I/N), alpha=0.1 latest
#   n_eff/N falls monotonically as alpha falls, at both K
#   rho*sqrt(n_eff) invariant across alpha  <- load-bearing; a miss falsifies H-K
#
# Measured mean top-class share per client (4 classes, 0.25 = IID):
#   alpha=0.1 -> 0.966   alpha=1 -> 0.734   alpha=100 -> 0.311
# So alpha=0.1 clients are effectively single-class: read those arms as a stress
# bound, not as evidence about the controller (convention: alpha=1 is primary).
# That share is also the independent x-axis to regress n_eff/N against.
#
# 4 arms x 4h vclock ~= 1.25h wall each ~= 5h total.
set -u

cd "$(dirname "$0")/.." || exit 1

A=(--only fluxtune --mode sim --yes --clean --force
   --server-update-audit --pool-split-half-audit
   --num-trainers 100 --num-gpus 8 --max-runtime-s 14400)

# alpha:K:c -- alpha=1 already covered by the 08-07 portfolio at both K.
# Run order interleaves K so a truncated session still yields both K at one alpha.
for ARM in 0.1:10:30 0.1:20:40 100:10:30 100:20:40; do
  IFS=: read -r ALPHA K C <<< "$ARM"
  # alpha is NOT in the run dir name (dataset.dirichlet_alpha is cosmetic), so
  # this banner is how the tmux scrollback maps runs back to arms.
  echo "=== [node1] alpha=$ALPHA K=$K c=$C  $(date -Is) ==="
  # Deliberately not `&&`: each arm is an independent probe, so one failure
  # must not cancel the rest (the 08-07 node-3 chain lost fwdllm_plus this way).
  ./run_sequential.sh "${A[@]}" \
      --partition-method "niid_label_clients=100_alpha=${ALPHA}" \
      --agg-goal "$K" --c "$C" \
    || echo "!!! [node1] arm alpha=$ALPHA K=$K FAILED (rc=$?); continuing"
done

echo "=== [node1] done $(date -Is) ==="
