#!/usr/bin/env bash
# SUPERSEDED 2026-08-09 by run_node2_probe.sh -- a free var gate moves N by ~10x
# alongside b^2, so these arms could not have isolated S-H (handoff §22.2).
# Node 3 -- S-H: assimilate all P probes instead of selecting one (handoff §11.3, §15.2)
#
# The largest single lever in the document for zero extra compute, zero extra
# bytes, and no change to eta, N or p. Selecting on |d| raises aim and step
# length by the same sqrt(E), so b^2/a = 1 and stability is untouched; averaging
# gives b^2/a = 1/P, a 10x improvement in rho/cos.
#
# PREDICTIONS at commit 1 (minutes in, before any accuracy signal):
#   arm 2 vs 1: rho drops by sqrt(E*P) = 5.45x, ||G|| by the same factor
#               (E=2.988 measured on 37k real selection events; the synthetic
#                check in expt_scripts/test_probe_combine.py reproduces 5.451)
#   arm 3: P=30 under mean -- rho drops a further sqrt(3), and unlike the 08-07
#          P=30 arm under `select` it must now HELP rather than hurt. That
#          reversal is the cleanest falsification of the b^2/a story.
#   arm 4: mean at alpha=0.1 -- does the gain survive extreme heterogeneity?
#
# Scored per §17: ||theta_tr||^2 vs commit, ||dtheta||, rho*sqrt(N), n_eff.
# NOTE: P=30 triples the forward passes, so arm 3 costs ~2-3x the wall of the others.
set -u

cd "$(dirname "$0")/.." || exit 1

A=(--only fluxtune --mode sim --yes --clean --force
   --server-update-audit --pool-split-half-audit
   --num-trainers 100 --num-gpus 8 --max-runtime-s 14400)

run () {  # label, then flags
  local label="$1"; shift
  echo "=== [node3] $label  $(date -Is) ==="
  ./run_sequential.sh "${A[@]}" "$@" \
    || echo "!!! [node3] $label FAILED (rc=$?); continuing"
}

run "select K=10 (anchor)"  --agg-goal 10 --c 30 --probe-combine select
run "mean   K=10"           --agg-goal 10 --c 30 --probe-combine mean
run "mean   K=10 P=30"      --agg-goal 10 --c 30 --probe-combine mean --perturbation-count 30
run "mean   K=10 alpha=0.1" --agg-goal 10 --c 30 --probe-combine mean \
    --partition-method 'niid_label_clients=100_alpha=0.1'

echo "=== [node3] done $(date -Is) ==="
