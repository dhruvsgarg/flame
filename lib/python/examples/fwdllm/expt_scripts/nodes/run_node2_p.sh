#!/usr/bin/env bash
# Node 2 -- S-I / H-G: is p a gradient-QUALITY parameter? (handoff §11.5, §15.3)
#
# The only lever in the document never tested, the only one that improves
# stability at NEGATIVE cost, and the only out-of-sample test of the criterion:
# every other sweep moved n, this moves the other half of cos ~ sqrt(n/p).
#
# pre_classifier is one 768x768 layer holding 56.7% of p and trainable by
# HuggingFace default, not by design. adapters_only freezes it:
#   p 1,040,932 -> 450,340   cos x1.52   rho/cos x0.433
#
# PREDICTIONS at commit ~10:
#   arm 2 vs 1: rho/cos improves 2.31x, norm doubling stretches ~2.3x
#   arm 3 (no FD rescale) isolates the §2.4 confound: h*||v|| drops 10.2 -> 6.7,
#          so arm 3 minus arm 2 is the FD side effect, NOT the p effect
#   peak accuracy unchanged is the REAL test (H-G) -- if it drops, p is load
#          bearing for the task and S-I is not free
#
# 4 arms x 4h vclock ~= 1.25h wall each ~= 5h total.
set -u

cd "$(dirname "$0")/.." || exit 1

A=(--only fluxtune --mode sim --yes --clean --force
   --server-update-audit --pool-split-half-audit
   --num-trainers 100 --num-gpus 8 --max-runtime-s 14400)

# exported, not prefixed: the flag is read inside the spawned TRAINER processes,
# so it has to be in the exported environment the launcher inherits and passes on.
# Verify per arm with:  grep -m1 '\[FD\]' <run>/*trainers.log
run () {  # fd_flag, label, then launcher flags
  export FWDLLM_FD_SCALE_INVARIANT="$1"; local label="$2"; shift 2
  echo "=== [node2] $label  FD_SCALE_INVARIANT=$FWDLLM_FD_SCALE_INVARIANT  $(date -Is) ==="
  ./run_sequential.sh "${A[@]}" "$@" \
    || echo "!!! [node2] $label FAILED (rc=$?); continuing"
}

# 1. baseline at full p, K=10 -- replicate of the 08-07 anchor, the reference row.
run 0 "p=full K=10 (anchor)" --agg-goal 10 --c 30

# 2. THE arm: p cut 57%, FD held at h*sqrt(p)=10.203 so only p moves.
run 1 "p=450k K=10 +FDfix" --agg-goal 10 --c 30 --trainable-scope adapters_only

# 3. same cut WITHOUT the FD rescale -- isolates the §2.4 side effect.
run 0 "p=450k K=10 noFDfix" --agg-goal 10 --c 30 --trainable-scope adapters_only

# 4. p cut where the gate is live: does a smaller p change what the gate targets?
run 1 "p=450k K=20 +FDfix" --agg-goal 20 --c 40 --trainable-scope adapters_only

echo "=== [node2] done $(date -Is) ==="
