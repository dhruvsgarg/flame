#!/usr/bin/env bash
# Node 4 -- p as a gradient-quality knob, via the ONLY p lever that exists
#   (handoff §11.5, and the 2026-08-09 correction to it)
#
# The old S-I arm (freeze pre_classifier) is dead: the trainer already replaces
# pre_classifier with nn.Sequential() before the probe is drawn, so production p
# has always been 450,340 and `trainable_scope=adapters_only` changes nothing.
# What is left is the adapter bottleneck -- adapters are 447,264 of the 450,340:
#
#   reduction_factor  16 -> 32 -> 64      p  450,340 -> 229,012 -> 118,348
#   cos ~ 1/sqrt(p)   1.00x   1.40x  1.95x        rho/cos  1.00x  0.51x  0.26x
#
# FWDLLM_FD_SCALE_INVARIANT=1 on every arm holds h*sqrt(p) at its production value
# 6.711, so the finite difference does not silently shrink with p (§2.4). On arm 1
# it is a no-op by construction (h stays 0.01), which is also its regression test.
#
# PREDICTIONS at commit ~10, against the raw-SGD anchor:
#   rho falls as sqrt(p/450340) * (13.33/||theta_tr||): 1.00 / ~0.71 / ~0.51
#   norm doubling time stretches ~2x per step of the ladder (rho/cos ∝ p)
#   PEAK ACCURACY IS THE REAL TEST (H-G). If 118k adapters learn agnews as well as
#   450k, p is free to spend on gradient quality and the §11.5 claim -- "for
#   forward gradients, PEFT rank is a gradient-quality parameter" -- is measured
#   rather than derived. If accuracy drops, the claim gets a capacity caveat.
set -u
. "$(dirname "$0")/_node_lib.sh"

export FWDLLM_FD_SCALE_INVARIANT=1     # read inside the spawned trainers

A=(--only fluxtune --mode sim --yes --clean --force
   --server-update-audit --pool-split-half-audit
   --num-trainers 100 --num-gpus 8 --max-runtime-s 14400 --agg-goal 10 --c 30)

node_run node4 "rf=16  p=450340 (anchor)" "${A[@]}" --adapter-reduction-factor 16
node_run node4 "rf=32  p=229012"          "${A[@]}" --adapter-reduction-factor 32
node_run node4 "rf=64  p=118348"          "${A[@]}" --adapter-reduction-factor 64

echo "=== [node4] done $(date -Is) ==="
