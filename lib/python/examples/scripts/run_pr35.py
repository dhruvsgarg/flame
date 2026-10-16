#!/usr/bin/env python3
"""PR35 overnight: FX-N93 step 1 (G1AS accuracy sims, six x both datasets) plus the one-code-state re-screen of every
non-streaming cell (G0U/G0UC, T3/T3C, P7); PR36 re-screens G0T/G0TC on the same code key (harness_pool.code_key).

Same runner as run_pr28.py (lanes, leases, probes, one Ctrl+C).
  run_pr35.py --plan                 # schedule simulation
  run_pr35.py [--hours 12] [--only A1,C1]
Claims and stop rules: FELIX_READINESS Next steps PR35. Score with readiness_score.py on this batch's pools only.
"""
import sys

import run_pr28 as rb

rb.BATCH = "pr35"
rb.GPU_IDS, rb.NODE_GPUS = "0,2,3,4,5,6,7", 7  # jayne, GPU 1 out (S0)
rb.CLIP_STAGES = ()
rb.LANE_POOL_ARGS["cpu2"] = rb.LANE_POOL_ARGS["cpu"]
rb.LANE_CPU_CAP.update(cpu=76, cpu2=24)  # cpu fits an n=165 mobiperf_3sts leg (as PR31)
rb.STAGES = [
    # GPU lane A: FX-N93.1 accuracy sims (4 GPUs each), speech first (longer, its targets are the open question).
    ("A1", "gpuA", "--tier G1AS --datasets google_speech", {}, "core",
     "FX-N93.1: speech G1AS six; felix/fedbuff >= 60% (run 18: 72/88 min), sync four round-bound or rising"),
    ("A2", "gpuA", "--tier G1AS --datasets cifar10", {}, "core",
     "FX-N93.1: cifar G1AS six; felix/fedbuff/oort_star >= 50%; first uncut oort + feddance (PR28 G5 was cut)"),
    # GPU lane B: unavailability re-screen + real replicate (floors, L13 votes); G0T,G0TC = PR36 on the same code key.
    ("B1", "gpuB", "--tier G0U,G0UC --datasets all", {}, "core",
     "one code state: G0U syn_50 + mobiperf six x both green (INV/EXACT + EV)"),
    # CPU lanes: stub tiers.
    ("C1", "cpu", "--tier T3 --datasets all", {}, "core", "one code state: T3 syn_0/20/50/mobiperf six x both green"),
    ("C2", "cpu2", "--tier T4 --datasets cifar10 --phases P7", {}, "core", "one code state: cifar P7 (stream_cpu) six green"),
    ("C3", "cpu", "--tier T3C --datasets all", {}, "core", "R7: a real replicate per T3 cell (floors on this code key)"),
]

if __name__ == "__main__":
    sys.exit(rb.main())
