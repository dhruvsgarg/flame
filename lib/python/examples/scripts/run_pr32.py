#!/usr/bin/env python3
"""PR32: the PR31 cells that never ran (A4/A5 gate aborts, fixed in run_gate) plus a same-code real floor for each
remaining PR31 red; every red is a pair rerun beside its real replicate, since floors pair only within one code state (FX-D132).

Same runner as run_pr28.py (lanes, leases, probes, one Ctrl+C).
  run_pr32.py --plan                 # schedule simulation
  run_pr32.py [--hours 6] [--only A4,F1]
Claims and stop rules: FELIX_READINESS Scoreboard -> Expected after PR32.
"""
import sys

import run_pr28 as rb

rb.BATCH = "pr32"
rb.GPU_IDS, rb.NODE_GPUS = "0,2,3,4,5,6,7", 7  # jayne, GPU 1 out (S0, operator 10-13)
rb.CLIP_STAGES = ("F2",)  # cifar oort_star G0T legs: clip from baseline_reference.yaml
rb.LANE_POOL_ARGS["cpu2"] = rb.LANE_POOL_ARGS["cpu"]
rb.LANE_CPU_CAP.update(cpu=80, cpu2=48)
rb.STAGES = [
    ("A4", "gpuA", "--tier G0U,G0UC --datasets all --baselines felix,fedbuff", {}, "core",
     "Regression: felix/fedbuff G0U syn_50 + mobiperf stay green (last graded PR28); G0UC floors the chaotic cells (FX-L53)"),
    ("A5", "gpuA", "--tier G0U,G0UC --datasets all --baselines feddance", {}, "core",
     "C20: feddance G0U syn_50 + mobiperf both datasets green on FX-D148/D150, floored by G0UC"),
    ("F1", "gpuB", "--tier G0T,G0TC --datasets cifar10 --baselines fedbuff,feddance --traces syn_50", {}, "core",
     "FX-L66: cifar fedbuff G0T lin/eve throughput + terminal_state and feddance eve trainers_at_n inside their real floor"),
    ("F2", "gpuB", "--tier G0T,G0TC --datasets cifar10 --baselines oort_star --traces syn_50", {}, "core",
     "FX-L53: cifar oort_star G0T lin syn_50s trainers_at_n inside its real floor"),
    ("F3", "gpuA", "--tier G0T,G0TC --datasets google_speech --baselines feddance --traces syn_50", {}, "core",
     "FX-L53: speech feddance G0T eve syn_50 trainers_at_n inside its real floor"),
    ("C1", "cpu", "--tier T3,T3C --datasets cifar10 --baselines oort_star --traces syn_20", {}, "core",
     "FX-L53: cifar oort_star T3 syn_20s trainers_at_n (37 vs 35) inside its real floor"),
    ("C2", "cpu2", "--tier T3,T3C --datasets google_speech --baselines oort --traces syn_20", {}, "core",
     "FX-N85 d: speech oort T3 syn_20s throughput inside its real floor at 600 s (300 s gave 10 matched rounds)"),
]

if __name__ == "__main__":
    sys.exit(rb.main())
