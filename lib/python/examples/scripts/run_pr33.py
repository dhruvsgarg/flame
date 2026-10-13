#!/usr/bin/env python3
"""PR33: confirm FX-D155-D158 + EV20 on the eight red cells with same-code floors (C19 minimum); the one-code-state re-screen of
every other cell the new code touches is PR34, launched only if PR33 holds (C18).

Same runner as run_pr28.py (lanes, leases, probes, one Ctrl+C).
  run_pr33.py --plan                 # schedule simulation
  run_pr33.py [--hours 10] [--only A1,B1]
Claims and stop rules: FELIX_READINESS Scoreboard -> Expected after PR33.
"""
import sys

import run_pr28 as rb

rb.BATCH = "pr33"
rb.GPU_IDS, rb.NODE_GPUS = "0,2,3,4,5,6,7", 7  # jayne, GPU 1 out (S0)
rb.CLIP_STAGES = ("A3",)  # cifar oort legs: clip from baseline_reference.yaml
rb.LANE_POOL_ARGS["gpuC"] = rb.LANE_POOL_ARGS["gpuA"]
rb.LANE_CPU_CAP.update(cpu=80)
rb.STAGES = [
    ("A1", "gpuA", "--tier G0U,G0UC --datasets all --baselines feddance", {}, "core",
     "FX-D155: cifar syn_50 EV20 PASS, cell green; FX-D158: speech syn_50 picks equal; mobiperf stays green"),
    ("A2", "gpuC", "--tier G0U,G0UC --datasets all --baselines felix,fedbuff", {}, "core",
     "FX-D157 + G0UC floor: fedbuff cifar syn_50 green or floor-gated; felix/fedbuff stay green on deterministic kernels"),
    ("B1", "gpuB", "--tier G0T,G0TC --datasets all --baselines feddance --traces syn_50", {}, "core",
     "FX-D156: cifar eve stale returns land before the next selection, trainers_at_n green; speech eve inside its floor"),
    ("A3", "gpuA", "--tier G0T,G0TC --datasets cifar10 --baselines oort --traces syn_50", {}, "core",
     "FX-D155 + EV20: cifar oort G0T syn_50s abandons on time, cells green"),
    ("C1", "cpu", "--tier T3,T3C --datasets all --baselines oort,oort_star,refl,feddance --traces 'syn_50 mobiperf_3st'", {},
     "core", "FX-D155/D156 on stub legs + EV20: T3 sync syn_50 + mobiperf green both datasets (oort speech syn_50s, cifar mobiperf_3sts)"),
]

if __name__ == "__main__":
    sys.exit(rb.main())
