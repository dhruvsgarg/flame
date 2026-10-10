#!/usr/bin/env python3
"""PR29: confirm FX-D127-D131 + clip (FX-D126 on for cifar Oort) and measure same-code floors for every PR28 timing red.

Same runner as run_pr28.py (lanes, leases, probes, one Ctrl+C); floors pair only within this batch (parity_ladder, L18).
  run_pr29.py --plan                 # schedule simulation
  run_pr29.py [--hours 5] [--only C1,G1]
Claims and stop rules: PARITY_READINESS run queue PR29.
"""
import sys

import run_pr28 as rb

NON_OORT = rb.NON_OORT
rb.BATCH = "pr29"
rb.CLIP_STAGES = ("C1", "B1")  # cifar Oort legs: clip comes from baseline_reference.yaml now
rb.LANE_POOL_ARGS["cpu2"] = rb.LANE_POOL_ARGS["cpu"]
rb.LANE_CPU_CAP.update(cpu=56, cpu2=48)  # two CPU lanes share the node (PL12)
rb.STAGES = [
    # CPU lane: each T3 pair packed with its T3C real replicate (floors pair within this batch only).
    ("C1", "cpu", "--tier T3,T3C --datasets cifar10 --baselines oort,oort_star,feddance --traces 'syn_20 syn_50'", {}, "core",
     "FX-D129 + clip: cifar Oort syn_20s/50s real EV17 PASS, coverage ~1, 0 update_rejected; floors: Oort, feddance syn_50"),
    ("C2", "cpu2", "--tier T4 --datasets cifar10 --phases P7", {}, "core",
     "P7 cifar: felix/oort graded (PR28 cut), feddance replicate 2 (FX-N79), Oort with clip"),
    ("C3", "cpu2", "--tier T3,T3C --datasets google_speech --baselines oort,oort_star,refl --traces 'syn_20 syn_50'", {}, "core",
     "floors: speech oort_star syn_20s (11% vs 3.9% n=2), refl syn_50; Oort syn_50s (PR28 C4 OOM)"),
    # GPU lane A: G0U pairs packed with G0UC replicates for the non-Oort reds, then the G1AS sims (RAM-bound, last).
    ("G1", "gpuA", "--tier G0U,G0UC --datasets google_speech --baselines feddance --traces 'syn_50 mobiperf_3st'", {}, "core",
     "FX-D130: speech feddance syn_50 sim EV16 PASS; floors syn_50 + mobiperf"),
    ("G2", "gpuA", "--tier G0U,G0UC --datasets cifar10 --baselines fedbuff,feddance --traces 'syn_50 mobiperf_3st'", {}, "core",
     "floors: cifar fedbuff/feddance syn_50 + mobiperf (also the G0T syn_50 cohort)"),
    ("G3", "gpuA", "--tier G1AS --datasets google_speech --baselines feddance", {}, "core",
     "FX-D127/D128: speech feddance sim_rate > 1, EV12 PASS, one eval per eval round"),
    ("G4", "gpuA", "--tier G1AS --datasets cifar10 --baselines oort,feddance", {}, "core",
     "FX-D127 + clip: cifar feddance/oort sims reach budget (EV12/EV0 PASS); accuracy (~400 GB RAM: waits, FX-D131)"),
    # GPU lane B: cifar Oort with clip, then the red streaming cells.
    ("B1", "gpuB", "--tier G0U --datasets cifar10 --baselines oort,oort_star", {}, "core",
     "clip default: cifar Oort syn_50s + mobiperf_3sts G0U (PR28 B5 rest), 0 update_rejected"),
    ("B2", "gpuB", "--tier G0T --datasets cifar10 --baselines felix,fedbuff,feddance", {}, "core",
     "streaming: felix lin syn_0, fedbuff eve syn_50, feddance lin syn_50 (syn_50 vs G2 floors); EV19"),
]

if __name__ == "__main__":
    sys.exit(rb.main())
