#!/usr/bin/env python3
"""PR29b: confirm FX-D136-D140 on PR29's open reds, re-screen every baseline whose sim path changed (FX-D137 fedbuff,
FX-D138 feddance/Oort/refl), and write the first resource ledgers (FX-D140).

Same runner as run_pr28.py (lanes, leases, probes, one Ctrl+C); floors pair only within this batch (parity_ladder, L18).
  run_pr29b.py --plan                 # schedule simulation
  run_pr29b.py [--hours 6] [--only C1,G1]
Claims and stop rules: PARITY_READINESS run queue PR29b.
"""
import sys

import run_pr28 as rb

rb.BATCH = "pr29b"
rb.CLIP_STAGES = ("B1",)  # cifar Oort legs: clip from baseline_reference.yaml
rb.LANE_POOL_ARGS["cpu2"] = rb.LANE_POOL_ARGS["cpu"]
rb.LANE_CPU_CAP.update(cpu=56, cpu2=48)  # two CPU lanes share the node (PL12)
rb.STAGES = [
    # CPU lanes: each T3 pair packed with its T3C real replicate.
    ("C1", "cpu", "--tier T3,T3C --datasets cifar10 --baselines fedbuff,feddance,oort --traces 'syn_50'", {}, "core",
     "FX-D137/D138: cifar fedbuff + feddance T3 syn_50 throughput within floor; Oort syn_50s no sim wait-K jump past a pick"),
    ("C2", "cpu2", "--tier T3,T3C --datasets google_speech --baselines oort,oort_star,feddance,fedbuff --traces 'syn_50'",
     {}, "core",
     "FX-D136: speech Oort syn_50s reals EV0/EV1/EV12 PASS (no OOM); FX-D137/D138 speech fedbuff/feddance syn_50 green"),
    # GPU lane A: the FX-D137/D138 GPU cells with same-batch real replicates.
    ("G1", "gpuA", "--tier G0U,G0UC --datasets cifar10 --baselines fedbuff,feddance --traces 'syn_50 mobiperf_3st'", {}, "core",
     "FX-D137: cifar fedbuff G0U mobiperf throughput within floor (was 14.6%); FX-D138: feddance no late sync commit"),
    ("G2", "gpuA", "--tier G0U,G0UC --datasets google_speech --baselines feddance --traces 'syn_50 mobiperf_3st'", {}, "core",
     "FX-D136: speech feddance G0U syn_50 graded (no DOOMED false positive); mobiperf on a same-batch floor"),
    # GPU lane B: the OOM rerun, then the streaming FX-D137/D138 cells.
    ("B1", "gpuB", "--tier G0U --datasets cifar10 --baselines oort_star --traces 'mobiperf_3st'", {}, "core",
     "FX-D136: cifar oort_star G0U mobiperf_3sts real EV0 PASS, no RAM_TIGHT/RAM_OVER, ledger peak <= estimate"),
    ("B2", "gpuB", "--tier G0T,G0TC --datasets cifar10 --baselines fedbuff,feddance --traces 'syn_50'", {}, "core",
     "FX-D137/D138: cifar fedbuff G0T lin syn_50 throughput within G0TC floor (was 8.5%); feddance lin/eve green"),
]

if __name__ == "__main__":
    sys.exit(rb.main())
