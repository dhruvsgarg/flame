#!/usr/bin/env python3
"""PR31: confirm FX-D141 (harness stall, RAM profile) and FX-D142 (eval copy at init) plus re-profiled sim charges
(FX-D143) on every open red, run the 8 untested speech streaming cells, and re-screen green cells the new charges touch.

Same runner as run_pr28.py (lanes, leases, probes, one Ctrl+C); floors pair only within this batch (parity_ladder, L18).
  run_pr31.py --plan                 # schedule simulation
  run_pr31.py [--hours 12] [--only B1,A1]
Claims and stop rules: PARITY_READINESS run queue PR31.
"""
import sys

import run_pr28 as rb

rb.BATCH = "pr31"
rb.CLIP_STAGES = ("B1", "B2", "B4")  # cifar Oort legs: clip from baseline_reference.yaml
rb.LANE_POOL_ARGS["cpu2"] = rb.LANE_POOL_ARGS["cpu"]
rb.LANE_CPU_CAP.update(cpu=56, cpu2=48)  # two CPU lanes share the node (PL12)
rb.STAGES = [
    # GPU lane B: cifar Oort first (2 reds + FX-D142), then the OOM rerun, speech Oort streaming, floors.
    ("B1", "gpuB", "--tier G0T,G0TC --datasets cifar10 --baselines oort,oort_star", {}, "core",
     "FX-D142: cifar Oort G0T eve syn_0s + lin syn_50s within floor (eval copy no longer delays round 2 in real only)"),
    ("B2", "gpuB", "--tier G0U --datasets cifar10 --baselines oort_star --traces mobiperf_3st", {}, "core",
     "FX-D136/D141: cifar oort_star G0U mobiperf_3sts real EV0 PASS (PR29b B1 never started)"),
    ("B3", "gpuB", "--tier G0T,G0TC --datasets google_speech --baselines oort,oort_star", {}, "core",
     "FX-L63: speech Oort G0T_*s (untested); EV19 + INV/EXACT green or within G0TC floor"),
    ("B4", "gpuB", "--tier G0TC --datasets cifar10 --baselines oort --traces syn_50", {}, "core",
     "R7: 3rd real leg for cifar Oort G0T syn_50s trainers_at_n (stall placement, FX-L53)"),
    # GPU lane A: non-Oort streaming leftovers, then the G0U regression screen on the new GPU charges.
    ("A1", "gpuA", "--tier G0T,G0TC --datasets cifar10 --baselines fedbuff,feddance --traces syn_50", {}, "core",
     "FX-D137: cifar fedbuff G0T lin syn_50 within G0TC floor (PR29b B2 skipped); feddance lin/eve green"),
    ("A2", "gpuA", "--tier G0T,G0TC --datasets google_speech --baselines refl,feddance", {}, "core",
     "FX-N30: speech G0T refl/feddance (untested); EV19 + INV/EXACT green or within G0TC floor"),
    ("A3", "gpuA", "--tier G0TC --datasets google_speech --baselines fedbuff --traces syn_50", {}, "core",
     "speech fedbuff G0TC syn_50 floor legs (PR30 deadline-cut MISSING)"),
    ("A4", "gpuA", "--tier G0U --datasets all --baselines felix,fedbuff,refl,feddance --traces syn_50", {}, "stretch",
     "FX-D143 regression: G0U syn_50 non-Oort both datasets stay green on the re-profiled GPU charges"),
    # CPU lanes: T3 re-screen on the re-profiled stub charges (FX-D143), each pair with its T3C replicate.
    ("C1", "cpu", "--tier T3,T3C --datasets cifar10 --traces 'syn_0 syn_20 syn_50'", {}, "core",
     "FX-D143: cifar feddance T3 syn_50 within floor (stub dispatch +5 ms, download +3 ms/round); no green T3 cell turns red"),
    ("C2", "cpu2", "--tier T3,T3C --datasets google_speech --traces 'syn_0 syn_20 syn_50'", {}, "core",
     "FX-D143: speech T3 stays green; speech Oort syn_50s trainers_at_n graded on a 3-leg floor (R7)"),
]

if __name__ == "__main__":
    sys.exit(rb.main())
