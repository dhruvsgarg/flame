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
rb.GPU_IDS, rb.NODE_GPUS = "0,1,2,3", 4  # shepherd, first 4 GPUs (operator 10-12; PL16)
rb.CLIP_STAGES = ("B1", "B2", "B4")  # cifar Oort legs: clip from baseline_reference.yaml
rb.LANE_POOL_ARGS["cpu2"] = rb.LANE_POOL_ARGS["cpu"]
rb.LANE_CPU_CAP.update(cpu=56, cpu2=48)  # two CPU lanes share the node (PL12)
rb.STAGES = [
    # ~5 h on 4 GPUs: reds first, pairs only; streaming on syn_50 unless a syn_0 cell is red (C19); replicates only for floors (R7).
    ("B1", "gpuB", "--tier G0T --datasets cifar10 --baselines oort,oort_star", {}, "core",
     "FX-D142: cifar Oort G0T eve syn_0s/syn_50s + oort_star lin syn_50s green (eval copy no longer delays round 2 in real only)"),
    ("B2", "gpuB", "--tier G0U --datasets cifar10 --baselines oort_star --traces mobiperf_3st", {}, "core",
     "FX-D136/D141: cifar oort_star G0U mobiperf_3sts real EV0 PASS (PR29 B1 OOM)"),
    ("B4", "gpuB", "--tier G0TC --datasets cifar10 --baselines oort --traces syn_50", {}, "core",
     "R7: 3rd real leg for cifar Oort G0T syn_50s trainers_at_n (stall placement, FX-L53)"),
    ("B3", "gpuB", "--tier G0T --datasets google_speech --baselines oort,oort_star --traces syn_50", {}, "core",
     "FX-L63: speech Oort G0T_*s syn_50 (untested); EV19 + INV/EXACT green"),
    ("A1", "gpuA", "--tier G0T --datasets cifar10 --baselines fedbuff,feddance --traces syn_50", {}, "core",
     "FX-D137: cifar fedbuff G0T lin syn_50 green; C20: feddance G0T syn_50 on FX-D148/D150 green"),
    ("A5", "gpuA", "--tier G0U --datasets all --baselines feddance", {}, "core",
     "C20: feddance G0U syn_50 + mobiperf both datasets green on FX-D148/D150 (cifar syn_50 terminal_state was red)"),
    ("A2", "gpuA", "--tier G0T --datasets google_speech --baselines refl,feddance --traces syn_50", {}, "core",
     "FX-N30 + C20: speech G0T syn_50 refl (untested), feddance on FX-D148/D150; EV19 + INV/EXACT green"),
    # CPU lanes: T3 re-screen on FX-D143 charges (syn_0 greens also check jayne charges on shepherd) + feddance (C20).
    ("C1", "cpu", "--tier T3 --datasets cifar10 --traces 'syn_0 syn_20 syn_50'", {}, "core",
     "FX-D143: no green cifar T3 cell turns red (syn_0: jayne charges hold on shepherd); C20: feddance T3 green"),
    ("C3", "cpu", "--tier T3C --datasets cifar10 --baselines feddance --traces syn_50", {}, "core",
     "R7: cifar feddance T3 syn_50 terminal_state on a same-batch floor"),
    ("C5", "cpu", "--tier T3 --datasets all --baselines feddance --traces mobiperf_3st", {}, "core",
     "C20: feddance T3 mobiperf both datasets green"),
    ("C6", "cpu", "--tier T4 --datasets cifar10 --baselines feddance --phases P7", {}, "core",
     "C20: cifar feddance P7 (stream_cpu) green"),
    ("C2", "cpu2", "--tier T3 --datasets google_speech --traces 'syn_0 syn_20 syn_50'", {}, "core",
     "FX-D143: no green speech T3 cell turns red; C20: feddance T3 green"),
    ("C4", "cpu2", "--tier T3C --datasets google_speech --baselines oort --traces syn_50", {}, "core",
     "R7: speech Oort T3 syn_50s trainers_at_n graded on a 3-leg floor"),
]

if __name__ == "__main__":
    sys.exit(rb.main())
