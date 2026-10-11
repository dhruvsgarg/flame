#!/usr/bin/env python3
"""PR30: the untested streaming cells: speech G0T all six (replaces speech P7, FX-N86 c) and cifar Oort-family G0T at K=10
(FX-L63), each with a same-code G0TC real replicate for its floor.

Same runner as run_pr28.py (lanes, leases, probes, one Ctrl+C); floors pair only within this batch (parity_ladder, L18).
  run_pr30.py --plan                 # schedule simulation
  run_pr30.py [--hours 6] [--only A1,B1]
Claims and stop rules: PARITY_READINESS run queue PR30.
"""
import sys

import run_pr28 as rb

rb.BATCH = "pr30"
rb.CLIP_STAGES = ("B1",)  # cifar Oort legs: clip from baseline_reference.yaml
rb.STAGES = [
    # GPU lane A: speech non-Oort (2 GPUs a leg), pair + real replicate packed per stage.
    ("A1", "gpuA", "--tier G0T,G0TC --datasets google_speech --baselines felix,fedbuff", {}, "core",
     "FX-N30: speech G0T felix/fedbuff lin+eve x syn_0/50 on HEAD; EV19 + INV/EXACT green or within G0TC floor"),
    ("A2", "gpuA", "--tier G0T,G0TC --datasets google_speech --baselines refl,feddance", {}, "core",
     "FX-N30: speech G0T refl/feddance (never run); EV19 + INV/EXACT green or within G0TC floor"),
    # GPU lane B: Oort family at K=10 (FX-L63).
    ("B1", "gpuB", "--tier G0T,G0TC --datasets cifar10 --baselines oort,oort_star", {}, "core",
     "FX-L63: cifar Oort G0T_*s with clip; EV19, 0 update_rejected, INV/EXACT green or within G0TC floor"),
    ("B2", "gpuB", "--tier G0T,G0TC --datasets google_speech --baselines oort,oort_star", {}, "core",
     "FX-L63: speech Oort G0T_*s; EV19 + INV/EXACT green or within G0TC floor"),
]

if __name__ == "__main__":
    sys.exit(rb.main())
