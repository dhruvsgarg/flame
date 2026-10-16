#!/usr/bin/env python3
"""PR34: the speech feddance syn_50 reds (G0U, G0T lin/eve) on FX-D159/D160 with n=3 real floors (T8), plus a cifar canary;
the one-code-state re-screen of every cell is the overnight run after 76/76 (C18).

Same runner as run_pr28.py (lanes, leases, probes, one Ctrl+C).
  run_pr34.py --plan                 # schedule simulation
  run_pr34.py [--hours 3] [--only A1,B1]
Claims and stop rules: FELIX_READINESS Scoreboard -> Expected after PR34.
"""
import sys

import run_pr28 as rb

rb.BATCH = "pr34"
rb.GPU_IDS, rb.NODE_GPUS = "0,2,3,4,5,6,7", 7  # jayne, GPU 1 out (S0)
rb.CLIP_STAGES = ()
rb.LANE_POOL_ARGS["gpuC"] = rb.LANE_POOL_ARGS["gpuA"]
SP = "--datasets google_speech --baselines feddance --traces syn_50"
rb.STAGES = [
    ("A1", "gpuA", f"--tier G0U,G0UC {SP}", {}, "core",
     "FX-D160 + n=3 floor: speech feddance G0U syn_50 green or floor-gated (trainers_at_n)"),
    ("B1", "gpuB", f"--tier G0T,G0TC {SP}", {}, "core",
     "FX-D160 + n=3 floors: speech feddance G0T lin/eve syn_50 green or floor-gated"),
    ("A2", "gpuC", f"--tier G0UC,G0TC {SP}", {}, "core",
     "3rd real per speech feddance syn_50 cell (same code state as A1/B1; floors take every replicate pair)"),
    ("C1", "gpuA", "--tier G0U --datasets cifar10 --baselines feddance,fedbuff --traces syn_50", {}, "core",
     "canary: FX-D159/D160 keep cifar feddance syn_50 green and fedbuff syn_50 green (K8 time within 8%)"),
]

if __name__ == "__main__":
    sys.exit(rb.main())
