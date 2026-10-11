#!/usr/bin/env python3
"""FedDance sprint (FELIX_READINESS "FedDance sprint"): FedDance on the EuroSys'26 setup, every baseline on one config.

Same runner as run_pr28.py (lanes, leases, probes, one Ctrl+C). Three steps, run in order:
  run_fd.py --only S2,S3,S4 --hours 2  # daytime sanity (~2 h), then fd_sanity.py / eurosys_export.py on its legs
  run_fd.py --only F4,F3 --hours 3     # validate: convergence pair (real vs sim) + fidelity vs random
  run_fd.py --only F5C,F5SP --hours 18 # experiment sims, cifar traces one at a time (398 GB each)
  run_fd.py --plan [--only ...]        # schedule simulation
"""
import sys

import run_pr28 as rb

rb.BATCH = "fd"
rb.CLIP_STAGES = ()
rb.STAGES = [
    # Step S (daytime sanity, ~2 h): each proves one thing cheaply before the long legs.
    ("S2", "gpuA", "--tier G1XS --datasets cifar10 --baselines feddance", {}, "core",
     "S2: shared config at n=50 real+sim (15 min): fd_sanity per-run checks PASS, pair losses equal, INV/EXACT green"),
    ("S3", "gpuA", "--tier G1XQ --datasets cifar10 --baselines feddance", {}, "core",
     "S3: n=300 sim mobiperf 2 h vclock: sim speed (vclock h per wall h) and an accuracy slope for the F5 budget"),
    ("S4", "gpuB", "--tier G1XFS --datasets cifar10 --baselines feddance,fedavg", {}, "core",
     "S4: paper config (LS=1) n=50 3000 s: feddance accuracy-per-round >= random (fedavg) early (F3 direction)"),
    # Step A (validate). F4 holds the node's RAM (n=300 real); F3's n=50 sims fit beside it.
    ("F4", "gpuA", "--tier G1XP --datasets cifar10 --baselines feddance", {}, "core",
     "F4: cifar feddance real vs sim accuracy curve on the shared config (60 min); sim within real<->real timing floor, same evals"),
    ("F3", "gpuB", "--tier G1XF --datasets cifar10 --baselines feddance,fedavg", {}, "core",
     "F3: FedDance paper config (LS=1), mobiperf, n=50: feddance reaches 30% / 40% in fewer rounds than random (fedavg)"),
    # Step B (experiment). One pool runs the four cifar traces back to back (398 GB each); speech after (stretch: ~1x sim speed).
    ("F5C", "gpuA", "--tier G1X --datasets cifar10 --baselines feddance", {}, "core",
     "F5: cifar feddance time-to-70% + comm on mobiperf, syn_0, syn_10, syn_50 (figures + e2e table row)"),
    ("F5SP", "gpuA", "--tier G1X --datasets google_speech --baselines feddance --traces mobiperf_3st", {}, "stretch",
     "F5: speech feddance MobiPerf time-to-60% + comm (e2e table row)"),
]

F5_EXPECTED_S = 4 * 3600  # per G1X leg: runs stop at target; the 20 h horizon / 1.6 estimate would skip every later stage
_stage_jobs = rb.stage_jobs


def stage_jobs(args):
    jobs = _stage_jobs(args)
    for j in jobs:
        if "G1X_" in j.jid:  # not G1XQ/G1XS: those have short horizons
            j.est_s = min(j.est_s, F5_EXPECTED_S)
    return jobs


rb.stage_jobs = stage_jobs

if __name__ == "__main__":
    sys.exit(rb.main())
