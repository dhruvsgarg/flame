#!/usr/bin/env python3
"""FedDance sprint (FELIX_READINESS "FedDance sprint"): FedDance on the EuroSys'26 setup, every baseline on one config.

Same runner as run_pr28.py (lanes, leases, probes, one Ctrl+C). Three steps, run in order:
  run_fd.py --only S2,S3,S4 --hours 2  # daytime sanity (~2 h), then fd_sanity.py / eurosys_export.py on its legs
  run_fd.py --only V1,V2,V3,F3 --hours 5  # short validation + fidelity vs random
  run_fd.py --only V4,F3 --hours 1.5   # confirm FX-D150/D151 + fidelity vs random
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
     "S4: paper config (LS=1) n=300 K=10 mobiperf 3000 s: feddance accuracy-per-round >= random (fedavg) early (F3 direction)"),
    # Step V (short validation, FX-D148/D149): unavailability + streaming parity, shared-config real==sim, speech sim speed.
    ("V1", "gpuA", "--tier G0U,G0T --datasets all --baselines feddance", {}, "core",
     "V1: feddance G0U (syn_50, mobiperf) + G0T (lin/eve x syn_0/syn_50) both datasets: EV + INV/EXACT green; fd_sanity per leg"),
    ("V2", "gpuB", "--tier G1XS --datasets all --baselines feddance", {}, "core",
     "V2: shared config n=50 real+sim syn_50 + syn_10 both datasets: fd_sanity --pair losses identical (FX-D149), INV/EXACT green"),
    ("V3", "gpuB", "--tier G1XQ --datasets google_speech --baselines feddance", {}, "core",
     "V3: speech n=100 sim mobiperf 30 min vclock: sim speed decides F5SP (needs >= 3x to fit 20 h vclock in ~7 h)"),
    ("V4", "gpuA", "--tier G0U --datasets cifar10 --baselines feddance", {}, "core",
     "V4: cifar feddance G0U syn_50 + mobiperf on FX-D150/D151: U6 green (real lag on absolute send), EV + INV/EXACT green"),
    # Step A (validate). n=300 cifar legs (398 GB each) run one at a time.
    ("F4", "gpuA", "--tier G1XP --datasets cifar10 --baselines feddance", {}, "core",
     "F4: cifar feddance real vs sim accuracy curve on the shared config (60 min); sim within real<->real timing floor, same evals"),
    ("F3", "gpuB", "--tier G1XF --datasets cifar10 --baselines feddance,fedavg", {}, "core",
     "F3: FedDance paper config (LS=1), n=300 K=10, mobiperf + syn_0, 12000 s: feddance reaches 30% / 40% in fewer rounds than random (fedavg)"),
    # Step B (experiment). One pool runs the four cifar traces back to back (398 GB each); speech after (stretch: ~1x sim speed).
    ("F5C", "gpuA", "--tier G1X --datasets cifar10 --baselines feddance", {}, "core",
     "F5: cifar feddance time-to-70% + comm on mobiperf, syn_0, syn_10, syn_50 (figures + e2e table row)"),
    ("F5SP", "gpuA", "--tier G1X --datasets google_speech --baselines feddance --traces mobiperf_3st", {}, "stretch",
     "F5: speech feddance MobiPerf time-to-60% + comm (e2e table row)"),
]

F5_EXPECTED_S = {"cifar10": 80 * 60, "google_speech": 135 * 60}  # 20 h vclock at measured 18x (F3) / 11.2x (V3), +20%
F3_EXPECTED_S = 30 * 60  # per G1XF leg: n=300 cifar sims run >= 18x (S3)
_stage_jobs = rb.stage_jobs


def stage_jobs(args):
    jobs = _stage_jobs(args)
    for j in jobs:
        if "G1X_" in j.jid:  # not G1XQ/G1XS: those have short horizons
            j.est_s = min(j.est_s, F5_EXPECTED_S["google_speech" if j.jid.startswith("gs_") else "cifar10"])
        elif "G1XF_" in j.jid:
            j.est_s = min(j.est_s, F3_EXPECTED_S)
    return jobs


rb.stage_jobs = stage_jobs

if __name__ == "__main__":
    sys.exit(rb.main())
