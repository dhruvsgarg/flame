#!/usr/bin/env python3
"""PR36: full re-screen on one code key (FX-D165/D166/D167 code): every non-convergence cell on syn_*_3st + mobiperf_3st, G1AS
six x both, and a third real for the PR35 draw cells. Two nodes, one runner each (same commit, PL18):

  run_pr36.py --node jayne --plan        # GPUs 0,2-7: G1AS, G0U/G0UC, G0TC, CPU tiers
  run_pr36.py --node shepherd --plan     # GPUs 0-3: G0T (streaming), cifar G0TC
  run_pr36.py --node <n> [--hours 12] [--only A1,C1]
Same runner as run_pr28.py (lanes, leases, probes, one Ctrl+C). Claims and stop rules: FELIX_READINESS Next steps PR36.
Score with readiness_score.py on this batch's pools only (copy shepherd's pools + runs to jayne first, FX-T48).
"""
import sys

import run_pr28 as rb

NODES = {
    "jayne": ("0,2,3,4,5,6,7", 7, [  # GPU 1 out (S0)
        ("A1", "gpuA", "--tier G1AS --datasets google_speech", {}, "core",
         "FX-N93.1 + FX-N98: speech G1AS six; felix/fedbuff >= 60%, felix sim >= 2x real (PR35 A1: 1.09x), overruns <= 2%"),
        ("A2", "gpuA", "--tier G1AS --datasets cifar10", {}, "core",
         "FX-N93.1: cifar G1AS six; felix/fedbuff/oort/oort_star >= 50%, no slower than PR35 A2"),
        ("B1", "gpuB", "--tier G0U --datasets all", {}, "core",
         "one code key: G0U syn_50_3st + mobiperf_3st six x both green (INV/EXACT + EV incl. EV21)"),
        ("B2", "gpuB", "--tier G0UC --datasets cifar10 --baselines oort_star --traces syn_50_3st", {}, "core",
         "L13: third real for cifar oort_star G0U syn_50_3sts (PR35 draw cell)"),
        ("D1", "gpuC", "--tier G0UC --datasets all", {}, "core", "R7: a real replicate per G0U cell (floors on this code key)"),
        ("B3", "gpuB", "--tier G0TC --datasets google_speech", {}, "core", "R7: a real replicate per speech G0T cell"),
        ("C1", "cpu", "--tier T3 --datasets all", {}, "core", "one code key: T3 syn_0/0b/20_3st/50_3st/mobiperf six x both green"),
        ("C2", "cpu2", "--tier T4 --datasets cifar10 --phases P7", {}, "core", "one code key: cifar P7 (stream_cpu) six green"),
        ("C3", "cpu", "--tier T3C --datasets all", {}, "core", "R7: a real replicate per T3 cell (floors on this code key)"),
        ("C5", "cpu", "--tier T3 --datasets all --baselines oort --traces syn_50_3st", {}, "core",
         "PR36-short rescreen: unaware oort T3 syn_50_3sts at full length (SHORT_KEEP; 240 s gave EV1)"),
        ("S1S", "gpuA", "--tier G0T --datasets google_speech --traces syn_50_3st", {}, "core",
         "PR36-short rescreen: speech G0T syn_50_3st (shepherd S1C/S1S0 run the rest of S1)"),
        ("C4", "cpu", "--tier T3C --datasets all --baselines oort,oort_star --traces syn_20_3st", {}, "core",
         "L13: third real for the T3 syn_20_3sts Oort draw cells (PR35: speech oort, cifar oort + oort_star)"),
    ]),
    "shepherd": ("0,1,2,3", 4, [
        ("S1", "gpuA", "--tier G0T --datasets all", {}, "core",
         "one code key: G0T lin/eve x syn_0/syn_50_3st six x both green (EV19 + INV/EXACT); jayne B3 runs speech G0TC"),
        ("S2", "gpuB", "--tier G0TC --datasets cifar10", {}, "core", "R7: a real replicate per cifar G0T cell"),
        ("S1C", "gpuA", "--tier G0T --datasets cifar10", {}, "core", "PR36-short rescreen: cifar G0T"),
        ("S1S0", "gpuB", "--tier G0T --datasets google_speech --traces syn_0", {}, "core", "PR36-short rescreen: speech G0T syn_0"),
    ]),
}

if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--node" not in argv or argv.index("--node") + 1 >= len(argv) or argv[argv.index("--node") + 1] not in NODES:
        sys.exit(f"usage: run_pr36.py --node {{{','.join(NODES)}}} [run_pr28 args]")
    i = argv.index("--node")
    node = argv[i + 1]
    del argv[i:i + 2]
    sys.argv[1:] = argv
    rb.BATCH = f"pr36_{node}"
    rb.GPU_IDS, rb.NODE_GPUS, rb.STAGES = NODES[node]
    rb.CLIP_STAGES = ()
    rb.LANE_POOL_ARGS["cpu2"] = rb.LANE_POOL_ARGS["cpu"]
    rb.LANE_POOL_ARGS["gpuC"] = rb.LANE_POOL_ARGS["gpuA"]  # GPU lanes lease 0.1 CPU/trainer (PL12)
    rb.LANE_CPU_CAP.update(cpu=76, cpu2=24)
    sys.exit(rb.main())
