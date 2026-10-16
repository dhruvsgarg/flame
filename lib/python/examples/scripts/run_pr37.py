#!/usr/bin/env python3
"""PR37: 2 h real<->sim convergence pairs, all six baselines at one n:c:K per dataset (harness_pool G3_SHAPE: cifar 75:15:10,
speech 50:15:10) x {syn_0, syn_50_3st, stream_lin syn_0} x both datasets, plus a real replicate (G3C) of the syn_0 and lin cells
(band reused for syn_50_3st; operator 10-15).
Gate: PR36 76/76 on one code key. Decision rule: FELIX Next steps PR37.

  run_pr37.py --node jayne --plan        # GPUs 0,2-7: speech G3, cifar G3C
  run_pr37.py --node shepherd --plan     # GPUs 0-3: cifar G3, speech G3C
  run_pr37.py --node <n> [--hours 30] [--only A1]
Same runner as run_pr28.py. Score with readiness_score.py / accuracy_table.py on this batch's pools (copy shepherd's to jayne).
"""
import sys

import run_pr28 as rb

LEAN = {}  # G3 legs lease --gpu-cpus-per-trainer 0.1 (PL12: real legs use p95 ~3-5 cores)
NODES = {
    "jayne": ("0,2,3,4,5,6,7", 7, [
        ("A1", "gpuA", "--tier G3 --datasets google_speech", LEAN, "core",
         "speech G3 six x {syn_0, syn_50_3st, lin}: real<->sim accuracy at matched rounds within max(2 pp, real<->real)"),
        ("B2", "gpuB", "--tier G3C --datasets cifar10 --traces syn_0", LEAN, "core", "cifar real replicate per G3 syn_0 + lin cell (band)"),
    ]),
    "shepherd": ("0,1,2,3", 4, [
        ("S1", "gpuA", "--tier G3 --datasets cifar10", LEAN, "core",
         "cifar G3 six x {syn_0, syn_50_3st, lin}: real<->sim accuracy at matched rounds within max(2 pp, real<->real)"),
        ("S2", "gpuB", "--tier G3C --datasets google_speech --traces syn_0", LEAN, "core", "speech real replicate per G3 syn_0 + lin cell (band)"),
    ]),
}

if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--node" not in argv or argv.index("--node") + 1 >= len(argv) or argv[argv.index("--node") + 1] not in NODES:
        sys.exit(f"usage: run_pr37.py --node {{{','.join(NODES)}}} [run_pr28 args]")
    i = argv.index("--node")
    node = argv[i + 1]
    del argv[i:i + 2]
    sys.argv[1:] = argv
    rb.BATCH = f"pr37_{node}"
    rb.GPU_IDS, rb.NODE_GPUS, rb.STAGES = NODES[node]
    rb.CLIP_STAGES = ()
    rb.LANE_POOL_ARGS.update(gpuA="--gpu-cpus-per-trainer 0.1", gpuB="--gpu-cpus-per-trainer 0.1")
    sys.exit(rb.main())
