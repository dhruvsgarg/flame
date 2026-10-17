#!/usr/bin/env python3
"""PR37: 2 h real<->sim convergence pairs, all six baselines at one n:c:K per dataset (harness_pool G3_SHAPE: cifar 75:15:10,
speech 50:15:10) x {syn_0, syn_50_3st, stream_lin syn_0} x both datasets. Pass 2 (operator 10-16): a real replicate (G3C) of the
syn_0 and lin cells, launched only after a clean pass 1 (band reused for syn_50_3st; operator 10-15).
Gate (operator 10-16): PR36 tail stages (T*) run first in the same queue. Decision rule: FELIX Next steps PR37.

  run_pr37.py --node jayne --plan        # pass 1, GPUs 0,2-7: PR36 tail TMP/TB1/TC5, speech G3 (not feddance)
  run_pr37.py --node shepherd --plan     # pass 1, GPUs 0-3: PR36 tail TDU/TDT, cifar G3, speech G3 feddance
  run_pr37.py --node jayne-rep|shepherd-rep    # pass 2 (G3C replicate reals) after a clean pass 1, same code key
  run_pr37.py --node jayne-s1b            # offload 2: cifar G3 felix syn_0
  run_pr37.py --node jayne-s1             # offload: cifar G3 syn_50_3st felix/fedbuff/feddance/refl (S1 tail)
  run_pr37.py --node <n> [--hours 30] [--only A1]
Same runner as run_pr28.py. Score with readiness_score.py / accuracy_table.py on this batch's pools (copy shepherd's to jayne).
"""
import sys

import run_pr28 as rb

LEAN = {}  # G3 legs lease --gpu-cpus-per-trainer 0.1 (PL12: real legs use p95 ~3-5 cores)
# PR36 tail (operator 10-16: no full PR36 rerun): cells PR36-short couldn't grade + draw cells with a replicate real, same code key.
TAIL_J = [
    ("TMP", "gpuC", "--tier G0U --datasets all --traces mobiperf_3st", {}, "core",
     "PR36 tail: G0U mobiperf_3st six x both at full length (240 s gave EV1; oort_star cifar lost to the abort)"),
    ("TB1", "gpuC", "--tier G0U --datasets google_speech --baselines felix,feddance,refl,oort_star --traces syn_50_3st", {}, "core",
     "PR36 tail: speech G0U syn_50_3st cells B1 lost to the FX-N33 abort"),
    ("TC5", "cpu", "--tier T3,T3C --datasets all --baselines oort --traces syn_50_3st", {}, "core",
     "PR36 tail: oort T3 syn_50_3sts pair + replicate real (draw class needs a floor)"),
]
TAIL_S = [
    ("TDU", "gpuC", "--tier G0U,G0UC --datasets cifar10 --baselines oort,fedbuff --traces syn_50_3st", {}, "core",
     "PR36 tail: cifar G0U oort syn_50_3sts / fedbuff syn_50_3st pair + replicate real (draw class)"),
    ("TDT", "gpuC", "--tier G0T,G0TC --datasets cifar10 --baselines oort --traces syn_50_3st", {}, "core",
     "PR36 tail: cifar G0T oort syn_50_3sts pair + replicate real (S1C draw)"),
]
G3_CLAIM = "real<->sim accuracy at matched rounds within 2 pp (pass 1); band from pass 2"
NODES = {  # pass 1: tail + one real + one sim per cell (operator 10-16)
    "jayne": ("0,2,3,4,5,6,7", 7, TAIL_J + [
        ("A1", "gpuA", "--tier G3 --datasets google_speech --baselines felix,fedbuff,refl,oort,oort_star", LEAN, "core", f"speech G3 six x {{syn_0, syn_50_3st, lin}}: {G3_CLAIM}"),
    ]),
    "shepherd": ("0,1,2,3", 4, TAIL_S + [
        ("S1", "gpuA", "--tier G3 --datasets cifar10", LEAN, "core", f"cifar G3 six x {{syn_0, syn_50_3st, lin}}: {G3_CLAIM}"),
        ("S2", "gpuB", "--tier G3 --datasets google_speech --baselines feddance", LEAN, "core", f"speech G3 feddance (rest on jayne A1): {G3_CLAIM}"),
    ]),
    # offload (operator 10-16): shepherd S1 runs the other cells first (units sort cell, baseline); jayne takes the syn_50_3st tail
    "jayne-s1": ("0,2,3,4,5,6,7", 7, [
        ("S1J", "gpuA", "--tier G3 --datasets cifar10 --baselines felix,fedbuff,feddance,refl --traces syn_50_3st", LEAN, "core",
         f"cifar G3 syn_50_3st felix/fedbuff/feddance/refl (S1's tail, moved from shepherd): {G3_CLAIM}"),
    ]),
    # offload 2 (operator 10-16): shepherd's last unstarted real (cifar felix syn_0) moves to idle jayne
    "jayne-s1b": ("0,2,3,4,5,6,7", 7, [
        ("S1K", "gpuA", "--tier G3 --datasets cifar10 --baselines felix --phases G3_syn_0", LEAN, "core",
         f"cifar G3 felix syn_0 (S1's last real, moved from shepherd): {G3_CLAIM}"),
    ]),
    # pass 2, only after pass 1 is clean on an unchanged code key: the real replicate (G3C) per syn_0 + lin cell (band)
    "jayne-rep": ("0,2,3,4,5,6,7", 7, [
        ("R1", "gpuA", "--tier G3C --datasets google_speech --traces syn_0", LEAN, "core", "speech real replicate per G3 syn_0 + lin cell"),
    ]),
    "shepherd-rep": ("0,1,2,3", 4, [
        ("R2", "gpuA", "--tier G3C --datasets cifar10 --traces syn_0", LEAN, "core", "cifar real replicate per G3 syn_0 + lin cell"),
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
    rb.BATCH = f"pr37_{node.replace('-', '_')}"
    rb.GPU_IDS, rb.NODE_GPUS, rb.STAGES = NODES[node]
    rb.CLIP_STAGES = ()
    rb.LANE_POOL_ARGS.update(gpuA="--gpu-cpus-per-trainer 0.1", gpuB="--gpu-cpus-per-trainer 0.1", gpuC="--gpu-cpus-per-trainer 0.1")
    sys.exit(rb.main())
