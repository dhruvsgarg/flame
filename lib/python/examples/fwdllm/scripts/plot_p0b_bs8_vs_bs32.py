"""P0b A/B: FluxTune train batch 8 vs 32 (DistilBERT, Yahoo, N=90). Accuracy vs vclock and vs commits."""
import glob
import json
import os
import sys

import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
EXPTS = os.path.join(HERE, "..", "experiments")
R = {"bs 8": ("run_20261018_195537", "#2a78d6"), "bs 32": ("run_20261018_221609", "#eb6834")}
out = {}
for k, (p, c) in R.items():
    f = glob.glob(f"{EXPTS}/{p}*/telemetry/aggregator_*.jsonl")[0]
    vc = 0; commits = 0; phi = 1; rows = []
    for l in open(f):
        if '"vclock_now"' in l: vc = json.loads(l)["vclock_now"]
        if '"server_update"' in l: commits += 1; phi = json.loads(l).get("phi", phi)
        if '"agg_eval"' in l:
            r = json.loads(l); rows.append((vc / 3600, commits, r["test-accuracy"], phi, r["ts"]))
    a = np.array(rows); out[k] = (a, c)
    t0 = a[0, 4]
    sm = np.convolve(a[:, 2], np.ones(5) / 5, "valid")
    print(f"{k}: evals={len(a)} commits={int(a[-1,1])} sim={a[-1,0]:.2f}h real={(a[-1,4]-t0)/3600:.2f}h "
          f"final={a[-1,2]:.3f} last5={a[-5:,2].mean():.3f} peak={a[:,2].max():.3f} phi={a[-1,3]:.2f}")
    for thr in (0.5, 0.6, 0.65):
        j = np.argmax(a[:, 2] >= thr) if (a[:, 2] >= thr).any() else None
        print(f"   first acc>={thr}: " + (f"sim {a[j,0]:.2f}h, commit {int(a[j,1])}" if j is not None else "never"))
# matched comparison at bs32's current sim time
h = out["bs 32"][0][-1, 0]
for k, (a, c) in out.items():
    m = a[a[:, 0] <= h]; print(f"{k} @ sim<={h:.2f}h: last5 acc={m[-5:,2].mean():.3f} commits={int(m[-1,1])}")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
for k, (a, c) in out.items():
    for ax, x in ((ax1, a[:, 0]), (ax2, a[:, 1])):
        ax.plot(x, a[:, 2], color=c, lw=0.8, alpha=0.3)
        ax.plot(x[4:], np.convolve(a[:, 2], np.ones(5) / 5, "valid"), color=c, lw=2, label=k)
ax1.set(xlabel="Simulated wall clock (h)", ylabel="Test accuracy (Yahoo)"); ax2.set(xlabel="Commits")
for ax in (ax1, ax2): ax.grid(alpha=0.3); ax.legend(frameon=False)
fig.suptitle("P0b: FluxTune Yahoo N=100, train batch 8 vs 32 (bs 32 stopped at c705)")
fig.tight_layout(); fig.savefig(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "p0b_bs8_vs_bs32.png"), dpi=120)
