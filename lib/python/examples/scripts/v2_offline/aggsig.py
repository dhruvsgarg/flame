"""C: which signals Felix already has predict the oracle's per-update weights? (random K=10 buffers at checkpoints)"""
import sys, random, torch, numpy as np, torch.nn.functional as F
sys.path.insert(0, "/home/dgarg39/flame/lib/python/examples/scripts/v2_offline")
import oracle_probe as op
from flame.optimizer.bn_buffers import is_bn_stat
pr = op.Probe(sys.argv[1], sys.argv[2]); feats = {}; W = []
def flat(d): return torch.cat([d[k].flatten().float() for k in sorted(d) if not is_bn_stat(k)])
for h in (1.5, 4, 8):
    ts, base, _, base_sd, rows = None, None, None, None, None
    ts, L0, A0, base_sd, rows = pr.at(h)
    rng = random.Random(int(h))
    for _ in range(8):
        buf = rng.sample(rows, 10)
        o = pr.agg(base_sd, buf, steps=60)
        w = np.array(o["oracle"][2]); W += list(w / max(w.mean(), 1e-9))
        vs = [flat(r["delta"]) for r in buf]; mean = sum(vs) / len(vs)
        for i, r in enumerate(buf):
            loo = (mean * len(vs) - vs[i]) / (len(vs) - 1)
            f = dict(n_samples=r["vis"], q_pre=r["q_pre64"], q_post=r["q_post"], q_drop=r["q_pre"] - r["q_post"], norm=float(vs[i].norm()),
                     cos_loo=float(F.cosine_similarity(vs[i], loo, dim=0)), single_value=r["v"])
            for k, x in f.items(): feats.setdefault(k, []).append(x)
W = np.array(W)
print(f"{len(W)} updates; oracle weight / buffer mean: p10 {np.percentile(W,10):.2f} median {np.median(W):.2f} p90 {np.percentile(W,90):.2f}; share ~0 (<0.1) {np.mean(W < 0.1):.2f}")
for k, x in feats.items():
    print(f"  spearman(oracle w, {k:12}) = {op.spearman(np.array(x), W):+.2f}")
