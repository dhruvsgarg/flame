"""Validate the probe: replay the run's real commits from a checkpoint (same clients, order, recorded staleness weights)."""
import sys, glob, json, copy, torch
sys.path.insert(0, "/home/dgarg39/flame/lib/python/examples/scripts/v2_offline")
import oracle_probe as op
from flame.optimizer.fedbuff import FedBuff
run, r0, nver, gpu = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
pr = op.Probe(run, gpu)
fb = FedBuff.__new__(FedBuff)
commits = {}
for line in open(glob.glob(run + "/telemetry/aggregator_*.jsonl")[0], "rb"):
    if b'"agg_round"' in line:
        e = json.loads(line)
        if r0 < e["round"] <= r0 + nver:
            for t, s, u in zip(e["contributing_trainers"], e["staleness"], e.get("stat_utility") or [0.0] * len(e["contributing_trainers"])):
                commits.setdefault(e["round"], []).append((t, s, u, float(e["vclock_now"])))
sd = torch.load(f"{run}/checkpoints/round_{r0:05d}.pt", map_location=pr.dev, weights_only=False)["state_dict"]
m = pr.b.model; m.load_state_dict(sd)
byid = {c.tid: c for c in pr.b.cl}
for random_sel in (False, True):
    m.load_state_dict(sd)
    import random; rng = random.Random(0)
    for v in range(r0 + 1, r0 + nver + 1):
        base = {k: x.clone() for k, x in m.state_dict().items()}
        acc = {k: torch.zeros_like(x, dtype=torch.float32) for k, x in base.items()}
        for t, s, u, ts in commits.get(v, []):
            c = byid[t] if not random_sel else rng.choice(pr.b.cl)
            idx = pr.train_idx(c, ts); c.idx, c.seen = idx, 0
            delta, *_ = pr.b.local_train(m, c, len(idx))
            w = fb.weight_factor(scale=0.4, staleness=s, a_exp=0.25, loss=u, b_exp=0.1, alpha_type="polynomial", beta_type="polynomial_upshift")
            for k in acc: acc[k] += w * delta[k].float()
        m.load_state_dict({k: base[k] + acc[k] / 10 for k in base})
        if (v - r0) % 10 == 0:
            print("random" if random_sel else "real  ", v, [round(x, 3) for x in pr.loss_acc(m, pr.B)], flush=True)
real = torch.load(f"{run}/checkpoints/round_{r0+nver:05d}.pt", map_location=pr.dev, weights_only=False)["state_dict"]
m.load_state_dict(real); print("run ckpt", r0 + nver, pr.loss_acc(m, pr.B))
