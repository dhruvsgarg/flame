import sys, torch, torch.nn.functional as F, random
sys.path.insert(0, "/home/dgarg39/flame/lib/python/examples/scripts/v2_offline")
import oracle_probe as op
pr = op.Probe(sys.argv[1], sys.argv[2])
full = [p for c in pr.b.cl for j, p in enumerate(c.order) if j % 10 != 9]
for frac in (0.25, 0.5, 1.0):
    idx = [p for c in pr.b.cl for p in [q for j, q in enumerate(c.order) if j % 10 != 9][:max(1, int(frac * len(c.order) * 0.9))]]
    m = op.ub.fl_data.CifarNet().to(pr.dev)
    opt = torch.optim.SGD(m.parameters(), lr=0.04, momentum=0.9, weight_decay=5e-4); best = 0
    for ep in range(40):
        m.train(); random.shuffle(idx)
        for b in range(0, len(idx), 32):
            s = idx[b:b + 32]; opt.zero_grad(); F.nll_loss(m(pr.b.X[s]), pr.b.Y[s]).backward(); opt.step()
        best = max(best, pr.loss_acc(m, pr.B)[1])
    print(f"shard fraction {frac}: {len(idx)} samples, centralized best {best:.3f}", flush=True)
