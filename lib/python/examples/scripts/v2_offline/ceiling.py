"""Centralized ceiling: train the leg's model on the union of all clients' train-visible data at t, report test accuracy."""
import sys, torch, torch.nn.functional as F, random
sys.path.insert(0, "/home/dgarg39/flame/lib/python/examples/scripts/v2_offline")
import oracle_probe as op
run, gpu = sys.argv[1], sys.argv[2]
pr = op.Probe(run, gpu)
for h in (0.1, 6, 12):
    idx = [j for c in pr.b.cl for j in pr.train_idx(c, h * 3600)]
    m = (op.ub.fl_data.CifarNet() if pr.ds == "cifar10" else op.ub.fl_data.ResNet34_1D()).to(pr.dev)
    opt = torch.optim.SGD(m.parameters(), lr=0.04, momentum=0.9, weight_decay=5e-4)
    best = 0
    for ep in range(40 if pr.ds == "cifar10" else 15):
        m.train(); random.shuffle(idx)
        for b in range(0, len(idx), 32):
            s = idx[b:b + 32]; opt.zero_grad()
            F.nll_loss(pr.b.fwd(m, pr.b.X[s]), pr.b.Y[s]).backward(); opt.step()
        best = max(best, pr.loss_acc(m, pr.B)[1])
    print(f"{pr.ds} t {h} h: {len(idx)} train-visible samples ({len({j for j in idx})} unique), centralized best test acc {best:.3f}", flush=True)
