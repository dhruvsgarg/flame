"""B: centralized ceiling vs share of the n=300 (or n=75) cifar population's shards visible (6-day usage stream: 24 h = 1/6)."""
import sys, os, random, yaml, torch, torch.nn.functional as F
sys.path.insert(0, "/home/dgarg39/flame/lib/python/examples/scripts")
import utility_bench as ub
n, gpu = int(sys.argv[1]), sys.argv[2]; dev = f"cuda:{gpu}"
spl = yaml.safe_load(open(os.path.join(ub.META, "dataset_splits", "cifar10_alpha0.1_n300.yaml")))["trainer_data_splits"]
keys = sorted(spl, key=lambda k: int(k.split("_")[-1]))[:n]
X, Y, Xt, Yt = ub.load_data("cifar10", None, None, dev)
for frac in (1/6, 1/6 + 1/12, 1/3, 1/2, 2/3, 1.0):
    idx = [j for k in keys for i, j in enumerate(spl[k][:max(1, int(frac * len(spl[k])))]) if i % 10 != 9]
    m = ub.fl_data.CifarNet().to(dev); opt = torch.optim.SGD(m.parameters(), lr=0.04, momentum=0.9, weight_decay=5e-4); best = 0
    for ep in range(30):
        m.train(); random.shuffle(idx)
        for b in range(0, len(idx), 64):
            s = idx[b:b + 64]; opt.zero_grad(); F.nll_loss(m(X[s]), Y[s]).backward(); opt.step()
        m.eval()
        with torch.no_grad(): best = max(best, sum(int((m(Xt[i:i+2000]).argmax(1) == Yt[i:i+2000]).sum()) for i in range(0, 10000, 2000)) / 10000)
    print(f"n {n}: shard share {frac:.2f} (day {frac*6:.1f}) {len(idx)} samples -> centralized best {best:.3f}", flush=True)
