#!/usr/bin/env python3
"""FX-N130 (V6): centralized ceiling C(t). One model trained from scratch on the union of the population's visible data at stream
time t (100% shards: what a holdout-0 baseline can use; each client's own `usage` schedule on the leg's trace, legs start at
FLAME_TRACE_OFFSET_S), best test accuracy over epochs. Same model, SGD and batch as the v2_offline numbers.

  ceiling.py --dataset cifar10 --n 75 --trace syn_50_3st --hours 0 12 24 48 72 96 120 --gpu 2 [--json-out C.json]
"""
import argparse
import json
import os
import random
import sys

import torch
import torch.nn.functional as F
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import utility_bench as ub  # noqa: E402

sys.path.insert(0, os.path.join(HERE, "..", "async_cifar10"))
import stream_schedule  # noqa: E402

EPOCHS = {"cifar10": 40, "google_speech": 30}  # speech: 15 under-trains with augmentation (68.7 vs 75.6% at 12 h)


def visible_idx(ds, n, trace, offset_s):
    """{stream clock s -> sample indices}: callable over t for the first n clients of the reference split."""
    spec = ub.DATA[ds]
    reg = yaml.safe_load(open(os.path.join(ub.META, "trainer_registry.yaml")))["trainers"]
    spl = yaml.safe_load(open(os.path.join(ub.META, "dataset_splits", spec["split"])))["trainer_data_splits"]
    keys = sorted(spl, key=lambda k: int(k.split("_")[-1]))[:n]
    cfg = {"enabled": "True", "mode": "usage", "initial_frac": 0.0, "full_data_available_after_s": 1, "trace": trace}
    cl = []
    for k in keys:
        tid = reg[k]["task_id"]
        idx = list(spl[k])
        seed = int(__import__("hashlib").sha256(str(tid).encode()).hexdigest(), 16) % (2 ** 31)
        order = torch.randperm(len(idx), generator=torch.Generator().manual_seed(seed)).tolist()  # the trainer's arrival order
        cl.append(([idx[j] for j in order], stream_schedule.from_config(cfg, tid, scale=1.0, offset=offset_s)))
    return lambda t: [p for order, s in cl for p in order[:s.visible(len(order), t)]], sorted({j for k in keys for j in spl[k]})


def ceiling(ds, idx, data, dev, epochs, batch=32, augment=True):
    X, Y, Xt, Yt = data
    m = (ub.fl_data.CifarNet() if ds == "cifar10" else ub.fl_data.ResNet34_1D()).to(dev)
    opt = torch.optim.SGD(m.parameters(), lr=0.04, momentum=0.9, weight_decay=5e-4)
    fwd = (lambda x: m(x)) if ds == "cifar10" else (lambda x: m(ub.fl_data.model_input(x)))
    best, idx = 0.0, list(idx)
    for _ in range(epochs):
        m.train()
        random.shuffle(idx)
        for b in range(0, len(idx), batch):
            s = idx[b:b + batch]
            x = ub.fl_data.augment(ds, ub.fl_data.model_input(X[s])) if augment else X[s]  # FX-N143: as the trainers
            opt.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=ds != "cifar10"):
                loss = F.nll_loss(fwd(x).float(), Y[s])
            loss.backward()
            opt.step()
        m.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=ds != "cifar10"):
            hit = sum(int((fwd(Xt[i:i + 1000]).float().argmax(1) == Yt[i:i + 1000]).sum()) for i in range(0, len(Yt), 1000))
        best = max(best, hit / len(Yt))
    return best


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="cifar10", choices=list(ub.DATA))
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--trace", default="syn_0")
    ap.add_argument("--hours", nargs="+", type=float, default=[0, 12, 24, 48, 72, 96, 120])
    ap.add_argument("--offset-h", type=float, default=24.0)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=32, help="cifar: 64 (v2_offline/ceiling300 recipe, best of the two)")
    ap.add_argument("--no-augment", action="store_true", help="pre-FX-N143 ceiling (no augmentation)")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--json-out")
    a = ap.parse_args(argv)
    dev = f"cuda:{a.gpu}"
    random.seed(0)
    torch.manual_seed(0)
    at, all_idx = visible_idx(a.dataset, a.n, a.trace, a.offset_h * 3600)
    data = ub.load_data(a.dataset, all_idx, os.path.expanduser("~/.cache/flame_utility_bench"), dev)
    out = []
    for h in a.hours:
        idx = at(h * 3600)
        acc = ceiling(a.dataset, idx, data, dev, a.epochs or EPOCHS[a.dataset], a.batch, not a.no_augment)
        out.append(dict(dataset=a.dataset, n=a.n, trace=a.trace, hour=h, samples=len(idx), ceiling=acc))
        print(f"{a.dataset} n {a.n} {a.trace} t {h:g} h: {len(idx)} visible samples -> ceiling {acc:.3f}", flush=True)
        if a.json_out:
            json.dump(out, open(a.json_out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
