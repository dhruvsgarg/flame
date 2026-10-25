#!/usr/bin/env python3
"""FL upper bound (operator 10-24): ideal FedAvg, every one of the n clients each round, full shards (= trace end), no
unavailability, 1 local epoch at Felix / FedBuff's client recipe (SGD lr 0.04, batch 32 / 16, per-batch augmentation), |D|-weighted
average (BN stats included), server lr 1. Runs until the best 5-eval mean stops rising; that plateau is the final FL target.

  fl_bound.py --dataset cifar10 --n 300 --gpu 0 [--rounds 1500] [--patience 100] [--json-out B.json]
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


def shards(ds, n):
    spl = yaml.safe_load(open(os.path.join(ub.META, "dataset_splits", ub.DATA[ds]["split"])))["trainer_data_splits"]
    return [list(spl[k]) for k in sorted(spl, key=lambda k: int(k.split("_")[-1]))[:n]]


def local_epochs(ds, dev, clients, lr, conn):
    """GPU worker: per round, one epoch per own client from the global state; returns the |D|-weighted state sum."""
    random.seed(hash((ds, dev)) & 0x7FFFFFFF)
    cl = shards(ds, clients[0])[clients[1]:clients[2]]
    allc = shards(ds, clients[0])
    X, Y, _, _ = ub.load_data(ds, sorted({j for c in allc for j in c}), os.path.expanduser("~/.cache/flame_utility_bench"), dev)
    bf16 = ds != "cifar10"
    w = (ub.fl_data.CifarNet() if ds == "cifar10" else ub.fl_data.ResNet34_1D()).to(dev)
    batch = ub.DATA[ds]["batch"]
    while True:
        msg = conn.recv()
        if msg is None:
            return
        gstate, tot = msg
        acc = {k: torch.zeros_like(v, dtype=torch.float32, device=dev) for k, v in gstate.items()}
        for c in cl:
            w.load_state_dict(gstate)
            opt = torch.optim.SGD(w.parameters(), lr=lr)
            w.train()
            idx = c[:]
            random.shuffle(idx)
            for b in range(0, len(idx), batch):
                s = idx[b:b + batch]
                x = ub.fl_data.augment(ds, ub.fl_data.model_input(X[s]) if bf16 else X[s])
                opt.zero_grad()
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                    loss = F.nll_loss(w(ub.fl_data.model_input(x)).float(), Y[s])
                loss.backward()
                opt.step()
            for k, v in w.state_dict().items():
                acc[k] += v.float() * (len(c) / tot)
        conn.send({k: v.cpu() for k, v in acc.items()})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="cifar10", choices=list(ub.DATA))
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--rounds", type=int, default=1500)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--patience", type=int, default=100, help="stop after this many rounds without a new best 5-eval mean")
    ap.add_argument("--lr", type=float, default=0.04)
    ap.add_argument("--gpu", default="0", help="comma list: >1 GPU splits each round's clients across worker GPUs")
    ap.add_argument("--json-out")
    a = ap.parse_args(argv)
    gpus = a.gpu.split(",")
    dev, ds = f"cuda:{gpus[0]}", a.dataset
    random.seed(0)
    torch.manual_seed(0)
    cl = shards(ds, a.n)
    X, Y, Xt, Yt = ub.load_data(ds, sorted({j for c in cl for j in c}), os.path.expanduser("~/.cache/flame_utility_bench"), dev)
    bf16 = ds != "cifar10"
    g = (ub.fl_data.CifarNet() if ds == "cifar10" else ub.fl_data.ResNet34_1D()).to(dev)
    w = (ub.fl_data.CifarNet() if ds == "cifar10" else ub.fl_data.ResNet34_1D()).to(dev)
    fwd = lambda m, x: m(ub.fl_data.model_input(x)) if bf16 else m(x)
    batch, tot = ub.DATA[ds]["batch"], sum(len(c) for c in cl)
    hist, best, best_r, out = [], 0.0, 0, []
    pipes = []
    if len(gpus) > 1:
        import torch.multiprocessing as mp
        ctx = mp.get_context("spawn")
        cuts = [round(i * len(cl) / len(gpus)) for i in range(len(gpus) + 1)]
        for i, gid in enumerate(gpus):
            here, there = ctx.Pipe()
            ctx.Process(target=local_epochs, args=(ds, f"cuda:{gid}", (a.n, cuts[i], cuts[i + 1]), a.lr, there), daemon=True).start()
            pipes.append(here)
    for r in range(1, a.rounds + 1):
        if pipes:
            st = {k: v.cpu() for k, v in g.state_dict().items()}
            for p in pipes:
                p.send((st, tot))
            parts = [p.recv() for p in pipes]
            acc = {k: sum(q[k] for q in parts).to(dev) for k in st}
        else:
            acc = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in g.state_dict().items()}
            for c in cl:
                w.load_state_dict(g.state_dict())
                opt = torch.optim.SGD(w.parameters(), lr=a.lr)
                w.train()
                idx = c[:]
                random.shuffle(idx)
                for b in range(0, len(idx), batch):
                    s = idx[b:b + batch]
                    x = ub.fl_data.augment(ds, ub.fl_data.model_input(X[s]) if bf16 else X[s])
                    opt.zero_grad()
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                        loss = F.nll_loss((w(x) if bf16 else fwd(w, x)).float(), Y[s])
                    loss.backward()
                    opt.step()
                for k, v in w.state_dict().items():
                    acc[k] += v.float() * (len(c) / tot)
        g.load_state_dict({k: v.to(g.state_dict()[k].dtype) for k, v in acc.items()})
        if r % a.eval_every == 0:
            g.eval()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                hit = sum(int((fwd(g, Xt[i:i + 1000]).float().argmax(1) == Yt[i:i + 1000]).sum()) for i in range(0, len(Yt), 1000))
            hist.append(hit / len(Yt))
            m5 = sum(hist[-5:]) / len(hist[-5:])
            if len(hist) >= 5 and m5 > best:
                best, best_r = m5, r
            out.append(dict(round=r, acc=hist[-1], mean5=m5))
            print(f"{ds} n {a.n} round {r}: acc {hist[-1]:.4f} 5-eval mean {m5:.4f} best {best:.4f} @ {best_r}", flush=True)
            if a.json_out:
                json.dump(dict(dataset=ds, n=a.n, bound=best, best_round=best_r, evals=out), open(a.json_out, "w"), indent=1)
            if best_r and r - best_r >= a.patience:
                break
    for p in pipes:
        p.send(None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
