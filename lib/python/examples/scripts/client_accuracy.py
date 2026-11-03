#!/usr/bin/env python3
"""FELIX A9: the global model's accuracy on each client's own shard (cifar), vs how often the client was picked.

  client_accuracy.py RUN_DIR|PHASE_DIR ... [--n 75] [--last 3]

Per leg, over the mean of its last `--last` checkpoints: client accuracy mean (and sample-weighted), worst-10% and p25 client accuracy,
Spearman of picks vs final client loss and vs shard size, clients with <= 2 train picks. A phase dir resolves via its summary.tsv.
"""
import argparse
import csv
import glob
import os
import statistics
import sys

import torch
import torch.nn.functional as F
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "async_cifar10"))
import fl_data  # noqa: E402
from selection_audit import spearman, trainer_rows  # noqa: E402

META = os.path.join(HERE, "..", "_metadata")
MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)


def clients(n):
    reg = yaml.safe_load(open(os.path.join(META, "trainer_registry.yaml")))["trainers"]
    spl = yaml.safe_load(open(os.path.join(META, "dataset_splits", "cifar10_alpha0.1_n300.yaml")))["trainer_data_splits"]
    return [(info["task_id"], spl[k]) for k, info in sorted(reg.items(), key=lambda kv: kv[1]["trainer_id"]) if k in spl][:n]


def run_dir(arg):
    tsv = os.path.join(arg, "summary.tsv")
    return next(csv.DictReader(open(tsv), delimiter="\t"))["sim_dir"] if os.path.exists(tsv) else arg


def leg(run, cl, X, Y, last, dev):
    cks = sorted(glob.glob(os.path.join(run, "checkpoints", "round_*.pt")))[-last:]
    if not cks:
        return None
    acc, loss = [0.0] * len(cl), [0.0] * len(cl)
    for c in cks:
        m = fl_data.CifarNet()
        m.load_state_dict(torch.load(c, map_location="cpu", weights_only=False)["state_dict"])
        m.to(dev).eval()
        with torch.no_grad():
            for j, (_, idx) in enumerate(cl):
                o = m(X[idx].to(dev))
                y = Y[idx].to(dev)
                acc[j] += (o.argmax(1) == y).float().mean().item() / len(cks)
                loss[j] += F.cross_entropy(o, y).item() / len(cks)
    picks = {k: len(v) for k, v in trainer_rows(run).items()}
    pk, n = [picks.get(t, 0) for t, _ in cl], [len(i) for _, i in cl]
    srt = sorted(acc)
    return dict(ck=os.path.basename(cks[-1]), mean=statistics.mean(acc), wmean=sum(a * k for a, k in zip(acc, n)) / sum(n),
                worst10=statistics.mean(srt[:max(1, len(srt) // 10)]), p25=srt[len(srt) // 4],
                picks_loss=spearman(pk, loss), picks_size=spearman(pk, n), starved=sum(p <= 2 for p in pk))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("legs", nargs="+")
    ap.add_argument("--n", type=int, default=75, help="population: the split's first n trainers (G3 cifar 75)")
    ap.add_argument("--last", type=int, default=3)
    a = ap.parse_args(argv)
    from torchvision.datasets import CIFAR10
    ds = CIFAR10(str(fl_data.dataset_dir("cifar10")), train=True, download=False)
    X = (torch.from_numpy(ds.data).float().div_(255).permute(0, 3, 1, 2) - torch.tensor(MEAN).view(1, 3, 1, 1)) \
        / torch.tensor(STD).view(1, 3, 1, 1)
    Y = torch.tensor(ds.targets)
    cl, dev = clients(a.n), "cuda" if torch.cuda.is_available() else "cpu"
    print("| leg | last ckpt | client acc mean (sample-wtd) | worst 10% | p25 | picks ~ final loss | picks ~ size | ≤ 2 picks |")
    print("|---|---|---|---|---|---|---|---|")
    for arg in a.legs:
        r = leg(run_dir(arg), cl, X, Y, a.last, dev)
        if r:
            print(f"| {os.path.basename(arg.rstrip('/'))} | {r['ck']} | {r['mean']:.3f} ({r['wmean']:.3f}) | {r['worst10']:.3f} "
                  f"| {r['p25']:.3f} | {r['picks_loss']:+.2f} | {r['picks_size']:+.2f} | {r['starved']} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
