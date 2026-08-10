#!/usr/bin/env python3
"""Rung-2 rig: how good is a held-out batch as a gradient reference?

Built to diagnose B1 (fl_fwd_ft_solution.md §6) and to size its fix. Imports
the production model and data path; no FL stack, no MQTT, no run.

    python probe_reference_quality.py --config <aggregator_config.json> [--train]

Reports cos(g_batch, g_ALL) for
  * the FIRST n rows -- exactly what FedSgdAggregator._cos_probe_gradient caches
  * a fixed-seed RANDOM n -- the fix
over n, plus the two-batch control cos(g_A, g_B) on random disjoint halves.

Why this exists: test_index_list is built by iterating clients in partition
order and extending with each client's test shard (base_data_manager.py:204-216),
with no shuffle. Under niid_label_clients=100_alpha=1 that makes the first 76
rows ONE client's Dirichlet-skewed shard, so the probe's 64-sample "global
reference" is 75% one class against a test set that is exactly balanced.

Measured at a backprop-trained model (acc 0.900): cos(g_first64, g_ALL) = -0.46
-- the production reference is ANTI-correlated with held-out truth, and its sign
swings with model state. Random-n alignment: 0.480 (64) / 0.794 (256) /
0.935 (1024) / 0.975 (2048).
"""
import argparse
import collections
import json
import math
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", ".."))
import torch                                                   # noqa: E402
from torch.nn import CrossEntropyLoss                          # noqa: E402

from examples.fwdllm.data_preprocessing.text_classification_preprocessor import (
    TLMPreprocessor,
)                                                              # noqa: E402
from examples.fwdllm.trainer.forward_training.tc_transformer_trainer_distribute import (
    ForwardTextClassificationTrainer,
)                                                              # noqa: E402
from examples.fwdllm.trainer.model_args_builder import build_model_args  # noqa: E402
from examples.fwdllm.data_manager.text_classification_data_manager import (
    TextClassificationDataManager,
)                                                              # noqa: E402
from examples.fwdllm.data_manager.base_data_manager import BaseDataManager  # noqa: E402
from examples.fwdllm.expts.initializer import create_model     # noqa: E402

CHUNK = 256


def build(cfg_path):
    hp = types.SimpleNamespace(**json.load(open(cfg_path))["hyperparameters"])
    attrs = BaseDataManager.load_attributes(hp.data_file_path)
    nl = len(attrs["label_vocab"])
    margs = build_model_args(hp, nl)
    _, model, tok = create_model(margs, formulation="classification")
    # the trainer __init__ drops pre_classifier (:217) -- this is what sets p
    ForwardTextClassificationTrainer(margs, 0, model, None, None, "rig")
    pre = TLMPreprocessor(args=margs, label_vocab=attrs["label_vocab"], tokenizer=tok)
    dm = TextClassificationDataManager(hp, margs, pre, 0, hp.data_loader_num_workers)
    return model, dm.load_federated_data(process_id=0, client_idx=hp.client_idx)[2], nl


def grad_over(model, X, Y, idx, nl, dev):
    """Mean-CE gradient over `idx`, chunked. fp32, eval mode -- mirrors B1."""
    model.eval()
    model.zero_grad(set_to_none=True)
    for s in range(0, len(idx), CHUNK):
        sel = idx[s:s + CHUNK]
        o = model(X[sel].to(dev))
        lg = o.logits if hasattr(o, "logits") else (
            o[0] if isinstance(o, (tuple, list)) else o)
        (CrossEntropyLoss(reduction="sum")(lg.view(-1, nl), Y[sel].view(-1).to(dev))
         / len(idx)).backward()
    g = [p.grad.detach().to("cpu", torch.float32).clone()
         for p in model.parameters() if p.requires_grad and p.grad is not None]
    model.zero_grad(set_to_none=True)
    return g


def cosv(a, b):
    dot = sum(float((x * y).sum()) for x, y in zip(a, b))
    na = math.sqrt(sum(float(x.pow(2).sum()) for x in a))
    nb = math.sqrt(sum(float(y.pow(2).sum()) for y in b))
    return dot / (na * nb)


@torch.no_grad()
def accuracy(model, X, Y, dev, n=2000):
    model.eval()
    ok = 0
    for s in range(0, n, CHUNK):
        o = model(X[s:s + CHUNK].to(dev))
        lg = o.logits if hasattr(o, "logits") else o[0]
        ok += int((lg.argmax(-1).cpu() == Y[s:s + CHUNK].view(-1)).sum())
    return ok / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="an aggregator_config.json")
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[64, 128, 256, 512, 1024, 2048])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--train", action="store_true",
                    help="backprop fine-tune first; the reference is worst at "
                         "high accuracy, where the gradient concentrates on few "
                         "hard examples")
    ap.add_argument("--epochs", type=int, default=3)
    a = ap.parse_args()

    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, tg, nl = build(a.config)
    model.to(dev)
    X, Y = tg.dataset.tensors[1], tg.dataset.tensors[4]
    n_all = X.shape[0]
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"[rig] p={sum(x.numel() for x in params)}  n_test={n_all}  device={dev}")
    print(f"[rig] full-set labels {dict(sorted(collections.Counter(Y.view(-1).tolist()).items()))}")

    if a.train:
        tr = torch.arange(n_all // 2, n_all)          # train on a disjoint tail
        opt = torch.optim.AdamW(params, lr=1e-3)
        g2 = torch.Generator().manual_seed(1)
        model.train()
        for ep in range(a.epochs):
            perm = tr[torch.randperm(len(tr), generator=g2)]   # tr is label-sorted
            for s in range(0, len(perm), 32):
                sel = perm[s:s + 32]
                o = model(X[sel].to(dev))
                lg = o.logits if hasattr(o, "logits") else o[0]
                CrossEntropyLoss()(lg.view(-1, nl), Y[sel].view(-1).to(dev)).backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
            print(f"  epoch {ep}: acc={accuracy(model, X, Y, dev):.4f}")
    print(f"[rig] accuracy = {accuracy(model, X, Y, dev):.4f}")

    g_all = grad_over(model, X, Y, torch.arange(n_all), nl, dev)
    gen = torch.Generator().manual_seed(7)

    print("\ncos(g_batch, g_ALL) -- 1.0 is a perfect reference")
    print(f"{'n':>6s}{'FIRST n (prod)':>17s}{'skew':>7s}{'RANDOM n':>11s}"
          f"{'min':>8s}{'max':>8s}{'atten':>8s}")
    for n in a.sizes:
        if n > n_all:
            continue
        cf = cosv(grad_over(model, X, Y, torch.arange(n), nl, dev), g_all)
        cnt = collections.Counter(Y.view(-1)[:n].tolist())
        cs = [cosv(grad_over(model, X, Y,
                             torch.randperm(n_all, generator=gen)[:n], nl, dev), g_all)
              for _ in range(a.reps)]
        m = sum(cs) / len(cs)
        print(f"{n:6d}{cf:17.4f}{max(cnt.values()) / n:6.0%}{m:11.4f}"
              f"{min(cs):8.4f}{max(cs):8.4f}{(1 / m if m > 0 else float('nan')):8.2f}")

    print("\ntwo-batch control cos(g_A, g_B), random disjoint halves")
    print(f"{'n':>6s}{'rep':>4s}{'cos':>10s}{'sqrt=atten':>12s}")
    for n in a.sizes[:4]:
        for r in range(min(3, a.reps)):
            idx = torch.randperm(n_all, generator=gen)[:2 * n]
            c = cosv(grad_over(model, X, Y, idx[:n], nl, dev),
                     grad_over(model, X, Y, idx[n:], nl, dev))
            print(f"{n:6d}{r:4d}{c:10.4f}"
                  f"{(math.sqrt(c) if c > 0 else float('nan')):12.3f}")


if __name__ == "__main__":
    main()
