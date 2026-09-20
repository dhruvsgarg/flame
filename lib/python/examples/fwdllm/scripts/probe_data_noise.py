#!/usr/bin/env python3
"""Rung-2 rig: where the factor D ~= 0.05 between measured and closed-form cos comes from.

Built for D-1 (fl_fwd_ft_solution.md 6.2). Imports the production model and the
real Dirichlet partition; no FL stack, no MQTT, no run.

    python probe_data_noise.py --config <aggregator_config.json> [--train]

The uploads a commit pools are directional derivatives of a CLIENT BIN's gradient,
not of the gradient we mean to descend. Because v is a raw Gaussian draw,
rms|d| = ||g_bin|| identically, so the pooled update is normalised by a quantity
~10x larger than the reference. That costs a constant factor in cos:

    cos(G, g*) = D * (a/b) * sqrt(N/p)
    L = rms||g_bin|| / ||g*||        length excess    -- in-run 8.5 .. 9.5
    S = <g_pool, ghat*> / ||g*||     shadow deficit   -- in-run 0.48
    D = S / L                                         -- in-run 0.050

Sweeps bin size B and distinct-bin count M and reports all three, against two
references: the production held-out probe batch, and a large pooled TRAIN
gradient. Splitting those separates a finite-pool effect from train/test drift.

PREDICTION (registered before the run): L ~ 1/sqrt(B) and is flat in M; S rises
with M*B toward a ceiling; hence D ~ sqrt(B) and is flat in M.
SINKING CONDITION: if D is flat in B, the shortfall is bias, not sampling -- bin
size buys nothing, and the sizing formula of 4.5 is dead rather than mis-scaled.

VALIDATION GATE: at M=10, B=8 -- the shipped operating point -- this rig must
return L ~= 9, S ~= 0.5, D ~= 0.05. Nothing below it counts otherwise.
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

CHUNK = 128


def build(cfg_path, n_clients):
    """Production model + the real per-client train shards."""
    hp = types.SimpleNamespace(**json.load(open(cfg_path))["hyperparameters"])
    attrs = BaseDataManager.load_attributes(hp.data_file_path)
    nl = len(attrs["label_vocab"])
    margs = build_model_args(hp, nl)
    _, model, tok = create_model(margs, formulation="classification")
    # the trainer __init__ drops pre_classifier (:217) -- this is what sets p
    ForwardTextClassificationTrainer(margs, 0, model, None, None, "rig")
    pre = TLMPreprocessor(args=margs, label_vocab=attrs["label_vocab"], tokenizer=tok)

    shards = []
    for cid in range(n_clients):
        # the ctor's process_id only picks a simulated worker slot and is capped by
        # data_loader_num_workers; client_idx is what actually selects the shard.
        dm = TextClassificationDataManager(hp, margs, pre, 1,
                                           hp.data_loader_num_workers)
        out = dm.load_federated_data(process_id=1, client_idx=cid)
        ds = out[4][cid].dataset                       # train_data_local_dict[cid]
        shards.append((ds.tensors[1], ds.tensors[4]))
    dm0 = TextClassificationDataManager(hp, margs, pre, 0, hp.data_loader_num_workers)
    test = dm0.load_federated_data(process_id=0, client_idx=None)[2].dataset
    return model, shards, (test.tensors[1], test.tensors[4]), nl


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


def norm(g):
    return math.sqrt(sum(float(x.pow(2).sum()) for x in g))


def dot(a, b):
    return sum(float((x * y).sum()) for x, y in zip(a, b))


def mean_of(gs):
    return [sum(t) / len(gs) for t in zip(*gs)]


@torch.no_grad()
def accuracy(model, X, Y, dev, n=2000):
    model.eval()
    ok = 0
    n = min(n, len(X))
    for s in range(0, n, CHUNK):
        o = model(X[s:s + CHUNK].to(dev))
        lg = o.logits if hasattr(o, "logits") else o[0]
        ok += int((lg.argmax(-1).cpu() == Y[s:s + CHUNK].view(-1)).sum())
    return ok / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="an aggregator_config.json")
    ap.add_argument("--bins", type=int, nargs="+", default=[8, 32, 128])
    ap.add_argument("--n-bins", type=int, nargs="+", default=[1, 10, 30])
    ap.add_argument("--clients", type=int, default=30)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--ref-size", type=int, default=1024,
                    help="production cos-probe batch (B17 default)")
    ap.add_argument("--train", action="store_true",
                    help="DO NOT USE for D-1. Backprop and forward-gradient reach "
                         "the same accuracy at very different points: at acc 0.57 a "
                         "backprop model has ||g_test(1024)|| = 1.95 where every real "
                         "arm reads 0.25-0.32, so L comes out 4x low and the rig "
                         "fails its own validation gate. ||g*|| is FLAT across a real "
                         "arm, so init is the representative state. Kept only to "
                         "reproduce that finding.")
    ap.add_argument("--target-acc", type=float, default=0.60,
                    help="stop training here; the arms that measured D sat at 0.37-0.60")
    a = ap.parse_args()

    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, shards, (Xt, Yt), nl = build(a.config, a.clients)
    model.to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    p_dim = sum(x.numel() for x in params)
    print(f"[rig] p={p_dim}  clients={len(shards)}  shard sizes="
          f"{[len(s[0]) for s in shards[:6]]}...  n_test={len(Xt)}  device={dev}")

    if a.train:
        Xa = torch.cat([s[0] for s in shards])
        Ya = torch.cat([s[1] for s in shards])
        opt = torch.optim.AdamW(params, lr=1e-3)
        g2 = torch.Generator().manual_seed(1)
        perm = torch.randperm(len(Xa), generator=g2)
        for s in range(0, len(perm), 32):
            model.train()
            sel = perm[s:s + 32]
            o = model(Xa[sel].to(dev))
            lg = o.logits if hasattr(o, "logits") else o[0]
            CrossEntropyLoss()(lg.view(-1, nl), Ya[sel].view(-1).to(dev)).backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            if (s // 32) % 4 == 0:
                ac = accuracy(model, Xt, Yt, dev, 1000)
                print(f"  step {s // 32}: test acc={ac:.4f}")
                if ac >= a.target_acc:
                    break
    print(f"[rig] test accuracy = {accuracy(model, Xt, Yt, dev):.4f}  "
          f"||theta_tr|| = {norm([x.detach().cpu() for x in params]):.3f}")

    gen = torch.Generator().manual_seed(7)

    # Bins are drawn from the FIRST half of every shard; the TRAIN reference is
    # pooled from the SECOND half. Overlap would inflate <g_pool, ghat*> through
    # shared sampling noise -- that is what put S above 1 on the first cut.
    half = [(Xc[:len(Xc) // 2], Yc[:len(Yc) // 2]) for Xc, Yc in shards]
    Xr = torch.cat([Xc[len(Xc) // 2:] for Xc, _ in shards])
    Yr = torch.cat([Yc[len(Yc) // 2:] for _, Yc in shards])

    ref_idx = torch.randperm(len(Xt), generator=gen)[:a.ref_size]
    g_test = grad_over(model, Xt, Yt, ref_idx, nl, dev)
    lab = collections.Counter(Yt.view(-1)[ref_idx].tolist())
    tr_idx = torch.randperm(len(Xr), generator=gen)[:a.ref_size * 4]
    g_train = grad_over(model, Xr, Yr, tr_idx, nl, dev)
    c_tt = dot(g_train, g_test) / (norm(g_train) * norm(g_test))
    print(f"[rig] ||g_test({a.ref_size})|| = {norm(g_test):.4f}  dominant class "
          f"{max(lab.values()) / a.ref_size:.2f}")
    print(f"[rig] ||g_train({len(tr_idx)}, held out from the bins)|| = {norm(g_train):.4f}"
          f"   cos(g_train, g_test) = {c_tt:.4f}  <- ceiling on cos(pool, g_test)")

    refs = (("TRAIN", g_train, norm(g_train)), ("TEST", g_test, norm(g_test)))
    print(f"\n{'M':>4s}{'B':>6s}{'samp':>7s} |" +
          "".join(f"{r[0] + ' L':>9s}{r[0] + ' S':>9s}{r[0] + ' D':>9s}"
                  f"{'cos':>8s}" for r in refs))
    for M in a.n_bins:
        if M > len(half):
            continue
        for B in a.bins:
            acc = {r[0]: ([], [], []) for r in refs}
            for _ in range(a.reps):
                cl = torch.randperm(len(half), generator=gen)[:M]
                gbs = []
                for c in cl.tolist():                 # one bin per client, as in-run
                    Xc, Yc = half[c]
                    sel = torch.randperm(len(Xc), generator=gen)[:min(B, len(Xc))]
                    gbs.append(grad_over(model, Xc, Yc, sel, nl, dev))
                rms = math.sqrt(sum(norm(g) ** 2 for g in gbs) / len(gbs))
                gp = mean_of(gbs)
                for nm, gstar, ng in refs:            # same bins, both references
                    acc[nm][0].append(rms / ng)
                    acc[nm][1].append(dot(gp, gstar) / ng ** 2)
                    acc[nm][2].append(dot(gp, gstar) / (norm(gp) * ng))
            row = f"{M:4d}{B:6d}{M * B:7d} |"
            for nm, _, _ in refs:
                L, S, C = (sum(v) / len(v) for v in acc[nm])
                row += f"{L:9.2f}{S:9.3f}{S / L:9.4f}{C:8.3f}"
            print(row)

    print("\nL should fall as 1/sqrt(B) and be flat in M; S should rise with M*B toward 1.")
    print("D = S/L is the factor the closed-form cos of model 4.5 is missing.")
    print("Production point is M=10, B=8; in-run against TEST it reads L~9, S~0.48, D~0.05.")


if __name__ == "__main__":
    main()
