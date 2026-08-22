#!/usr/bin/env python3
"""Rung-2 rig for Q1 and Q2: does inflation itself destroy the classifier?

Both blockers reduce to one question -- what does growing ||theta_tr|| do to a
TRAINED model? The runs answer it by waiting ~330 commits to accumulate the
growth. The growth is a sum of steps in random directions, so it can be injected
directly instead, in minutes.

    python probe_inflation_damage.py --config <aggregator_config.json> --rf 16
    python probe_inflation_damage.py --config <aggregator_config.json> --rf 64

Train to a realistic peak, then add Gaussian noise to the trainable slice scaled
so ||theta_tr|| grows by Phi, and read accuracy back.

  Q2 (causal vs symptom): if accuracy collapses near the observed Phi ~ 3.6, the
      inflation IS the mechanism and weight decay should help -- which INVERTS
      the registered prediction for node 1. If accuracy survives, the norm is a
      symptom and the null is confirmed for free.
  Q1 (absolute vs relative): run at two values of p. If the knee sits at the same
      Phi, the budget is RELATIVE; if at the same ||theta_tr||, it is ABSOLUTE.
      That is exactly the discriminator the p-ladder arm is meant to provide.

Two controls separate the mechanisms:
  * `scale`  -- multiply theta_tr by Phi instead of adding noise. Grows the norm
                without adding junk, so it isolates norm-per-se from misaim.
  * `signal` -- add noise ONLY in the gradient direction. Same norm growth, but
                coherent, so it isolates direction from magnitude.

Caveat, stated so the result is not over-read: injected noise reproduces the
inflation faithfully (the uploads really are random directions) but jumps to the
endpoint rather than walking there, and omits the learning happening alongside.
This pre-answers Q1/Q2 and re-aims their predictions; it does not replace them.
"""
import argparse
import collections
import json
import math
import os
import sys
import tempfile

import torch
from torch.nn import CrossEntropyLoss

sys.path.insert(0, "/home/dgarg39/flame/lib/python")
from examples.fwdllm.expts.bmax_probe import knee  # noqa: E402
from examples.fwdllm.scripts.probe_reference_quality import (  # noqa: E402
    build, accuracy, grad_over, CHUNK,
)

PHIS = [1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 3.6, 4.5, 6.0, 9.0]


def tr_params(model):
    return [p for p in model.parameters() if p.requires_grad]


def tr_norm(model):
    return math.sqrt(sum(float(p.detach().pow(2).sum()) for p in tr_params(model)))


@torch.no_grad()
def head_stats(model, X, Y, dev, n=2000):
    """Accuracy plus the collapse fingerprints the runs log at agg_eval."""
    model.eval()
    ok, ent, ln, preds = 0, 0.0, 0.0, []
    for s in range(0, n, CHUNK):
        o = model(X[s:s + CHUNK].to(dev))
        lg = (o.logits if hasattr(o, "logits") else o[0]).float()
        p = lg.softmax(-1)
        ok += int((lg.argmax(-1).cpu() == Y[s:s + CHUNK].view(-1)).sum())
        ent += float(-(p * (p + 1e-12).log()).sum())
        ln += float(lg.norm(dim=-1).sum())
        preds += lg.argmax(-1).cpu().tolist()
    c = collections.Counter(preds)
    return dict(acc=ok / n, entropy=ent / n, logit_norm=ln / n,
                top_class_share=max(c.values()) / n)


def perturb(model, base, phi, mode, gen, gdir=None):
    """Restore `base`, then grow ||theta_tr|| to phi * ||theta_tr||_base."""
    ps = tr_params(model)
    with torch.no_grad():
        for p, b in zip(ps, base):
            p.copy_(b)
        if phi <= 1.0:
            return
        n0 = tr_norm(model)
        if mode == "scale":
            for p in ps:
                p.mul_(phi)
            return
        target = n0 * math.sqrt(phi ** 2 - 1.0)     # ||eps|| for orthogonal noise
        if mode == "signal":                        # coherent: along the gradient
            gn = math.sqrt(sum(float(g.pow(2).sum()) for g in gdir))
            for p, g in zip(ps, gdir):
                p.add_(g.to(p.device) * (target / gn))
            return
        eps = [torch.randn(p.shape, generator=gen) for p in ps]   # isotropic
        en = math.sqrt(sum(float(e.pow(2).sum()) for e in eps))
        for p, e in zip(ps, eps):
            p.add_(e.to(p.device) * (target / en))
        if mode == "noise_renorm":
            # Exactly what perfect weight decay leaves behind: the SAME direction
            # as the inflated model, shrunk back to the original norm. Recovery
            # here means the damage is magnitude (decay helps, Q2 = causal);
            # no recovery means it is the junk:signal ratio, which decay cannot
            # touch because it scales signal and junk alike (Q2 = symptom).
            k = n0 / tr_norm(model)
            for p in ps:
                p.mul_(k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--rf", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--modes", nargs="+", default=["noise", "scale", "signal"])
    ap.add_argument("--phis", default=",".join(str(p) for p in PHIS),
                     help="comma-separated Phi grid (default: today's byte-identical list)")
    a = ap.parse_args()
    phis = [float(x) for x in a.phis.split(",")]

    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = json.load(open(a.config))
    cfg["hyperparameters"]["adapter_reduction_factor"] = a.rf
    # unique per invocation -- a fixed /tmp/rig_cfg_rf{rf}.json collides when
    # two datasets at the same rf run concurrently (B-1's 3-dataset sweep).
    fd, tmp = tempfile.mkstemp(prefix=f"rig_cfg_rf{a.rf}_", suffix=".json")
    os.close(fd)
    json.dump(cfg, open(tmp, "w"))

    model, tg, nl = build(tmp)
    model.to(dev)
    X, Y = tg.dataset.tensors[1], tg.dataset.tensors[4]
    p_count = sum(x.numel() for x in tr_params(model))
    print(f"[rig] rf={a.rf}  p={p_count}  ||theta_tr||_init={tr_norm(model):.3f}")

    # train to a realistic peak, on a disjoint half so the eval head is untouched
    tr = torch.arange(X.shape[0] // 2, X.shape[0])
    opt = torch.optim.AdamW(tr_params(model), lr=1e-3)
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

    base = [p.detach().clone() for p in tr_params(model)]
    n0 = tr_norm(model)
    b = head_stats(model, X, Y, dev)
    print(f"[rig] trained: acc={b['acc']:.4f}  ||theta_tr||={n0:.3f}  "
          f"top_class_share={b['top_class_share']:.3f}")
    gdir = grad_over(model, X, Y, torch.arange(X.shape[0]), nl, dev)

    knees = {}
    for mode in a.modes:
        print(f"\n=== mode = {mode} "
              f"({'isotropic junk' if mode == 'noise' else mode}) ===")
        print(f"{'Phi':>6s}{'||theta_tr||':>13s}{'acc':>18s}{'d_acc':>8s}"
              f"{'top_cls':>9s}{'entropy':>9s}{'logit_n':>9s}")
        gen = torch.Generator().manual_seed(11)
        curve = []
        for phi in phis:
            reps = a.reps if mode.startswith("noise") and phi > 1.0 else 1
            accs, st = [], None
            for _ in range(reps):
                perturb(model, base, phi, mode, gen, gdir)
                st = head_stats(model, X, Y, dev)
                accs.append(st["acc"])
            m = sum(accs) / len(accs)
            curve.append(m)
            spread = f"+-{(max(accs) - min(accs)) / 2:.3f}" if reps > 1 else "      "
            shown = tr_norm(model)   # noise_renorm pins this back at n0
            print(f"{phi:6.1f}{shown:13.2f}{m:12.4f} {spread:>5s}"
                  f"{m - b['acc']:+8.3f}{st['top_class_share']:9.3f}"
                  f"{st['entropy']:9.3f}{st['logit_norm']:9.2f}")
        # Row P1's deliverable: the same arithmetic the LIVE sensor uses, so a
        # number here is directly comparable to a `[BmaxProbe] Phi_knee=` line.
        # Grid points at or below 1.0 are the unperturbed model and carry no
        # information about where the knee is; knee() supplies that anchor itself.
        _pts = [(f, c) for f, c in zip(phis, curve) if f > 1.0]
        knees[mode] = (knee([f for f, _ in _pts], [c for _, c in _pts],
                            b["acc"], nl) if _pts else None)

    print(f"\n[rig] rf={a.rf}  p={p_count}  num_labels={nl}  "
          f"base_acc={b['acc']:.4f}")
    for mode, k in knees.items():
        print(f"  Phi_knee ({mode:<12}) = "
              + (f"{k:.3f}" if k else "n/a -- never crossed the 0.5 level")
              + ("   <-- BRACKETED" if k and phis[0] < k < phis[-1]
                 else "   <-- NOT bracketed: re-range --phis"))
    print("  compare against Phi* = 2.82 / 3.00 / 2.91 on agnews / yahoo / yelp-p")

    # leave the model as trained, not perturbed
    perturb(model, base, 1.0, "noise", torch.Generator())


if __name__ == "__main__":
    main()
