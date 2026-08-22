#!/usr/bin/env python3
"""Row P1': the noise-THEN-REFIT probe -- the instrument `B_max` actually needs.

    python probe_inflation_refit.py --config <aggregator_config.json> \
        --ms 0,10,50,150 --phis 1.5,2,2.5,3,3.5,4 --rf 16

The shipped probe (`expts/bmax_probe.py`, and `probe_inflation_damage.py`
offline) injects `||theta||*sqrt(Phi^2-1)` of isotropic noise in ONE SHOT and
reads a FROZEN model. A run adds the same total length in ~1,000 increments with
the head re-fitting between every one. Same end `||theta||`, same drift angle off
`theta_0`, opposite verdict -- 60 degrees of injected drift is fatal, 70 degrees
of earned drift is optimal, and buildplan §5.9 measures the gap at **~1.8x in
Phi**. That is why the live sensor reads 1.25 while every run peaks near 2.9.

This changes exactly one thing: inject at `Phi`, run `m` steps, THEN read.

  * `m` = 0 must reproduce the shipped probe's readings -- the positive control.
    If it does not, the difference is the rig and not the re-fit.
  * **Predicted:** the knee rises with `m` and lands near 2.9 by `m` ~ 50, then
    stops moving.
  * **Falsified if** the knee is still <= 1.6 at `m` = 150 -- then the junk a
    trajectory carries is NOT isotropic, §5.9's mechanism is wrong, and `Phi*` is
    unmeasurable without a training run (which makes row E's saturation stop the
    primary sensor rather than a backstop -- §5.6b).

**What `m` is, stated so the result is not over-read.** These are the rig's own
AdamW steps at batch 32, not FL commits: the question is whether the damage is
RE-FITTABLE, not how many federated commits it would take. A knee that moves with
`m` at all is the finding; where exactly it saturates is a rig number.

The knee is read with `expts/bmax_probe.knee` -- the same arithmetic the live
sensor uses, on chance-normalized accuracy, so a number here is directly
comparable to a `[BmaxProbe] Phi_knee=` line in a run.
"""
import argparse
import json
import os
import sys
import tempfile

import torch
from torch.nn import CrossEntropyLoss

sys.path.insert(0, "/home/dgarg39/flame/lib/python")
from examples.fwdllm.expts.bmax_probe import knee  # noqa: E402
from examples.fwdllm.scripts.probe_inflation_damage import (  # noqa: E402
    head_stats, perturb, tr_norm, tr_params,
)
from examples.fwdllm.scripts.probe_reference_quality import build, accuracy  # noqa: E402

PHIS = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
MS = [0, 10, 50, 150]


def refit(model, X, Y, idx, nl, dev, steps, lr, gen, batch=32):
    """`steps` ordinary optimizer steps on the TRAIN half. A fresh optimizer each
    time: the run's re-fit carries no Adam state across an injection either."""
    if steps <= 0:
        return
    opt = torch.optim.AdamW(tr_params(model), lr=lr)
    model.train()
    for s in range(steps):
        sel = idx[torch.randint(len(idx), (batch,), generator=gen)]
        o = model(X[sel].to(dev))
        lg = o.logits if hasattr(o, "logits") else o[0]
        CrossEntropyLoss()(lg.view(-1, nl), Y[sel].view(-1).to(dev)).backward()
        opt.step()
        opt.zero_grad(set_to_none=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--rf", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--phis", default=",".join(str(p) for p in PHIS))
    ap.add_argument("--ms", default=",".join(str(m) for m in MS))
    a = ap.parse_args()
    phis = [float(x) for x in a.phis.split(",")]
    ms = [int(x) for x in a.ms.split(",")]

    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = json.load(open(a.config))
    cfg["hyperparameters"]["adapter_reduction_factor"] = a.rf
    fd, tmp = tempfile.mkstemp(prefix=f"refit_cfg_rf{a.rf}_", suffix=".json")
    os.close(fd)
    json.dump(cfg, open(tmp, "w"))

    model, tg, nl = build(tmp)
    model.to(dev)
    X, Y = tg.dataset.tensors[1], tg.dataset.tensors[4]
    p_count = sum(x.numel() for x in tr_params(model))
    print(f"[refit] rf={a.rf}  p={p_count}  num_labels={nl}  "
          f"||theta_tr||_init={tr_norm(model):.3f}")

    # Train to peak on the SECOND half; the first half stays the eval set, and
    # the re-fit below reuses the same train half so it can add no eval signal.
    tr = torch.arange(X.shape[0] // 2, X.shape[0])
    opt = torch.optim.AdamW(tr_params(model), lr=a.lr)
    g2 = torch.Generator().manual_seed(1)
    model.train()
    for ep in range(a.epochs):
        perm = tr[torch.randperm(len(tr), generator=g2)]
        for s in range(0, len(perm), 32):
            sel = perm[s:s + 32]
            o = model(X[sel].to(dev))
            lg = o.logits if hasattr(o, "logits") else o[0]
            CrossEntropyLoss()(lg.view(-1, nl), Y[sel].view(-1).to(dev)).backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        print(f"  epoch {ep}: acc={accuracy(model, X, Y, dev):.4f}")

    base = [p.detach().clone() for p in tr_params(model)]
    b = head_stats(model, X, Y, dev)
    print(f"[refit] trained: acc={b['acc']:.4f}  ||theta_tr||={tr_norm(model):.3f}\n")

    print(f"{'m':>5s}" + "".join(f"{'Phi=' + str(p):>10s}" for p in phis)
          + f"{'knee':>9s}")
    print("-" * (5 + 10 * len(phis) + 9))
    knees = {}
    for m in ms:
        gen = torch.Generator().manual_seed(11)
        rg = torch.Generator().manual_seed(7)
        accs = []
        for phi in phis:
            reps = a.reps if phi > 1.0 else 1
            got = []
            for _ in range(reps):
                perturb(model, base, phi, "noise", gen)
                refit(model, X, Y, tr, nl, dev, m, a.lr, rg)
                got.append(head_stats(model, X, Y, dev)["acc"])
            accs.append(sum(got) / len(got))
        k = knee(phis, accs, b["acc"], nl)
        knees[m] = k
        print(f"{m:5d}" + "".join(f"{v:10.4f}" for v in accs)
              + (f"{k:9.3f}" if k else f"{'n/a':>9s}"))

    # restore, so a caller can keep using the model
    perturb(model, base, 1.0, "noise", torch.Generator())

    print("\nknee vs m:  " + "  ".join(
        f"m={m}: {('%.3f' % knees[m]) if knees[m] else 'n/a'}" for m in ms))
    first, last = knees.get(ms[0]), knees.get(ms[-1])
    if last and last <= 1.6:
        print(f"FALSIFIED (P1'): knee still {last:.2f} <= 1.6 at m={ms[-1]} -- the "
              f"trajectory's junk is not isotropic and Phi* needs a training run")
    elif last and first and last > first:
        print(f"the knee MOVES with the re-fit: {first:.2f} -> {last:.2f}. "
              f"Compare against Phi* = 2.82 / 3.00 / 2.91 (§5.5)")


if __name__ == "__main__":
    main()
