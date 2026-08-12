#!/usr/bin/env python3
"""Rung-2 rig: is the shipped finite difference still measuring a directional derivative?

Built for H-S (fl_fwd_ft_solution.md 6.3). Imports the PRODUCTION FD -- this
calls `calculate_jvp` itself, it does not re-implement it (P7 rule 1).

    python probe_fd_chord.py --config <aggregator_config.json>

D-1 showed the 20x gap between the closed-form and measured cos is data-side and
constant, but the rig reproduces only L. It returns S = 1.68 where the arms read
0.48, so a 3.5x SHADOW loss sits somewhere in the pipeline rather than the data.

The suspect: `v` is a raw Gaussian draw, so the probe displacement is
h*||v|| = h*sqrt(p) = 6.711 -- against ||theta_tr|| = 6.75. Every probe steps
roughly a FULL PARAMETER NORM. Over a chord that long, (L(x+hv)-L(x-hv))/(2h) is
an average slope across the chord, not <g,v> at theta. A chord-averaged slope is
still Gaussian across v, which is why the JVP distribution check in P3 passed it
and why 1.3's ruling-out of the FD -- as a cause of DIVERGENCE, not of shadow
loss -- does not apply.

Same `v`, same bin, both ways: d_fd from the production FD, d_true = <g,v> from
a backward pass. Swept over displacement, with the shipped 6.711 marked.

PREDICTION (registered before the run): cos(d_fd, d_true) ~= 0.3 at the shipped
displacement if the chord is the cause, and -> 1 as the displacement shrinks.
SINKING CONDITION: cos >= 0.9 at 6.711 means the FD is faithful, H-S is wrong,
and the 3.5x is elsewhere -- omega weighting, oort selection, or staleness.
"""
import argparse
import json
import math
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", ".."))
import functorch as fc                                          # noqa: E402
import torch                                                    # noqa: E402
from torch.nn import CrossEntropyLoss                           # noqa: E402

from examples.fwdllm.data_preprocessing.text_classification_preprocessor import (
    TLMPreprocessor,
)                                                               # noqa: E402
from examples.fwdllm.trainer.forward_training.tc_transformer_trainer_distribute import (
    ForwardTextClassificationTrainer,
)                                                               # noqa: E402
from examples.fwdllm.trainer.model_args_builder import build_model_args   # noqa: E402
from examples.fwdllm.data_manager.text_classification_data_manager import (
    TextClassificationDataManager,
)                                                               # noqa: E402
from examples.fwdllm.data_manager.base_data_manager import BaseDataManager  # noqa: E402
from examples.fwdllm.expts.initializer import create_model      # noqa: E402
from examples.fwdllm.trainer.forward_training import fwdgrad_utils as FG   # noqa: E402


def build(cfg_path):
    hp = types.SimpleNamespace(**json.load(open(cfg_path))["hyperparameters"])
    attrs = BaseDataManager.load_attributes(hp.data_file_path)
    nl = len(attrs["label_vocab"])
    margs = build_model_args(hp, nl)
    _, model, tok = create_model(margs, formulation="classification")
    ForwardTextClassificationTrainer(margs, 0, model, None, None, "rig")  # drops pre_classifier
    pre = TLMPreprocessor(args=margs, label_vocab=attrs["label_vocab"], tokenizer=tok)
    dm = TextClassificationDataManager(hp, margs, pre, 1, hp.data_loader_num_workers)
    ds = dm.load_federated_data(process_id=1, client_idx=0)[4][0].dataset
    return model, (ds.tensors[1], ds.tensors[4]), nl, int(hp.train_batch_size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--displacements", type=float, nargs="+",
                    default=[6.711, 2.0, 0.6, 0.2, 0.06, 0.02])
    ap.add_argument("--probes", type=int, default=200)
    ap.add_argument("--bins", type=int, default=4)
    a = ap.parse_args()

    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, (X, Y), nl, B = build(a.config)
    model.to(dev).eval()
    fmodel, params, buffers = fc.make_functional_with_buffers(model)
    params = [p.to(dev) for p in params]
    tidx = [i for i, p in enumerate(params) if p.requires_grad]
    p_dim = sum(params[i].numel() for i in tidx)
    tw = math.sqrt(sum(float(params[i].detach().float().pow(2).sum()) for i in tidx))
    print(f"[rig] p={p_dim}  ||theta_tr||={tw:.3f}  bin={B}  probes/bin={a.probes}  dev={dev}")
    print(f"[rig] shipped displacement h*sqrt(p) = {0.01 * math.sqrt(450340):.3f}"
          f"  ({100 * 0.01 * math.sqrt(450340) / tw:.0f}% of ||theta_tr||)")

    os.environ["FWDLLM_FD_SCALE_INVARIANT"] = "1"   # so displacement is what we set

    print(f"\n{'h*sqrt(p)':>10s}{'% of |th|':>10s}{'cos(d_fd,d_true)':>18s}"
          f"{'rms d_fd':>11s}{'rms d_true':>12s}{'ratio':>8s}")
    for disp in a.displacements:
        FG._FD_REF_DISPLACEMENT = disp          # the one knob calculate_jvp reads
        FG._fd_p_cache.clear()
        dots = na = nb = 0.0
        rf = rt = 0.0
        n = 0
        for b in range(a.bins):
            sel = torch.arange(b * B, (b + 1) * B)
            x, t = X[sel].to(dev), Y[sel].view(-1).to(dev)

            # true gradient on this bin, fp32 -- the reference d_true = <g,v>
            pg = [params[i].detach().clone().requires_grad_(True) for i in tidx]
            full = list(params)
            for j, i in enumerate(tidx):
                full[i] = pg[j]
            CrossEntropyLoss()(fmodel(tuple(full), buffers, x)[0].view(-1, nl), t).backward()
            g = [q.grad.detach().clone() for q in pg]

            func = lambda P: FG.functional_get_loss(P, fmodel, x, t, nl, buffers)
            for _ in range(a.probes):
                v = [torch.zeros_like(q) for q in params]
                for j, i in enumerate(tidx):
                    v[i] = torch.randn_like(params[i])
                _, d_fd = FG.calculate_jvp(func, tuple(params), v, trainable_idx=tidx)
                d_true = sum(float((g[j] * v[i]).sum()) for j, i in enumerate(tidx))
                d_fd = float(d_fd)
                dots += d_fd * d_true; na += d_fd ** 2; nb += d_true ** 2
                rf += d_fd ** 2; rt += d_true ** 2; n += 1
        c = dots / math.sqrt(na * nb) if na > 0 and nb > 0 else float("nan")
        print(f"{disp:10.3f}{100 * disp / tw:10.0f}{c:18.4f}"
              f"{math.sqrt(rf / n):11.4f}{math.sqrt(rt / n):12.4f}"
              f"{math.sqrt(rf / rt):8.3f}")

    print("\nRead the top row: that is the shipped probe. cos ~= 0.3 => the chord is")
    print("eating the shadow and h becomes a first-class knob; cos >= 0.9 => H-S is")
    print("wrong and the missing 3.5x is in omega, selection or staleness.")


if __name__ == "__main__":
    main()
