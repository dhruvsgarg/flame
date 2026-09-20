#!/usr/bin/env python3
"""Steps 1-2 of §5.8's porting order, for any model. Free -- no data, no run.

    ./probe_port_init.py --model-type roberta-large --model-name roberta-large --rf 16
    ./probe_port_init.py --rf 16 --rf 32 --rf 64          # the DistilBERT ladder

Builds the model exactly as `initializer.create_model` does for a real run --
same adapter config, same `pre_classifier` drop on distilbert -- and reports the
two numbers a port needs before anything else, plus the one ratio that decides
whether the FD probe steps the same relative distance on the new model:

    p               trainable parameter count, the PRODUCTION one
    ||theta_tr||    its norm at init
    chord ON/OFF    h*sqrt(p)/||theta_tr||, under both FWDLLM_FD_SCALE_INVARIANT states

**What this measured (2026-08-22, row N5a).** `||theta_tr||/sqrt(p)` is constant
at 0.0196-0.0199 across a 35.7x range in `p` AND across two architectures. That
makes the flag's effect mechanical and backwards from its intent: holding the
ABSOLUTE displacement `h*sqrt(p)` fixed (flag ON) makes the DIMENSIONLESS chord
go as `1/sqrt(p)` -- 6.1x across the four configs below -- while leaving `h` at
0.01 (flag OFF) holds it at 0.501-0.511, constant to 1%.

    config                  p        ||th_tr||   ||th||/sqrt(p)   ON      OFF
    distilbert rf=16        450,340     13.347      0.01989      0.503   0.503
    distilbert rf=32        229,012      9.471      0.01979      0.709   0.505
    distilbert rf=64        118,348      6.732      0.01957      0.997   0.511
    roberta-large rf=16   4,225,540     40.991      0.01994      0.164   0.501

The ON column reproduces §5.8's predicted 0.50 / 0.70 / 0.98 on the ladder, which
is what closed N5a. Read the caveats before acting on it: this is at INIT
(`||theta_tr||` grows 13.35 -> 61.9 over an agnews run, so the chord falls through
any run either way), and it says the flag does not do what its docstring claims --
NOT what that costs in gradient quality. That is H-S / `scripts/probe_fd_chord.py`.
"""
import argparse
import math
import sys
import types

sys.path.insert(0, "/home/dgarg39/flame/lib/python")
import torch  # noqa: E402

from examples.fwdllm.expts.initializer import create_model  # noqa: E402

# The invariant the flag holds fixed: h=0.01 at fluxtune's own trainable p.
_FD_REF_P = 450340
_FD_REF_DISPLACEMENT = 0.01 * math.sqrt(_FD_REF_P)


def measure(model_type, model_name, rf, num_labels):
    args = types.SimpleNamespace(
        model_type=model_type, model_name=model_name, num_labels=num_labels,
        config={"num_labels": num_labels}, do_lower_case=True,
        peft_method="adapter", adapter_reduction_factor=rf,
        trainable_scope="adapters_head")
    _cfg, model, _tok = create_model(args, formulation="classification")
    # The trainer drops pre_classifier before the probe is drawn, so production p
    # excludes it (tc_transformer_trainer_distribute.py:217).
    if model_type == "distilbert":
        model.add_module("pre_classifier", torch.nn.Sequential())
    tr = [p for p in model.parameters() if p.requires_grad]
    p = sum(x.numel() for x in tr)
    n = math.sqrt(sum(float(x.detach().pow(2).sum()) for x in tr))
    return p, n, sum(x.numel() for x in model.parameters())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-type", default="distilbert")
    ap.add_argument("--model-name", default=None,
                    help="defaults to --model-type, or distilbert-base-uncased")
    ap.add_argument("--rf", type=int, action="append",
                    help="repeat for a ladder; default 16")
    ap.add_argument("--num-labels", type=int, default=4)
    a = ap.parse_args()
    name = a.model_name or ("distilbert-base-uncased"
                            if a.model_type == "distilbert" else a.model_type)
    rfs = a.rf or [16]

    print(f"{'config':<24}{'p':>11}{'||th_tr||':>11}{'||th||/sqrt(p)':>16}"
          f"{'chord ON':>10}{'chord OFF':>11}")
    print("-" * 83)
    rows = []
    for rf in rfs:
        p, n, total = measure(a.model_type, name, rf, a.num_labels)
        on, off = _FD_REF_DISPLACEMENT / n, 0.01 * math.sqrt(p) / n
        rows.append((on, off))
        print(f"{a.model_type + ' rf=' + str(rf):<24}{p:>11,}{n:>11.3f}"
              f"{n / math.sqrt(p):>16.5f}{on:>10.3f}{off:>11.3f}")
    if len(rows) > 1:
        o = [r[0] for r in rows]
        f = [r[1] for r in rows]
        print(f"\nflag ON  spans {min(o):.3f}-{max(o):.3f} = {max(o) / min(o):.1f}x")
        print(f"flag OFF spans {min(f):.3f}-{max(f):.3f} = {max(f) / min(f):.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
