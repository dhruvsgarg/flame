#!/usr/bin/env python3
"""Fluxtune/fwdllm forward-grad JVP optimization profiler (TEMP, real-code reuse).

Isolated latency + correctness profiler for the perturbation/JVP compute that
dominates fluxtune GPU time (`_train_one_batch` in
`trainer/forward_training/tc_transformer_trainer_distribute.py`). Measures the
gain and validates the numerical fidelity of each optimization stage before
changing the real trainer.

Deliberately REUSES the production code so a validated speedup transfers directly:
the real model (`expts.initializer.create_model` → DistilBERT-base + AdapterHub
bottleneck adapters, backbone frozen) and the real JVP math
(`fwdgrad_utils.calculate_jvp` / `functional_get_loss`). Targets only the current
fluxtune/fwdllm config (distilbert-base-uncased, agnews→4 labels, seq 192,
batch 8, adapter PEFT, perturbation_count 10) — not generic.

STAGES (each vs Stage 0 as the numerical ground truth):
  0 baseline   the real `calculate_jvp` in a Python loop over perturbations
               (central finite difference, 2 forward passes each = the real code)
  1 vmap-FD    the SAME finite-difference JVP, all perturbations batched via
               torch.func.vmap (frozen params broadcast, only adapters vary)
  2 fwd-AD     exact forward-mode-AD JVP (torch.func.jvp) — DIFFERENT math
               (true directional derivative, not a finite difference); a small
               systematic gap vs stage 0 is EXPECTED (the O(h^2) FD error).

Per stage: forward-pass count, latency (mean±std over repeats, warmup +
cuda.sync), speedup vs baseline, max|Δjvp| / max rel-diff vs baseline, and a
verdict: BIT-IDENTICAL / WITHIN-TOL / DIVERGED.

Run in the test_fwdllm env:
    conda run -n test_fwdllm python scripts/profile_jvp_opt.py
    conda run -n test_fwdllm python scripts/profile_jvp_opt.py --no-autocast
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from functools import partial
from pathlib import Path

import torch

# lib/python on the path so `examples.fwdllm...` imports resolve
_LIB_PYTHON = Path(__file__).resolve().parents[3]
if str(_LIB_PYTHON) not in sys.path:
    sys.path.insert(0, str(_LIB_PYTHON))

import functorch as fc  # noqa: E402
from torch.func import jvp as func_jvp  # noqa: E402
from torch.func import vmap  # noqa: E402

from examples.fwdllm.expts.initializer import create_model, set_seed  # noqa: E402
from examples.fwdllm.trainer.model.transformer.model_args import (  # noqa: E402
    ClassificationArgs,
)
from examples.fwdllm.trainer.forward_training.fwdgrad_utils import (  # noqa: E402
    calculate_jvp,
    functional_get_loss,
)


# ─────────────────────────── real model (reused) ───────────────────────────
def build_model(num_labels, seq, attn="sdpa"):
    """Exactly the trainer's construction path (main.py): DistilBERT-base +
    AdapterHub adapter, backbone frozen, only adapters + head trainable.

    `attn`: "sdpa" (the real default, fastest) or "eager" — forward-mode AD
    (stage 2) is not implemented for scaled_dot_product_attention, so it needs
    eager attention to run at all."""
    ma = ClassificationArgs()
    ma.model_name = "distilbert-base-uncased"
    ma.model_type = "distilbert"
    ma.load(ma.model_name)  # no-op for a hub name (no local model_args.json)
    ma.num_labels = num_labels
    ma.client_idx = 0
    ma.config["attn_implementation"] = attn
    ma.update_from_dict(
        {
            "peft_method": "adapter",       # BnConfig, output_adapter per layer
            "do_lower_case": True,
            "max_seq_length": seq,
            "train_batch_size": 8,
            "manual_seed": 42,
            "var_control": True,
            "perturbation_sampling": True,
            "select_perturbation_using_jvp": True,   # fluxtune JVP-selection path
            "fp16": True,
            "reprocess_input_data": False,
            "overwrite_output_dir": True,
        }
    )
    ma.config["num_labels"] = num_labels
    _, model, _ = create_model(ma, formulation="classification")
    return model


# ─────────────────────────── JVP stage implementations ───────────────────────────
def _loss_partial(fmodel, buffers, num_labels, x, labels):
    # == _compute_forward_jvp's `f`: functional_get_loss(params, ...)
    return partial(
        functional_get_loss,
        model=fmodel,
        buffers=buffers,
        num_classes=num_labels,
        x=x,
        t=labels,
    )


def stage0_baseline(f, params, V, h):
    """The real code: calculate_jvp per perturbation (2 forward passes each)."""
    jvps = []
    for v in V:                       # V: list[tuple(dir per param)]
        _loss, jvp = calculate_jvp(f, params, v)
        jvps.append(jvp)
    return torch.stack([j.float().reshape(()) for j in jvps])


def stage0b_trainable_only(fmodel, buffers, num_labels, x, labels, params, V,
                           train_idx, h, device):
    """Same central-FD JVP but perturb ONLY the trainable params (frozen p-h*0=p
    is skipped — the frozen tensors are reused as-is). Bit-identical to stage 0
    (p-0.0=p in IEEE), avoids copying the 98.5% frozen backbone twice/perturbation."""
    jvps = []
    with torch.no_grad():
        for v in V:
            plus = list(params)
            minus = list(params)
            for i in train_idx:
                plus[i] = params[i] + h * v[i]
                minus[i] = params[i] - h * v[i]
            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                lp = functional_get_loss(tuple(plus), model=fmodel, buffers=buffers,
                                         num_classes=num_labels, x=x, t=labels)
                lm = functional_get_loss(tuple(minus), model=fmodel, buffers=buffers,
                                         num_classes=num_labels, x=x, t=labels)
            jvps.append((lp - lm) / (2 * h))
    return torch.stack([j.float().reshape(()) for j in jvps])


def stage1_vmap_fd(fmodel, buffers, num_labels, x, labels, params, train_idx,
                   V_train, h, autocast_on, device):
    """Same central-FD JVP, all perturbations batched via vmap. Only the
    trainable adapter dirs get a leading P dim (frozen params broadcast) so we
    never materialise P copies of the 66M frozen backbone."""
    def jvp_one(vt_tuple):            # vt_tuple: per-trainable dir (no P dim inside vmap)
        plus = list(params)
        minus = list(params)
        for k, i in enumerate(train_idx):
            plus[i] = params[i] + h * vt_tuple[k]
            minus[i] = params[i] - h * vt_tuple[k]
        with torch.autocast(device_type=device.type, enabled=autocast_on):
            lp = functional_get_loss(tuple(plus), model=fmodel, buffers=buffers,
                                     num_classes=num_labels, x=x, t=labels)
            lm = functional_get_loss(tuple(minus), model=fmodel, buffers=buffers,
                                     num_classes=num_labels, x=x, t=labels)
        return ((lp - lm) / (2 * h)).float().reshape(())

    in_dims = (tuple(0 for _ in train_idx),)
    return vmap(jvp_one, in_dims=in_dims)(V_train)


def stage2_forward_mode(f, params, V, h):
    """Exact forward-mode AD (torch.func.jvp): the true directional derivative,
    not a finite difference. Different math -> a small O(h^2) gap vs stage 0 is
    expected; answers 'is exact fwd-AD faster/more accurate than 2 FD passes?'."""
    jvps = []
    for v in V:
        _loss, jv = func_jvp(f, (params,), (v,))
        jvps.append(jv)
    return torch.stack([j.float().reshape(()) for j in jvps])


# ─────────────────────────────── harness ───────────────────────────────
def _time(fn, warmup, repeats, device):
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.mean(ts), (statistics.stdev(ts) if len(ts) > 1 else 0.0)


def _measure(fn, warmup, repeats, device):
    """Latency (mean±std ms) AND peak CUDA memory (MB) for one call of fn."""
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    peak = (torch.cuda.max_memory_allocated() / 1024**2) if device.type == "cuda" else 0.0
    return statistics.mean(ts), (statistics.stdev(ts) if len(ts) > 1 else 0.0), peak


def backprop_ref(num_labels, seq, batch, h, seed, attn, autocast_on, device):
    """Standard BACKPROP fine-tune step (1 forward + 1 backward on the trainable
    adapters/head) — the memory/time contrast forward-grad is chosen to avoid.
    Fresh model so functorch's make_functional does not interfere."""
    model = build_model(num_labels, seq, attn).to(device).eval()
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randint(0, 30522, (batch, seq), generator=g).to(device)
    labels = torch.randint(0, num_labels, (batch,), generator=g).to(device)

    def step():
        model.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=autocast_on):
            out = model(x)
            logits = out[0] if not hasattr(out, "logits") else out.logits
            loss = torch.nn.functional.cross_entropy(logits.float(), labels)
        loss.backward()
    return step


def _verdict(ref, got):
    if ref.shape != got.shape:
        return "SHAPE-MISMATCH", float("nan"), float("nan")
    if torch.equal(ref, got):
        return "BIT-IDENTICAL", 0.0, 0.0
    mad = (ref - got).abs().max().item()
    rel = ((ref - got).abs() / (ref.abs() + 1e-12)).max().item()
    tag = "WITHIN-TOL" if torch.allclose(ref, got, rtol=1e-3, atol=1e-4) else "DIVERGED"
    return tag, mad, rel


def _fp64_check(attn, num_labels, seq, batch, P, h, seed, device):
    """Prove the fp32 vmap-FD divergence is FINITE-DIFFERENCE CANCELLATION, not
    a batching bug: in fp64 the sequential and vmap FD JVPs should agree to
    ~machine eps. Also prints the JVP signal magnitude so 'DIVERGED' on tiny
    (lp-lm) values is interpretable."""
    model = build_model(num_labels, seq, attn).to(device).double().eval()
    mask = [p.requires_grad for p in model.parameters()]
    fmodel, params, buffers = fc.make_functional_with_buffers(model)
    train_idx = [i for i, m in enumerate(mask) if m]
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randint(0, 30522, (batch, seq), generator=g).to(device)
    labels = torch.randint(0, num_labels, (batch,), generator=g).to(device)
    gp = torch.Generator(device=device).manual_seed(seed + 1)
    V, per_train = [], [[] for _ in train_idx]
    for _ in range(P):
        v = [torch.randn(p.shape, generator=gp, device=device, dtype=p.dtype)
             if mask[i] else torch.zeros_like(p) for i, p in enumerate(params)]
        V.append(tuple(v))
        for k, i in enumerate(train_idx):
            per_train[k].append(v[i])
    V_train = tuple(torch.stack(per_train[k]) for k in range(len(train_idx)))
    f = _loss_partial(fmodel, buffers, num_labels, x, labels)
    seq_jvp = stage0_baseline(f, params, V, h).double()
    vmap_jvp = stage1_vmap_fd(fmodel, buffers, num_labels, x, labels, params,
                              train_idx, V_train, h, False, device).double()
    tag, mad, rel = _verdict(seq_jvp, vmap_jvp)
    print(f"  fp64  seq-FD vs vmap-FD : {tag}  max|Δ|={mad:.2e} rel={rel:.2e}")
    print(f"  jvp magnitude (fp64)   : |jvp| in "
          f"[{seq_jvp.abs().min():.2e}, {seq_jvp.abs().max():.2e}]  "
          f"(FD subtracts losses ~O(1) -> cancellation floors fp32 precision)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--perturbations", type=int, default=10)   # perturbation_count
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=192)
    ap.add_argument("--labels", type=int, default=4)           # agnews
    ap.add_argument("--h", type=float, default=0.01)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--no-autocast", action="store_true",
                    help="fp32 (default: autocast, matching the real fp16 code)")
    ap.add_argument("--attn", choices=["sdpa", "eager"], default="sdpa",
                    help="eager needed for forward-mode AD (stage 2); sdpa is the real default")
    ap.add_argument("--fp64-check", action="store_true",
                    help="run the fp64 seq-vs-vmap cancellation diagnosis")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autocast_on = (not args.no_autocast) and device.type == "cuda"
    set_seed(args.seed)

    model = build_model(args.labels, args.seq, args.attn).to(device).eval()
    trainable_mask = [p.requires_grad for p in model.parameters()]
    fmodel, params, buffers = fc.make_functional_with_buffers(model)
    train_idx = [i for i, m in enumerate(trainable_mask) if m]

    total = sum(p.numel() for p in params)
    trainable = sum(params[i].numel() for i in train_idx)

    # fixed-seed inputs (the JVP-METHOD comparison is data-independent) — the
    # real forward is model(params, buffers, x)[0]; x = token ids, no mask.
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    x = torch.randint(0, 30522, (args.batch, args.seq), generator=g).to(device)
    labels = torch.randint(0, args.labels, (args.batch,), generator=g).to(device)

    # perturbations: randn on trainable, zero on frozen (== _prepare_perturbation_tensors)
    gp = torch.Generator(device=device).manual_seed(args.seed + 1)
    V = []
    V_train = []  # trainable-only stacked view for vmap
    per_train = [[] for _ in train_idx]
    for _ in range(args.perturbations):
        v = []
        for i, p in enumerate(params):
            if trainable_mask[i]:
                d = torch.randn(p.shape, generator=gp, device=device, dtype=p.dtype)
            else:
                d = torch.zeros_like(p)
            v.append(d)
        V.append(tuple(v))
        for k, i in enumerate(train_idx):
            per_train[k].append(v[i])
    V_train = tuple(torch.stack(per_train[k]) for k in range(len(train_idx)))  # each (P, *shape)

    f = _loss_partial(fmodel, buffers, args.labels, x, labels)

    print("=" * 94)
    print("fluxtune/fwdllm JVP optimization profiler  (TEMP — reuses create_model + calculate_jvp)")
    print(f"device={device.type}  autocast={autocast_on}  seed={args.seed}")
    print(f"model: distilbert-base + AdapterHub adapters | params={total/1e6:.1f}M  "
          f"trainable={trainable/1e6:.3f}M ({100*trainable/total:.1f}%)  [{len(train_idx)} tensors]")
    print(f"batch={args.batch}  seq={args.seq}  P(perturbations)={args.perturbations}  "
          f"h={args.h}  repeats={args.repeats} (+{args.warmup} warmup)")
    print("=" * 94)
    print(f"{'stage':22s} {'fwd/batch':>9s} {'ms/batch':>13s} {'speedup':>8s} "
          f"{'max|Δjvp|':>11s} {'max rel':>10s}  verdict")
    print("-" * 94)

    stages = [
        ("0 baseline (calc_jvp)", 2 * args.perturbations,
         lambda: stage0_baseline(f, params, V, args.h)),
        ("0b FD trainable-only", 2 * args.perturbations,
         lambda: stage0b_trainable_only(fmodel, buffers, args.labels, x, labels,
                                        params, V, train_idx, args.h, device)),
        ("1 vmap-FD (all pert)", 2 * args.perturbations,
         lambda: stage1_vmap_fd(fmodel, buffers, args.labels, x, labels, params,
                                train_idx, V_train, args.h, autocast_on, device)),
        ("2 forward-mode AD", 1 * args.perturbations,
         lambda: stage2_forward_mode(f, params, V, args.h)),
    ]

    base = None
    for name, n_pass, fn in stages:
        try:
            out = fn().detach().float()
            ms, sd = _time(fn, args.warmup, args.repeats, device)
        except Exception as e:
            print(f"{name:22s} {n_pass:>9d} {'ERR':>13s}         "
                  f"{type(e).__name__}: {str(e)[:44]}")
            continue
        if base is None:
            base = (ms, out)
            print(f"{name:22s} {n_pass:>9d} {ms:>9.2f}±{sd:>4.1f} {'1.00x':>8s} "
                  f"{'—':>11s} {'—':>10s}  REFERENCE")
        else:
            verdict, mad, rel = _verdict(base[1], out)
            print(f"{name:22s} {n_pass:>9d} {ms:>9.2f}±{sd:>4.1f} {base[0]/ms:>6.2f}x "
                  f"{mad:>11.2e} {rel:>10.2e}  {verdict}")

    print("-" * 94)

    # ── COST COMPARISON: sync vs fluxtune (perturbation sweep) vs backprop ──
    # Both baselines run the SAME _train_one_batch; the difference is the
    # perturbation-SELECTION cost: fwdllm/sync uses cos-sim (0 forward passes,
    # 1 final JVP=2 passes); fluxtune does JVP-selection (2*P passes) + 1 final.
    # We retain ONLY fidelity-preserving (bit-identical) cuts: drop the 3
    # diagnostic-only passes (both) + reuse the winner JVP (fluxtune) => the
    # per-batch forward-pass count below.
    # Measure the two per-forward-pass unit costs + peak memory (memory is flat
    # in P: no autograd graph, one forward held at a time). Totals are then
    # passes x per-pass — transparent projection.
    P = args.perturbations
    fms, _, fmem = _measure(lambda: stage0_baseline(f, params, V[:P], args.h),
                            args.warmup, args.repeats, device)
    oms, _, omem = _measure(
        lambda: stage0b_trainable_only(fmodel, buffers, args.labels, x, labels,
                                       params, V[:P], train_idx, args.h, device),
        args.warmup, args.repeats, device)
    full_pp, opt_pp = fms / (2 * P), oms / (2 * P)   # ms per forward pass
    print("COST MODEL — forward-grad (finite-difference JVP), per training batch")
    print(f"per-pass: full-param={full_pp:.2f}ms  trainable-only={opt_pp:.2f}ms "
          f"(1.{round(100*(full_pp/opt_pp-1)):02d}x, BIT-IDENTICAL) | peak mem "
          f"full={fmem:.0f}MB opt={omem:.0f}MB (flat in P)")
    print(f"{'path':32s} {'fwd passes':>10s} {'ms/batch':>10s}  note")
    print("-" * 94)
    # (passes, per-pass, note): current uses full-param; opt uses trainable-only
    rows = [
        ("sync fwdllm (current)", 5, full_pp, "cos-sim select(0) + 1 JVP + 3 diag"),
        ("sync fwdllm (opt)", 2, opt_pp, "1 JVP trainable-only; -3 diag  [RETAIN]"),
        ("fluxtune P=1 (opt)", 2, opt_pp, "== sync when P=1"),
        ("fluxtune P=2 (opt)", 4, opt_pp, ""),
        ("fluxtune P=5 (opt)", 10, opt_pp, ""),
        (f"fluxtune P={P} (current)", 2 * P + 5, full_pp, "2P + 2 redundant + 3 diag"),
        (f"fluxtune P={P} (opt)", 2 * P, opt_pp, "2P trainable-only; winner reused; -3 diag  [RETAIN]"),
    ]
    for name, npass, pp, note in rows:
        print(f"{name:32s} {npass:>10d} {npass*pp:>8.1f}  {note}")

    bp = backprop_ref(args.labels, args.seq, args.batch, args.h, args.seed,
                      args.attn, autocast_on, device)
    bms, bsd, bmem = _measure(bp, args.warmup, args.repeats, device)
    print(f"{'backprop ref (1 fwd+1 bwd)':30s} {'1f+1b':>10s} {bms:>9.1f}±{bsd:>3.1f} {bmem:>9.0f}"
          f"  contrast: stores activations")
    print("-" * 94)
    print(f"forward-grad peak mem ≈ model+1 forward (no autograd graph); backprop "
          f"stores activations ({bmem:.0f} MB).")
    print("-" * 94)

    if args.fp64_check:
        print("numerical diagnosis:")
        _fp64_check(args.attn, args.labels, args.seq, args.batch,
                    args.perturbations, args.h, args.seed, device)
        print("-" * 94)
    print("stage 0 = numerical ground truth (the real calculate_jvp loop). Real fluxtune per")
    print("batch also does +2 redundant + 3 diagnostic-only passes (removable, bit-identical)")
    print("on top of P*2 selection passes.  fwdllm uses cos-sim selection (no JVP-selection loop).")
    print("forward-mode AD (stage 2) needs --attn eager (not impl for SDPA).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
