#!/usr/bin/env python3
"""H12 probe: is the forward-gradient pipeline reproducible run-to-run, and does
fp32/strict-determinism make it so? Minutes on one GPU, NO FL run.

Two same-seed real replicates of `felix_round` disagree by 13.3% on iters/bin and
11.2 accuracy points at peak, and it is not a seeding bug: the RNG stream and its
position match on every trainer's first task. What differs is arithmetic --
8 of 30 trainers reproduce their loss bit-exactly, 22 differ at ~1.6e-3 (fp16
autocast), and `calculate_jvp`'s central difference at h=0.01 amplifies that by a
median 72x into the gradient. This isolates both halves without the FL loop:

  SEED      -- does the same forward pass reproduce bit-exactly across processes?
  AMPLIFIER -- how far does the measured loss spread move the jvp?

DECISION RULE, fixed before running (simulate_fwdllm.md D-9, and printed by the
tool):
  * some arm goes bit-exact         => a BUG. Fix it; don't widen tolerances.
  * spread unchanged on every arm   => IRREDUCIBLE. Write the invariant, widen
                                       each tolerance to its floor, buy seeds.
  * fp32 alone closes it            => the amplifier is the whole story.
  * `base` itself bit-exact         => the PROBE failed to reproduce and says
                                       nothing about any arm. Fix the probe.

Measures spread ACROSS concurrently-launched processes, not repeats inside one:
co-location is what varies kernel selection, and a single process reuses one
kernel plan so it can read bit-exact while two trainers still disagree.

    python probe_jvp_determinism.py --sweep --replicas 8 --repeats 20
    python probe_jvp_determinism.py --repeats 20 --tag base      # one arm

Calls the REAL `calculate_jvp`, so whatever the trainer does, this does.
`--model real` loads the actual DistilBERT+adapter stack; default `--model proxy`
is a small transformer on the same autocast/GEMM path. Results print as a table
and land in `probe_out/` as JSON (`sweep.json` = the merged comparison).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_FWDLLM = _HERE.parent
sys.path.insert(0, str(_FWDLLM))
sys.path.insert(0, str(_FWDLLM / "trainer"))

_ARMS = {
    "base":      {},
    "fp32":      {"FWDLLM_JVP_FP32": "1"},
    "determ":    {"FWDLLM_STRICT_DETERMINISM": "1"},
    "fp32determ": {"FWDLLM_JVP_FP32": "1", "FWDLLM_STRICT_DETERMINISM": "1"},
}


def _build_proxy(device, seed, d_model=256, n_layers=4, vocab=1024, seq=128):
    """Small transformer on the same autocast/GEMM path as DistilBERT."""
    import torch
    import torch.nn as nn
    g = torch.Generator(device="cpu").manual_seed(seed)

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(vocab, d_model)
            layer = nn.TransformerEncoderLayer(
                d_model, nhead=4, dim_feedforward=4 * d_model,
                batch_first=True, dropout=0.0)
            self.enc = nn.TransformerEncoder(layer, n_layers)
            self.head = nn.Linear(d_model, 4)

        def forward(self, x):
            return (self.head(self.enc(self.emb(x)).mean(1)),)

    torch.manual_seed(seed)
    model = M().to(device).eval()
    x = torch.randint(0, vocab, (8, seq), generator=g).to(device)
    t = torch.randint(0, 4, (8,), generator=g).to(device)
    return model, x, t


def _run_arm(args) -> dict:
    """One arm, in THIS process. Env flags must already be set."""
    import torch
    from expts.initializer import set_seed, strict_determinism_enabled
    from forward_training.fwdgrad_utils import calculate_jvp, jvp_fp32_enabled

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available()
                          or args.device == "cpu" else "cpu")
    model, x, t = _build_proxy(device, args.seed)
    params = [p.detach().clone() for p in model.parameters()]

    # ONE fixed perturbation, drawn on CPU from a seeded generator -- identical in
    # every repeat and every arm, so anything that moves is arithmetic, not RNG.
    g = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    v = [torch.randn(p.shape, generator=g).to(device) for p in params]

    ce = torch.nn.CrossEntropyLoss()

    def f(ps):
        out = torch.func.functional_call(
            model, {n: p for (n, _), p in zip(model.named_parameters(), ps)}, (x,))
        return ce(out[0], t)

    losses, jvps = [], []
    for _ in range(args.repeats):
        loss, jvp = calculate_jvp(f, params, v)
        losses.append(float(loss))
        jvps.append(float(jvp))

    def spread(vals):
        lo, hi = min(vals), max(vals)
        return abs(hi - lo) / max(abs(hi), 1e-30)

    exact = sum(1 for l in losses if l == losses[0]) / len(losses)
    return {
        "arm": args.tag,
        "env": {k: os.environ.get(k, "") for k in
                ("FWDLLM_JVP_FP32", "FWDLLM_STRICT_DETERMINISM",
                 "CUBLAS_WORKSPACE_CONFIG")},
        "jvp_fp32": jvp_fp32_enabled(),
        "strict_determinism": strict_determinism_enabled(),
        "device": str(device),
        "torch": torch.__version__,
        "cuda_device": (torch.cuda.get_device_name(0)
                        if torch.cuda.is_available() else None),
        "repeats": args.repeats,
        "loss_mean": st.mean(losses),
        "loss_exact_frac": exact,
        "loss_rel_spread": spread(losses),
        "jvp_mean": st.mean(jvps),
        "jvp_exact_frac": sum(1 for j in jvps if j == jvps[0]) / len(jvps),
        "jvp_rel_spread": spread(jvps),
        # The condition number the central difference actually realised.
        "amplification": (spread(jvps) / spread(losses)) if spread(losses) > 0 else None,
        "losses": losses,
        "jvps": jvps,
    }


def _merge_replicas(arm: str, reps: list) -> dict:
    """Pool N concurrent PROCESSES into one arm row.

    The cross-process spread is the quantity that matters: a single process
    reuses one kernel plan for every repeat, so it can read bit-exact while two
    separately-launched trainers still disagree -- which is exactly the live
    signature (8 of 30 trainers bit-identical, 22 differing at ~1.6e-3).
    """
    def spread(vals):
        lo, hi = min(vals), max(vals)
        return abs(hi - lo) / max(abs(hi), 1e-30)

    losses = [r["losses"][0] for r in reps]      # first repeat of each process
    jvps = [r["jvps"][0] for r in reps]
    ls, js = spread(losses), spread(jvps)
    return {
        "arm": arm,
        "replicas": len(reps),
        "repeats": reps[0]["repeats"],
        "device": reps[0]["device"],
        "cuda_device": reps[0].get("cuda_device"),
        "jvp_fp32": reps[0]["jvp_fp32"],
        "strict_determinism": reps[0]["strict_determinism"],
        # Within one process (kernel plan fixed) -- the weaker signal.
        "within_loss_exact_frac": st.mean(r["loss_exact_frac"] for r in reps),
        "within_jvp_rel_spread": st.mean(r["jvp_rel_spread"] for r in reps),
        # ACROSS processes -- the replicate-floor analogue, and the decision input.
        "loss_exact_frac": sum(1 for l in losses if l == losses[0]) / len(losses),
        "loss_rel_spread": ls,
        "jvp_exact_frac": sum(1 for j in jvps if j == jvps[0]) / len(jvps),
        "jvp_rel_spread": js,
        "amplification": (js / ls) if ls > 0 else None,
        "losses": losses,
        "jvps": jvps,
    }


def _report(rows: list) -> None:
    print("\n  ACROSS concurrent processes (the replicate-floor analogue):")
    print(f"{'arm':<12}{'loss exact':>12}{'loss spread':>14}{'jvp exact':>11}"
          f"{'jvp spread':>13}{'amplif':>10}")
    for r in rows:
        amp = f"{r['amplification']:.0f}x" if r.get("amplification") else "--"
        print(f"{r['arm']:<12}{r['loss_exact_frac']:>11.0%}{r['loss_rel_spread']:>14.2e}"
              f"{r['jvp_exact_frac']:>10.0%}{r['jvp_rel_spread']:>13.2e}{amp:>10}")
    base = next((r for r in rows if r["arm"] == "base"), None)
    print()
    # No baseline spread => the probe never reproduced the phenomenon, so NOTHING
    # here can be read as a fix. Say so instead of crediting every arm.
    if not base or base["jvp_rel_spread"] == 0.0:
        print("  INCONCLUSIVE: the `base` arm is already bit-exact, so this run "
              "did not reproduce H12 and no arm can be credited with fixing it.\n"
              "  Reproduce first (GPU, --replicas >= 4, ideally under load); only "
              "then do the other arms mean anything.")
        return
    for r in rows:
        if r["arm"] == "base":
            continue
        if r["loss_exact_frac"] == 1.0 and r["jvp_exact_frac"] == 1.0:
            print(f"  {r['arm']}: BIT-EXACT across processes where base spread "
                  f"{base['jvp_rel_spread']:.2e} -- H12 is a BUG on this arm; "
                  f"fix rather than widen tolerances.")
        else:
            cut = 1.0 - r["jvp_rel_spread"] / base["jvp_rel_spread"]
            print(f"  {r['arm']}: jvp spread {cut:+.0%} vs base "
                  f"({base['jvp_rel_spread']:.2e} -> {r['jvp_rel_spread']:.2e})")
    if rows and all(r.get("replicas", 1) < 2 for r in rows):
        print("\n  NOTE: --replicas 1 measures only WITHIN-process repeatability, "
              "which is not the live signature. Re-run with --replicas 4+.")
    if rows and rows[0].get("device", "").startswith("cpu"):
        print("\n  WARNING: ran on CPU -- autocast is disabled there, so this "
              "cannot reproduce the fp16 effect. Use a GPU node.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model", choices=("proxy", "real"), default="proxy")
    ap.add_argument("--tag", default="base")
    ap.add_argument("--out-dir", default=str(_HERE / "probe_out"))
    ap.add_argument("--replicas", type=int, default=4,
                    help="concurrent PROCESSES per arm (--sweep). The "
                         "cross-process spread is the decision input; 1 measures "
                         "only within-process repeatability")
    ap.add_argument("--sweep", action="store_true",
                    help="re-exec once per arm (flags must precede CUDA init), "
                         "then print the comparison")
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    if args.sweep:
        rows = []
        for arm, env in _ARMS.items():
            print(f"[probe] arm={arm} env={env or '{}'} replicas={args.replicas}")
            # Replicas run CONCURRENTLY on one GPU: co-location is what makes
            # cuBLAS/autocast kernel selection vary, and a serial re-run would
            # measure a quieter machine than production (100 trainers/8 GPUs).
            procs = []
            for i in range(args.replicas):
                cmd = [sys.executable, __file__, "--tag", f"{arm}.r{i}",
                       "--repeats", str(args.repeats), "--seed", str(args.seed),
                       "--device", args.device, "--model", args.model,
                       "--out-dir", args.out_dir]
                procs.append(subprocess.Popen(cmd, env={**os.environ, **env}))
            if any(p.wait() != 0 for p in procs):
                print(f"[probe] arm {arm} FAILED")
                continue
            reps = [json.load(open(os.path.join(args.out_dir, f"{arm}.r{i}.json")))
                    for i in range(args.replicas)]
            rows.append(_merge_replicas(arm, reps))
        _report(rows)
        json.dump(rows, open(os.path.join(args.out_dir, "sweep.json"), "w"), indent=1)
        return 0

    res = _run_arm(args)
    path = os.path.join(args.out_dir, f"{args.tag}.json")
    json.dump(res, open(path, "w"), indent=1)
    print(f"[probe] {args.tag}: loss_exact={res['loss_exact_frac']:.0%} "
          f"loss_spread={res['loss_rel_spread']:.3e} "
          f"jvp_exact={res['jvp_exact_frac']:.0%} "
          f"jvp_spread={res['jvp_rel_spread']:.3e} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
