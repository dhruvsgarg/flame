#!/usr/bin/env python3
"""H13 probe: is the forward-gradient pipeline reproducible, and what breaks it?
Minutes on one GPU, NO FL run.

Two same-seed real replicates of `felix_round` disagree by 13.3% on iters/bin and
11.2 accuracy points at peak, with data, dispatch order, iteration, model_version
and the per-client perturbation RNG all verified identical. Runs 1-2 falsified
arithmetic as the source (§E). What is left is DROPOUT: `create_model` +
`train_adapter` leaves 13 of DistilBERT's 20 `nn.Dropout` modules in training mode
at p=0.1, so `calculate_jvp` evaluates its two finite-difference passes under
DIFFERENT masks drawn from the process-global RNG -- which nothing seeds per task.

  SOURCE    -- does the same forward pass reproduce, in the mode production runs?
  AMPLIFIER -- how far does the measured loss spread move the jvp? (h=0.01)

DECISION RULE, fixed before running (simulate_fwdllm.md D-9, and printed by the
tool):
  * some arm goes bit-exact         => a BUG. Fix it; don't widen tolerances.
  * spread unchanged on every arm   => IRREDUCIBLE. Write the invariant, widen
                                       each tolerance to its floor, buy seeds.
  * `evalmode` alone closes it      => dropout is the source; the fix is in the
                                       trainer, not in tolerances.
  * `base` itself bit-exact         => the PROBE failed to reproduce and says
                                       nothing about any arm. Fix the probe.

Reports spread both WITHIN one process (repeats) and ACROSS concurrently-launched
ones: dropout shows up in the first, kernel selection only in the second.

    python probe_jvp_determinism.py --sweep --model real --replicas 8 --repeats 20
    python probe_jvp_determinism.py --repeats 20 --tag base      # one arm

Calls the REAL `calculate_jvp`, so whatever the trainer does, this does. Builds
the model in the mode production ACTUALLY runs it -- `--eval-mode` (arm
`evalmode`) is the A/B, not the default. Runs 1-2 forced `.eval()` and so read
bit-exact on every arm; that is what made them inconclusive.
`--model real` loads the actual DistilBERT+adapter stack; default `--model proxy`
is a small transformer on the same autocast/GEMM/dropout path. Results print as a
table and land in `probe_out/` as JSON (`sweep.json` = the merged comparison).
"""
from __future__ import annotations

import argparse
import glob
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
    "evalmode":  {"FWDLLM_PROBE_EVAL": "1"},
    "fp32":      {"FWDLLM_JVP_FP32": "1"},
    "determ":    {"FWDLLM_STRICT_DETERMINISM": "1"},
    "fp32determ": {"FWDLLM_JVP_FP32": "1", "FWDLLM_STRICT_DETERMINISM": "1"},
}


def _eval_mode_requested() -> bool:
    """`evalmode` arm: run the model in eval(). Off by default -- production does
    NOT call eval(), and forcing it is what made runs 1-2 read bit-exact."""
    return os.environ.get("FWDLLM_PROBE_EVAL", "").strip().lower() in ("1", "true", "yes")


def _dropout_census(model) -> int:
    """Dropout modules live (p>0 AND training) in the graph the probe evaluates."""
    import torch.nn as nn
    return sum(1 for m in model.modules()
               if isinstance(m, nn.Dropout) and m.p > 0 and m.training)


def _build_proxy(device, seed, eval_mode=False, d_model=256, n_layers=4,
                 vocab=1024, seq=128, dropout=0.1):
    """Small transformer on the same autocast/GEMM/dropout path as DistilBERT."""
    import torch
    import torch.nn as nn
    g = torch.Generator(device="cpu").manual_seed(seed)

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(vocab, d_model)
            layer = nn.TransformerEncoderLayer(
                d_model, nhead=4, dim_feedforward=4 * d_model,
                batch_first=True, dropout=dropout)
            self.enc = nn.TransformerEncoder(layer, n_layers)
            self.head = nn.Linear(d_model, 4)

        def forward(self, x):
            return (self.head(self.enc(self.emb(x)).mean(1)),)

    torch.manual_seed(seed)
    model = M().to(device)
    if eval_mode:
        model.eval()
    x = torch.randint(0, vocab, (8, seq), generator=g).to(device)
    t = torch.randint(0, 4, (8,), generator=g).to(device)
    return model, x, t


def _build_real(device, seed, eval_mode=False, seq=192, batch=8, n_labels=4):
    """The ACTUAL DistilBERT+adapter stack the trainer runs, in its mode.

    `train_adapter` leaves 13 of the 20 `nn.Dropout` modules training at p=0.1
    even though `model.training` reads False, so DO NOT call `.eval()` here: that
    silences the mechanism under test (`eval_mode` is the A/B arm, default off).

    Size is the point: the proxy's GEMMs are small enough that cuBLAS picks a
    single deterministic kernel every time, and its fp16 error is ~100x smaller
    than production's. Token ids are random -- the probe measures arithmetic, not
    accuracy, so only the SHAPES have to match the real batch.

    The "adapters available but none are activated" warning is EXPECTED and must
    not be "fixed": every production trainer logs it too (100/run), so silencing
    it here would make the probe diverge from the path under test.
    """
    import torch
    from expts.initializer import create_model
    from model.transformer.model_args import ClassificationArgs

    a = ClassificationArgs()
    a.model_name, a.model_type = "distilbert-base-uncased", "distilbert"
    a.load(a.model_name)
    a.num_labels = n_labels
    a.update_from_dict({"peft_method": "adapter", "do_lower_case": True,
                        "max_seq_length": seq, "manual_seed": seed})
    a.config = dict(getattr(a, "config", {}) or {})
    a.config["num_labels"] = n_labels
    torch.manual_seed(seed)
    _, model, _ = create_model(a, formulation="classification")
    model = model.to(device)
    if eval_mode:
        model.eval()
    g = torch.Generator(device="cpu").manual_seed(seed)
    vocab = model.get_input_embeddings().num_embeddings
    x = torch.randint(0, vocab, (batch, seq), generator=g).to(device)
    t = torch.randint(0, n_labels, (batch,), generator=g).to(device)
    return model, x, t


def _run_arm(args) -> dict:
    """One arm, in THIS process. Env flags must already be set."""
    import torch
    from expts.initializer import set_seed, strict_determinism_enabled
    from forward_training.fwdgrad_utils import calculate_jvp, jvp_fp32_enabled

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available()
                          or args.device == "cpu" else "cpu")
    build = _build_real if args.model == "real" else _build_proxy
    eval_mode = args.eval_mode or _eval_mode_requested()
    model, x, t = build(device, args.seed, eval_mode=eval_mode)
    # Perturb only trainable params, as the trainer does under `peft_method`.
    params = [p.detach().clone() for p in model.parameters()]
    trainable = [i for i, p in enumerate(model.parameters()) if p.requires_grad]
    _tset = set(trainable)

    # ONE fixed perturbation, drawn on CPU from a seeded generator -- identical in
    # every repeat and every arm, so anything that moves is arithmetic, not RNG.
    g = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    v = [(torch.randn(p.shape, generator=g) if i in _tset
          else torch.zeros(p.shape)).to(device) for i, p in enumerate(params)]

    ce = torch.nn.CrossEntropyLoss()

    def f(ps):
        out = torch.func.functional_call(
            model, {n: p for (n, _), p in zip(model.named_parameters(), ps)}, (x,))
        return ce(out[0], t)

    losses, jvps = [], []
    for _ in range(args.repeats):
        loss, jvp = calculate_jvp(f, params, v, trainable_idx=trainable)
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
        "model": args.model,
        "seed": args.seed,
        "eval_mode": eval_mode,
        "live_dropout": _dropout_census(model),
        "n_params": sum(p.numel() for p in params),
        "n_trainable": sum(params[i].numel() for i in trainable),
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
        "model": reps[0].get("model"),
        "cuda_device": reps[0].get("cuda_device"),
        "jvp_fp32": reps[0]["jvp_fp32"],
        "strict_determinism": reps[0]["strict_determinism"],
        "eval_mode": reps[0].get("eval_mode"),
        "live_dropout": reps[0].get("live_dropout"),
        # Replicas hold DIFFERENT seeds under --hetero, so their cross-process
        # spread is work, not nondeterminism; the report must not read it.
        "hetero": len({r.get("seed") for r in reps}) > 1,
        # Within one process (same seed, same kernel plan): the only thing that
        # can move here is per-pass RNG -- i.e. dropout.
        "within_loss_exact_frac": st.mean(r["loss_exact_frac"] for r in reps),
        "within_loss_rel_spread": st.mean(r["loss_rel_spread"] for r in reps),
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
    hetero = any(r.get("hetero") for r in rows)
    print("\n  WITHIN one process, same input repeated (dropout shows up here):")
    print(f"{'arm':<12}{'live drop':>10}{'loss exact':>12}{'loss spread':>14}"
          f"{'jvp spread':>13}")
    for r in rows:
        print(f"{r['arm']:<12}{str(r.get('live_dropout', '?')):>10}"
              f"{r['within_loss_exact_frac']:>11.0%}"
              f"{r['within_loss_rel_spread']:>14.2e}{r['within_jvp_rel_spread']:>13.2e}")
    if hetero:
        # Under --hetero every replica holds a different seed, so the table below
        # measures different WORK. Printing it as a determinism number is how a
        # `base` arm reads non-exact for a reason that has nothing to do with H13.
        print("\n  ACROSS concurrent processes: SUPPRESSED -- --hetero gives each "
              "replica its own seed,\n  so cross-process spread is different work, "
              "not nondeterminism. Use --compare for the\n  same-work diff, and drop "
              "--hetero to read this table.")
    else:
        print("\n  ACROSS concurrent processes (the replicate-floor analogue):")
        print(f"{'arm':<12}{'loss exact':>12}{'loss spread':>14}{'jvp exact':>11}"
              f"{'jvp spread':>13}{'amplif':>10}")
        for r in rows:
            amp = f"{r['amplification']:.0f}x" if r.get("amplification") else "--"
            print(f"{r['arm']:<12}{r['loss_exact_frac']:>11.0%}{r['loss_rel_spread']:>14.2e}"
                  f"{r['jvp_exact_frac']:>10.0%}{r['jvp_rel_spread']:>13.2e}{amp:>10}")
    base = next((r for r in rows if r["arm"] == "base"), None)
    fp32 = next((r for r in rows if r["arm"] == "fp32"), None)
    # The AMPLIFIER isolates the central difference's condition number by moving
    # ONE thing (fp16 -> fp32). Live dropout moves a second, so the ratio is only
    # readable with dropout off -- runs 1-2 measured it clean at 189x.
    if base and fp32 and base.get("live_dropout"):
        print("\n  AMPLIFIER: not readable -- dropout is live, so base and fp32 do "
              "not share an input.\n    Re-read it from the `evalmode` arm, or "
              "cite the 189x already on record (§B).")
    elif base and fp32:
        dl = abs(base["losses"][0] - fp32["losses"][0]) / max(abs(fp32["losses"][0]), 1e-30)
        dj = abs(base["jvps"][0] - fp32["jvps"][0]) / max(abs(fp32["jvps"][0]), 1e-30)
        if dl > 0:
            print("\n  AMPLIFIER (fp16 vs fp32, same input -- independent of any "
                  f"nondeterminism):\n    rel d(loss) {dl:.2e}  ->  rel d(jvp) "
                  f"{dj:.2e}   = {dj/dl:.0f}x")
        else:
            print("\n  AMPLIFIER: fp16 and fp32 agree bit-for-bit (nothing to amplify).")
    print()
    # Read the verdict off the column that actually shows spread. Within-process
    # wins ties: it is the tighter control (one process, one seed, one input), and
    # under --hetero it is the only determinism measure in the table.
    if base and (hetero or base.get("within_jvp_rel_spread", 0.0) > 0):
        key, scope = "within_jvp_rel_spread", "within process"
    else:
        key, scope = "jvp_rel_spread", "across processes"
    if (base and base.get("within_jvp_rel_spread", 0.0) > 0
            and not hetero and base["jvp_rel_spread"] == 0.0):
        print("  Cross-process reads exact only because every replica restarts the "
              "same global RNG stream;\n  production advances it by a "
              "task/eval count that timing decides. Read the within column.")
    # No baseline spread => the probe never reproduced the phenomenon, so NOTHING
    # here can be read as a fix. Say so instead of crediting every arm.
    if not base or base[key] == 0.0:
        print(f"  INCONCLUSIVE: the `base` arm is already bit-exact ({scope}), so "
              "this run did not reproduce\n  the phenomenon and no arm can be "
              "credited with fixing it. Check `live drop` above: 0 there means the "
              "model was built in eval mode and dropout, the standing source (H13), "
              "was silenced.")
        return
    for r in rows:
        if r["arm"] == "base":
            continue
        if r[key] == 0.0:
            print(f"  {r['arm']}: BIT-EXACT {scope} where base spread "
                  f"{base[key]:.2e} -- that is a BUG on this arm; "
                  f"fix rather than widen tolerances.")
        else:
            # Signed percentages read as improvements either way round; name the
            # direction. fp32 came back LARGER than base on probe C.
            ratio = r[key] / base[key]
            verb = "smaller than" if ratio < 1 else "LARGER than"
            print(f"  {r['arm']}: jvp spread {abs(1.0 - ratio):.0%} {verb} base "
                  f"({base[key]:.2e} -> {r[key]:.2e})")
    if base.get("live_dropout"):
        print("  With dropout live each arm draws its own mask stream, so only "
              "`evalmode` reaching 0 is\n  a signal -- do not read the other arms' "
              "spreads against each other.")
    if not hetero and rows and all(r.get("replicas", 1) < 2 for r in rows):
        print("\n  NOTE: --replicas 1 measures only WITHIN-process repeatability, "
              "which is not the live signature. Re-run with --replicas 4+.")
    if rows and rows[0].get("device", "").startswith("cpu"):
        print("\n  WARNING: ran on CPU -- autocast is disabled there, so the fp16 "
              "arms say nothing. Dropout still reproduces; use a GPU for the rest.")


def _compare_dirs(da: str, db: str) -> int:
    """Replica-by-replica diff of two sweeps -- the comparison production makes.

    Within one sweep, replicas are neighbours; across two sweeps, replica i is
    the SAME trainer run twice. Under `--hetero` only the cross-run form is
    meaningful, because neighbours deliberately hold different seeds.
    """
    print(f"\n  CROSS-RUN (replica i of A vs replica i of B -- same work, two launches):")
    print(f"{'arm':<12}{'replicas':>9}{'loss exact':>12}{'loss spread':>14}"
          f"{'jvp exact':>11}{'jvp spread':>13}")
    worst = 0.0
    for arm in _ARMS:
        pairs = []
        for fa in sorted(glob.glob(os.path.join(da, f"{arm}.r*.json"))):
            fb = os.path.join(db, os.path.basename(fa))
            if os.path.exists(fb):
                pairs.append((json.load(open(fa)), json.load(open(fb))))
        if not pairs:
            continue
        le = sum(1 for a, b in pairs if a["losses"][0] == b["losses"][0]) / len(pairs)
        je = sum(1 for a, b in pairs if a["jvps"][0] == b["jvps"][0]) / len(pairs)
        ls = max(abs(a["losses"][0] - b["losses"][0]) / max(abs(a["losses"][0]), 1e-30)
                 for a, b in pairs)
        js = max(abs(a["jvps"][0] - b["jvps"][0]) / max(abs(a["jvps"][0]), 1e-30)
                 for a, b in pairs)
        worst = max(worst, js)
        print(f"{arm:<12}{len(pairs):>9}{le:>11.0%}{ls:>14.2e}{je:>10.0%}{js:>13.2e}")
    if worst == 0.0:
        print("\n  Every arm reproduces across launches. Production's divergence is "
              "NOT arithmetic -- stop probing and hunt the differing INPUT.")
    else:
        print("\n  Reproduced across launches. The arm whose spread collapses is the fix.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model", choices=("proxy", "real"), default="proxy")
    ap.add_argument("--eval-mode", action="store_true",
                    help="build the model in eval() -- dropout off. NOT what "
                         "production does; this is the `evalmode` arm's A/B")
    ap.add_argument("--tag", default="base")
    ap.add_argument("--out-dir", default=str(_HERE / "probe_out"))
    ap.add_argument("--hetero", action="store_true",
                    help="give each replica seed+i so co-tenants do DIFFERENT "
                         "work, as production's 12-per-GPU trainers do. Compare "
                         "two --sweep runs with --compare, not replicas to each "
                         "other (their seeds differ by design)")
    ap.add_argument("--compare", nargs=2, metavar=("DIR_A", "DIR_B"),
                    help="diff two sweep out-dirs replica-by-replica: the "
                         "same-trainer-across-two-runs comparison production makes")
    ap.add_argument("--replicas", type=int, default=4,
                    help="concurrent PROCESSES per arm (--sweep). The "
                         "cross-process spread is the decision input; 1 measures "
                         "only within-process repeatability")
    ap.add_argument("--sweep", action="store_true",
                    help="re-exec once per arm (flags must precede CUDA init), "
                         "then print the comparison")
    args = ap.parse_args(argv)
    if args.compare:
        return _compare_dirs(*args.compare)
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
                       "--repeats", str(args.repeats),
                       "--seed", str(args.seed + i if args.hetero else args.seed),
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
