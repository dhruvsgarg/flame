#!/usr/bin/env python3
# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Render the SOCC-2026 paper figures from a baseline→run-dir mapping.

The mapping is data: `figs.yaml` maps each baseline to a run dir, overridable with
`--run KEY=PATH`. Missing baselines are skipped. Output goes straight into
`paper_figs/` (stable basenames, overwritten each render) + a manifest.json.

    python make_paper_figs.py [--run fluxtune=/data/run_...] [--figures e1_acc_vs_time,...]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from plotlib import baselines as B          # noqa: E402
from plotlib import figures as F            # noqa: E402
from plotlib import reducers as R           # noqa: E402
from plotlib import style as S              # noqa: E402

# run dirs in the manifest are resolved relative to the fwdllm example dir
EXAMPLE_DIR = os.path.abspath(os.path.join(HERE, ".."))


def _resolve(path: str) -> str:
    if os.path.isabs(path):
        return path
    for base in (os.getcwd(), EXAMPLE_DIR):
        cand = os.path.join(base, path)
        if os.path.exists(cand):
            return os.path.abspath(cand)
    return os.path.abspath(os.path.join(EXAMPLE_DIR, path))  # best effort


def _load_manifest(path: str) -> dict:
    if not path or not os.path.exists(path):
        return {}
    try:
        import yaml
        with open(path) as fh:
            return yaml.safe_load(fh) or {}
    except Exception as e:  # noqa: BLE001
        print(f"  [paper-figs] could not read manifest {path}: {e}", file=sys.stderr)
        return {}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=os.path.join(HERE, "figs.yaml"),
                    help="baseline→run-dir manifest (default: expt_scripts/figs.yaml)")
    ap.add_argument("--run", action="append", default=[], metavar="KEY=PATH",
                    help="add/override one baseline's run dir (repeatable)")
    ap.add_argument("--out-root", default=os.path.join(HERE, "paper_figs"),
                    help="output dir (flat, overwritten each render)")
    ap.add_argument("--figures", default=None,
                    help="comma list of figure names (default: all)")
    ap.add_argument("--target-acc", type=float, default=None)
    ap.add_argument("--baseline-key", default=None,
                    help="'true baseline' for the E1/E3 speedup callouts "
                         "(default: fwdllm, or the manifest's speedup_baseline)")
    ap.add_argument("--champion-key", default=None,
                    help="our system for the E1/E3 speedup callouts "
                         "(default: fluxtune, or the manifest's speedup_champion)")
    ap.add_argument("--smooth", type=float, default=0.7,
                    help="EMA line-smoothing factor ∈ [0,1) for learning curves "
                         "(visual only; 0 = raw). Default 0.7")
    ap.add_argument("--loss-plateau-rel", type=float, default=0.01,
                    help="cut each run at its last cumulative test-loss drop of this "
                         "relative size — the deterministic end of productive "
                         "learning (default 0.01 = 1%%; the stall-guard rule)")
    ap.add_argument("--post-peak-grace-min", type=float, default=0.0,
                    help="minutes of tail to keep past the loss-plateau point "
                         "(default 0 — the plateau is deterministic, no buffer needed)")
    ap.add_argument("--no-cutoff", action="store_true",
                    help="disable the cutoff (use full telemetry)")
    ap.add_argument("--cutoff-mode", choices=["peak_acc", "loss_plateau"],
                    default="peak_acc",
                    help="where to clip each run: 'peak_acc' (default) at its "
                         "highest-accuracy eval — every baseline stops very close "
                         "to its best accuracy, clipping post-peak drift/divergence; "
                         "'loss_plateau' at the last significant test-loss drop. "
                         "A run that never learned (flat loss) is never clipped.")
    args = ap.parse_args()
    grace_s = None if args.no_cutoff else args.post_peak_grace_min * 60.0

    manifest = _load_manifest(args.manifest)
    runs_map = dict(manifest.get("runs", {}) or {})
    target = args.target_acc if args.target_acc is not None else manifest.get("target_acc")
    baseline_key = args.baseline_key or manifest.get("speedup_baseline") or "fwdllm"
    champion_key = args.champion_key or manifest.get("speedup_champion") or "fluxtune"
    savings_from = manifest.get("bandwidth_savings_from") or "fwdllm"
    savings_to = manifest.get("bandwidth_savings_to") or "felix_round"
    for spec in args.run:                      # CLI overrides the manifest
        if "=" not in spec:
            print(f"  [paper-figs] ignoring malformed --run '{spec}' (want KEY=PATH)",
                  file=sys.stderr)
            continue
        k, v = spec.split("=", 1)
        runs_map[k.strip()] = v.strip()
    if not runs_map:
        print("  [paper-figs] no baselines mapped — populate figs.yaml or pass --run",
              file=sys.stderr)
        return 1

    # load each mapped run (streaming); skip + warn on missing/empty telemetry
    loaded = {}
    for key in B.ordered(runs_map.keys()):
        rr = R.load_run(_resolve(runs_map[key]), key=key, post_peak_grace_s=grace_s,
                        loss_plateau_rel=args.loss_plateau_rel,
                        cutoff_mode=args.cutoff_mode)
        if rr is None or not rr.evals:
            print(f"  [paper-figs] WARN no usable telemetry for '{key}' "
                  f"({runs_map[key]}) — skipping", file=sys.stderr)
            continue
        loaded[key] = rr
        cut = " (no cutoff — never learned)" if rr.cutoff_ts is None else (
            f", cut @{args.cutoff_mode} {(rr.cutoff_ts - rr.t0)/3600:.2f}h")
        print(f"  [paper-figs] loaded {key}: {len(rr.evals)} evals, "
              f"{rr.n_trainers} trainers, max_acc={100*(rr.max_accuracy() or 0):.2f}%{cut}")
    if not loaded:
        print("  [paper-figs] nothing loaded — nothing to plot.", file=sys.stderr)
        return 1

    ordered_runs = [loaded[k] for k in B.ordered(loaded.keys())]
    which = ([f.strip() for f in args.figures.split(",")]
             if args.figures else list(F.FIG_BUILDERS))

    S.use_paper_style()
    out_dir = S.render_outdir(args.out_root)
    written = []
    for name in which:
        builder = F.FIG_BUILDERS.get(name)
        if builder is None:
            print(f"  [paper-figs] unknown figure '{name}' — skipping", file=sys.stderr)
            continue
        fig = builder(ordered_runs, target=target, smooth=args.smooth,
                     baseline_key=baseline_key, champion_key=champion_key,
                     savings_from=savings_from, savings_to=savings_to)
        if fig is None:
            print(f"  [paper-figs] {name}: no data across baselines — skipped")
            continue
        S.save_pdf(fig, out_dir, name)
        written.append(name)

    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump({
            "runs": {k: loaded[k].run_dir for k in loaded},
            "baselines_order": B.ordered(loaded.keys()),
            "target_acc": target,
            "speedup_baseline": baseline_key,
            "speedup_champion": champion_key,
            "bandwidth_savings_from": savings_from,
            "bandwidth_savings_to": savings_to,
            "figures": written,
            "system_label": B.SYSTEM_LABEL,
            "smooth": args.smooth,
            "cutoff_mode": None if args.no_cutoff else args.cutoff_mode,
            "loss_plateau_rel": None if args.no_cutoff else args.loss_plateau_rel,
            "post_peak_grace_min": None if args.no_cutoff else args.post_peak_grace_min,
            "cutoff_h": {k: (None if loaded[k].cutoff_ts is None
                             else round((loaded[k].cutoff_ts - loaded[k].t0) / 3600, 3))
                         for k in loaded},
        }, fh, indent=2)

    print(f"\n  wrote {len(written)} figure(s) → {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
