#!/usr/bin/env python3
# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Render the FluxTune paper figures from a manifest (EXPERIMENTS.md §7).

    python make_paper_figs.py --manifest figs_e2e.yaml
    python make_paper_figs.py --manifest figs_ablation.yaml
    python make_paper_figs.py --manifest figs_availability.yaml
    python make_paper_figs.py --manifest figs_e2e.yaml --synthetic   # layout preview

Manifest kinds:
  e2e       cells: [{id, label, target_acc, n_clients, arms: {key: [run_dir, ...]}}]
            -> <prefix>_<cell>.pdf per cell, <prefix>_grid.pdf, and figures: b1/b2/b3 per cell
  ablation  cell: {...}; groups: [{id, title, figures, arms: [{key, label, sync, runs}]}]
            -> c_<group>.pdf (+ c_<group>_step_size.pdf if listed)
Each run list is one dir per seed. Empty lists are skipped. Writes summary.csv + manifest.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from plotlib import paper as P              # noqa: E402
from plotlib import reducers as R           # noqa: E402
from plotlib import style as S              # noqa: E402
from plotlib import synthetic as SY         # noqa: E402

EXAMPLE_DIR = os.path.abspath(os.path.join(HERE, ".."))   # run dirs resolve from here
E2E_FIGS = {"b1": P.fig_b1_participation, "b2": P.fig_b2_diversity, "b3": P.fig_b3_progression}


def _resolve(path: str) -> str:
    if os.path.isabs(path):
        return path
    for base in (os.getcwd(), EXAMPLE_DIR):
        cand = os.path.join(base, path)
        if os.path.exists(cand):
            return os.path.abspath(cand)
    return os.path.abspath(os.path.join(EXAMPLE_DIR, path))


def _load_runs(key, paths, args, cache):
    out = []
    for p in paths or []:
        rp = _resolve(p)
        if rp not in cache:
            cache[rp] = R.load_run(rp, key=key, post_peak_grace_s=args.grace_s,
                                   cutoff_mode=args.cutoff_mode)
        rr = cache[rp]
        if rr is None or not rr.evals:
            print(f"  [paper-figs] WARN no usable telemetry: {key} {p}", file=sys.stderr)
            continue
        out.append(rr)
    return out


def _build_arms(specs, args, cache, n_clients, champion):
    """specs: list of (key, label, sync, run_paths). Synthetic mode fakes 3 seeds per arm."""
    arms = []
    for i, (key, label, sync, paths) in enumerate(specs):
        st = P.arm_style(key, label, i, len(specs), sync=bool(sync))
        if args.synthetic:
            runs = SY.fake_arm_runs(key, i, len(specs), n_clients=n_clients or 100,
                                    champion=champion)
        else:
            runs = _load_runs(key, paths, args, cache)
        if not runs:
            print(f"  [paper-figs] missing runs: {key}")
        arms.append(P.Arm(key=key, label=st.label, style=st, runs=runs))
    return arms


def _save(fig, out_dir, name, args, written):
    if fig is None:
        print(f"  [paper-figs] {name}: no data — skipped")
        return
    if args.synthetic:
        P.watermark(fig)
    S.save_pdf(fig, out_dir, name, png=args.png)
    written.append(name)


def render_e2e(man, args, out_dir, cache, written, rows):
    opts = _opts(man, args)
    prefix = man.get("prefix", "e2e")
    figs = man.get("figures", ["row", "grid"])
    grid = []
    for cell in man.get("cells", []):
        specs = [(k, None, k.startswith("fwdllm"), v) for k, v in (cell.get("arms") or {}).items()]
        arms = _build_arms(specs, args, cache, cell.get("n_clients"), opts["champion"])
        grid.append((cell, arms))
        live = [a for a in arms if a.runs]
        rows += P.summary_rows(prefix, cell, live, opts)
        if "row" in figs:
            _save(P.fig_e2e_row(live, cell, opts), out_dir, f"{prefix}_{cell['id']}", args, written)
        for f in figs:
            if f in E2E_FIGS and live:
                _save(E2E_FIGS[f](live, cell, opts), out_dir, f"{f}_{cell['id']}", args, written)
    if "grid" in figs:
        _save(P.fig_e2e_grid(grid, opts), out_dir, f"{prefix}_grid", args, written)


def render_ablation(man, args, out_dir, cache, written, rows):
    opts = _opts(man, args)
    cell = man.get("cell", {})
    for g in man.get("groups", []):
        specs = [(a["key"], a.get("label"), a.get("sync", False), a.get("runs"))
                 for a in g.get("arms", [])]
        arms = _build_arms(specs, args, cache, cell.get("n_clients"), opts["champion"])
        live = [a for a in arms if a.runs]
        rows += P.summary_rows(f"c_{g['id']}", cell, live, opts)
        figs = g.get("figures", ["group"])
        if "group" in figs:
            _save(P.fig_c_group(live, cell, opts, title=g.get("title")), out_dir,
                  f"c_{g['id']}", args, written)
        if "step_size" in figs:
            _save(P.fig_c_step_size(live, cell, opts), out_dir,
                  f"c_{g['id']}_step_size", args, written)


def _opts(man, args):
    return {"consecutive": int(man.get("tta_consecutive", 1)),
            "smooth": args.smooth,
            "baseline": man.get("baseline", "fwdllm"),
            "champion": man.get("champion", "fluxtune")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-root", default=None,
                    help="output dir (default: paper_figs/<manifest stem>)")
    ap.add_argument("--synthetic", action="store_true",
                    help="fake 3 seeds per arm to preview layouts (watermarked)")
    ap.add_argument("--smooth", type=float, default=0.6, help="EMA for curves (visual only)")
    ap.add_argument("--cutoff-mode", choices=["none", "peak_acc", "loss_plateau"],
                    default="none", help="clip each run's tail (default: full run)")
    ap.add_argument("--png", action="store_true", help="also write a .png next to each .pdf")
    args = ap.parse_args()
    args.grace_s = None if args.cutoff_mode == "none" else 0.0
    args.cutoff_mode = "peak_acc" if args.cutoff_mode == "none" else args.cutoff_mode

    import yaml
    with open(args.manifest) as fh:
        man = yaml.safe_load(fh) or {}
    stem = os.path.splitext(os.path.basename(args.manifest))[0]
    out_dir = S.render_outdir(args.out_root or os.path.join(HERE, "paper_figs", stem))

    S.use_paper_style()
    written, rows, cache = [], [], {}
    kind = man.get("kind", "e2e")
    if kind == "e2e":
        render_e2e(man, args, out_dir, cache, written, rows)
    elif kind == "ablation":
        render_ablation(man, args, out_dir, cache, written, rows)
    else:
        print(f"  [paper-figs] unknown manifest kind '{kind}'", file=sys.stderr)
        return 1

    if rows:
        with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump({"manifest": os.path.abspath(args.manifest), "synthetic": args.synthetic,
                   "figures": written, "runs": sorted({r["run_dir"] for r in rows})},
                  fh, indent=2)
    print(f"\n  wrote {len(written)} figure(s) → {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
