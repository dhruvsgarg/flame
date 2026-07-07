#!/usr/bin/env python3
# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Decompose fwdllm trainer wall time by compute step, for GPU optimization.

Two granularities, both read from a run's ``telemetry/trainer_*.jsonl``:

  * FINE  — the ``step_timing`` events emitted by ``@timer_decorator`` (P2-4):
    per-function wall (``_make_model_functional``, ``_setup_training_state`` /
    perturbation selection, ``_train_one_batch`` / JVP forward passes,
    ``_emulate_training_delay``). This is the "where does the GPU time go" view
    that tells us WHAT to optimize (e.g. fluxtune's 20-pass JVP at 7.57 s).
  * COARSE — the per-round phase fields already on ``trainer_round``
    (``gpu_compute_s``, ``pre_train_s``, ``weights_to_gpu_s``,
    ``weights_to_ram_s``, ``mqtt_fetch_s``, ``post_train_s``). Always present;
    a fallback when a run predates the step_timing telemetry.

Prints a breakdown table and (unless ``--no-plot``) writes PNGs next to the run.

Usage:
  python plot_step_timing.py                     # latest sim run per baseline
  python plot_step_timing.py --run-dir DIR ...   # explicit run dir(s)
  python plot_step_timing.py --baselines fluxtune
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import pathlib
import re
import sys

_HERE = pathlib.Path(__file__).resolve().parent
_DEFAULT_EXPERIMENTS = _HERE.parent / "experiments"

# Same run-dir grammar as run_parity._RUN_RE — the `_n\d+_smoke` anchor makes the
# baseline token EXACT, so "fwdllm" does not swallow "fwdllm_plus" (a plain
# substring match does — that collision made the two report identical numbers).
_RUN_RE = re.compile(
    r"^run_(?P<ts>\d{8}_\d{6})_(?P<baseline>.+)_n(?P<n>\d+)_smoke"
    r"(?:_(?P<trace>.+))?_(?P<variant>real|sim)$"
)

# Coarse trainer_round phase fields, in pipeline order.
_PHASES = ("mqtt_fetch_s", "weights_to_ram_s", "weights_to_gpu_s",
           "pre_train_s", "gpu_compute_s", "post_train_s")


def _iter_trainer_events(run_dir: str):
    """Yield (short_id, event_dict) for every trainer telemetry record."""
    tdir = pathlib.Path(run_dir) / "telemetry"
    for f in sorted(tdir.glob("trainer_*.jsonl")):
        sid = f.stem[-4:]
        with open(f) as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield sid, json.loads(line)
                except json.JSONDecodeError:
                    continue


def _collect(run_dir: str) -> dict:
    """Aggregate step_timing (fine) + trainer_round phases (coarse) for a run."""
    step_tot: dict = collections.defaultdict(float)
    step_n: dict = collections.defaultdict(int)
    phase_tot: dict = collections.defaultdict(float)
    phase_n: dict = collections.defaultdict(int)
    n_rounds = 0
    for _sid, e in _iter_trainer_events(run_dir):
        ev = e.get("event")
        if ev == "step_timing":
            fn, d = e.get("func"), e.get("duration_s")
            if fn is not None and d is not None:
                step_tot[fn] += float(d)
                step_n[fn] += 1
        elif ev == "trainer_round":
            n_rounds += 1
            for p in _PHASES:
                v = e.get(p)
                if v is not None:
                    phase_tot[p] += float(v)
                    phase_n[p] += 1
    return {"step_tot": dict(step_tot), "step_n": dict(step_n),
            "phase_tot": dict(phase_tot), "phase_n": dict(phase_n),
            "n_rounds": n_rounds}


def _fmt_table(label: str, agg: dict) -> str:
    lines = [f"\n=== {label}  ({agg['n_rounds']} trainer rounds) ==="]
    # coarse phases (always present)
    lines.append("  COARSE trainer_round phases (mean s/round | total s):")
    ptot = agg["phase_tot"]
    for p in _PHASES:
        if p in ptot:
            mean = ptot[p] / max(agg["phase_n"].get(p, 1), 1)
            lines.append(f"    {p:<18} {mean:8.3f} | {ptot[p]:9.1f}")
    # fine steps (present only on runs with step_timing telemetry)
    st = agg["step_tot"]
    if st:
        lines.append("  FINE step_timing (mean s/call | calls | total s), "
                     "by total desc:")
        for fn in sorted(st, key=st.get, reverse=True):
            n = agg["step_n"].get(fn, 0)
            mean = st[fn] / max(n, 1)
            lines.append(f"    {fn:<30} {mean:7.3f} | {n:6d} | {st[fn]:8.1f}")
    else:
        lines.append("  FINE step_timing: NONE — run predates step_timing "
                     "telemetry (re-run to populate the per-step GPU breakdown).")
    return "\n".join(lines)


def _plot(label: str, agg: dict, out_path: str) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"  [plot skipped] matplotlib unavailable: {exc}")
        return False

    st = agg["step_tot"]
    ptot = agg["phase_tot"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # left: coarse phase mean s/round
    ph = [(p, ptot[p] / max(agg["phase_n"].get(p, 1), 1))
          for p in _PHASES if p in ptot]
    if ph:
        names, vals = zip(*ph)
        axes[0].barh(range(len(names)), vals, color="#4C72B0")
        axes[0].set_yticks(range(len(names)))
        axes[0].set_yticklabels(names)
        axes[0].invert_yaxis()
        axes[0].set_xlabel("mean seconds / round")
        axes[0].set_title(f"{label}: coarse phase (trainer_round)")

    # right: fine step_timing mean s/call, top 12 by total
    ax = axes[1]
    if st:
        top = sorted(st, key=st.get, reverse=True)[:12]
        means = [st[fn] / max(agg["step_n"].get(fn, 1), 1) for fn in top]
        ax.barh(range(len(top)), means, color="#C44E52")
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("mean seconds / call")
        ax.set_title(f"{label}: fine step (step_timing)")
    else:
        ax.text(0.5, 0.5, "no step_timing telemetry\n(re-run to populate)",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return True


def _discover_latest(experiments_dir: str, baselines, side: str) -> dict:
    """{baseline -> newest run_dir} for the requested side, EXACT baseline match
    (via _RUN_RE, so fwdllm != fwdllm_plus). ts is fixed-width → lexical == time."""
    want = set(baselines)
    best: dict = {}
    root = pathlib.Path(experiments_dir)
    if not root.is_dir():
        return {}
    for d in root.iterdir():
        if not d.is_dir():
            continue
        m = _RUN_RE.match(d.name)
        if not m or m["variant"] != side or m["baseline"] not in want:
            continue
        base = m["baseline"]
        if base not in best or m["ts"] > best[base][0]:
            best[base] = (m["ts"], str(d))
    return {b: p for b, (_ts, p) in best.items()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", nargs="+", default=None,
                    help="explicit run dir(s); default = latest per baseline")
    ap.add_argument("--experiments-dir", default=str(_DEFAULT_EXPERIMENTS))
    ap.add_argument("--baselines", nargs="+",
                    default=["fwdllm", "fwdllm_plus", "fluxtune"])
    ap.add_argument("--side", default="sim", choices=["sim", "real"],
                    help="which side to profile when auto-discovering (default sim)")
    ap.add_argument("--out-dir", default=None,
                    help="where to write PNGs (default <experiments>/_timing_plots)")
    ap.add_argument("--no-plot", action="store_true", help="table only, no PNGs")
    args = ap.parse_args(argv)

    if args.run_dir:
        runs = {os.path.basename(d.rstrip("/")): d for d in args.run_dir}
    else:
        runs = _discover_latest(args.experiments_dir, args.baselines, args.side)
    if not runs:
        print("No run dirs found. Pass --run-dir or check --experiments-dir.")
        return 1

    out_dir = args.out_dir or os.path.join(args.experiments_dir, "_timing_plots")
    if not args.no_plot:
        os.makedirs(out_dir, exist_ok=True)

    for label, rdir in runs.items():
        agg = _collect(rdir)
        print(_fmt_table(label, agg))
        if not args.no_plot:
            pdf = os.path.join(out_dir, f"step_timing_{label}.pdf")
            if _plot(label, agg, pdf):
                print(f"  -> {pdf}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
