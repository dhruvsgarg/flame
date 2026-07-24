#!/usr/bin/env python3
# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Cross-baseline metric reducer (EXPERIMENTS.md WS4).

Discovers the latest run per baseline (exact-token regex, so `fwdllm` never captures
`fwdllm_plus`) and emits one comparison table + CSV (+ optional overlay plots). The
five experiments are computed by `plotlib.reducers` (shared with plot_run.py and
make_paper_figs.py); this module owns only discovery, the mix-guard and the table.
`--plots` renders via `plotlib.figures` so the look matches the paper. Missing
baselines / metrics degrade to skipped / None, never a crash.

    python compare_baselines.py [--variant real|sim] [--target-acc 0.84] [--plots]
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from plotlib import reducers as R          # noqa: E402  (single source of the metrics)

# `run_<YYYYMMDD>_<HHMMSS>_<baseline>_n<N>_smoke[_<trace>]_<real|sim>`
_RUN_RE = re.compile(
    r"^run_(?P<ts>\d{8}_\d{6})_(?P<baseline>.+)_n(?P<n>\d+)_smoke"
    r"(?:_(?P<trace>.+))?_(?P<variant>real|sim)$"
)
_DEFAULT_BASELINES = ["fwdllm", "fwdllm_plus", "fluxtune"]

_BASELINES_YAML = os.path.join(HERE, "..", "..", "_metadata", "baselines.yaml")


def _is_async(baseline: str) -> bool:
    """selector.kwargs.is_async from _metadata/baselines.yaml (single source of
    truth) — NOT a hardcoded `baseline == "fluxtune"` guess, which silently
    mislabeled every new async baseline (fedbuff_round/it_*, felix_round/it) as
    sync (same bug class run_sequential.sh's `_BL_INTERNALS` lookup already
    fixed this session). Unreadable/missing key -> False (safe default: a sync
    baseline mislabeled async is the more visible failure of the two)."""
    try:
        import yaml
        bl = yaml.safe_load(open(_BASELINES_YAML, encoding="utf-8"))
        bl = bl.get("baselines", bl)
        if baseline not in bl:
            print(f"  [compare] WARN baseline '{baseline}' not in {_BASELINES_YAML} "
                  f"— defaulting to sync", file=sys.stderr)
            return False
        sel = (bl.get(baseline) or {}).get("aggregator", {}).get("selector", {})
        return bool((sel.get("kwargs", {}) or {}).get("is_async"))
    except Exception as e:
        print(f"  [compare] WARN could not resolve is_async for '{baseline}' "
              f"from {_BASELINES_YAML}: {e} — defaulting to sync", file=sys.stderr)
        return False


# --------------------------------------------------------------------------- #
# discovery
# --------------------------------------------------------------------------- #
def discover(experiments_dir: str, variant: str) -> dict:
    """{baseline: (n, trace, path)} — latest run per baseline for the variant.

    Exact-token match on baseline avoids the fwdllm ⊃ fwdllm_plus glob trap."""
    latest: dict = {}
    for path in glob.glob(os.path.join(experiments_dir, "run_*")):
        m = _RUN_RE.match(os.path.basename(path))
        if not m or m["variant"] != variant:
            continue
        b = m["baseline"]
        prev = latest.get(b)
        if prev is None or m["ts"] > prev[0]:
            latest[b] = (m["ts"], int(m["n"]), m["trace"] or "", path)
    return {b: (n, tr, p) for b, (ts, n, tr, p) in latest.items()}


# --------------------------------------------------------------------------- #
# small stats helpers
# --------------------------------------------------------------------------- #
def _pct(values, q: float):
    xs = sorted(v for v in values if v is not None)
    if not xs:
        return None
    if len(xs) == 1:
        return xs[0]
    pos = (q / 100.0) * (len(xs) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac


def _p(values):
    return {"p50": _pct(values, 50), "p90": _pct(values, 90), "p99": _pct(values, 99)}


# --------------------------------------------------------------------------- #
# the five experiments — PURE over a RunResult (from plotlib.reducers)
# --------------------------------------------------------------------------- #
def expt1_time_to_target(rr: R.RunResult, target: float, window: int) -> dict:
    max_acc, final_acc = rr.max_accuracy(), rr.final_accuracy()
    # N3: prefer converge_watch.py's own verdict over the agg_eval reconstruction
    # below -- the watcher is the process that actually decided WHEN to stop the
    # run, so it can't silently diverge from what gets reported. Only trusted
    # when its target/window match what's being asked for here (a converge.json
    # from a different --target-acc/--window is not this metric).
    cj = rr.converge_json
    if cj is not None and cj.get("converged") and \
            cj.get("target_accuracy") == target and cj.get("window") == window:
        return {
            "reached": True, "target": target, "window": window,
            "wall_s": cj.get("time_to_converge_wall_s"),
            "vclock_s": cj.get("time_to_converge_vclock_s"),
            "rounds": cj.get("rounds_at_converge"),
            "data_bins": cj.get("n_bins_completed"),
            "max_accuracy": max_acc, "final_accuracy": final_acc,
            "source": "converge.json",
        }
    e = rr.target_event(target, window)
    if e is not None:
        return {
            "reached": True, "target": target, "window": window,
            "wall_s": round(e["ts"] - rr.t0, 2) if rr.t0 is not None else None,
            "vclock_s": rr.vclock_at(e["ts"]), "rounds": e.get("round"),
            # cumulative bin-evals to convergence (monotonic; data_id cycles per round)
            "data_bins": rr.evals.index(e) + 1,
            "max_accuracy": max_acc, "final_accuracy": final_acc,
            "source": "reconstructed",
        }
    return {
        "reached": False, "target": target, "window": window,
        "wall_s": None, "vclock_s": None, "rounds": None, "data_bins": None,
        "max_accuracy": max_acc, "final_accuracy": final_acc,
        "source": "reconstructed",
    }


def expt2_utilization(rr: R.RunResult) -> dict:
    tp = _p(rr.busy_frac)
    # N5: mqtt_fetch_s (emitted per trainer_round, previously unread) splits the
    # old single "idle = 1-busy_frac" bucket into network-wait (waiting on the
    # aggregator's weight payload) vs. the true residual idle (waiting to be
    # selected again) -- barrier_wait_s/drain_tail_s stay agg-side-only (real ~0
    # in sim), so they're not a trainer idle signal and aren't used here.
    nwp = _p(rr.net_wait_frac)
    idle_frac = [max(0.0, 1.0 - b - n) for b, n in zip(rr.busy_frac, rr.net_wait_frac)]
    ip = _p(idle_frac)
    w = rr.agg_wall_s
    frac = lambda x: (round(x / w, 4) if (w and w > 0 and x is not None) else None)
    return {
        "trainer_busy_frac_p50": tp["p50"], "trainer_busy_frac_p90": tp["p90"],
        "trainer_busy_frac_p99": tp["p99"], "n_trainers_measured": len(rr.busy_frac),
        "trainer_net_wait_frac_p50": nwp["p50"], "trainer_net_wait_frac_p90": nwp["p90"],
        "trainer_net_wait_frac_p99": nwp["p99"],
        "trainer_idle_frac_p50": ip["p50"], "trainer_idle_frac_p90": ip["p90"],
        "trainer_idle_frac_p99": ip["p99"],
        "agg_busy_frac": frac(rr.agg_compute_s), "agg_barrier_wait_frac": frac(rr.agg_barrier_s),
        "agg_drain_frac": frac(rr.agg_drain_s),
        "agg_wall_s": round(w, 2) if w else None,
    }


def expt3_productivity(rr: R.RunResult) -> dict:
    losses = [e["loss"] for e in rr.evals if e["loss"] is not None]
    first_loss = losses[0] if losses else None
    final_loss = losses[-1] if losses else None
    delta = rr.delta_loss()
    gpu_total = rr.gpu_s_total()
    return {
        "first_loss": first_loss, "final_loss": final_loss, "delta_loss": delta,
        "trainer_gpu_s": round(rr.trainer_gpu_s, 2), "agg_compute_s": round(rr.agg_compute_s, 2),
        "gpu_s_total": round(gpu_total, 2),
        "delta_loss_per_gpu_s": (round(delta / gpu_total, 6) if (delta and gpu_total) else None),
        "forward_passes_total": rr.fwd_total if rr.have_fwd else None,
        "perturbations_total": rr.pert_total if rr.have_fwd else None,
        "delta_loss_per_forward_pass": (
            round(delta / rr.fwd_total, 9) if (delta and rr.have_fwd and rr.fwd_total) else None),
    }


def expt4_network(rr: R.RunResult) -> dict:
    if not rr.have_comm and not rr.up_sizes and not rr.down_sizes:
        return {"comm_telemetry": False}
    a_sizes, c_sizes = rr.down_sizes, rr.up_sizes
    ap, cp = _p(a_sizes), _p(c_sizes)
    return {
        "comm_telemetry": True,
        "msgs_agg_to_trainer": len(a_sizes), "msgs_trainer_to_agg": len(c_sizes),
        "bytes_agg_to_trainer": sum(a_sizes), "bytes_trainer_to_agg": sum(c_sizes),
        "bytes_total": sum(a_sizes) + sum(c_sizes),
        "msg_size_agg_p50": ap["p50"], "msg_size_agg_p90": ap["p90"], "msg_size_agg_p99": ap["p99"],
        "msg_size_client_p50": cp["p50"], "msg_size_client_p90": cp["p90"], "msg_size_client_p99": cp["p99"],
    }


def expt5_sessions(rr: R.RunResult, is_async: bool) -> dict:
    # N6: sync baselines report round_span_durs (per-data_id engagement, the
    # actual "one round span" the label already claimed) instead of
    # session_durs (per-contribution dispatch->commit, which for a
    # multi-iteration sync round only measured the LAST iteration's trip, not
    # the cohort's full time-in-round). Async is untouched (already validated).
    durs = rr.session_durs if is_async else rr.round_span_durs
    sp = _p(durs)
    return {
        "session_def": "dispatch_to_commit" if is_async else "one_round_span",
        "session_s_p50": sp["p50"], "session_s_p90": sp["p90"], "session_s_p99": sp["p99"],
        "n_sessions": len(durs),
        "part_rounds_p50": _pct(rr.part_rounds, 50), "part_rounds_p90": _pct(rr.part_rounds, 90),
        "part_databins_p50": _pct(rr.part_bins, 50), "part_databins_p90": _pct(rr.part_bins, 90),
        "part_iters_p50": _pct(rr.part_iters, 50), "part_iters_p90": _pct(rr.part_iters, 90),
        "part_iters_total": sum(rr.part_iters),
    }


# --------------------------------------------------------------------------- #
# config fingerprint mix-guard
# --------------------------------------------------------------------------- #
def _fingerprint(run_dir: str, n: int, trace: str) -> dict:
    """Shared condition axes that MUST match across baselines (N/partition/trace).
    agg_goal/c/k/selector legitimately differ per baseline, so they're excluded."""
    part = None
    cfg = os.path.join(run_dir, "aggregator_config.json")
    if os.path.exists(cfg):
        try:
            h = json.load(open(cfg)).get("hyperparameters", {})
            part = h.get("partition_method")
        except (ValueError, OSError):
            pass
    return {"N": n, "trace": trace, "partition": part}


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def _fmt(v):
    if v is None:
        return "–"
    if isinstance(v, float):
        return f"{v:.4g}"
    if isinstance(v, int) and abs(v) >= 100000:
        return f"{v/1e6:.2f}M"
    return str(v)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiments-dir", default=os.path.join(HERE, "..", "experiments"))
    ap.add_argument("--variant", choices=["real", "sim"], default="real")
    ap.add_argument("--baselines", default=",".join(_DEFAULT_BASELINES))
    ap.add_argument("--target-acc", type=float, default=0.84)
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--out", default=None, help="output dir for CSV (+plots); default = experiments/_compare")
    ap.add_argument("--plots", action="store_true", help="also write paper overlay plots (via plotlib)")
    ap.add_argument("--smooth", type=float, default=0.7,
                    help="EMA smoothing factor for overlay learning curves (0 = raw)")
    ap.add_argument("--loss-plateau-rel", type=float, default=0.01,
                    help="cut each run at its last cumulative test-loss drop of this "
                         "relative size (default 0.01 = 1%%; end of productive learning)")
    ap.add_argument("--post-peak-grace-min", type=float, default=0.0,
                    help="minutes to keep past the loss-plateau point (default 0)")
    ap.add_argument("--no-cutoff", action="store_true", help="use full telemetry (no cutoff)")
    args = ap.parse_args()
    grace_s = None if args.no_cutoff else args.post_peak_grace_min * 60.0

    exp_dir = os.path.abspath(args.experiments_dir)
    baselines = [b.strip() for b in args.baselines.split(",") if b.strip()]
    out_dir = os.path.abspath(args.out) if args.out else os.path.join(exp_dir, "_compare")
    os.makedirs(out_dir, exist_ok=True)

    found = discover(exp_dir, args.variant)
    rows, fps, results = {}, {}, {}
    for b in baselines:
        if b not in found:
            print(f"  [compare] WARN no {args.variant} run found for baseline '{b}' — skipping", file=sys.stderr)
            continue
        n, trace, path = found[b]
        rr = R.load_run(path, key=b, post_peak_grace_s=grace_s,
                        loss_plateau_rel=args.loss_plateau_rel)
        if rr is None or not rr.evals:
            print(f"  [compare] WARN empty aggregator telemetry for '{b}' ({path}) — skipping", file=sys.stderr)
            continue
        results[b] = rr
        is_async = _is_async(b)
        rows[b] = {
            "run_dir": os.path.basename(path), "N": n, "trace": trace or "syn_0",
            **{f"e1_{k}": v for k, v in expt1_time_to_target(rr, args.target_acc, args.window).items()},
            **{f"e2_{k}": v for k, v in expt2_utilization(rr).items()},
            **{f"e3_{k}": v for k, v in expt3_productivity(rr).items()},
            **{f"e4_{k}": v for k, v in expt4_network(rr).items()},
            **{f"e5_{k}": v for k, v in expt5_sessions(rr, is_async).items()},
        }
        fps[b] = _fingerprint(path, n, trace)

    if not rows:
        print("  [compare] no baselines with telemetry found — nothing to compare.", file=sys.stderr)
        return 1

    # mix-guard: shared condition axes must agree across baselines
    uniq = {tuple(sorted(fp.items())) for fp in fps.values()}
    if len(uniq) > 1:
        print("  [compare] ⚠ MIX-GUARD: baselines differ on a SHARED condition axis "
              "(N/partition/trace) — comparison may be apples-to-oranges:", file=sys.stderr)
        for b, fp in fps.items():
            print(f"      {b}: {fp}", file=sys.stderr)

    cols = []
    for b in baselines:
        for k in rows.get(b, {}):
            if k not in cols:
                cols.append(k)

    csv_path = os.path.join(out_dir, "compare_baselines.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["baseline"] + cols)
        for b in baselines:
            if b in rows:
                w.writerow([b] + [rows[b].get(c, "") for c in cols])

    headline = [
        ("e1_reached", "conv?"), ("e1_wall_s", "t→acc wall_s"), ("e1_data_bins", "t→acc bins"),
        ("e1_max_accuracy", "max_acc"), ("e2_trainer_busy_frac_p50", "trn_busy_p50"),
        ("e2_agg_busy_frac", "agg_busy"), ("e3_delta_loss", "Δloss"),
        ("e3_delta_loss_per_gpu_s", "Δloss/gpu_s"), ("e3_delta_loss_per_forward_pass", "Δloss/fwd"),
        ("e4_bytes_total", "net_bytes"), ("e4_msgs_trainer_to_agg", "msgs↑"),
        ("e5_session_s_p50", "sess_p50"), ("e5_part_iters_total", "iters_tot"),
    ]
    w1 = max(len(b) for b in rows)
    print(f"\n=== CROSS-BASELINE COMPARISON ({args.variant}, τ={args.target_acc}, W={args.window}) ===")
    hdr = f"{'baseline':<{w1}}  " + "  ".join(f"{lbl:>13}" for _, lbl in headline)
    print(hdr)
    print("-" * len(hdr))
    for b in baselines:
        if b not in rows:
            continue
        cells = "  ".join(f"{_fmt(rows[b].get(k)):>13}" for k, _ in headline)
        print(f"{b:<{w1}}  {cells}")
    print(f"\nCSV:   {csv_path}")

    if args.plots:
        _plots(results, out_dir, args.target_acc, args.smooth)
    print(f"Out:   {out_dir}")
    return 0


def _plots(results: dict, out_dir: str, target, smooth: float = 0.0):
    """Overlay plots via the shared plotlib (same look as the paper figures)."""
    try:
        from plotlib import baselines as B
        from plotlib import figures as F
        from plotlib import style as S
    except Exception as e:  # noqa: BLE001
        print(f"  [compare] plots skipped (plotlib/matplotlib unavailable: {e})", file=sys.stderr)
        return
    S.use_paper_style()
    ordered = [results[k] for k in B.ordered(results.keys())]
    n = 0
    for name, builder in F.FIG_BUILDERS.items():
        fig = builder(ordered, target=target, smooth=smooth)
        if fig is None:
            continue
        S.save_pdf(fig, out_dir, name)
        n += 1
    print(f"  [compare] {n} overlay plot(s) written to {out_dir}")


if __name__ == "__main__":
    sys.exit(main())
