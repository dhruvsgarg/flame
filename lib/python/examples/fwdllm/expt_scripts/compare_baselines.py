#!/usr/bin/env python3
# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Cross-baseline metric reducer (EXPERIMENTS.md WS4).

Walks the fwdllm/fwdllm_plus/fluxtune run dirs, computes the five experiments'
metrics as PURE FUNCTIONS over each run's telemetry, and emits one comparison
table + CSV (+ optional overlay plots). Every experiment is a *view* over the
same run-set (EXPERIMENTS.md sec 0) — no run is re-executed here.

    python compare_baselines.py [--experiments-dir DIR] [--variant real|sim]
        [--baselines fwdllm,fwdllm_plus,fluxtune] [--target-acc 0.88]
        [--out DIR] [--plots]

Reuses the run-discovery regex from run_parity.py (so the `fwdllm` glob never
captures `fwdllm_plus`). Reads telemetry field names verified against live runs
(see EXPERIMENTS.md sec 5). Degrades gracefully: a metric with no source data is
reported as None rather than crashing, and missing comm/perturbation telemetry
(pre-WS3 runs) is flagged, not fatal.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys

# `run_<YYYYMMDD>_<HHMMSS>_<baseline>_n<N>_smoke[_<trace>]_<real|sim>`
_RUN_RE = re.compile(
    r"^run_(?P<ts>\d{8}_\d{6})_(?P<baseline>.+)_n(?P<n>\d+)_smoke"
    r"(?:_(?P<trace>.+))?_(?P<variant>real|sim)$"
)
_DEFAULT_BASELINES = ["fwdllm", "fwdllm_plus", "fluxtune"]


# --------------------------------------------------------------------------- #
# discovery + loading
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


def _load_jsonl(path: str) -> list:
    out = []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # partial trailing line
    except OSError:
        pass
    return out


def load_run(run_dir: str) -> dict:
    """Load a run's telemetry into {agg: [...], trainers: {tid: [...]}}."""
    tdir = os.path.join(run_dir, "telemetry")
    agg_files = glob.glob(os.path.join(tdir, "aggregator_*.jsonl"))
    agg = _load_jsonl(agg_files[0]) if agg_files else []
    trainers = {}
    for f in glob.glob(os.path.join(tdir, "trainer_*.jsonl")):
        tid = os.path.basename(f)[len("trainer_"):-len(".jsonl")]
        trainers[tid] = _load_jsonl(f)
    return {"agg": agg, "trainers": trainers, "dir": run_dir}


# --------------------------------------------------------------------------- #
# small stats helpers (stdlib only)
# --------------------------------------------------------------------------- #
def _pct(values, q: float):
    """Linear-interpolation percentile (q in [0,100]). None if empty."""
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


def _t0(events) -> float | None:
    ts = [e.get("ts") for e in events if e.get("ts") is not None]
    return min(ts) if ts else None


# --------------------------------------------------------------------------- #
# Experiment 1 — time to target accuracy
# --------------------------------------------------------------------------- #
def _acc_loss_by_bin(agg):
    """(data_id -> accuracy, data_id -> (loss, ts, round)) from agg_eval, last wins."""
    acc, meta = {}, {}
    for e in agg:
        if e.get("event") != "agg_eval":
            continue
        d = e.get("data_id")
        a = e.get("test-accuracy")
        if d is None:
            continue
        d = int(d)
        if a is not None:
            acc[d] = float(a)
        meta[d] = (e.get("test-loss"), e.get("ts"), e.get("round"))
    return acc, meta


def expt1_time_to_target(agg, target: float, window: int) -> dict:
    acc, meta = _acc_loss_by_bin(agg)
    max_acc = max(acc.values()) if acc else None
    final_acc = acc[max(acc)] if acc else None
    # first data bin at which the trailing `window` consecutive bins are all >= target
    trigger = None
    if acc:
        for top in sorted(acc):
            streak, d = 0, top
            while d in acc and acc[d] >= target:
                streak += 1
                d -= 1
            if streak >= window:
                trigger = top
                break
    t0 = _t0(agg)
    if trigger is not None and t0 is not None:
        _loss, ts, rnd = meta.get(trigger, (None, None, None))
        # vclock at/just before the trigger (nearest agg_round)
        vclock = None
        for e in agg:
            if e.get("event") == "agg_round" and e.get("vclock_now") is not None:
                if e.get("ts") is not None and ts is not None and e["ts"] <= ts:
                    vclock = float(e["vclock_now"])
        return {
            "reached": True, "target": target, "window": window,
            "wall_s": round(ts - t0, 2) if ts else None,
            "vclock_s": vclock, "rounds": rnd, "data_bins": trigger,
            "max_accuracy": max_acc, "final_accuracy": final_acc,
        }
    return {
        "reached": False, "target": target, "window": window,
        "wall_s": None, "vclock_s": None, "rounds": None, "data_bins": None,
        "max_accuracy": max_acc, "final_accuracy": final_acc,
    }


# --------------------------------------------------------------------------- #
# Experiment 2 — resource utilization (derived busy/idle fraction)
# --------------------------------------------------------------------------- #
def expt2_utilization(run) -> dict:
    agg, trainers = run["agg"], run["trainers"]
    # trainer busy fraction across trainers
    busy_fracs = []
    for tid, evs in trainers.items():
        rounds = [e for e in evs if e.get("event") == "trainer_round"]
        if not rounds:
            continue
        gpu = sum(e.get("gpu_compute_s") or 0.0 for e in rounds)
        ts = [e.get("ts") for e in evs if e.get("ts") is not None]
        wall = (max(ts) - min(ts)) if len(ts) >= 2 else None
        if wall and wall > 0:
            busy_fracs.append(min(1.0, gpu / wall))
    # aggregator busy vs wait
    agg_rounds = [e for e in agg if e.get("event") == "agg_round"]
    agg_wall = None
    we = [e.get("wall_elapsed_s") for e in agg_rounds if e.get("wall_elapsed_s") is not None]
    if we:
        agg_wall = max(we)
    else:
        ats = [e.get("ts") for e in agg if e.get("ts") is not None]
        agg_wall = (max(ats) - min(ats)) if len(ats) >= 2 else None
    # Aggregator busy = the NON-OVERLAPPING agg_round phase decomposition
    # (aggregate + eval), NOT summed step_timing — timed funcs nest via
    # timer_decorator, so their sum double-counts and can exceed the wall.
    agg_compute = sum(
        (e.get("aggregate_fedavg_s") or 0.0) + (e.get("eval_s") or 0.0) for e in agg_rounds
    )
    barrier = sum(e.get("barrier_wait_s") or 0.0 for e in agg_rounds if e.get("barrier_wait_s"))
    drain = sum(e.get("drain_tail_s") or 0.0 for e in agg_rounds if e.get("drain_tail_s"))
    frac = lambda x: (round(x / agg_wall, 4) if (agg_wall and agg_wall > 0 and x is not None) else None)
    tp = _p(busy_fracs)
    return {
        "trainer_busy_frac_p50": tp["p50"], "trainer_busy_frac_p90": tp["p90"],
        "trainer_busy_frac_p99": tp["p99"], "n_trainers_measured": len(busy_fracs),
        "agg_busy_frac": frac(agg_compute), "agg_barrier_wait_frac": frac(barrier),
        "agg_drain_frac": frac(drain), "agg_wall_s": round(agg_wall, 2) if agg_wall else None,
    }


# --------------------------------------------------------------------------- #
# Experiment 3 — compute productivity (Δloss per unit compute)
# --------------------------------------------------------------------------- #
def expt3_productivity(run) -> dict:
    agg, trainers = run["agg"], run["trainers"]
    _acc, meta = _acc_loss_by_bin(agg)
    losses = [(d, meta[d][0]) for d in sorted(meta) if meta[d][0] is not None]
    first_loss = losses[0][1] if losses else None
    final_loss = losses[-1][1] if losses else None
    delta = (first_loss - final_loss) if (first_loss is not None and final_loss is not None) else None
    # compute: trainer GPU seconds + aggregator compute (aggregate + eval)
    trainer_gpu = sum(
        e.get("gpu_compute_s") or 0.0
        for evs in trainers.values() for e in evs if e.get("event") == "trainer_round"
    )
    agg_compute = sum(
        (e.get("aggregate_fedavg_s") or 0.0) + (e.get("eval_s") or 0.0)
        for e in agg if e.get("event") == "agg_round"
    )
    gpu_total = trainer_gpu + agg_compute
    # forward passes (WS3-b): per-trainer cumulative max, summed
    fwd_total, pert_total, have_fwd = 0, 0, False
    for evs in trainers.values():
        fp = [e.get("forward_passes_total") for e in evs if e.get("forward_passes_total") is not None]
        pt = [e.get("perturbations_total") for e in evs if e.get("perturbations_total") is not None]
        if fp:
            fwd_total += max(fp); have_fwd = True
        if pt:
            pert_total += max(pt)
    return {
        "first_loss": first_loss, "final_loss": final_loss, "delta_loss": delta,
        "trainer_gpu_s": round(trainer_gpu, 2), "agg_compute_s": round(agg_compute, 2),
        "gpu_s_total": round(gpu_total, 2),
        "delta_loss_per_gpu_s": (round(delta / gpu_total, 6) if (delta and gpu_total) else None),
        "forward_passes_total": fwd_total if have_fwd else None,
        "perturbations_total": pert_total if have_fwd else None,
        "delta_loss_per_forward_pass": (
            round(delta / fwd_total, 9) if (delta and have_fwd and fwd_total) else None),
    }


# --------------------------------------------------------------------------- #
# Experiment 4 — data transmitted over the network (WS3-a)
# --------------------------------------------------------------------------- #
def expt4_network(run) -> dict:
    agg, trainers = run["agg"], run["trainers"]
    agg_comm = [e for e in agg if e.get("event") == "comm"]
    trainer_comm = [e for evs in trainers.values() for e in evs if e.get("event") == "comm"]
    if not agg_comm and not trainer_comm:
        return {"comm_telemetry": False}
    a_sizes = [e.get("size_bytes") for e in agg_comm if e.get("direction") == "agg_to_trainer"]
    c_sizes = [e.get("size_bytes") for e in trainer_comm if e.get("direction") == "trainer_to_agg"]
    ap, cp = _p(a_sizes), _p(c_sizes)
    return {
        "comm_telemetry": True,
        "msgs_agg_to_trainer": len(a_sizes), "msgs_trainer_to_agg": len(c_sizes),
        "bytes_agg_to_trainer": sum(s for s in a_sizes if s),
        "bytes_trainer_to_agg": sum(s for s in c_sizes if s),
        "bytes_total": sum(s for s in a_sizes if s) + sum(s for s in c_sizes if s),
        "msg_size_agg_p50": ap["p50"], "msg_size_agg_p90": ap["p90"], "msg_size_agg_p99": ap["p99"],
        "msg_size_client_p50": cp["p50"], "msg_size_client_p90": cp["p90"], "msg_size_client_p99": cp["p99"],
    }


# --------------------------------------------------------------------------- #
# Experiment 5 — client session durations & participation
# --------------------------------------------------------------------------- #
def expt5_sessions(run, is_async: bool) -> dict:
    agg, trainers = run["agg"], run["trainers"]
    # session durations: async = dispatch->commit (contributor_intervals);
    # sync = one-round span (same intervals when present — the agg emits them for
    # both; falls back to per-trainer round span if absent).
    durations = []
    for e in agg:
        if e.get("event") != "agg_round":
            continue
        for iv in (e.get("contributor_intervals") or []):
            d = iv.get("dispatch_ts")
            c = iv.get("commit_ts")
            if d is not None and c is not None and c >= d:
                durations.append(c - d)
    # participation counts per client (rounds / data_bins / iterations)
    rounds_pp, bins_pp, iters_pp = [], [], []
    for evs in trainers.values():
        tr = [e for e in evs if e.get("event") == "trainer_round"]
        if not tr:
            continue
        rounds_pp.append(len({e.get("round") for e in tr if e.get("round") is not None}))
        bins_pp.append(len({e.get("data_id") for e in tr if e.get("data_id") is not None}))
        iters_pp.append(len(tr))
    sp = _p(durations)
    return {
        "session_def": "dispatch_to_commit" if is_async else "one_round_span",
        "session_s_p50": sp["p50"], "session_s_p90": sp["p90"], "session_s_p99": sp["p99"],
        "n_sessions": len(durations),
        "part_rounds_p50": _pct(rounds_pp, 50), "part_rounds_p90": _pct(rounds_pp, 90),
        "part_databins_p50": _pct(bins_pp, 50), "part_databins_p90": _pct(bins_pp, 90),
        "part_iters_p50": _pct(iters_pp, 50), "part_iters_p90": _pct(iters_pp, 90),
        "part_iters_total": sum(iters_pp),
        "session_durations": durations,   # kept for the histogram
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
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiments-dir", default=os.path.join(here, "..", "experiments"))
    ap.add_argument("--variant", choices=["real", "sim"], default="real")
    ap.add_argument("--baselines", default=",".join(_DEFAULT_BASELINES))
    ap.add_argument("--target-acc", type=float, default=0.88)
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--out", default=None, help="output dir for CSV (+plots); default = experiments/_compare")
    ap.add_argument("--plots", action="store_true", help="also write overlay plots (needs matplotlib)")
    args = ap.parse_args()

    exp_dir = os.path.abspath(args.experiments_dir)
    baselines = [b.strip() for b in args.baselines.split(",") if b.strip()]
    out_dir = os.path.abspath(args.out) if args.out else os.path.join(exp_dir, "_compare")
    os.makedirs(out_dir, exist_ok=True)

    found = discover(exp_dir, args.variant)
    rows = {}
    fps = {}
    for b in baselines:
        if b not in found:
            print(f"  [compare] WARN no {args.variant} run found for baseline '{b}' — skipping", file=sys.stderr)
            continue
        n, trace, path = found[b]
        run = load_run(path)
        if not run["agg"]:
            print(f"  [compare] WARN empty aggregator telemetry for '{b}' ({path}) — skipping", file=sys.stderr)
            continue
        is_async = (b == "fluxtune")
        rows[b] = {
            "run_dir": os.path.basename(path), "N": n, "trace": trace or "syn_0",
            **{f"e1_{k}": v for k, v in expt1_time_to_target(run["agg"], args.target_acc, args.window).items()},
            **{f"e2_{k}": v for k, v in expt2_utilization(run).items()},
            **{f"e3_{k}": v for k, v in expt3_productivity(run).items()},
            **{f"e4_{k}": v for k, v in expt4_network(run).items()},
            **{f"e5_{k}": v for k, v in expt5_sessions(run, is_async).items()},
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

    # drop the bulky histogram list before serializing the flat table
    flat = {b: {k: v for k, v in r.items() if not isinstance(v, list)} for b, r in rows.items()}
    cols = []
    for b in baselines:
        for k in flat.get(b, {}):
            if k not in cols:
                cols.append(k)

    # CSV: one row per baseline
    csv_path = os.path.join(out_dir, "compare_baselines.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["baseline"] + cols)
        for b in baselines:
            if b in flat:
                w.writerow([b] + [flat[b].get(c, "") for c in cols])

    # headline table to stdout
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
        _plots(rows, out_dir, args)
    print(f"Out:   {out_dir}")
    return 0


def _plots(rows, out_dir, args):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  [compare] plots skipped (matplotlib unavailable: {e})", file=sys.stderr)
        return
    # session-duration histogram overlay (Experiment 5)
    fig, ax = plt.subplots(figsize=(7, 4))
    any_data = False
    for b, r in rows.items():
        durs = r.get("e5_session_durations") or []
        if durs:
            ax.hist(durs, bins=30, alpha=0.5, label=b)
            any_data = True
    if any_data:
        ax.set_xlabel("client session duration (s)")
        ax.set_ylabel("count")
        ax.set_title("Experiment 5 — client session durations")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "e5_session_hist.pdf"))
    plt.close(fig)
    # network-bytes bar (Experiment 4)
    fig, ax = plt.subplots(figsize=(7, 4))
    bs = [b for b in rows if rows[b].get("e4_bytes_total") is not None]
    if bs:
        ax.bar([f"{b}\n↓agg" for b in bs], [rows[b].get("e4_bytes_agg_to_trainer", 0) for b in bs], alpha=0.7)
        ax.bar([f"{b}\n↑cli" for b in bs], [rows[b].get("e4_bytes_trainer_to_agg", 0) for b in bs], alpha=0.7)
        ax.set_ylabel("bytes")
        ax.set_title("Experiment 4 — data transmitted")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "e4_network_bytes.pdf"))
    plt.close(fig)
    print(f"  [compare] plots written to {out_dir}")


if __name__ == "__main__":
    sys.exit(main())
