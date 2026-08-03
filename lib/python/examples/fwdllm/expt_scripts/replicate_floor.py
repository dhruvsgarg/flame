#!/usr/bin/env python3
"""Measures the pipeline's own run-to-run REPRODUCIBILITY floor from same-mode,
same-seed, same-config replicate runs — the number every DIST parity tolerance
has to clear to be meaningful.

Why this exists: a parity rung compares real against sim and calls a gap a bug.
But the fwdllm cadence loop is a feedback system (variance gate -> iterations ->
model updates -> variance), so two runs of the SAME mode with the SAME seed do
not land on the same numbers either. Until that self-variance is measured, a
rung tolerance is a guess, and a rung tighter than the floor manufactures fails
nobody can ever fix (`v2_var_trajectory` shipped at 2% against a ~2% floor).
This is §D-5's "absolute, mode-independent sanity check", instantiated.

Reads only telemetry already on disk — no runs required. Compare its output
against the corresponding rung tolerance in
`async_cifar10/scripts/parity/checks.py`:

    metric              rung                 tolerance field
    committed bins      throughput           tol_rel
    iters/bin           v1_iter_per_data_id  mean_tol_rel
    mean var            v2_var_trajectory    mean_tol_rel
    cycles              cohort_sequence      count_tol

Usage:
    python replicate_floor.py                          # all baselines, real legs
    python replicate_floor.py --baselines fedbuff_round --mode real
    python replicate_floor.py --min-duration 3000      # only compare like durations

Replicates must be comparable: runs are grouped by (baseline, trace, mode,
max_runtime_s) and only groups with >= 2 members are reported. Runs of different
lengths are NEVER pooled — `iters/bin` rises monotonically with run length
(9.82 @1800s -> 10.85 @3600s -> 12.39 @5400s on fedbuff_round real), so mixing
durations measures the training curve, not reproducibility.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics as st
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_DEFAULT_EXPERIMENTS = _HERE.parent / "experiments"

_RUN_RE = re.compile(
    r"^run_(?P<ts>\d{8}_\d{6})_(?P<baseline>.+)_n(?P<n>\d+)_smoke"
    r"(?:_(?P<trace>.+))?_(?P<variant>real|sim)$"
)

# Metric -> the parity rung + tolerance field it calibrates.
_CALIBRATES = {
    "committed_bins": ("throughput", "tol_rel", 0.05),
    "cycles": ("cohort_sequence", "count_tol", 0.05),
    "iters_per_bin": ("v1_iter_per_data_id", "mean_tol_rel", 0.15),
    "mean_var": ("v2_var_trajectory", "mean_tol_rel", 0.02),
}


def _agg_events(run_dir: str) -> list:
    paths = glob.glob(os.path.join(run_dir, "telemetry", "aggregator_*.jsonl"))
    if not paths:
        return []
    out = []
    for line in open(paths[0], errors="replace"):
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e.get("event") == "agg_round":
            out.append(e)
    return out


def _max_runtime_s(run_dir: str):
    cfg = os.path.join(run_dir, "aggregator_config.json")
    if not os.path.exists(cfg):
        return None
    try:
        h = json.load(open(cfg)).get("hyperparameters", {})
        return h.get("max_runtime_s") or h.get("maxRuntimeS")
    except (ValueError, OSError):
        return None


def _seed(run_dir: str):
    cfg = os.path.join(run_dir, "aggregator_config.json")
    if not os.path.exists(cfg):
        return None
    try:
        return json.load(open(cfg)).get("hyperparameters", {}).get("seed")
    except (ValueError, OSError):
        return None


def achieved_span_s(run_dir: str):
    """Wall span the aggregator ACTUALLY covered.

    `max_runtime_s` is what the run was ASKED for; a run killed early still
    reports it, and pooling one reads its truncation as irreproducibility.
    """
    ts = [e["ts"] for e in _agg_events(run_dir) if isinstance(e.get("ts"), (int, float))]
    return (max(ts) - min(ts)) if len(ts) >= 2 else None


def metrics(run_dir: str):
    """The four run-level quantities the DIST rungs grade, from one run."""
    ev = _agg_events(run_dir)
    if not ev:
        return None
    committed, iters, variances = set(), {}, []
    for e in ev:
        cid = e.get("cycle_data_id")
        if cid is None:
            continue
        key = (e.get("round") or 0, cid)
        if e.get("var_good_enough") is True:
            committed.add(key)
        it = e.get("iteration_per_data_id")
        if it is not None:
            iters[key] = max(iters.get(key, 0), it + 1)
        if e.get("var") is not None:
            variances.append(e["var"])
    if not committed or not iters:
        return None
    return {
        "committed_bins": float(len(committed)),
        "cycles": float(len(ev)),
        "iters_per_bin": st.mean(iters.values()),
        "mean_var": st.mean(variances) if variances else float("nan"),
    }


def _jvp_eval_mode(path: str) -> bool:
    """Was this leg trained with dropout off (`jvp_eval_mode`, H13)?

    Read from the trainer log because the knob is written to NO config file in
    the run dir -- it lives in the trainer's `config_overrides`, which the runner
    does not dump. Absent means the run predates the flag, i.e. the code default:
    dropout LIVE. Pooling an ON leg with an OFF one would measure the flag rather
    than the floor, which is exactly what §D-45 forbids.
    """
    for lg in glob.glob(os.path.join(path, "*trainers.log")):
        try:
            with open(lg, errors="ignore") as fh:
                for i, line in enumerate(fh):
                    if "jvp_eval_mode=" in line:
                        return "jvp_eval_mode=True" in line
                    if i > 50000:      # the knob logs at trainer init or never
                        break
        except OSError:
            continue
    return False


def discover(experiments_dir: str, baselines, mode: str) -> dict:
    """{(baseline, trace, mode, max_runtime_s, jvp_eval_mode): [(ts, path), ...]}"""
    groups: dict = {}
    for path in sorted(glob.glob(os.path.join(experiments_dir, "run_*"))):
        m = _RUN_RE.match(os.path.basename(path))
        if not m or m["variant"] != mode:
            continue
        if baselines and m["baseline"] not in baselines:
            continue
        key = (m["baseline"], m["trace"] or "", mode, _max_runtime_s(path),
               _jvp_eval_mode(path))
        groups.setdefault(key, []).append((m["ts"], path))
    return groups


def _spread(vals: list) -> float:
    """Max pairwise relative spread — the floor a 2-sided tolerance must clear."""
    lo, hi = min(vals), max(vals)
    return (hi - lo) / hi if hi else 0.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiments-dir", default=str(_DEFAULT_EXPERIMENTS))
    ap.add_argument("--baselines", nargs="*", default=None)
    ap.add_argument("--mode", choices=("real", "sim"), default="real",
                    help="which side's replicates to pool (default real)")
    ap.add_argument("--min-duration", type=float, default=0.0,
                    help="skip groups whose max_runtime_s is below this")
    ap.add_argument("--span-tol", type=float, default=0.05,
                    help="drop a leg whose ACHIEVED span is this far below the "
                         "group's longest (default 0.05 = 5%%); a truncated run "
                         "is a shorter run, not a replicate")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args(argv)

    groups = discover(args.experiments_dir, args.baselines, args.mode)
    report, any_group = {}, False
    for key in sorted(groups, key=lambda k: (k[0], k[3] or 0, k[4])):
        baseline, trace, mode, maxrt, jvp_eval = key
        # Name the training config in the header: two groups of the same baseline
        # and duration now appear, and reading the wrong one inverts the verdict.
        cfg = f"  jvp_eval_mode={jvp_eval}"
        runs = groups[key]
        if len(runs) < 2 or (maxrt or 0) < args.min_duration:
            continue
        rows = []
        for ts, path in runs:
            m = metrics(path)
            if m:
                rows.append((ts, _seed(path), m, achieved_span_s(path)))
        # A leg killed early is a shorter run wearing the same `max_runtime_s`,
        # not a replicate -- keep the longest-span cohort.
        spans = [s for _, _, _, s in rows if s]
        dropped = []
        if spans:
            ref = max(spans)
            keep = []
            for row in rows:
                if row[3] is not None and row[3] < ref * (1.0 - args.span_tol):
                    dropped.append(row)
                else:
                    keep.append(row)
            rows = keep
        label = "/".join(x for x in (baseline, trace) if x)
        # Report drops BEFORE the too-few-replicates bail, else a group that fell
        # below 2 from a truncated leg reads as "no replicates found", unexplained.
        if dropped:
            print(f"\n=== {label}  mode={mode}  max_runtime_s={maxrt}{cfg}")
            for ts, _s, _m, span in dropped:
                print(f"    {ts}  DROPPED — achieved span {span:.0f}s is >{args.span_tol:.0%} "
                      f"short of {max(spans):.0f}s (truncated run, not a replicate)")
        if len(rows) < 2:
            if dropped:
                print(f"    only {len(rows)} full-length leg(s) left — no floor for this group")
            continue
        seeds = {s for _, s, _, _ in rows}
        any_group = True
        print((f"    n_replicates={len(rows)}  seeds={sorted(seeds)}" if dropped else
               f"\n=== {label}  mode={mode}  max_runtime_s={maxrt}{cfg}  "
               f"n_replicates={len(rows)}  seeds={sorted(seeds)}")
              + ("   ⚠ MIXED SEEDS — not a reproducibility floor" if len(seeds) > 1 else ""))
        for ts, seed, m, span in rows:
            print(f"    {ts}  bins={m['committed_bins']:.0f}  cycles={m['cycles']:.0f}  "
                  f"iters/bin={m['iters_per_bin']:.2f}  var={m['mean_var']:.4f}"
                  + (f"  span={span:.0f}s" if span else "  span=?"))
        print(f"    {'metric':16s} {'floor':>8s}   {'rung':<22s} {'tol':>7s}  verdict")
        entry = {}
        for metric, (rung, field, tol) in _CALIBRATES.items():
            vals = [m[metric] for _, _, m, _ in rows]
            if any(v != v for v in vals):     # NaN
                continue
            floor = _spread(vals)
            verdict = ("OK" if tol > floor * 1.5 else
                       "TIGHT — within 1.5x of the floor" if tol > floor else
                       "BELOW FLOOR — grades noise")
            print(f"    {metric:16s} {floor:7.1%}   {rung:<22s} {tol:6.1%}  {verdict}")
            entry[metric] = {"floor_rel": round(floor, 4), "rung": rung,
                             "tolerance_field": field, "tolerance": tol,
                             "verdict": verdict}
        report[label + f"@{maxrt}"] = {
            "mode": mode, "jvp_eval_mode": jvp_eval,
            "n_replicates": len(rows), "seeds": sorted(seeds),
            "achieved_span_s": [s for _, _, _, s in rows],
            "dropped_truncated": [t for t, _, _, _ in dropped], "metrics": entry}

    if not any_group:
        print("No replicate groups found (need >= 2 runs sharing "
              "baseline/trace/mode/max_runtime_s).")
        return 1
    print("\nA tolerance at or below its floor cannot be closed by any code change — "
          "raise it to the floor or grade the metric differently.")
    if args.json_out:
        json.dump(report, open(args.json_out, "w"), indent=2)
        print(f"json: {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
