#!/usr/bin/env python3
"""Measure how much of the aggregator's aggregate() compute overlaps other
trainers' GPU passes, from a REAL-mode run's banked telemetry (§J step 1).

Sim's own trainer concurrency is exactly what #15 is repairing (trainers
idling in recv), so it cannot be the source of truth for how much of
aggregate() is genuinely hidden behind other trainers' GPU work -- this
script only reads REAL-mode telemetry.

For each `agg_round` event that carries `agg_compute_start_wall` /
`agg_compute_end_wall`, take every OTHER (non this-cycle-committing)
trainer's `trainer_round` events and union the overlap of their
[gpu_pass_start_wall, gpu_pass_end_wall] windows against the aggregate()
window. overlap_fraction = overlapped_s / aggregate_fedavg_s.

Usage: python measure_agg_overlap.py --run <real_run_dir> [--verbose]
"""
from __future__ import annotations
import argparse, glob, json
from pathlib import Path

_EXP = Path(__file__).resolve().parent.parent / "experiments"


def _resolve_run_dir(run: str) -> Path:
    p = Path(run)
    if p.is_dir():
        return p
    p2 = _EXP / run
    if p2.is_dir():
        return p2
    raise SystemExit(f"run dir not found: {run}")


def _agg_windows(run_dir: Path):
    f = next(iter(glob.glob(str(run_dir / "telemetry" / "aggregator_*.jsonl"))), None)
    if f is None:
        raise SystemExit(f"no aggregator_*.jsonl under {run_dir / 'telemetry'}")
    out = []
    for line in open(f):
        try:
            e = json.loads(line)
        except Exception:
            continue
        if e.get("event") != "agg_round":
            continue
        start, end = e.get("agg_compute_start_wall"), e.get("agg_compute_end_wall")
        fedavg_s = e.get("aggregate_fedavg_s")
        if start is None or end is None or not fedavg_s:
            continue  # pre-§J-step-1 telemetry -- skip, don't fake it
        out.append({
            "round": e.get("round"),
            "data_id": e.get("cycle_data_id"),  # pre-mutation snapshot (K-D9)
            "start": start,
            "end": end,
            "fedavg_s": fedavg_s,
            "cohort": set(e.get("contributing_trainers") or []),
        })
    return out


def _trainer_windows(run_dir: Path):
    """end_id -> sorted [(start, end)] GPU-pass windows."""
    windows = {}
    for f in glob.glob(str(run_dir / "telemetry" / "trainer_*.jsonl")):
        end_id = None
        spans = []
        for line in open(f):
            try:
                e = json.loads(line)
            except Exception:
                continue
            if e.get("event") != "trainer_round":
                continue
            s, en = e.get("gpu_pass_start_wall"), e.get("gpu_pass_end_wall")
            if s is None or en is None:
                continue
            end_id = e.get("end_id")
            spans.append((s, en))
        if end_id and spans:
            windows[end_id] = sorted(spans)
    return windows


def _merged_overlap_s(intervals, lo, hi):
    """Union length of `intervals` clipped to [lo, hi] (avoids double-
    counting when two other trainers' GPU passes overlap each other)."""
    clipped = []
    for s, e in intervals:
        cs, ce = max(s, lo), min(e, hi)
        if ce > cs:
            clipped.append((cs, ce))
    if not clipped:
        return 0.0
    clipped.sort()
    merged_s = 0.0
    cur_s, cur_e = clipped[0]
    for s, e in clipped[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            merged_s += cur_e - cur_s
            cur_s, cur_e = s, e
    merged_s += cur_e - cur_s
    return merged_s


def measure(run_dir: Path):
    aggs = _agg_windows(run_dir)
    trainer_windows = _trainer_windows(run_dir)
    if not aggs:
        raise SystemExit(
            "no agg_round events carry agg_compute_start_wall/agg_compute_end_wall -- "
            "this run predates the §J step-1 telemetry; bank a fresh real-mode pair."
        )
    rows = []
    for a in aggs:
        others = [w for end_id, spans in trainer_windows.items()
                  for w in spans if end_id not in a["cohort"]]
        overlapped_s = _merged_overlap_s(others, a["start"], a["end"])
        rows.append({
            **a,
            "overlapped_s": overlapped_s,
            "overlap_fraction": overlapped_s / a["fedavg_s"],
        })
    total_fedavg = sum(r["fedavg_s"] for r in rows)
    total_overlap = sum(r["overlapped_s"] for r in rows)
    run_mean_fraction = total_overlap / total_fedavg if total_fedavg else 0.0
    return rows, run_mean_fraction


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="run dir path or name under experiments/")
    ap.add_argument("--verbose", action="store_true", help="print every commit's overlap")
    args = ap.parse_args()

    run_dir = _resolve_run_dir(args.run)
    rows, run_mean_fraction = measure(run_dir)

    print(f"run: {run_dir.name}")
    print(f"commits with §J telemetry: {len(rows)}")
    if args.verbose:
        for r in rows:
            print(f"  round={r['round']!s:<4} data_id={r['data_id']!s:<4} "
                  f"aggregate_fedavg_s={r['fedavg_s']:.3f} overlapped_s={r['overlapped_s']:.3f} "
                  f"overlap_fraction={r['overlap_fraction']:.3f}")
    print(f"\nrun-mean overlap_fraction = {run_mean_fraction:.4f}")
    print("-> sim_agg_compute_overlap_fraction (§J step 2 hyperparameter)")


if __name__ == "__main__":
    main()
