#!/usr/bin/env python3
"""P2-6 (simulate_fwdllm.md §B fluxtune item 3): correlate `tb_prepare_
perturbation`'s real/sim duration gap (KS 0.267, mean_rel 45.5%, real 1.8ms vs
sim 3.4ms) against (a) which branch it took -- index a cached `v_buffer` vs
`torch.randn_like` a fresh one -- and (b) concurrent-trainer GPU density at
call time, same overlap method as `measure_agg_overlap.py` (union of OTHER
trainers' `trainer_round` `[gpu_pass_start_wall, gpu_pass_end_wall]` windows
against this event's own window -- no new telemetry needed for density,
`tb_prepare_perturbation` already carries `branch` as of this change).

Verdict this script is meant to produce (§B P2-6's own plan):
- density correlates with duration independent of branch -> exempt like
  `eval_model` (§G 07-16), same GPU-contention class.
- branch-rate (real/sim taking the two branches at different frequencies)
  explains the gap instead -> no separate fix, rides the cadence-mismatch
  item already tracked.

Usage: python analyze_tb_prepare_perturbation.py --run <run_dir> [--verbose]
Works on either a real or sim run dir; run it on both sides of a pair and
compare the printed summaries by hand (small, deliberately not automated into
a pass/fail rung yet -- this is the A/B, not the gate).
"""
from __future__ import annotations
import argparse
import glob
import json
import statistics
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


def _load_events(run_dir: Path):
    """(tb_events, gpu_windows) -- tb_events: list of {trainer_id, ts,
    duration_s, branch}; gpu_windows: end_id -> sorted [(start, end)]."""
    tb_events = []
    gpu_windows = {}
    for f in glob.glob(str(run_dir / "telemetry" / "trainer_*.jsonl")):
        end_id = None
        spans = []
        for line in open(f):
            try:
                e = json.loads(line)
            except Exception:
                continue
            ev = e.get("event")
            if ev == "step_timing" and e.get("func") == "tb_prepare_perturbation":
                if e.get("ts") is None or e.get("duration_s") is None:
                    continue
                tb_events.append({
                    "trainer_id": e.get("trainer_id") or e.get("end_id"),
                    "ts": float(e["ts"]),
                    "duration_s": float(e["duration_s"]),
                    "branch": e.get("branch"),  # None on pre-this-change telemetry
                })
            elif ev == "trainer_round":
                s, en = e.get("gpu_pass_start_wall"), e.get("gpu_pass_end_wall")
                if s is None or en is None:
                    continue
                end_id = e.get("end_id")
                spans.append((s, en))
        if end_id and spans:
            gpu_windows[end_id] = sorted(spans)
    return tb_events, gpu_windows


def _overlap_count(windows_by_trainer: dict, exclude: str, lo: float, hi: float) -> int:
    """Count of OTHER trainers with >=1 GPU-pass window overlapping [lo, hi]."""
    n = 0
    for end_id, spans in windows_by_trainer.items():
        if end_id == exclude:
            continue
        if any(max(s, lo) < min(e, hi) for s, e in spans):
            n += 1
    return n


def _mean(xs):
    return statistics.mean(xs) if xs else float("nan")


def analyze(run_dir: Path):
    tb_events, gpu_windows = _load_events(run_dir)
    if not tb_events:
        raise SystemExit(
            f"no tb_prepare_perturbation step_timing events under {run_dir} -- "
            "bank a fresh pair after this change, or check telemetry is enabled."
        )
    have_branch = sum(1 for e in tb_events if e["branch"] is not None)
    for e in tb_events:
        # ts is stamped at _stage_timer's finally (call END); this event's own
        # window is [ts - duration_s, ts], not [ts, ts + duration_s].
        lo, hi = e["ts"] - e["duration_s"], e["ts"]
        e["concurrency"] = _overlap_count(gpu_windows, e["trainer_id"], lo, hi)
    return tb_events, have_branch


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run dir path or name under experiments/")
    ap.add_argument("--verbose", action="store_true", help="print every event")
    args = ap.parse_args()

    run_dir = _resolve_run_dir(args.run)
    events, have_branch = analyze(run_dir)

    durs_ms = [e["duration_s"] * 1e3 for e in events]
    print(f"run: {run_dir.name}")
    print(f"tb_prepare_perturbation events: {len(events)} "
          f"(branch field present on {have_branch})")
    print(f"duration_ms: mean={_mean(durs_ms):.3f} "
          f"median={statistics.median(durs_ms):.3f}")

    if have_branch:
        print("\n-- by branch --")
        for branch in ("cached", "fresh"):
            xs = [e["duration_s"] * 1e3 for e in events if e["branch"] == branch]
            if xs:
                print(f"  {branch:>6}: n={len(xs):<6} mean={_mean(xs):.3f}ms "
                      f"median={statistics.median(xs):.3f}ms")
            else:
                print(f"  {branch:>6}: n=0")

    print("\n-- by concurrent-trainer GPU density --")
    max_c = max((e["concurrency"] for e in events), default=0)
    # Bucket into terciles if density varies, else just report the single value.
    buckets = sorted(set(e["concurrency"] for e in events))
    if len(buckets) <= 6:
        for c in buckets:
            xs = [e["duration_s"] * 1e3 for e in events if e["concurrency"] == c]
            print(f"  concurrency={c:<3} n={len(xs):<6} mean={_mean(xs):.3f}ms")
    else:
        lo_thr, hi_thr = buckets[len(buckets) // 3], buckets[2 * len(buckets) // 3]
        for label, pred in (
            ("low", lambda c: c <= lo_thr),
            ("mid", lambda c: lo_thr < c <= hi_thr),
            ("high", lambda c: c > hi_thr),
        ):
            xs = [e["duration_s"] * 1e3 for e in events if pred(e["concurrency"])]
            print(f"  {label:>4} (n={len(xs)}): mean={_mean(xs):.3f}ms")
    print(f"  (max observed concurrency: {max_c})")

    if args.verbose:
        print("\n-- events --")
        for e in events:
            print(f"  trainer={str(e['trainer_id'])[-4:]:<6} "
                  f"dur_ms={e['duration_s']*1e3:.3f} branch={e['branch']} "
                  f"concurrency={e['concurrency']}")

    print(
        "\nVerdict guide (§B P2-6): if mean duration tracks concurrency "
        "independent of branch -> GPU-density artifact, exempt like eval_model. "
        "If it tracks branch (real/sim take branches at different rates) "
        "instead -> rides the cadence-mismatch item, no separate fix."
    )


if __name__ == "__main__":
    main()
