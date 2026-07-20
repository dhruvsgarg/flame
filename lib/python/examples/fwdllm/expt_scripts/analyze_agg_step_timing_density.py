#!/usr/bin/env python3
"""P1 item 2/4 (simulate_fwdllm.md §B, 2026-07-19): correlate the still-open
`agg_step_timing_breakdown` residual (fluxtune's `_compute_var`/`_distribute_
weights_async`/`_prepare_round_state`/`_process_aggregation_goal_met`; fwdllm/
fwdllm_plus's `_apply_weighted_update`/`_compute_var`/etc) against concurrent-
trainer GPU density at call time -- same overlap method as `measure_agg_
overlap.py` / `analyze_tb_prepare_perturbation.py` (union of trainers'
`trainer_round` `[gpu_pass_start_wall, gpu_pass_end_wall]` windows against the
aggregator event's own window). No new telemetry needed -- these functions are
already individually `@timer_decorator`-wrapped (`FedSgdAggregator.py`); this
reads the existing aggregator `step_timing` events and existing trainer
`trainer_round` events.

Verdict this script is meant to produce:
- density correlates with duration -> GPU/CPU contention from sim's denser
  trainer pool (same class as the already-accepted `eval_model`/
  `tb_prepare_perturbation` artifacts) -- candidate for `mean_tol_rel` widening
  or exemption, not an algorithmic bug.
- no density correlation -> the gap is NOT explained by contention burstiness;
  the `c=agg_goal=10` zero-slack hypothesis (P1 item 2) needs a different test.

Usage: python analyze_agg_step_timing_density.py --run <run_dir> [--funcs f1,f2,...]
Works on either a real or sim run dir; run it on both sides of a pair and
compare the printed summaries by hand -- not automated into a pass/fail rung.
"""
from __future__ import annotations
import argparse
import glob
import json
import statistics
from pathlib import Path

_EXP = Path(__file__).resolve().parent.parent / "experiments"

_DEFAULT_FUNCS = (
    "_compute_var", "_apply_weighted_update", "_prepare_round_state",
    "_distribute_weights_async", "_distribute_weights_sync",
    "_process_aggregation_goal_met", "_replay_buffered_cohort_contribs",
)


def _resolve_run_dir(run: str) -> Path:
    p = Path(run)
    if p.is_dir():
        return p
    p2 = _EXP / run
    if p2.is_dir():
        return p2
    raise SystemExit(f"run dir not found: {run}")


def _load_events(run_dir: Path, funcs: set):
    """(func -> [{ts, duration_s}]), gpu_windows: end_id -> sorted [(start, end)]."""
    agg_files = glob.glob(str(run_dir / "telemetry" / "aggregator_*.jsonl"))
    if not agg_files:
        raise SystemExit(f"no aggregator_*.jsonl under {run_dir}/telemetry")
    events_by_func: dict = {f: [] for f in funcs}
    for line in open(agg_files[0]):
        try:
            e = json.loads(line)
        except Exception:
            continue
        if e.get("event") != "step_timing":
            continue
        func = e.get("func")
        if func not in funcs or e.get("ts") is None or e.get("duration_s") is None:
            continue
        rec = {"ts": float(e["ts"]), "duration_s": float(e["duration_s"])}
        if e.get("cpu_duration_s") is not None:
            rec["cpu_duration_s"] = float(e["cpu_duration_s"])
        events_by_func[func].append(rec)

    gpu_windows: dict = {}
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
            gpu_windows[end_id] = sorted(spans)
    return events_by_func, gpu_windows


def _overlap_count(windows_by_trainer: dict, lo: float, hi: float) -> int:
    """Count of trainers with >=1 GPU-pass window overlapping [lo, hi]. The
    aggregator has no `end_id` of its own to exclude -- count ALL trainers."""
    n = 0
    for spans in windows_by_trainer.values():
        if any(max(s, lo) < min(e, hi) for s, e in spans):
            n += 1
    return n


def _mean(xs):
    return statistics.mean(xs) if xs else float("nan")


def _bucket_print(events: list):
    max_c = max((e["concurrency"] for e in events), default=0)
    buckets = sorted(set(e["concurrency"] for e in events))
    if len(buckets) <= 6:
        for c in buckets:
            xs = [e["duration_s"] * 1e3 for e in events if e["concurrency"] == c]
            print(f"    concurrency={c:<3} n={len(xs):<6} mean={_mean(xs):.4f}ms")
    else:
        lo_thr, hi_thr = buckets[len(buckets) // 3], buckets[2 * len(buckets) // 3]
        for label, pred in (
            ("low", lambda c: c <= lo_thr),
            ("mid", lambda c: lo_thr < c <= hi_thr),
            ("high", lambda c: c > hi_thr),
        ):
            xs = [e["duration_s"] * 1e3 for e in events if pred(e["concurrency"])]
            print(f"    {label:>4} (n={len(xs)}): mean={_mean(xs):.4f}ms")
    print(f"    (max observed concurrency: {max_c})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run dir path or name under experiments/")
    ap.add_argument("--funcs", default=None,
                     help=f"comma-separated func names (default: {','.join(_DEFAULT_FUNCS)})")
    args = ap.parse_args()

    funcs = set(args.funcs.split(",")) if args.funcs else set(_DEFAULT_FUNCS)
    run_dir = _resolve_run_dir(args.run)
    events_by_func, gpu_windows = _load_events(run_dir, funcs)

    print(f"run: {run_dir.name}")
    for func, events in events_by_func.items():
        if not events:
            continue
        for e in events:
            # ts is stamped at timer_decorator's end (call END); window is
            # [ts - duration_s, ts], not [ts, ts + duration_s].
            lo, hi = e["ts"] - e["duration_s"], e["ts"]
            e["concurrency"] = _overlap_count(gpu_windows, lo, hi)
        durs_ms = [e["duration_s"] * 1e3 for e in events]
        print(f"\n{func}: n={len(events)} mean={_mean(durs_ms):.4f}ms "
              f"median={statistics.median(durs_ms):.4f}ms")
        cpu_ms = [e["cpu_duration_s"] * 1e3 for e in events if "cpu_duration_s" in e]
        if cpu_ms:
            wall_mean = _mean(durs_ms)
            cpu_mean = _mean(cpu_ms)
            ratio = f"{cpu_mean / wall_mean:.2f}" if wall_mean else "n/a"
            print(f"    cpu_duration: n={len(cpu_ms)} mean={cpu_mean:.4f}ms cpu/wall={ratio}")
        else:
            print("    cpu_duration: no data (run predates cpu_duration_s telemetry, re-run to get it)")
        _bucket_print(events)

    print(
        "\nVerdict guide (concurrency): if mean duration climbs with concurrency "
        "-> GPU/CPU contention from sim's denser trainer pool, same class as "
        "eval_model/tb_prepare_perturbation. Flat/backwards across buckets -> "
        "contention doesn't explain this function's gap, look elsewhere."
        "\nVerdict guide (cpu/wall, now thread-local `time.thread_time()`, §G "
        "07-20): ratio near 1 -> thread was busy the whole window, sim is doing "
        "genuinely more computation, look for a code-path difference. Ratio "
        "well below 1 -> thread was blocked/idle part of the window (contention, "
        "not the gpu_pass-window kind already refuted). Ratio > 1 should no "
        "longer happen -- treat as a bug, not a signal."
    )


if __name__ == "__main__":
    main()
