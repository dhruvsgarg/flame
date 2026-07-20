#!/usr/bin/env python3
"""§B tracker row 2 (simulate_fwdllm.md, 2026-07-20 pm-5): correlate sim's
`per_round_advance` tail outliers against that round's own aggregator
`step_timing` wall cost, to test whether the still-open matched-window KS
shape gap (FW/FW+, `per_round_advance`) is a `vclock` hygiene bug (aggregator
wall-time leaking onto the virtual clock, violating §F-1's "never put
overhead on the vclock") or something else. No new telemetry -- both series
(`vclock_now` per `agg_round`, `step_timing` per aggregator function) are
already on disk for any banked pair.

Per-round advance is Δvclock_now between consecutive progress-axis events
(`_per_round_advances`, use_vclock=True); a round's step_timing wall sum is
every `step_timing` event whose `ts` falls in (prev_event.ts, curr_event.ts].
Correlating the two on SIM only, since `per_round_advance`'s residual is a
sim-side shape gap (KS on the matched window), not a real-side one.

Verdict this script is meant to produce:
- advance correlates with step_timing sum -> aggregator wall-time IS leaking
  onto the vclock somewhere -- a real bug against §F-1, needs a code fix, not
  a tolerance change.
- no correlation -> the residual KS gap is NOT a vclock-hygiene bug; look at
  dispatch-order/tie-breaking randomness in the reorder buffer instead.

Usage: python analyze_per_round_advance_vs_step_timing.py --sim-run <dir>
"""
from __future__ import annotations
import argparse
import glob
import os
import statistics
import sys
from pathlib import Path

_LIB_PYTHON = Path(__file__).resolve().parents[3]  # lib/python
_PARITY_SCRIPTS = _LIB_PYTHON / "examples" / "async_cifar10" / "scripts"
for _p in (str(_LIB_PYTHON), str(_PARITY_SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from parity.checks import (  # noqa: E402
    load_agg_jsonl, _progress_axis, _per_progress_last_event,
)

_EXP = Path(__file__).resolve().parent.parent / "experiments"


def _resolve_run_dir(run: str) -> Path:
    p = Path(run)
    if p.is_dir():
        return p
    p2 = _EXP / run
    if p2.is_dir():
        return p2
    raise SystemExit(f"run dir not found: {run}")


def _load(run_dir: Path) -> dict:
    tel = glob.glob(os.path.join(str(run_dir), "telemetry", "aggregator_*.jsonl"))
    if not tel:
        raise SystemExit(f"no aggregator_*.jsonl under {run_dir}/telemetry")
    return load_agg_jsonl(tel[0])


def _pearson(xs: list, ys: list) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = statistics.mean(xs), statistics.mean(ys)
    sx = statistics.pstdev(xs)
    sy = statistics.pstdev(ys)
    if sx == 0 or sy == 0:
        return float("nan")
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / n
    return cov / (sx * sy)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim-run", required=True)
    args = ap.parse_args()

    sim_dir = _resolve_run_dir(args.sim_run)
    sim = _load(sim_dir)

    axis = _progress_axis(sim["agg_rounds"])
    by_round = _per_progress_last_event(sim["agg_rounds"], axis)
    rounds_sorted = sorted(by_round.keys())
    if len(rounds_sorted) < 2:
        raise SystemExit("fewer than 2 progress-axis rounds -- run too short to measure")

    step_events = sorted(
        (e for e in sim["step_timing"] if e.get("ts") is not None and e.get("duration_s") is not None),
        key=lambda e: e["ts"])
    step_ts = [e["ts"] for e in step_events]

    import bisect

    def _step_sum(lo: float, hi: float) -> float:
        i = bisect.bisect_left(step_ts, lo)
        j = bisect.bisect_right(step_ts, hi)
        return sum(step_events[k]["duration_s"] for k in range(i, j))

    advances, step_sums = [], []
    for i in range(1, len(rounds_sorted)):
        e_prev = by_round[rounds_sorted[i - 1]]
        e_curr = by_round[rounds_sorted[i]]
        v_prev, v_curr = e_prev.get("vclock_now"), e_curr.get("vclock_now")
        t_prev, t_curr = e_prev.get("ts"), e_curr.get("ts")
        if v_prev is None or v_curr is None or t_prev is None or t_curr is None:
            continue
        adv = v_curr - v_prev
        if adv <= 0:
            continue
        advances.append(adv)
        step_sums.append(_step_sum(t_prev, t_curr))

    n = len(advances)
    print(f"sim: {sim_dir.name}")
    print(f"rounds compared: {n}")
    if n < 3:
        raise SystemExit("too few matched rounds to correlate")

    r = _pearson(step_sums, advances)
    print(f"\nPearson r(step_timing_sum, per_round_advance) = {r:.3f}")

    # Tail-outlier bucket: top decile of advance vs the rest.
    paired = sorted(zip(advances, step_sums), key=lambda p: p[0])
    cut = max(1, n // 10)
    tail = paired[-cut:]
    rest = paired[:-cut]
    tail_step = [s for _, s in tail]
    rest_step = [s for _, s in rest]
    print(f"\ntop-decile advance rounds (n={len(tail)}): "
          f"mean step_timing_sum={statistics.mean(tail_step) * 1e3:.3f}ms")
    print(f"remaining rounds        (n={len(rest)}): "
          f"mean step_timing_sum={statistics.mean(rest_step) * 1e3:.3f}ms")

    print(
        "\nVerdict guide: |r| notably > 0 (or the tail bucket's mean step_timing_sum "
        "clearly exceeds the rest) -> aggregator wall-time is leaking onto the "
        "vclock somewhere, a real bug against §F-1 ('never put overhead on the "
        "vclock') -- needs a code fix. r near 0 and buckets comparable -> the "
        "vclock is properly isolated from aggregator wall-time; the residual KS "
        "shape gap is NOT a vclock-hygiene bug -- look at dispatch-order/"
        "tie-breaking randomness in the reorder buffer instead."
    )


if __name__ == "__main__":
    main()
