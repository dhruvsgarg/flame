#!/usr/bin/env python3
"""Compare virtual-time trajectories between a real run and a simulated run.

Two distinct "speedup" concepts (both printed clearly):

  sim_rate     = vclock / wall_sim   — virtual-seconds per sim wall-second.
                 < 1 means sim is computing slower than real-time.
                 This does NOT say whether sim finishes a given virtual task
                 faster than real; it just measures how fast vclock ticks.

  wall_speedup = real_wall(V) / sim_wall(V)  — for matched virtual time V,
                 how much less wall time sim uses.  > 1 means sim is faster
                 than real (correct behaviour).  This is the true "sim speed
                 advantage" and what we care about for experiment efficiency.

Usage:
    python compare_clock_parity.py <real_run_dir> <sim_run_dir> [options]
    python compare_clock_parity.py --real <dir> --sim <dir> [--tolerance 0.05]
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
from plotters._annot import annotate_percentiles, flush_percentile_table


def load_agg_events(run_dir: str) -> list[dict]:
    pattern = os.path.join(run_dir, "telemetry", "aggregator_*.jsonl")
    files = glob.glob(pattern)
    if not files:
        print(f"  WARNING: no aggregator telemetry in {run_dir}/telemetry/", file=sys.stderr)
        return []
    events = []
    for fpath in files:
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                events.append(obj)
    return events


def load_trainer_events(run_dir: str) -> list[dict]:
    pattern = os.path.join(run_dir, "telemetry", "trainer_*.jsonl")
    events = []
    for fpath in glob.glob(pattern):
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("event") == "trainer_round":
                    events.append(obj)
    return events


def _agg_round_timeline(events: list[dict]) -> list[tuple[int, float, float]]:
    """Return (round, ts, max_trainer_speed_s) from agg_round events, sorted by round.

    ts is the absolute wall timestamp of the aggregation event.
    max_trainer_speed_s is the committed k-th trainer's speed (= vclock advance in sim).
    """
    result = []
    for ev in events:
        if ev.get("event") != "agg_round":
            continue
        r = ev.get("round")
        ts = ev.get("ts")
        speed = ev.get("trainer_speed_s")
        if r is None or ts is None:
            continue
        max_speed = max(speed) if speed else None
        result.append((int(r), float(ts), max_speed))
    return sorted(result, key=lambda x: x[0])


def detect_failsafe_triggered(events: list[dict]) -> tuple[bool, str]:
    """Return (triggered, reason_tag) from aggregator log markers."""
    for ev in events:
        stop = ev.get("stop_reason", "")
        if "SIM_WALL_CEILING" in str(stop):
            return True, "SIM_WALL_CEILING"
        if "WALL_CLOCK_FAILSAFE" in str(stop):
            return True, "WALL_CLOCK_FAILSAFE"
    return False, ""


def report(
    real_dir: str,
    sim_dir: str,
    real_events: list[dict],
    sim_events: list[dict],
    tolerance: float,
    out_dir: str,
) -> None:
    real_label = os.path.basename(real_dir.rstrip("/"))
    sim_label = os.path.basename(sim_dir.rstrip("/"))

    real_tl = _agg_round_timeline(real_events)
    sim_tl = _agg_round_timeline(sim_events)

    sim_failsafe, failsafe_tag = detect_failsafe_triggered(sim_events)

    print(f"\n{'='*72}")
    print(f"  Clock parity: real={real_label}  sim={sim_label}")
    print(f"{'='*72}")
    print(f"  Sim wall-ceiling triggered: "
          f"{'YES [' + failsafe_tag + '] — sim stopped at wall, not vclock' if sim_failsafe else 'NO (correct)'}")

    # ── Compute cumulative virtual time for both modes ──────────────────
    # Real mode: virtual_time = wall elapsed since round-0 timestamp.
    # Sim mode:  virtual_time = cumulative sum of max(trainer_speed_s) per round
    #            (= sum of vclock advances; each advance = kth committed sct).
    real_ts0 = real_tl[0][1] if real_tl else None
    real_vt_by_round: dict[int, float] = {}
    if real_ts0:
        for r, ts, _ in real_tl:
            real_vt_by_round[r] = ts - real_ts0

    sim_vclock: float = 0.0
    sim_vt_by_round: dict[int, float] = {}
    for r, ts, max_speed in sim_tl:
        if max_speed is not None:
            sim_vclock += max_speed
        sim_vt_by_round[r] = sim_vclock

    # Wall time for each mode (last ts - first ts)
    real_wall_total = (real_tl[-1][1] - real_tl[0][1]) if len(real_tl) >= 2 else None
    sim_wall_total = (sim_tl[-1][1] - sim_tl[0][1]) if len(sim_tl) >= 2 else None
    real_vclock_final = max(real_vt_by_round.values()) if real_vt_by_round else None
    sim_vclock_final = max(sim_vt_by_round.values()) if sim_vt_by_round else None

    # sim_rate = vclock / wall  (virtual-s per wall-s, NOT the wall speedup)
    if sim_wall_total and sim_vclock_final:
        sim_rate = sim_vclock_final / sim_wall_total
        print(f"\n  sim_rate = {sim_vclock_final:.0f}s vclock / {sim_wall_total:.0f}s wall"
              f" = {sim_rate:.3f} virtual-s/wall-s")
        print(f"  (sim_rate < 1 means sim ticks slower than real-time; "
              f"doesn't directly imply sim is slower than real for the same task)")

    # wall_speedup: for matched virtual time V, real_wall(V) / sim_wall(V)
    # We match at the sim's final vclock (= V_matched).
    if real_vt_by_round and sim_vclock_final:
        # find real wall elapsed when vclock first reaches sim's final vclock
        V_matched = sim_vclock_final
        real_wall_at_V = None
        for r in sorted(real_vt_by_round):
            if real_vt_by_round[r] >= V_matched:
                real_wall_at_V = real_vt_by_round[r]
                break
        if real_wall_at_V and sim_wall_total:
            wall_speedup = real_wall_at_V / sim_wall_total
            print(f"\n  wall_speedup = real_wall({V_matched:.0f}s vclock) / sim_wall"
                  f" = {real_wall_at_V:.0f}s / {sim_wall_total:.0f}s"
                  f" = {wall_speedup:.2f}x")
            verdict = "FASTER (correct)" if wall_speedup > 1 else "SLOWER (bug iii-c)"
            print(f"  Sim is {verdict} than real for the same virtual work.")

    # ── Per-round virtual-time deviation ───────────────────────────────────
    common = sorted(set(real_vt_by_round) & set(sim_vt_by_round))
    if common:
        deviations = []
        print(f"\n  Per-round virtual-time comparison (tolerance={tolerance*100:.0f}%):")
        print(f"  {'Round':>6}  {'real_vt':>9}  {'sim_vt':>9}  {'dev%':>7}  {'OK?':>5}")
        violations = []
        step = max(1, len(common) // 30)
        for r in common:
            real_vt = real_vt_by_round[r]
            sim_vt = sim_vt_by_round[r]
            dev = abs(sim_vt - real_vt) / real_vt if real_vt > 0 else 0.0
            deviations.append(dev)
            ok = dev <= tolerance
            if not ok:
                violations.append((r, real_vt, sim_vt, dev))
            if r % step == 0 or not ok:
                print(f"  {r:>6}  {real_vt:>9.1f}  {sim_vt:>9.1f}  {dev*100:>6.1f}%  {'OK' if ok else 'WARN':>5}")
        if violations:
            print(f"\n  VIOLATIONS (>{tolerance*100:.0f}%): {len(violations)} / {len(common)} rounds")
        else:
            print(f"\n  All {len(common)} rounds within {tolerance*100:.0f}% tolerance.")
    else:
        print("\n  No common rounds to compare — check telemetry.")

    _plot(real_dir, sim_dir, real_label, sim_label,
          real_vt_by_round, sim_vt_by_round, out_dir)


def _plot(real_dir, sim_dir, real_label, sim_label,
          real_vt: dict[int, float], sim_vt: dict[int, float],
          out_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    c_real, c_sim = "#4e79a7", "#e15759"

    # Left: virtual-time trajectory per round
    ax = axes[0]
    if real_vt:
        rs = sorted(real_vt)
        ax.plot(rs, [real_vt[r] for r in rs], color=c_real,
                lw=1.5, label=f"{real_label} (real: wall elapsed = virtual time)")
        annotate_percentiles(ax, list(real_vt.values()), color=c_real,
                             label=real_label, below=True)
    if sim_vt:
        rs = sorted(sim_vt)
        ax.plot(rs, [sim_vt[r] for r in rs], color=c_sim, ls="--",
                lw=1.5, label=f"{sim_label} (sim: cumulative vclock)")
        annotate_percentiles(ax, list(sim_vt.values()), color=c_sim,
                             label=sim_label, below=True)
    ax.set_xlabel("Round")
    ax.set_ylabel("Cumulative virtual time (s)")
    ax.set_title("Virtual-time trajectory per round\n(P50/P90/P99 in table below)")
    ax.legend(fontsize=8)
    flush_percentile_table(ax)

    # Right: per-round advance comparison (real wall Δ vs sim vclock Δ)
    ax2 = axes[1]
    if real_vt:
        rounds_r = sorted(real_vt)
        adv_r = [real_vt[rounds_r[i]] - real_vt[rounds_r[i-1]]
                 for i in range(1, len(rounds_r))]
        ax2.plot(rounds_r[1:], adv_r, color=c_real, lw=1, alpha=0.7,
                 label=f"{real_label} Δvt/round")
    if sim_vt:
        rounds_s = sorted(sim_vt)
        adv_s = [sim_vt[rounds_s[i]] - sim_vt[rounds_s[i-1]]
                 for i in range(1, len(rounds_s))]
        ax2.plot(rounds_s[1:], adv_s, color=c_sim, lw=1, alpha=0.7, ls="--",
                 label=f"{sim_label} Δvclock/round")
    ax2.set_xlabel("Round")
    ax2.set_ylabel("Virtual-time advance per round (s)")
    ax2.set_title("Per-round virtual-time advance\n(should match for parity)")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    out = os.path.join(out_dir, f"clock_parity_{real_label}_vs_{sim_label}.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("positional", nargs="*", help="real_run_dir sim_run_dir")
    parser.add_argument("--real", metavar="DIR")
    parser.add_argument("--sim", metavar="DIR")
    parser.add_argument("--tolerance", type=float, default=0.05,
                        help="Per-round virtual-time deviation tolerance (default 5%%)")
    parser.add_argument("--out-dir", metavar="DIR", default=None)
    args = parser.parse_args()

    real_dir = args.real
    sim_dir = args.sim
    if not real_dir and len(args.positional) >= 2:
        real_dir, sim_dir = args.positional[0], args.positional[1]
    if not real_dir or not sim_dir:
        parser.print_help()
        sys.exit(1)

    out_dir = args.out_dir or os.path.join(sim_dir, "plots")

    print(f"Loading real run:  {real_dir}")
    real_events = load_agg_events(real_dir)
    print(f"  {len(real_events)} agg events")

    print(f"Loading sim run:   {sim_dir}")
    sim_events = load_agg_events(sim_dir)
    print(f"  {len(sim_events)} agg events")

    report(real_dir, sim_dir, real_events, sim_events, args.tolerance, out_dir)


if __name__ == "__main__":
    main()
