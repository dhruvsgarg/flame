#!/usr/bin/env python3
"""Compare virtual-time trajectories between a real run and a simulated run.

Reads aggregator JSONL telemetry from two run directories and produces:
  1. Per-round virtual-time deviation (real wall-elapsed vs sim vclock).
  2. Sim wall speedup (virtual_budget / sim_wall_time).
  3. Whether the sim was cut by the wall failsafe or stopped correctly at vclock≈T.

The key invariant: for both real and sim modes the virtual-time value at each
round should follow the same trajectory.  For real mode virtual-time = wall
elapsed; for sim mode virtual-time = vclock.now.

Usage:
    python compare_clock_parity.py <real_run_dir> <sim_run_dir> [options]
    python compare_clock_parity.py --real <dir> --sim <dir> [--tolerance 0.05]
"""

import argparse
import glob
import json
import math
import os
import sys
from collections import defaultdict


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


def extract_vclock_progress(events: list[dict]) -> list[tuple[int, float, float]]:
    """Extract (round, vclock_s, wall_s) from VCLOCK_PROGRESS log lines embedded
    as run_meta or other events, or from agg_round events that carry vclock_now."""
    result = []
    for ev in events:
        # agg_round events may carry vclock_now
        if ev.get("event") == "agg_round":
            r = ev.get("round")
            vclock = ev.get("vclock_now")
            wall = ev.get("wall_elapsed_s")
            if r is not None and vclock is not None and wall is not None:
                result.append((int(r), float(vclock), float(wall)))
    return sorted(result, key=lambda x: x[0])


def extract_round_wall_times(events: list[dict]) -> dict[int, float]:
    """For real mode: map round -> wall_elapsed_s from agg_round events."""
    result = {}
    for ev in events:
        if ev.get("event") == "agg_round":
            r = ev.get("round")
            wall = ev.get("wall_elapsed_s")
            if r is not None and wall is not None:
                result[int(r)] = float(wall)
    return result


def extract_sim_completion_by_round(trainer_events: list[dict]) -> dict[int, list[float]]:
    """Map round -> [sim_completion_ts, ...] from trainer_round events."""
    by_round: dict[int, list[float]] = defaultdict(list)
    for ev in trainer_events:
        r = ev.get("round")
        sct = ev.get("sim_completion_ts")
        if r is not None and sct is not None:
            by_round[int(r)].append(float(sct))
    return dict(by_round)


def detect_failsafe_triggered(events: list[dict]) -> bool:
    """Check aggregator log for WALL_CLOCK_FAILSAFE being the stop reason."""
    for ev in events:
        if ev.get("event") == "run_meta":
            stop = ev.get("stop_reason", "")
            if "WALL_CLOCK_FAILSAFE" in str(stop):
                return True
    return False


def report(
    real_dir: str,
    sim_dir: str,
    real_events: list[dict],
    sim_events: list[dict],
    real_trainers: list[dict],
    sim_trainers: list[dict],
    tolerance: float,
    out_dir: str,
) -> None:
    real_label = os.path.basename(real_dir.rstrip("/"))
    sim_label = os.path.basename(sim_dir.rstrip("/"))

    # Real mode: virtual time = wall elapsed at each round
    real_wall_by_round = extract_round_wall_times(real_events)
    # Sim mode: virtual time = vclock at each round
    sim_vclock_progress = extract_vclock_progress(sim_events)

    # Detect failsafe
    sim_failsafe = detect_failsafe_triggered(sim_events)

    # Sim completion times from trainer telemetry (sanity check on vclock)
    sim_sct = extract_sim_completion_by_round(sim_trainers)
    real_sct = extract_sim_completion_by_round(real_trainers)

    print(f"\n{'='*72}")
    print(f"  Clock parity: real={real_label}  sim={sim_label}")
    print(f"{'='*72}")
    print(f"  Sim failsafe triggered: {'YES (bug — sim stopped at wall, not vclock)' if sim_failsafe else 'NO (correct)'}")

    # Wall time for sim
    sim_wall = None
    if sim_events:
        wall_vals = [ev.get("wall_elapsed_s") for ev in sim_events
                     if ev.get("event") == "agg_round" and ev.get("wall_elapsed_s")]
        if wall_vals:
            sim_wall = max(float(v) for v in wall_vals)
    sim_vclock_final = None
    if sim_vclock_progress:
        sim_vclock_final = sim_vclock_progress[-1][1]
    elif sim_sct:
        # approximate vclock from max sim_completion_ts
        all_sct = [v for vals in sim_sct.values() for v in vals]
        if all_sct:
            sim_vclock_final = max(all_sct)

    if sim_wall and sim_vclock_final:
        speedup = sim_vclock_final / sim_wall
        print(f"  Sim speedup: vclock={sim_vclock_final:.0f}s / wall={sim_wall:.0f}s = {speedup:.2f}x")
    else:
        print(f"  Sim speedup: insufficient data (vclock={sim_vclock_final}, wall={sim_wall})")

    # Per-round virtual-time deviation
    if real_wall_by_round and sim_vclock_progress:
        print(f"\n  Per-round virtual-time comparison (tolerance={tolerance*100:.0f}%):")
        print(f"  {'Round':>6}  {'real_vt':>9}  {'sim_vt':>9}  {'deviation':>10}  {'OK?':>5}")
        common_rounds = sorted(
            set(real_wall_by_round.keys()) &
            {r for r, _, _ in sim_vclock_progress}
        )
        violations = []
        for r in common_rounds:
            real_vt = real_wall_by_round[r]
            # find closest sim round
            sim_vt_row = next(((vt, w) for rr, vt, w in sim_vclock_progress if rr == r), None)
            if sim_vt_row is None:
                continue
            sim_vt = sim_vt_row[0]
            if real_vt > 0:
                dev = abs(sim_vt - real_vt) / real_vt
            else:
                dev = 0.0
            ok = dev <= tolerance
            if not ok:
                violations.append((r, real_vt, sim_vt, dev))
            if r % max(1, len(common_rounds) // 20) == 0 or not ok:
                print(f"  {r:>6}  {real_vt:>9.1f}  {sim_vt:>9.1f}  {dev*100:>9.1f}%  {'OK' if ok else 'WARN':>5}")
        if violations:
            print(f"\n  VIOLATIONS (>{tolerance*100:.0f}%): {len(violations)} rounds")
            for r, rv, sv, d in violations[:10]:
                print(f"    round={r} real_vt={rv:.1f} sim_vt={sv:.1f} dev={d*100:.1f}%")
        else:
            print(f"\n  All {len(common_rounds)} compared rounds within {tolerance*100:.0f}% tolerance.")
    else:
        print("\n  Insufficient vclock_now data in telemetry for per-round comparison.")
        print("  (Add vclock_now to agg_round events or rely on VCLOCK_PROGRESS logs.)")

    # Plot
    _plot(real_dir, sim_dir, real_label, sim_label, real_wall_by_round,
          sim_vclock_progress, out_dir)


def _plot(real_dir, sim_dir, real_label, sim_label,
          real_wall: dict[int, float], sim_vclock: list[tuple],
          out_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: virtual-time trajectory
    ax = axes[0]
    if real_wall:
        rounds_r = sorted(real_wall.keys())
        ax.plot(rounds_r, [real_wall[r] for r in rounds_r],
                label=f"{real_label} (real: wall elapsed)", color="#4e79a7", lw=1.5)
    if sim_vclock:
        rounds_s = [r for r, _, _ in sim_vclock]
        vclocks = [v for _, v, _ in sim_vclock]
        ax.plot(rounds_s, vclocks,
                label=f"{sim_label} (sim: vclock)", color="#e15759", lw=1.5, ls="--")
    ax.set_xlabel("Round")
    ax.set_ylabel("Virtual time (s)")
    ax.set_title("Virtual-time trajectory per round")
    ax.legend(fontsize=8)

    # Right: sim wall elapsed vs vclock
    ax2 = axes[1]
    if sim_vclock:
        walls = [w for _, _, w in sim_vclock]
        vcs = [v for _, v, _ in sim_vclock]
        ax2.plot(walls, vcs, color="#59a14f", lw=1.5)
        # ideal line: vclock == wall (1× speedup)
        mx = max(max(walls), max(vcs)) if walls else 1
        ax2.plot([0, mx], [0, mx], "k--", lw=0.8, label="1× speedup")
        ax2.set_xlabel("Wall elapsed (s)")
        ax2.set_ylabel("vclock (s)")
        ax2.set_title(f"{sim_label}: vclock vs wall")
        ax2.legend(fontsize=8)

    fig.tight_layout()
    out = os.path.join(out_dir, f"clock_parity_{real_label}_vs_{sim_label}.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"\n  Plot: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare virtual-time trajectories between a real and a simulated run"
    )
    parser.add_argument("positional", nargs="*", help="real_run_dir sim_run_dir")
    parser.add_argument("--real", metavar="DIR", help="Real-mode run directory")
    parser.add_argument("--sim", metavar="DIR", help="Simulated-mode run directory")
    parser.add_argument(
        "--tolerance", type=float, default=0.05,
        help="Per-round virtual-time deviation tolerance (default: 0.05 = 5%%)",
    )
    parser.add_argument(
        "--out-dir", metavar="DIR", default=None,
        help="Output directory for plots (default: sim_run_dir/plots/)",
    )
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
    real_trainers = load_trainer_events(real_dir)
    print(f"  {len(real_events)} agg events, {len(real_trainers)} trainer_round events")

    print(f"Loading sim run:   {sim_dir}")
    sim_events = load_agg_events(sim_dir)
    sim_trainers = load_trainer_events(sim_dir)
    print(f"  {len(sim_events)} agg events, {len(sim_trainers)} trainer_round events")

    report(real_dir, sim_dir, real_events, sim_events, real_trainers, sim_trainers,
           args.tolerance, out_dir)


if __name__ == "__main__":
    main()
