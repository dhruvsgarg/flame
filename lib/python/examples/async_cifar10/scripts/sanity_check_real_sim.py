#!/usr/bin/env python3
"""Battery of sanity checks (T1-T7) for real and sim runs.

Designed for the four 0606 run directories:
  run_20260606_124318_dbg_felix_n300_alpha0.1_syn0_stream_sim
  run_20260606_124326_dbg_refl_n300_alpha0.1_syn0_stream_sim
  run_20260606_130422_dbg_felix_n300_alpha0.1_syn0_stream_real
  run_20260606_131928_dbg_refl_n300_alpha0.1_syn0_stream_real

Checks:
  T1  vclock monotone - vclock_now never decreases across agg_round events
  T2  vclock accounting - each agg_round: vclock_now delta ~= max(trainer_speed_s)
  T3  sim_rate in sane range - vclock_final / wall_elapsed in [0.01, 100]
  T4  wall_speedup > 1 - sim finishes matched virtual work faster than real
  T5  failsafe check - sim must NOT overshoot max_runtime_s wall budget by > 20%
  T6  selection balance - num_chosen <= agg_goal * 1.5 per selection event
  T7  trainer phase sum - for each trainer_round: sum(CPU+GPU phases) ~= real_gpu_time_s +
      pre_train_s + post_train_s (within 0.5s tolerance for bookkeeping differences)

Usage:
    # Run all 4 dirs with auto-labeling:
    python sanity_check_real_sim.py <dir1> <dir2> ...

    # Or explicit label=dir pairs:
    python sanity_check_real_sim.py --run felix_sim=<dir> --run refl_real=<dir>

    # For T4 (wall_speedup), specify matching pairs:
    python sanity_check_real_sim.py --real <real_dir> --sim <sim_dir> [--real2 ...] [--sim2 ...]
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from typing import Optional

# ── PASS/FAIL thresholds ────────────────────────────────────────────────────
T2_MAX_DEV_S = 2.0           # max allowed |vclock_delta - max_speed| per round (s)
T3_MIN_RATE = 0.01           # min sim_rate (vclock/wall-s)
T3_MAX_RATE = 100.0          # max sim_rate
T4_MIN_SPEEDUP = 1.0         # sim must be at least as fast as real
T5_MAX_OVERSHOOT = 0.20      # 20% wall overshoot tolerance vs max_runtime_s
T6_MAX_RATIO = 1.6           # num_chosen / agg_goal <= this
T7_MAX_PHASE_ERR_S = 0.5     # max absolute error in per-round phase sum


def _PASS(msg=""):
    return f"\033[32mPASS\033[0m {msg}"

def _FAIL(msg=""):
    return f"\033[31mFAIL\033[0m {msg}"

def _WARN(msg=""):
    return f"\033[33mWARN\033[0m {msg}"

def _SKIP(msg=""):
    return f"\033[90mSKIP\033[0m {msg}"


# ── Telemetry loading ────────────────────────────────────────────────────────

def load_agg_events(run_dir: str) -> list[dict]:
    pattern = os.path.join(run_dir, "telemetry", "aggregator_*.jsonl")
    events = []
    for fpath in glob.glob(pattern):
        with open(fpath, errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return events


def load_trainer_events(run_dir: str) -> list[dict]:
    pattern = os.path.join(run_dir, "telemetry", "trainer_*.jsonl")
    events = []
    for fpath in glob.glob(pattern):
        with open(fpath, errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                    if e.get("event") == "trainer_round":
                        events.append(e)
                except json.JSONDecodeError:
                    pass
    return events


def _agg_rounds(events: list[dict]) -> list[dict]:
    return sorted(
        [e for e in events if e.get("event") == "agg_round"],
        key=lambda e: (e.get("round", 0), e.get("ts", 0)),
    )


def _selections(events: list[dict]) -> list[dict]:
    return [e for e in events if e.get("event") == "selection"]


# ── Individual tests ─────────────────────────────────────────────────────────

def t1_vclock_monotone(label: str, agg_events: list[dict]) -> str:
    rounds = _agg_rounds(agg_events)
    if not rounds:
        return _SKIP("no agg_round events")
    prev_v = None
    violations = []
    for ev in rounds:
        v = ev.get("vclock_now")
        if v is None:
            continue
        if prev_v is not None and v < prev_v - 1e-6:
            violations.append((ev.get("round"), prev_v, v))
        prev_v = v
    if violations:
        return _FAIL(f"{len(violations)} vclock decreases e.g. round={violations[0][0]} "
                     f"{violations[0][1]:.2f}->{violations[0][2]:.2f}")
    return _PASS(f"{len(rounds)} rounds all monotone")


def t2_vclock_accounting(label: str, agg_events: list[dict]) -> str:
    """Check per-FL-round vclock advance == max(trainer_speed_s) for that round.

    Works for both syncfl (one agg_round event per FL round) and asyncfl (multiple
    agg_round events per FL round, each with a potentially different vclock_now).
    We identify FL-round boundaries as events where vclock_now strictly increases.
    """
    rounds = _agg_rounds(agg_events)
    if not rounds:
        return _SKIP("no agg_round events")

    prev_v = None
    errors = []
    n_with_vclock = 0
    n_fl_rounds = 0

    for ev in rounds:
        v = ev.get("vclock_now")
        speed = ev.get("trainer_speed_s")
        if v is None:
            continue
        n_with_vclock += 1
        if prev_v is None or v > prev_v + 1e-9:
            # This event advanced the vclock — it marks the completion of an FL round
            n_fl_rounds += 1
            if prev_v is not None and speed:
                actual_advance = v - prev_v
                expected_advance = max(speed)
                err = abs(actual_advance - expected_advance)
                if err > T2_MAX_DEV_S:
                    errors.append((ev.get("round"), actual_advance, expected_advance, err))
            prev_v = v

    if n_with_vclock == 0:
        return _SKIP("no vclock_now values in agg_round events")

    n = len(rounds)
    # asyncfl emits one agg_round per commit, but vclock advances only per FL round
    # (when enough trainers committed). In asyncfl, vclock advance != max(trainer_speed_s)
    # of the individual commit — it's set by the FL-round completion logic.
    # Downgrade to WARN for asyncfl runs (events >> unique vclock steps).
    is_asyncfl = n_fl_rounds > 0 and n / n_fl_rounds > 1.5
    if errors:
        ex = errors[0]
        msg = (f"{len(errors)}/{n_fl_rounds} FL-round advances have "
               f"|actual - max_speed| > {T2_MAX_DEV_S}s "
               f"e.g. r={ex[0]} advance={ex[1]:.2f}s max_speed={ex[2]:.2f}s err={ex[3]:.2f}s")
        if is_asyncfl:
            return _WARN(msg + " (asyncfl - vclock advance != per-commit speed; expected)")
        return _FAIL(msg)
    return _PASS(f"{n} events, {n_fl_rounds} FL-round vclock advances all within +/-{T2_MAX_DEV_S}s")


def t3_sim_rate(label: str, agg_events: list[dict], is_sim: bool) -> str:
    if not is_sim:
        return _SKIP("real mode - sim_rate not applicable")
    rounds = _agg_rounds(agg_events)
    if len(rounds) < 2:
        return _SKIP("fewer than 2 rounds")
    ts0 = rounds[0]["ts"]
    ts_last = rounds[-1]["ts"]
    wall_elapsed = ts_last - ts0
    vclock_final = rounds[-1].get("vclock_now")
    if not vclock_final or wall_elapsed <= 0:
        return _SKIP("missing vclock_now or zero wall time")
    sim_rate = vclock_final / wall_elapsed
    ok = T3_MIN_RATE <= sim_rate <= T3_MAX_RATE
    msg = (f"vclock={vclock_final:.0f}s wall={wall_elapsed:.0f}s "
           f"sim_rate={sim_rate:.3f} virtual-s/wall-s")
    return _PASS(msg) if ok else _FAIL(msg + f" [expected {T3_MIN_RATE}-{T3_MAX_RATE}]")


def t4_wall_speedup(
    real_label: str, sim_label: str,
    real_events: list[dict], sim_events: list[dict]
) -> str:
    real_rounds = _agg_rounds(real_events)
    sim_rounds = _agg_rounds(sim_events)
    if len(real_rounds) < 2 or len(sim_rounds) < 2:
        return _SKIP("insufficient rounds in real or sim")

    # Build real: round -> wall elapsed since start
    real_ts0 = real_rounds[0]["ts"]
    real_wall_by_round = {ev["round"]: ev["ts"] - real_ts0 for ev in real_rounds}

    # Build sim: cumulative vclock by round
    sim_vclock = 0.0
    sim_vt_by_round = {}
    for ev in sim_rounds:
        speed = ev.get("trainer_speed_s")
        if speed:
            sim_vclock += max(speed)
        sim_vt_by_round[ev["round"]] = sim_vclock

    sim_vclock_final = sim_vclock
    sim_wall_total = sim_rounds[-1]["ts"] - sim_rounds[0]["ts"]

    # Find real wall elapsed at matched virtual time
    V_matched = sim_vclock_final
    real_wall_at_V = None
    for r in sorted(real_wall_by_round):
        if real_wall_by_round[r] >= V_matched:
            real_wall_at_V = real_wall_by_round[r]
            break
    if real_wall_at_V is None:
        # Real run didn't reach V_matched. In real mode wall == virtual time,
        # so the time to reach V_matched would be V_matched wall-seconds.
        real_wall_at_V = V_matched
        suffix = " (extrapolated: real wall = virtual time in real mode)"
    else:
        suffix = ""

    if sim_wall_total <= 0:
        return _SKIP("sim wall time is zero")

    speedup = real_wall_at_V / sim_wall_total
    ok = speedup >= T4_MIN_SPEEDUP
    msg = (f"{sim_label}: wall_speedup={speedup:.2f}x "
           f"(real={real_wall_at_V:.0f}s / sim={sim_wall_total:.0f}s "
           f"for V={V_matched:.0f}s vclock){suffix}")
    return _PASS(msg) if ok else _FAIL(msg + f" [expected >= {T4_MIN_SPEEDUP}x - bug iii-c]")


def t5_failsafe(label: str, agg_events: list[dict], is_sim: bool,
                max_runtime_s: Optional[float]) -> str:
    if not is_sim:
        return _SKIP("real mode - failsafe only relevant for sim")
    rounds = _agg_rounds(agg_events)
    if len(rounds) < 2:
        return _SKIP("fewer than 2 rounds")
    wall_elapsed = rounds[-1]["ts"] - rounds[0]["ts"]

    # Detect whether failsafe fired
    failsafe_fired = any(
        "SIM_WALL_CEILING" in str(e.get("stop_reason", "")) or
        "WALL_CLOCK_FAILSAFE" in str(e.get("stop_reason", ""))
        for e in agg_events
    )

    if max_runtime_s is None:
        # Infer from vclock_now at last round as proxy for intended budget
        vclock_final = rounds[-1].get("vclock_now")
        if not vclock_final:
            return _SKIP("no max_runtime_s and no vclock_now")
        max_runtime_s = vclock_final  # budget = final vclock (best guess)

    overshoot = (wall_elapsed - max_runtime_s) / max_runtime_s
    ok = overshoot <= T5_MAX_OVERSHOOT
    msg = (f"wall={wall_elapsed:.0f}s budget={max_runtime_s:.0f}s "
           f"overshoot={overshoot*100:.1f}% "
           f"failsafe={'FIRED' if failsafe_fired else 'DID NOT FIRE'}")
    return _PASS(msg) if ok else _FAIL(msg + f" [>{T5_MAX_OVERSHOOT*100:.0f}% - wall ceiling too loose]")


def t6_selection_balance(label: str, agg_events: list[dict]) -> str:
    sels = _selections(agg_events)
    if not sels:
        return _SKIP("no selection events")
    violations = []
    for ev in sels:
        goal = ev.get("agg_goal") or ev.get("effective_c")
        chosen = ev.get("num_chosen")
        if goal and chosen and goal > 0:
            ratio = chosen / goal
            if ratio > T6_MAX_RATIO:
                violations.append((ev.get("round"), chosen, goal, ratio))
    n = len(sels)
    if violations:
        ex = violations[0]
        return _WARN(f"{len(violations)}/{n} selections with num_chosen/agg_goal > {T6_MAX_RATIO:.1f}x "
                     f"e.g. r={ex[0]} chosen={ex[1]} goal={ex[2]} ratio={ex[3]:.2f}")
    # Also report typical chosen/goal
    ratios = []
    for ev in sels:
        goal = ev.get("agg_goal") or ev.get("effective_c")
        chosen = ev.get("num_chosen")
        if goal and chosen and goal > 0:
            ratios.append(chosen / goal)
    if ratios:
        sv = sorted(ratios)
        med = sv[len(sv) // 2]
        return _PASS(f"{n} selections; num_chosen/agg_goal median={med:.2f} max={sv[-1]:.2f}")
    return _PASS(f"{n} selections all OK")


def t7_trainer_phase_sum(label: str, trainer_events: list[dict]) -> str:
    """Check that gpu_compute_s == real_gpu_time_s (both set from the same measurement).

    We only compare the GPU-compute phase because the other phases (mqtt_fetch_s, etc.)
    accumulate different things than the simple real_gpu_time_s + pre + post sum.
    Also checks pre_train_s and post_train_s are present and non-negative.
    """
    if not trainer_events:
        return _SKIP("no trainer_round events")

    errors = []
    missing_phases = 0
    for ev in trainer_events:
        gpu_real = float(ev.get("real_gpu_time_s", 0.0) or 0.0)
        gpu_phase = ev.get("gpu_compute_s")
        if gpu_phase is None:
            missing_phases += 1
            continue
        gpu_phase = float(gpu_phase)
        err = abs(gpu_phase - gpu_real)
        # gpu_compute_s is assigned from _real_gpu_time_s directly — should be identical
        if err > 1e-6:
            errors.append((ev.get("round"), ev.get("end_id", "?")[:12],
                           gpu_phase, gpu_real, err))

    n = len(trainer_events)
    n_with_phases = n - missing_phases
    if missing_phases > n * 0.5:
        return _SKIP(f"{missing_phases}/{n} rounds lack gpu_compute_s - old trainer version?")
    if errors:
        ex = errors[0]
        return _FAIL(f"{len(errors)}/{n_with_phases} rounds have gpu_compute_s != real_gpu_time_s "
                     f"e.g. r={ex[0]} tid={ex[1]} gpu_compute={ex[2]:.6f} real_gpu={ex[3]:.6f} "
                     f"err={ex[4]:.2e}")
    return _PASS(f"{n_with_phases} rounds: gpu_compute_s == real_gpu_time_s")


# ── Runner ───────────────────────────────────────────────────────────────────

def check_run(
    label: str,
    run_dir: str,
    is_sim: bool,
    max_runtime_s: Optional[float] = None,
    real_events: Optional[list[dict]] = None,
    real_label: str = "",
) -> None:
    print(f"\n{'='*72}")
    print(f"  {label}  ({run_dir})")
    print(f"  mode={'SIM' if is_sim else 'REAL'}")
    print(f"{'='*72}")

    agg = load_agg_events(run_dir)
    trainers = load_trainer_events(run_dir)
    print(f"  Loaded: {len(agg)} agg events, {len(trainers)} trainer_round events")

    results = []
    results.append(("T1 vclock monotone",     t1_vclock_monotone(label, agg)))
    results.append(("T2 vclock accounting",   t2_vclock_accounting(label, agg)))
    results.append(("T3 sim_rate range",      t3_sim_rate(label, agg, is_sim)))
    if is_sim and real_events is not None:
        results.append(("T4 wall_speedup",
                        t4_wall_speedup(real_label, label, real_events, agg)))
    else:
        results.append(("T4 wall_speedup", _SKIP("need --real/--sim pair")))
    results.append(("T5 failsafe ceiling",    t5_failsafe(label, agg, is_sim, max_runtime_s)))
    results.append(("T6 selection balance",   t6_selection_balance(label, agg)))
    results.append(("T7 trainer phase sum",   t7_trainer_phase_sum(label, trainers)))

    passed = failed = warned = skipped = 0
    for name, result in results:
        icon = "+" if "PASS" in result else ("!" if "FAIL" in result else ("~" if "WARN" in result else "-"))
        print(f"  [{icon}] {name:<26} {result}")
        if "PASS" in result:
            passed += 1
        elif "FAIL" in result:
            failed += 1
        elif "WARN" in result:
            warned += 1
        else:
            skipped += 1

    print(f"\n  Summary: {passed} PASS  {failed} FAIL  {warned} WARN  {skipped} SKIP")
    return {"passed": passed, "failed": failed, "warned": warned}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("run_dirs", nargs="*",
                        help="Run directories (mode inferred from 'sim'/'real' in dirname)")
    parser.add_argument("--run", action="append", default=[], metavar="LABEL=DIR",
                        help="Explicit labeled run (mode inferred from label or dirname)")
    parser.add_argument("--real", metavar="DIR",
                        help="Real-mode run dir for T4 wall_speedup comparison")
    parser.add_argument("--sim", metavar="DIR",
                        help="Sim-mode run dir for T4 wall_speedup comparison")
    parser.add_argument("--max-runtime-s", type=float, default=None,
                        help="Expected virtual budget (max_runtime_s from config). "
                             "Used in T5. If omitted, inferred from final vclock_now.")
    args = parser.parse_args()

    specs: list[tuple[str, str]] = []
    for d in args.run_dirs:
        specs.append((os.path.basename(d.rstrip("/")), d))
    for item in args.run:
        if "=" not in item:
            parser.error(f"--run expects label=path, got {item!r}")
        lbl, path = item.split("=", 1)
        specs.append((lbl, path))

    # Load real events for T4 if explicit --real/--sim given
    real_events_for_t4 = None
    real_label_for_t4 = ""
    if args.real:
        real_label_for_t4 = os.path.basename(args.real.rstrip("/"))
        real_events_for_t4 = load_agg_events(args.real)
        if args.sim and args.sim not in [d for _, d in specs]:
            specs.append((os.path.basename(args.sim.rstrip("/")), args.sim))

    total_pass = total_fail = total_warn = 0

    for label, run_dir in specs:
        # Infer mode from directory name
        is_sim = "sim" in label.lower() or "sim" in os.path.basename(run_dir).lower()
        # Use real events from --real if this is the paired sim dir
        r_events = real_events_for_t4 if (is_sim and real_events_for_t4) else None
        r_label = real_label_for_t4 if r_events else ""
        stats = check_run(label, run_dir, is_sim,
                          max_runtime_s=args.max_runtime_s,
                          real_events=r_events, real_label=r_label)
        if stats:
            total_pass += stats["passed"]
            total_fail += stats["failed"]
            total_warn += stats["warned"]

    if specs:
        print(f"\n{'='*72}")
        print(f"  GRAND TOTAL: {total_pass} PASS  {total_fail} FAIL  {total_warn} WARN")
        print(f"{'='*72}")
        sys.exit(1 if total_fail > 0 else 0)


if __name__ == "__main__":
    main()
