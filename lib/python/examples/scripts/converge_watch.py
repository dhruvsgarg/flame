#!/usr/bin/env python3
# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Convergence-stop watcher (EXPERIMENTS.md WS2).

Polls the aggregator telemetry of the run that `expt_launch` just started and
terminates it the moment the convergence condition is met:

    the last W consecutive data bins are ALL >= target accuracy tau.

Data bins complete in order (data_id = 0,1,2,...) and emit exactly one
``agg_eval`` per bin at completion (verified), so "W continuous bins all >= tau"
is exactly a trailing run of W consecutive-data_id bins each with test-accuracy
>= tau. We track the trailing streak; a bin below tau resets it.

On convergence we write ``converge.json`` (time-to-converge in wall + vclock +
data_id + round) and SIGTERM->SIGKILL the run's process group. If the run exits
on its own first (hit a safety cap / crashed), the pgid vanishes and we exit
WITHOUT writing converge.json -> the harness reports DID_NOT_CONVERGE.

Entirely side-car: it only reads telemetry and signals the process group. When
``--target-acc`` is not passed the harness never launches this, so default runs
are byte-identical.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import signal
import sys
import time


def _newest_agg_jsonl(exp_dir: str, marker: str) -> str | None:
    """Newest run_*/telemetry/aggregator_*.jsonl newer than the launch marker."""
    try:
        marker_mtime = os.path.getmtime(marker)
    except OSError:
        marker_mtime = 0.0
    best, best_mtime = None, marker_mtime - 1.0
    for run_dir in glob.glob(os.path.join(exp_dir, "run_*")):
        try:
            if os.path.getmtime(run_dir) < marker_mtime - 5.0:
                continue  # pre-existing run dir, not ours
        except OSError:
            continue
        for f in glob.glob(os.path.join(run_dir, "telemetry", "aggregator_*.jsonl")):
            try:
                m = os.path.getmtime(f)
            except OSError:
                continue
            if m > best_mtime:
                best, best_mtime = f, m
    return best


def _scan(agg_path: str):
    """Return (data_id->accuracy, latest_vclock, max_round) from agg telemetry."""
    acc_by_bin: dict[int, float] = {}
    latest_vclock = None
    max_round = 0
    with open(agg_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue  # partial trailing line from a line-buffered writer
            ev = r.get("event")
            if r.get("round") is not None:
                try:
                    max_round = max(max_round, int(r["round"]))
                except (TypeError, ValueError):
                    pass
            if ev == "agg_eval":
                d = r.get("data_id")
                a = r.get("test-accuracy")
                if d is not None and a is not None:
                    acc_by_bin[int(d)] = float(a)  # last eval for the bin wins
            elif ev == "agg_round":
                v = r.get("vclock_now")
                if v is not None:
                    latest_vclock = float(v)
    return acc_by_bin, latest_vclock, max_round


def _trailing_streak(acc_by_bin: dict[int, float], target: float):
    """Length of the trailing run of consecutive data_ids all >= target, and the
    first data_id in that run. Bins must be contiguous (no gap) to count."""
    if not acc_by_bin:
        return 0, None
    top = max(acc_by_bin)
    streak, first = 0, None
    d = top
    while d in acc_by_bin and acc_by_bin[d] >= target:
        streak += 1
        first = d
        d -= 1
    return streak, first


def _pgid_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False
    except OSError:
        return False


def _terminate(pgid: int, grace_s: float) -> None:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pgid, sig)
        except OSError:
            return
        if sig is signal.SIGTERM:
            deadline = time.time() + grace_s
            while time.time() < deadline and _pgid_alive(pgid):
                time.sleep(1.0)
            if not _pgid_alive(pgid):
                return


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", required=True, help="<example>/experiments")
    ap.add_argument("--marker", required=True, help="expt_launch pre-launch marker file")
    ap.add_argument("--target-acc", type=float, required=True)
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--pgid", type=int, required=True, help="run process-group id to kill on convergence")
    ap.add_argument("--converge-json", required=True)
    ap.add_argument("--poll", type=float, default=15.0)
    ap.add_argument("--grace", type=float, default=20.0)
    # Stall guard: terminate early (before the wall ceiling) if the run is clearly
    # not learning — best accuracy has not improved by >= --stall-min-delta within
    # any --stall-window-s wall window. 0 window ⇒ disabled.
    ap.add_argument("--stall-window-s", type=float, default=0.0,
                    help="terminate if best acc hasn't gained --stall-min-delta in this many wall s (0=off)")
    ap.add_argument("--stall-min-delta", type=float, default=0.01,
                    help="minimum accuracy gain that counts as progress (default 0.01 = 1%%)")
    ap.add_argument("--stall-json", default=None, help="written on a stall termination")
    args = ap.parse_args()

    try:
        start_wall = os.path.getmtime(args.marker)
    except OSError:
        start_wall = time.time()

    label = os.path.basename(args.converge_json)
    _stall_desc = (f"; stall if <{args.stall_min_delta:.3f} acc gain in {args.stall_window_s/3600:.1f}h"
                   if args.stall_window_s > 0 else "")
    print(f"  [converge] watching for {args.window} consecutive bins >= {args.target_acc:.4f} acc"
          f"{_stall_desc}", flush=True)

    # Stall tracking: milestone_acc = best acc at the last recorded improvement;
    # last_improve_ts = when that improvement happened. A gain >= min_delta resets
    # the clock; the clock also runs from launch, so a run that produces NO eval
    # (or no progress) within the window is caught too.
    milestone_acc = None
    last_improve_ts = time.time()

    while _pgid_alive(args.pgid):
        agg_path = _newest_agg_jsonl(args.exp_dir, args.marker)
        if agg_path is None:
            time.sleep(args.poll)
            continue
        try:
            acc_by_bin, vclock, max_round = _scan(agg_path)
        except OSError:
            time.sleep(args.poll)
            continue
        streak, first = _trailing_streak(acc_by_bin, args.target_acc)
        best = max(acc_by_bin.values()) if acc_by_bin else None
        now = time.time()
        # progress bookkeeping for the stall guard
        if best is not None and (milestone_acc is None or best - milestone_acc >= args.stall_min_delta):
            milestone_acc = best
            last_improve_ts = now
        no_improve_s = now - last_improve_ts
        if acc_by_bin:
            top = max(acc_by_bin)
            _stall_note = (f" no_improve={no_improve_s/60:.0f}m/{args.stall_window_s/60:.0f}m"
                           if args.stall_window_s > 0 else "")
            print(f"  [converge] bins={len(acc_by_bin)} top_bin={top} "
                  f"streak>={args.target_acc:.3f}={streak}/{args.window} "
                  f"best_acc={best:.4f}{_stall_note}", flush=True)
        if streak >= args.window:
            wall_s = time.time() - start_wall
            payload = {
                "converged": True,
                "target_accuracy": args.target_acc,
                "window": args.window,
                "trigger_data_id": max(acc_by_bin),
                "first_bin_in_window": first,
                "n_bins_completed": len(acc_by_bin),
                "time_to_converge_wall_s": round(wall_s, 3),
                "time_to_converge_vclock_s": vclock,
                "rounds_at_converge": max_round,
                "agg_telemetry": agg_path,
                "window_accuracies": [acc_by_bin[b] for b in
                                      range(first, max(acc_by_bin) + 1)],
            }
            try:
                with open(args.converge_json, "w", encoding="utf-8") as fh:
                    json.dump(payload, fh, indent=2)
            except OSError as e:
                print(f"  [converge] WARN could not write {args.converge_json}: {e}", flush=True)
            print(f"  [{label}] CONVERGED at data_id={max(acc_by_bin)} "
                  f"(wall={wall_s:.0f}s vclock={vclock}) -> terminating run", flush=True)
            _terminate(args.pgid, args.grace)
            return 0
        # Stall termination: no >=min_delta gain within the window (and NOT converged).
        if args.stall_window_s > 0 and no_improve_s >= args.stall_window_s:
            payload = {
                "stalled": True,
                "reason": "no_accuracy_improvement",
                "stall_window_s": args.stall_window_s,
                "stall_min_delta": args.stall_min_delta,
                "no_improve_s": round(no_improve_s, 1),
                "best_accuracy": best,
                "milestone_accuracy": milestone_acc,
                "target_accuracy": args.target_acc,
                "n_bins_completed": len(acc_by_bin),
                "wall_s": round(now - start_wall, 1),
                "rounds": max_round,
                "agg_telemetry": agg_path,
            }
            if args.stall_json:
                try:
                    with open(args.stall_json, "w", encoding="utf-8") as fh:
                        json.dump(payload, fh, indent=2)
                except OSError as e:
                    print(f"  [converge] WARN could not write {args.stall_json}: {e}", flush=True)
            print(f"  [{label}] STALLED: best_acc={best} gained <{args.stall_min_delta} in "
                  f"{args.stall_window_s/3600:.1f}h -> terminating run (not learning)", flush=True)
            _terminate(args.pgid, args.grace)
            return 0
        time.sleep(args.poll)

    print(f"  [converge] run process group ended before convergence", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
