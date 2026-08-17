#!/usr/bin/env python3
"""Side-car that kills an arm that has stopped making progress, so a node does not
burn hours on a run that is already void.

**Why not the harness's own stall guard.** `converge_watch.py` arms on held-out
accuracy and requires `--target-acc`, which would also end the arm on convergence
-- and a controller arm that does not end on `[BudgetStop] reason=budget` is void
(P4.7 defect 1). Worse, its stall signal is wrong here: reaching a plateau and
holding it is precisely what the controller is *supposed* to do, so an accuracy
stall guard would kill the success case. What actually fails on these arms is
mechanical, and all three modes have already cost a node:

  * **hang** -- commits stop entirely (`--stall-window-s`, default 20 min);
  * **gate starvation** -- `N_req ~ rho_t^2` under `gate_rho_ref=annealed`, so an
    annealing rho demands less pooling every commit until `I` floors at 1 and the
    server path has no trainer work amortising it. G-2's `003648` died this way
    with `I` floored on 98% of its commits, and the launch projection could not
    see it because it prices law C off the `ln 2` prior (defect 3);
  * **zero steps** -- `rho_star == 0`, a requirement of zero rather than an absent
    one, which was 23-48% of the 2026-08-16 arms (defect 2).

Both rate checks wait out `--grace-commits` (default 200, the horizon §6 already
says to read) so early noise cannot trip them.

    ./watch_arm.py --exp-dir <experiments/> --since <ts> --pgid <pid> [--kill]

Writes `arm_stall.json` next to the run it killed. Reads telemetry only.
"""
import argparse
import glob
import json
import os
import signal
import sys
import time


def newest_run(exp_dir, since):
    """Newest run_*/ created after `since` -- ours, not a pre-existing one."""
    best, best_m = None, since
    for d in glob.glob(os.path.join(exp_dir, "run_*")):
        try:
            m = os.path.getmtime(d)
        except OSError:
            continue
        if m >= best_m:
            best, best_m = d, m
    return best


def scan(run_dir):
    """(commits, trips, zero_rho, trips_in_last_window) from one pass of the log.

    Both events come off the same file in emission order, so trips-per-commit
    needs no counter of its own -- the interleaving is the measurement.
    """
    commits = trips = zero = 0
    tail = []          # trips between each of the last commits
    pending = 0
    for f in sorted(glob.glob(os.path.join(run_dir, "telemetry",
                                           "aggregator_*.jsonl"))):
        try:
            fh = open(f, errors="ignore")
        except OSError:
            continue
        with fh:
            for line in fh:
                if '"agg_round"' in line:
                    trips += 1
                    pending += 1
                elif '"server_update"' in line:
                    commits += 1
                    tail.append(pending)
                    pending = 0
                    if len(tail) > 200:
                        tail.pop(0)
                    if '"rho_star": 0,' in line or '"rho_star": 0.0,' in line:
                        zero += 1
    recent = (sum(tail) / len(tail)) if tail else None
    return commits, trips, zero, recent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", required=True)
    ap.add_argument("--since", type=float, required=True,
                    help="epoch seconds; only run dirs newer than this are ours")
    ap.add_argument("--pgid", type=int, default=0, help="process group to kill")
    ap.add_argument("--kill", action="store_true",
                    help="actually signal on a breach (default: report only)")
    ap.add_argument("--poll", type=float, default=60.0)
    ap.add_argument("--stall-window-s", type=float, default=1200.0,
                    help="no new commit for this long => hang")
    ap.add_argument("--grace-commits", type=int, default=200,
                    help="rate checks only apply past this many commits")
    ap.add_argument("--trips-floor", type=float, default=3.0)
    ap.add_argument("--max-hours", type=float, default=0.0,
                    help="hard cap on this arm's wall clock; 0 = none")
    a = ap.parse_args()

    t0 = time.time()
    last_commits, last_change = 0, time.time()
    run = None
    print(f"[watch] exp_dir={a.exp_dir} stall={a.stall_window_s / 60:.0f}m "
          f"grace={a.grace_commits} trips_floor={a.trips_floor} "
          f"kill={'on' if a.kill else 'off (report only)'}", flush=True)

    while True:
        time.sleep(a.poll)
        if a.pgid and not _alive(a.pgid):
            print("[watch] run exited on its own -- done", flush=True)
            return 0
        run = newest_run(a.exp_dir, a.since) or run
        if run is None:
            if time.time() - t0 > a.stall_window_s:
                return _fire(a, None, "no run directory appeared")
            continue

        commits, trips, zero, recent = scan(run)
        if commits > last_commits:
            last_commits, last_change = commits, time.time()
        idle = time.time() - last_change
        print(f"[watch] {os.path.basename(run)} commits={commits} trips={trips} "
              f"trips/commit(recent)={recent if recent is None else round(recent, 2)} "
              f"rho_star==0:{zero} idle={idle / 60:.0f}m", flush=True)

        if a.max_hours and (time.time() - t0) > a.max_hours * 3600:
            return _fire(a, run, f"exceeded --max-hours {a.max_hours}")
        if idle > a.stall_window_s:
            return _fire(a, run, f"no new commit for {idle / 60:.0f} min "
                                 f"(hang; last commit #{commits})")
        if commits >= a.grace_commits:
            if zero:
                return _fire(a, run, f"{zero} commits took a step of length zero "
                                     f"(P4.7 defect 2)")
            if recent is not None and recent < a.trips_floor:
                return _fire(a, run, f"trips/commit {recent:.2f} over the last "
                                     f"{min(commits, 200)} commits is below "
                                     f"{a.trips_floor:g} -- the gate is starving "
                                     f"(P4.7 defect 3, G-2's death)")


def _alive(pgid):
    try:
        os.killpg(pgid, 0)
        return True
    except (ProcessLookupError, PermissionError, OSError):
        return False


def _fire(a, run, why):
    print(f"\n!!! [watch] BREACH: {why}", flush=True)
    if run:
        try:
            with open(os.path.join(run, "arm_stall.json"), "w") as fh:
                json.dump({"stalled": True, "reason": why,
                           "at": time.strftime("%FT%T")}, fh, indent=2)
        except OSError:
            pass
    if a.kill and a.pgid:
        print(f"!!! [watch] terminating pgid {a.pgid}", flush=True)
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(a.pgid, sig)
            except OSError:
                break
            time.sleep(20)
            if not _alive(a.pgid):
                break
    return 1


if __name__ == "__main__":
    sys.exit(main())
