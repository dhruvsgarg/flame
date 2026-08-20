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


class Scan:
    """Running commit/trip counts for ONE run, read incrementally.

    TWO commit signals. `server_update` exists only under `--server-update-audit`,
    which the real profiling arm deliberately does not set -- the audit's per-commit
    work lands inside the `fedavg` span that arm exists to price. Reading it alone
    saw commits=0 on a healthy 61-commit yahoo arm and killed it at the
    pre-first-commit grace, 82% into its budget (2026-08-20). `version_bump_census`
    is unconditional and fires once per commit in the same `var_good_enough` branch,
    so it is the signal that is always there.

    The LARGER stream wins, never their sum: with the audit on both fire once per
    commit, and max() is then still the commit count.

    Both events come off the same file in emission order, so trips-per-commit
    needs no counter of its own -- the interleaving is the measurement.

    INCREMENTAL, because re-reading the file each poll is O(run^2): the aggregator
    writes ~4.5 MB/min, so a 14 h scored arm (run_node_p4.sh's CEIL) ends at ~4 GB
    and 60 s polls would re-read ~1.6 TB, contending with the run's own I/O. Keep a
    per-file offset and consume only whole lines that are new.
    """

    _CAP = 200                       # trips-per-commit window, in commits

    def __init__(self, run_dir):
        self.run_dir = run_dir
        self._off = {}               # path -> bytes already consumed
        self.trips = self.zero = 0
        self.commits = {"audit": 0, "census": 0}
        self._tail = {"audit": [], "census": []}
        self._pending = {"audit": 0, "census": 0}
        self._recent_i = []          # iteration_per_data_id, last _CAP commits
        self._recent_starved = []    # n_eff < n_req, i.e. gate not met

    def g2_signature(self):
        """(I-floored fraction, gate-unmet fraction, n) over the recent window.

        The DIRECT test for G-2's death, which `trips/commit` only proxies: G-2
        floored `I` at 1 on 98% of its commits with the pool demand unmet. Both
        fields ride on `server_update`, so this reads (None, None, 0) on an arm
        without --server-update-audit -- unread, never silently passing.
        """
        n = len(self._recent_i)
        if not n:
            return None, None, 0
        floored = sum(1 for i in self._recent_i if i is not None and i <= 1) / n
        starved = ((sum(self._recent_starved) / len(self._recent_starved))
                   if self._recent_starved else None)
        return floored, starved, n

    def update(self):
        """(commits, trips, zero_rho, trips_in_last_window, signal) as of now."""
        for f in sorted(glob.glob(os.path.join(self.run_dir, "telemetry",
                                               "aggregator_*.jsonl"))):
            off = self._off.get(f, 0)
            try:
                if os.path.getsize(f) < off:      # rotated/truncated: start over
                    off = 0
                with open(f, "rb") as fh:
                    fh.seek(off)
                    data = fh.read()
            except OSError:
                continue
            # Whole lines only; a half-written record waits for the next poll.
            cut = data.rfind(b"\n")
            if cut < 0:
                continue
            self._off[f] = off + cut + 1
            for line in data[:cut].split(b"\n"):
                self._consume(line)
        k = ("audit" if self.commits["audit"] >= self.commits["census"]
             else "census")
        tail = self._tail[k]
        recent = (sum(tail) / len(tail)) if tail else None
        return self.commits[k], self.trips, self.zero, recent, k

    def _consume(self, line):
        if b'"agg_round"' in line:
            self.trips += 1
            self._pending["audit"] += 1
            self._pending["census"] += 1
            return
        if b'"server_update"' in line:
            k = "audit"
            if b'"rho_star": 0,' in line or b'"rho_star": 0.0,' in line:
                self.zero += 1
            self._note_gate(line)
        elif b'"version_bump_census"' in line:
            k = "census"
        else:
            return
        self.commits[k] += 1
        self._tail[k].append(self._pending[k])
        self._pending[k] = 0
        if len(self._tail[k]) > self._CAP:
            self._tail[k].pop(0)

    def _note_gate(self, line):
        """Keep this commit's I and whether the pool demand was met."""
        try:
            d = json.loads(line)
        except ValueError:
            return
        self._recent_i.append(d.get("iteration_per_data_id"))
        if len(self._recent_i) > self._CAP:
            self._recent_i.pop(0)
        n_eff, n_req = d.get("n_eff"), d.get("n_req")
        if n_eff is not None and n_req is not None:
            self._recent_starved.append(n_eff < n_req)
            if len(self._recent_starved) > self._CAP:
                self._recent_starved.pop(0)


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
                    help="no new commit for this long => hang (once commit 1 lands)")
    ap.add_argument("--first-commit-grace-s", type=float, default=2700.0,
                    help="separate, much longer budget for reaching commit 1: a "
                         "REAL-mode arm at seq 256 legitimately takes tens of "
                         "minutes to its first commit, and the steady-state stall "
                         "window is far too tight for it")
    ap.add_argument("--grace-commits", type=int, default=200,
                    help="rate checks only apply past this many commits")
    ap.add_argument("--trips-floor", type=float, default=3.0,
                    help="trips/commit below this AND the pool demand unmet => the "
                         "gate is starving. The conjunction is the point: "
                         "trips/commit is n_req/agg_goal, and a landing controller "
                         "drives n_req down BY DESIGN (agnews 2026-08-20 projects to "
                         "n_req~5 at its stop), so on its own this floor voids every "
                         "controller arm at any setting above ~0.5.")
    ap.add_argument("--i-floor-frac", type=float, default=0.9,
                    help="kill when iteration_per_data_id is 1 on this fraction of "
                         "the recent window -- G-2's actual death (98%). Direct, "
                         "where --trips-floor is a proxy. Needs --server-update-audit.")
    ap.add_argument("--max-hours", type=float, default=0.0,
                    help="hard cap on this arm's wall clock; 0 = none")
    a = ap.parse_args()

    t0 = time.time()
    last_commits, last_change = 0, time.time()
    run, scanner = None, None
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

        if scanner is None or scanner.run_dir != run:
            scanner = Scan(run)          # a new run dir invalidates every counter
        commits, trips, zero, recent, signal = scanner.update()
        if commits > last_commits:
            last_commits, last_change = commits, time.time()
        idle = time.time() - last_change
        print(f"[watch] {os.path.basename(run)} commits={commits}({signal}) "
              f"trips={trips} "
              f"trips/commit(recent)={recent if recent is None else round(recent, 2)} "
              f"rho_star==0:{zero} idle={idle / 60:.0f}m", flush=True)

        if a.max_hours and (time.time() - t0) > a.max_hours * 3600:
            return _fire(a, run, f"exceeded --max-hours {a.max_hours}")
        # Before commit 1 there is no cadence to be "stalled" against -- the arm
        # is still spinning up 100 trainers. Killing on the steady-state window
        # here false-positived a real yelp-p run at commit 0 (2026-08-17).
        _budget = a.stall_window_s if commits > 0 else a.first_commit_grace_s
        if idle > _budget:
            return _fire(a, run, f"no new commit for {idle / 60:.0f} min "
                                 f"(limit {_budget / 60:.0f} min"
                                 f"{'' if commits else ', pre-first-commit'}; "
                                 f"last commit #{commits})")
        if commits >= a.grace_commits:
            if zero:   # audit-only signal; vacuous under signal="census"
                return _fire(a, run, f"{zero} commits took a step of length zero "
                                     f"(P4.7 defect 2)")
            # G-2's death, tested directly rather than through trips/commit.
            i_frac, starved, n_i = scanner.g2_signature()
            if n_i >= a.grace_commits and i_frac is not None and i_frac >= a.i_floor_frac:
                return _fire(a, run, f"iteration_per_data_id floored at 1 on "
                                     f"{i_frac:.0%} of the last {n_i} commits "
                                     f"(limit {a.i_floor_frac:.0%})"
                                     + (f", pool demand unmet on {starved:.0%}"
                                        if starved else "")
                                     + " -- P4.7 defect 3, G-2's death")
            # Starving means the gate is NOT BEING MET, not merely that it asks
            # for less. A commit with n_eff < n_req is one the pool could not
            # satisfy (a force-commit); low trips/commit with the demand met is
            # just a small step correctly costed by the n_target gate.
            if (recent is not None and recent < a.trips_floor
                    and starved is not None and starved > 0.5):
                return _fire(a, run, f"trips/commit {recent:.2f} over the last "
                                     f"{min(commits, 200)} commits is below "
                                     f"{a.trips_floor:g} AND the pool demand went "
                                     f"unmet on {starved:.0%} -- the gate is starving "
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
