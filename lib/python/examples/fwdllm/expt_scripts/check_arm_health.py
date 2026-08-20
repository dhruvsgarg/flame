#!/usr/bin/env python3
"""The four in-flight gates of buildplan §6, as one command, on a live or finished arm.

"A clean `--dry-run` is a prior, not a guarantee" (§4). Each of these four cost a
node, and none is visible at launch:

  1. `[DataBins]` derived from the registry, at 100% coverage, confirmed by a trainer
     -- 150 hardcoded gave yahoo 8.6% of its data and nothing raised (P4.7 defect 4);
  2. no commit with `rho_star == 0` -- a zero step is a requirement of zero, not an
     absent one, and it was 23-48% of the 2026-08-16 arms (defect 2);
  3. round trips per commit >= 3 IN EVERY QUINTILE -- `N_req` ~ rho_t^2 under
     `gate_rho_ref=annealed`, so an annealing rho demands less pooling every commit
     until `I` floors at 1 and the server path has no trainer work amortising it.
     The launch projection prices law C off the `ln 2` prior, so an arm can pass at
     launch and breach in flight (defect 3);
  4. a controller arm ends on `[BudgetStop] reason=budget`, never `max_runtime_s`
     -- that rule alone voided all four 2026-08-16 arms (defect 1).

Plus the `[BmaxProbe]` trajectory, whose FIRST FIRING COMMIT is itself a result on
yahoo (the sensor declines below chance, so early commits run the `ln 2` prior).

    ./check_arm_health.py <run_dir> [--quintiles 5] [--expect-controller]

Exit 0 = every gate that can be read yet holds; 1 = a gate is breached. Reads
telemetry and logs only; safe against a run in progress.
"""
import argparse
import glob
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))

from examples.fwdllm.expts import dataset_registry as dsreg  # noqa: E402

OK, WARN, BAD = "ok  ", "WARN", "FAIL"


def _events(run_dir):
    """server_update, version_bump_census and agg_round, in file order (= commit order).

    `server_update` carries every per-commit field below, but only exists under
    `--server-update-audit` -- which the scored arms set and the real profiling arm
    deliberately does not (its per-commit work would land in the `fedavg` span that
    arm exists to price). `version_bump_census` is unconditional and fires once per
    commit, so it is what tells a genuinely committing arm apart from a dead one:
    without it this script read 0 commits on a healthy 61-commit yahoo arm and still
    printed a green verdict (2026-08-20).
    """
    su, vb, rounds, last_v = [], [], 0, 0.0
    for f in sorted(glob.glob(os.path.join(run_dir, "telemetry",
                                           "aggregator_*.jsonl"))):
        with open(f, errors="ignore") as fh:
            for line in fh:
                if ('"server_update"' not in line and '"agg_round"' not in line
                        and '"version_bump_census"' not in line):
                    continue
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                if d.get("event") == "server_update":
                    su.append(d)
                elif d.get("event") == "version_bump_census":
                    vb.append(d)
                elif d.get("event") == "agg_round":
                    rounds += 1
                    last_v = d.get("vclock_now") or last_v
    for d in su:                       # stamp each commit with the vclock it saw
        d.setdefault("vclock_now", None)
    return su, vb, rounds, last_v


def _log_lines(run_dir, tag):
    out = []
    for f in glob.glob(os.path.join(run_dir, "*aggregator.log")):
        with open(f, errors="ignore") as fh:
            out += [ln.rstrip() for ln in fh if tag in ln]
    return out


def _trips_per_commit(run_dir, n_quintiles):
    """agg_round events between consecutive commits, bucketed over the run.

    Both come off the same file in emission order, so the split is exact without
    needing either side's own counter."""
    both = {"audit": [], "census": []}
    for f in sorted(glob.glob(os.path.join(run_dir, "telemetry",
                                           "aggregator_*.jsonl"))):
        with open(f, errors="ignore") as fh:
            for line in fh:
                if '"agg_round"' in line:
                    both["audit"].append("r")
                    both["census"].append("r")
                elif '"server_update"' in line:
                    both["audit"].append("c")
                elif '"version_bump_census"' in line:
                    both["census"].append("c")
    # Larger stream, never the sum -- see _events(); with the audit on both fire
    # once per commit, so the max is still the commit count.
    seq = max(both.values(), key=lambda v: v.count("c"))
    commits = seq.count("c")
    if commits < n_quintiles:
        return None, commits, seq.count("r")
    edges = [round(commits * i / n_quintiles) for i in range(n_quintiles + 1)]
    seen_c = b = 0
    counts = [[0, 0] for _ in range(n_quintiles)]     # [trips, commits]
    for tok in seq:
        if tok == "r":
            counts[min(b, n_quintiles - 1)][0] += 1
        else:
            seen_c += 1
            counts[min(b, n_quintiles - 1)][1] += 1
            while b + 1 < n_quintiles and seen_c >= edges[b + 1]:
                b += 1
    per_bucket = [(t / c if c else float("nan")) for t, c in counts]
    return per_bucket, commits, sum(t for t, _ in counts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--quintiles", type=int, default=5)
    ap.add_argument("--trips-floor", type=float, default=3.0)
    ap.add_argument("--target-commits", type=int, default=898,
                    help="commits law C is projected to need; used to size the "
                         "next arm's vclock budget from this arm's measured rate")
    ap.add_argument("--expect-controller", action="store_true",
                    help="also require a [BudgetStop] ending (gate 4). Off by "
                         "default so the same command reads a control arm.")
    a = ap.parse_args()
    run = a.run_dir.rstrip("/")
    print(f"[arm] {os.path.basename(run)}")

    cfg = os.path.join(run, "aggregator_config.json")
    hp = json.load(open(cfg))["hyperparameters"] if os.path.exists(cfg) else {}
    ds = hp.get("dataset")
    fails = []

    # --- gate 1: DataBins ---------------------------------------------------
    bins_lines = _log_lines(run, "[DataBins]")
    want = None
    if ds:
        try:
            want = dsreg.total_data_bins(ds, int(hp.get("client_num_in_total") or 100),
                                         int(hp.get("train_batch_size") or 8))
        except Exception:
            want = None
    got = next((int(m.group(1)) for ln in bins_lines
                if (m := re.search(r"total_data_bins=(\d+)", ln))), None)
    src = next((m.group(1) for ln in bins_lines
                if (m := re.search(r"source=(\w+)", ln))), "?")
    confirmed = any("confirmed by trainer" in ln for ln in bins_lines)
    mismatch = any("MISMATCH" in ln for ln in bins_lines)
    undercount = any("UNDERCOUNT" in ln for ln in bins_lines)
    exact = any(re.search(r"\(100\.00%\)", ln) for ln in bins_lines)
    lvl = OK
    if got is None:
        lvl = WARN
    elif (want and got != want) or src != "registry" or mismatch or undercount:
        lvl = BAD
    elif not confirmed or not exact:
        # No trainer has reported yet, or the arm predates the coverage line --
        # unread, not breached.
        lvl = WARN
    print(f"  [{lvl}] 1. DataBins        bins={got} (registry says {want}) "
          f"source={src} coverage="
          f"{'100%' if exact else ('UNDERCOUNT' if undercount else 'not logged')} "
          f"trainer-confirmed={confirmed}")
    if lvl == BAD:
        fails.append("DataBins")

    su, vb, _, vclock_last = _events(run)

    # --- gate 2: no rho_star == 0 -------------------------------------------
    zeros = [d for d in su if d.get("rho_star") == 0]
    if not su and vb:
        # Unreadable, not passing: rho_star only exists on the audit record.
        print(f"  [{WARN}] 2. rho_star != 0   UNREADABLE -- arm ran without "
              f"--server-update-audit ({len(vb)} commits seen via "
              f"version_bump_census)")
    else:
        lvl = OK if not zeros else BAD
        print(f"  [{lvl}] 2. rho_star != 0   {len(zeros)}/{len(su)} commits took a "
              f"step of length zero (must be 0)")
    if zeros:
        fails.append("rho_star==0")

    # --- gate 3: trips/commit per quintile ----------------------------------
    q, commits, trips = _trips_per_commit(run, a.quintiles)
    if q is None:
        print(f"  [{WARN}] 3. trips/commit    only {commits} commits so far "
              f"({trips} trips) -- too few to bucket")
    else:
        low = min(q)
        lvl = OK if low >= a.trips_floor else BAD
        print(f"  [{lvl}] 3. trips/commit    " +
              " ".join(f"Q{i + 1}={v:.2f}" for i, v in enumerate(q)) +
              f"   (floor {a.trips_floor:g}; overall {trips / commits:.2f})")
        if lvl == BAD:
            fails.append("trips/commit")

    # trips/commit is n_req/agg_goal, so at K=10 a floor of 3 demands n_req>=30 --
    # a bound a landing controller must cross on its way down. G-2's actual death was
    # `I` floored at 1 on 98% of commits with the pool demand unmet; print that
    # directly so a gate-3 FAIL can be told apart from a controller annealing on plan.
    _i = [d.get("iteration_per_data_id") for d in su[-200:]]
    _g = [(d.get("n_eff"), d.get("n_req")) for d in su[-200:]]
    _g = [(e, r) for e, r in _g if e is not None and r is not None]
    if _i:
        _fl = sum(1 for i in _i if i is not None and i <= 1) / len(_i)
        _st = (sum(1 for e, r in _g if e < r) / len(_g)) if _g else None
        print(f"  [    ] +  G-2 signature   I==1 on {_fl:.0%} of the last {len(_i)} "
              f"commits (G-2: 98%)"
              + (f"; pool demand unmet on {_st:.0%}" if _st is not None else "")
              + (f"; n_req {su[-1]['n_req']:.1f}"
                 if su[-1].get("n_req") is not None else ""))

    # --- gate 4: how it ended -----------------------------------------------
    stop = _log_lines(run, "[BudgetStop]")
    reason = next((m.group(1) for ln in stop
                   if (m := re.search(r"reason=(\w+)", ln))), None)
    action = next((m.group(1) for ln in stop
                   if (m := re.search(r"action=(\w+)", ln))), None)
    if reason:
        lvl = OK if (reason == "budget" or not a.expect_controller) else WARN
        print(f"  [{lvl}] 4. ending          [BudgetStop] reason={reason} "
              f"action={action}")
        if lvl != OK:
            fails.append("stop reason")
    elif a.expect_controller:
        print(f"  [{WARN}] 4. ending          no [BudgetStop] yet -- a controller "
              f"arm that ends on max_runtime_s instead is VOID")
    else:
        print(f"  [{OK}] 4. ending          no [BudgetStop] (not expected on a control)")

    # --- the B_max sensor ---------------------------------------------------
    probes = _log_lines(run, "[BmaxProbe]")
    fired = [ln for ln in probes if "->" in ln]
    declined = [ln for ln in probes if "too close to chance" in ln]
    first = next((int(m.group(1)) for ln in fired
                  if (m := re.search(r"commit=(\d+)", ln))), None)
    print(f"  [{'ok  ' if fired else 'WARN'}] +  BmaxProbe       "
          f"{len(fired)} fires, {len(declined)} declined at chance; "
          f"first firing commit={first}"
          + ("  (running the ln 2 prior until then)" if declined else ""))
    below = [d for d in su
             if d.get("budget_b_max") is not None and d.get("budget_b") is not None
             and d["budget_b_max"] < d["budget_b"]]
    if below:
        print(f"  [{BAD}] +  B_max origin   {len(below)} commits with B_max < B "
              f"-- the two origins were subtracted again (P4.7 defect 2)")
        fails.append("B_max < B")

    # --- the rate, which is what sizes the next arm's budget -----------------
    # The wall-clock preflight prices every commit at a dataset-independent
    # 4.41 s; yahoo measured 45.6 (79 commits/h against agnews' 350), so a budget
    # sized off the projection alone under-books the seq-256 datasets ~10x. Read
    # the rate off a SHORT arm and size the long one from it.
    # Either record carries `ts`, so the rate reads off an un-audited arm too --
    # and the profiling arm, which is the SHORT arm this line exists to size from,
    # is exactly the one that never sets the audit flag.
    rate_src = su if len(su) > 5 else vb
    if len(rate_src) > 5:
        span_h = (rate_src[-1]["ts"] - rate_src[0]["ts"]) / 3600.0
        vmax = vclock_last
        if span_h > 0:
            print(f"  [    ] +  rate            {len(rate_src) / span_h:.0f} commits/h"
                  + (f", {vmax / span_h:,.0f} vclock/h" if vmax else "")
                  + f"  ({3600 * span_h / len(rate_src):.1f} s/commit)")
            if vmax:
                print(f"  [    ] +  budget sizing   a {a.target_commits}-commit arm "
                      f"needs ~{a.target_commits * vmax / len(rate_src):,.0f} vclock "
                      f"and ~{a.target_commits / (len(rate_src) / span_h):.1f} h at this rate")

    # --- progress, exact at any horizon -------------------------------------
    if su:
        last = su[-1]
        B = last.get("budget_b")
        rho = [d.get("rho") or 0.0 for d in su]
        maxbin = max((d.get("data_id") or 0) for d in su)
        print(f"  [    ] +  progress        commits={len(su)} "
              f"B={B:.4g}/{last.get('budget_b_max'):.4g} "
              f"({100.0 * (last.get('budget_frac') or 0):.1f}% of B_max) "
              f"Phi={math.exp(B):.4g} rho_last={rho[-1]:.4g}")
        if got:
            print(f"  [{OK if maxbin == got - 1 else WARN}] +  bin sweep       "
                  f"max data_id={maxbin} of {got - 1} "
                  f"({'every bin visited' if maxbin == got - 1 else 'still lapping'})")
    elif vb:
        # B/rho live on the audit record only; commits and the bin sweep do not.
        print(f"  [    ] +  progress        commits={len(vb)} "
              f"(no B/rho -- arm ran without --server-update-audit)")
        if got:
            maxbin = max((d.get("data_id") or 0) for d in vb)
            print(f"  [{OK if maxbin == got - 1 else WARN}] +  bin sweep       "
                  f"max data_id={maxbin} of {got - 1} "
                  f"({'every bin visited' if maxbin == got - 1 else 'still lapping'})")

    print()
    if fails:
        print(f"[VERDICT] BREACHED: {', '.join(fails)} -- read §6 before letting "
              f"this arm run unattended")
        return 1
    print("[VERDICT] every readable gate holds")
    return 0


if __name__ == "__main__":
    sys.exit(main())
