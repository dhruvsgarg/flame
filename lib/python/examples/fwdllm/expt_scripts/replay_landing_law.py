#!/usr/bin/env python3
"""Rung-1 replay: does the budget-landing law compose with the commit gate?

Buildplan task T5, run BEFORE any Phase-3 code lands in `aggregator/`. Answers
three things, none of which needs a GPU:

  A. ARM VALIDATION -- on arms already on disk, is `Lambda = 2B/s` (solution
     doc S4.6a) actually the identity it looks like? If it is, the rho SCHEDULE
     is Lambda-neutral and only the commit COUNT distinguishes the two candidate
     landing laws. Also measures per-commit and per-round-trip wall cost, which
     Part D projects with.

  B. LAW C (fixed-rate) -- rho*_t = sqrt(2*(B_max - B_t)/T_res) with T_res a
     CONSTANT, never decremented. B approaches B_max as B_max*(1-e^{-t/T_res}),
     so a run stops at B >= f*B_max. Reports where the gate's I floors at 1 and
     where it breaches max_iter -- the composition failure mode S8 item 5 warns
     about (`N_req ~ rho_t^2`, so annealing rho demands LESS pooling).

  C. LAW A (fixed-horizon) -- the same law with T_res - t in the denominator.
     Under perfect tracking rho* is exactly CONSTANT and B lands on B_max at
     t = T_res, so it is a constant-rho policy with a deadline, not an anneal.

  D. PROJECTION -- Part B/C commit counts x Part A's measured costs.

    ./replay_landing_law.py RUN_DIR [RUN_DIR ...]
    ./replay_landing_law.py --gate-rho-ref setpoint RUN_DIR ...

Nothing here reads or writes production config; it is arithmetic plus the two
telemetry events `replay_scoring.py` already slices.
"""
import argparse
import glob
import json
import math
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from replay_scoring import slice_run, meta, load, enrich   # noqa: E402

# B-1 (2026-08-13, model S7.1): knees are erratic across task, so each dataset
# carries its own sensed value. ln 2 is D1's Phase-A prior -- the weakest
# non-trivial claim ("the model survives its weights doubling"), not a fit.
B_MAX = {"prior": math.log(2.0), "agnews": math.log(3.15), "yahoo": math.log(2.15)}
P_OF = {"prior": 450340, "agnews": 450340, "yahoo": 454954}


def gate(rho, s, p, g_rule, K, max_iter):
    """The shipped gate, exactly: N_req = p*(rho/s)^2/G_rule, I = ceil(N_req/K).

    `I` floors at 1 (one round trip is indivisible) and is capped by max_iter
    (`ceil(n_req/K) <= max_iter` is the standing launch assertion, P9.1).
    """
    n_req = p * (rho / s) ** 2 / g_rule
    # Tolerance, not cosmetics: rho_gate_cap() solves n_req == max_iter*K
    # exactly, and a bare ceil() on the sqrt/square round trip returns
    # max_iter + 1 for the very rho the cap was built to permit.
    i_want = math.ceil(n_req / K - 1e-9) if n_req > 0 else 1
    i = max(1, min(i_want, max_iter))
    return n_req, i, i_want, K * i


def rho_gate_cap(s, p, g_rule, K, max_iter):
    """Largest rho* the gate can actually reach: `ceil(n_req/K) <= max_iter` is
    the standing launch assertion (P9.1), and n_req = p*(rho/s)^2/G_rule. Solved
    for rho it is a MECHANICAL cap -- no operator input, no fitted constant.

    This replaces a rho* <= rho*_0 monotonicity clamp, which would be wrong: it
    would pin rho* to the Phase-A PRIOR (ln 2) and block 3.1's re-sense from
    ever spending the budget it just discovered, defeating the probe entirely.
    Within a fixed B_max, law C is monotone non-increasing by construction
    anyway (B_rem only shrinks), so the only thing that can raise rho* is a
    re-sense -- which is exactly when it should rise.
    """
    return s * math.sqrt(max_iter * K * g_rule / p)


def simulate(law, b_max, t_res, s, p, g_rule, K, max_iter, f, gate_ref, t_cap,
             prior=None, resense_at=0):
    """One trajectory of the landing law. Arm-independent: under trust_ratio the
    realised rho IS the setpoint (P3, to 8.7e-5), so the schedule is closed-form.

    `prior`/`resense_at` model D1's two-phase start: run at the ln 2 prior until
    commit `resense_at`, then 3.1's probe lands and B_max jumps to the sensed
    value mid-flight. `f` is always measured against the CURRENT B_max.
    """
    cap = rho_gate_cap(s, p, g_rule, K, max_iter)
    b, lam, rows, trips = 0.0, 0.0, [], 0
    b_now = prior if prior is not None else b_max
    rho0 = min(cap, math.sqrt(2.0 * b_now / t_res))
    for t in range(t_cap):
        if prior is not None and t == resense_at:
            b_now = b_max                       # 3.1 fires: B_max moves, T_res does not
        b_rem = max(0.0, b_now - b)
        if law == "C":
            rho = math.sqrt(2.0 * b_rem / t_res)
        else:                                   # law A: horizon counts down
            t_rem = max(1.0, t_res - t)
            rho = math.sqrt(2.0 * b_rem / t_rem)
        rho = min(cap, rho)
        if rho <= 0:
            break
        # `annealed` sizes the pool from the CURRENT rho, `setpoint` from rho*_0
        # -- which tracks B_max, so a re-sense moves it too.
        ref = rho if gate_ref == "annealed" else min(
            cap, math.sqrt(2.0 * b_now / t_res))
        n_req, i, i_want, N = gate(ref, s, p, g_rule, K, max_iter)
        cos = math.sqrt(g_rule * N / p)
        lam += rho * cos
        b += 0.5 * math.log1p(rho ** 2)
        trips += i
        rows.append(dict(t=t, rho=rho, n_req=n_req, I=i, I_want=i_want, N=N,
                         B=b, Lam=lam, s_eff=rho / cos))
        if b >= f * b_now:
            break
    return rho0, rows, trips


# Pre-registered before the sweep was run, so a config that breaches one is
# re-picked rather than argued with.
#   TRIPS_PER_COMMIT -- the metric that actually discriminates. Flooring `I` at 1
#     is SAFE (N > n_req means rho/cos < s, conservative) but it strands the
#     server-side per-commit cost with no trainer work to amortise it. G-2's
#     annealed leg 003648 died on [SIM_WALL_CEILING] at 1.02 trips/commit and
#     8.40 s/trip; the healthy 145729 ran 8.0 trips/commit at 1.75 s/trip.
#   LAM_FLOOR -- P4 read 1: peak accuracy ~0.865 by Lambda ~0.95.
TRIPS_PER_COMMIT_MIN = 3.0
LAM_FLOOR = 0.95
WALL_CEIL_H = 10.0                      # one overnight slot per node


def summarise(tag, rho0, rows, trips, b_max, s, K):
    if not rows:
        print(f"  {tag:34s}  (no commits)")
        return None
    last = rows[-1]
    floored = sum(1 for r in rows if r["I"] == 1 and r["I_want"] <= 1)
    capped = sum(1 for r in rows if r["I_want"] > r["I"])
    tpc = trips / len(rows)
    bad = []
    if capped:
        bad.append("CAP")                # breaches ceil(n_req/K) <= max_iter
    if tpc < TRIPS_PER_COMMIT_MIN:
        bad.append("TPC")
    if last["Lam"] < LAM_FLOOR:
        bad.append("LAM")
    print(f"  {tag:34s}{rho0:8.4f}{len(rows):8d}{trips:9d}{tpc:7.2f}"
          f"{last['B'] / b_max:8.2f}{last['Lam']:8.3f}{2 * last['B'] / s:9.3f}"
          f"{100 * floored / len(rows):7.0f}%{100 * capped / len(rows):7.0f}%"
          f"{rows[0]['I']:5d}{last['I']:5d}  "
          f"{'ok' if not bad else ' '.join(bad)}")
    return dict(commits=len(rows), trips=trips, tpc=tpc, lam=last["Lam"],
                B=last["B"], floored=floored / len(rows),
                capped=capped / len(rows), bad=bad)


def arm_costs(run_dir):
    """commits, round trips, vclock-h and real-wall-h, off `agg_round`."""
    src = glob.glob(os.path.join(run_dir, "telemetry", "aggregator_*.jsonl"))
    vmax = wmax = 0.0
    for path in src:
        with open(path, errors="ignore") as fh:
            for line in fh:
                if '"agg_round"' not in line:
                    continue
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                vmax = max(vmax, float(r.get("vclock_now") or 0))
                wmax = max(wmax, float(r.get("wall_elapsed_s") or 0))
    return vmax, wmax


def part_a(runs, cache):
    """Is Lambda = 2B/s on real arms? And what does a commit actually cost?"""
    print("=== A. arm validation (arms on disk) ===")
    print(f"  {'rid':8s}{'sched':7s}{'T':>6s}{'trips':>7s}{'B':>8s}{'Lam':>8s}"
          f"{'s_med':>8s}{'2B/s':>8s}{'err%':>7s}{'I=1':>6s}"
          f"{'vclk_h':>8s}{'wall_h':>8s}{'s/cmt':>7s}{'s/trip':>7s}")
    out = []
    for run in runs:
        rid, path, cfg = slice_run(run, cache)
        if not path:
            print(f"  {rid:8s}  no telemetry", file=sys.stderr)
            continue
        m = meta(cfg, run)
        commits, _ = load(path)
        if not commits:
            continue
        rows = enrich(m, commits)
        h = cfg.get("hyperparameters", {})
        sched = str(h.get("rho_schedule", "?"))
        s_t = [r["rho"] / r["cos_pred"] for r in rows if r["cos_pred"] > 0]
        s_med = statistics.median(s_t)
        B, lam = rows[-1]["B"], rows[-1]["Lam"]
        pred = 2 * B / s_med
        trips = sum(r["N"] // m["K"] for r in rows)
        floored = sum(1 for r in rows if r["N"] // m["K"] <= 1) / len(rows)
        vclk, wall = arm_costs(run)
        print(f"  {rid:8s}{sched:7s}{len(rows):6d}{trips:7d}{B:8.4f}{lam:8.3f}"
              f"{s_med:8.3f}{pred:8.3f}{100 * (pred - lam) / lam:7.1f}"
              f"{100 * floored:5.0f}%{vclk / 3600:8.2f}{wall / 3600:8.2f}"
              f"{wall / len(rows):7.2f}{wall / max(1, trips):7.2f}")
        out.append(dict(rid=rid, commits=len(rows), trips=trips, wall=wall,
                        s_cmt=wall / len(rows), s_trip=wall / max(1, trips)))
    print("\n  Lambda = 2B/s is the S4.6a identity. It holds only where the gate")
    print("  HOLDS s; an arm whose s drifts (rm at pinned N, or a free var gate)")
    print("  is exactly where err% opens up -- which is S4.6's time-law miss.")
    return out


HDR = (f"  {'law / dataset':34s}{'rho*_0':>8s}{'commits':>8s}{'trips':>9s}"
       f"{'tr/cmt':>7s}{'B/Bmax':>8s}{'Lam':>8s}{'2B/s':>9s}{'I=1':>8s}"
       f"{'I>cap':>8s}{'I_0':>5s}{'I_T':>5s}  verdict")


def part_bc(args):
    res = {}
    for law, title in (("C", "B. law C -- fixed-rate (T_res never decremented)"),
                       ("A", "C. law A -- fixed-horizon (T_res - t)")):
        print(f"\n=== {title} ===")
        print(f"  gate_rho_ref={args.gate_rho_ref}  s={args.s}  K={args.K}  "
              f"G_rule={args.g_rule}  max_iter={args.max_iter}  f={args.f}"
              f"   [gates: tr/cmt>={TRIPS_PER_COMMIT_MIN}, Lam>={LAM_FLOOR}, "
              f"no CAP]")
        print(HDR)
        for ds in ("prior", "agnews", "yahoo"):
            for t_res in args.t_res:
                rho0, rows, trips = simulate(
                    law, B_MAX[ds], t_res, args.s, P_OF[ds], args.g_rule,
                    args.K, args.max_iter, args.f, args.gate_rho_ref, args.t_cap)
                tag = f"{ds} B_max={B_MAX[ds]:.3f} T_res={t_res}"
                r = summarise(tag, rho0, rows, trips, B_MAX[ds], args.s, args.K)
                if r:
                    res[(law, ds, t_res)] = r
    return res


def part_f(args, t_res):
    """f is the least-evidenced constant in the law -- sweep it explicitly.

    Under law C, B never reaches B_max, so the stop is B >= f*B_max at
    t = T_res*ln(1/(1-f)). Lambda ~= 2*f*B_max/s, so f trades commits for the
    last few percent of budget on a log curve.
    """
    print(f"\n=== f sweep (law C, T_res={t_res}, gate_rho_ref="
          f"{args.gate_rho_ref}) ===")
    print(HDR)
    for ds in ("agnews", "yahoo"):
        for f in args.f_sweep:
            rho0, rows, trips = simulate(
                "C", B_MAX[ds], t_res, args.s, P_OF[ds], args.g_rule, args.K,
                args.max_iter, f, args.gate_rho_ref, args.t_cap)
            summarise(f"{ds} f={f}", rho0, rows, trips, B_MAX[ds], args.s, args.K)


def fit_cost(costs):
    """Least-squares `wall = a*commits + b*trips` over the arms.

    The two terms are separable here only because the portfolio spans 1.02 to
    8.0 trips/commit -- 003648 is almost pure per-commit cost and 035045 almost
    pure per-trip, so the design matrix is well conditioned.
    """
    sxx = sum(c["commits"] ** 2 for c in costs)
    syy = sum(c["trips"] ** 2 for c in costs)
    sxy = sum(c["commits"] * c["trips"] for c in costs)
    sxw = sum(c["commits"] * c["wall"] for c in costs)
    syw = sum(c["trips"] * c["wall"] for c in costs)
    det = sxx * syy - sxy * sxy
    if abs(det) < 1e-9:
        return None, None
    return (syy * sxw - sxy * syw) / det, (sxx * syw - sxy * sxw) / det


def part_e(args, t_res):
    """D1's actual two-phase trajectory: start at the ln 2 prior, 3.1 re-senses
    at `resense_at`, B_max jumps, T_res does not move. This is the thing the
    overnight nodes would run, so it is the one that has to pass the gates.
    """
    print(f"\n=== E. two-phase (prior ln2 -> sensed B_max), law C, "
          f"T_res={t_res}, f={args.f} ===")
    print(HDR)
    out = {}
    for ds in ("agnews", "yahoo"):
        for at in args.resense_at:
            rho0, rows, trips = simulate(
                "C", B_MAX[ds], t_res, args.s, P_OF[ds], args.g_rule, args.K,
                args.max_iter, args.f, args.gate_rho_ref, args.t_cap,
                prior=B_MAX["prior"], resense_at=at)
            r = summarise(f"{ds} re-sense @ commit {at}", rho0, rows, trips,
                          B_MAX[ds], args.s, args.K)
            if r:
                out[("E", ds, at)] = r
                jump = [x for x in rows if x["t"] in (at - 1, at)]
                if len(jump) == 2:
                    print(f"      rho* {jump[0]['rho']:.4f} -> {jump[1]['rho']:.4f}"
                          f"   I {jump[0]['I']} -> {jump[1]['I']}"
                          f"   (B={jump[0]['B']:.3f} spent under the prior, "
                          f"{100 * jump[0]['B'] / B_MAX[ds]:.0f}% of the sensed B_max)")
    return out


def part_d(res, costs, args):
    print("\n=== D. projection to real wall ===")
    if len(costs) < 2:
        print("  fewer than 2 arms; skipped")
        return
    a, b = fit_cost(costs)
    if a is None:
        print("  degenerate fit; skipped")
        return
    # The audit is charged per COMMIT (85 s / stride 25, P9.2), so it lives
    # entirely in `a`. Subtracting it is what an audit-off arm would cost.
    audit = 85.0 / 25.0
    a_off = max(args.audit_off_s_per_commit, a - audit)
    print(f"  fitted over {len(costs)} arms: {a:.2f} s/commit + {b:.2f} s/trip"
          f"   (audit ON, stride 25)")
    for c in costs:
        pred = a * c["commits"] + b * c["trips"]
        print(f"    {c['rid']}  wall {c['wall'] / 3600:5.2f} h  "
              f"predicted {pred / 3600:5.2f} h  ({100 * (pred - c['wall']) / c['wall']:+.0f}%)")
    print(f"  audit OFF projection uses {a_off:.2f} s/commit + {b:.2f} s/trip "
          f"(P9.2 floor {args.audit_off_s_per_commit})")
    print(f"\n  {'law / dataset':34s}{'commits':>8s}{'trips':>8s}"
          f"{'h audit-on':>12s}{'h audit-off':>13s}")
    for (law, ds, t_res), r in sorted(res.items(), key=lambda kv: str(kv[0])):
        if ds == "prior":
            continue
        on = (a * r["commits"] + b * r["trips"]) / 3600
        off = (a_off * r["commits"] + b * r["trips"]) / 3600
        flag = "" if off <= WALL_CEIL_H else "  WALL"
        tag = f"{law}: {ds} T_res={t_res}"
        print(f"  {tag:34s}{r['commits']:8d}{r['trips']:8d}{on:12.1f}"
              f"{off:13.1f}{flag}")
    print(f"\n  Overnight budget is ~{WALL_CEIL_H:.0f} h/node.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--s", type=float, default=1.5, help="gate_safety_s")
    ap.add_argument("--K", type=int, default=10, help="agg_goal")
    ap.add_argument("--g-rule", type=float, default=10.0, help="P under `mean`")
    ap.add_argument("--max-iter", type=int, default=20)
    ap.add_argument("--f", type=float, default=0.95,
                    help="stop at B >= f*B_max (law C never reaches B_max)")
    ap.add_argument("--t-res", type=int, nargs="+", default=[200, 500])
    ap.add_argument("--f-sweep", type=float, nargs="+",
                    default=[0.70, 0.80, 0.85, 0.90, 0.95])
    ap.add_argument("--f-sweep-t-res", type=int, default=300)
    ap.add_argument("--resense-at", type=int, nargs="+", default=[50, 150, 300])
    ap.add_argument("--audit-off-s-per-commit", type=float, default=1.64,
                    help="P9.2's measured audit-off commit path")
    ap.add_argument("--t-cap", type=int, default=20000, help="simulation cutoff")
    ap.add_argument("--gate-rho-ref", choices=("annealed", "setpoint"),
                    default="annealed")
    ap.add_argument("--cache", default="/tmp/fwd_replay_cache")
    a = ap.parse_args()

    costs = part_a(a.runs, a.cache) if a.runs else []
    res = part_bc(a)
    part_f(a, a.f_sweep_t_res)
    res.update(part_e(a, a.f_sweep_t_res))
    part_d(res, costs, a)


if __name__ == "__main__":
    main()
