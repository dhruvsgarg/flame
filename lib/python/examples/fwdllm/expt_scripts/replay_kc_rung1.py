#!/usr/bin/env python3
"""K-C rung 1 (fl_fwd_ft_buildplan.md 0.6): replay commits/vclock-h and
staleness on the existing K/C arms against the two rival models
(fl_fwd_ft_practice.md P5.3 K-C):

    C-model (K-C):      commit_rate ~ C / (n_req * tau)
    K-model (pooling):  commit_rate ~ K / tau(K),  tau(K) = tau_ref*(K/K_ref)^exponent

`C` moved WITH `K` on every arm run so far (`C/K` = 3/2/2) -- by construction
this rung can only falsify arithmetic, never decide between the two models;
it exists to size K-1 (fl_fwd_ft_practice.md P5.1), which holds `C` fixed.

    ./replay_kc_rung1.py RUN_DIR [RUN_DIR ...]

Reads `agg_round` events (staleness[], pastdated_commits, vclock_now) and
`server_update` events (commit count) directly from telemetry -- no GPU.
"""
import argparse
import glob
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from replay_scoring import slice_run, meta  # noqa: E402 -- reuse p/K/gate resolution

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "..", "..", ".."))
from examples.fwdllm.expts.wall_clock_preflight import (  # noqa: E402
    g_rule, tau_round_s, RHO_STAR_DEFAULT, GATE_SAFETY_S_DEFAULT,
    PERTURBATION_COUNT_DEFAULT, PROBE_COMBINE_DEFAULT)


def load_agg_rounds(run_dir):
    files = glob.glob(os.path.join(run_dir, "telemetry", "aggregator_*.jsonl"))
    rounds = []
    for f in files:
        with open(f, errors="ignore") as fh:
            for line in fh:
                if '"event": "agg_round"' not in line:
                    continue
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                rounds.append(r)
    rounds.sort(key=lambda r: r["ts"])
    return rounds


def count_commits(run_dir):
    files = glob.glob(os.path.join(run_dir, "telemetry", "aggregator_*.jsonl"))
    n = 0
    for f in files:
        with open(f, errors="ignore") as fh:
            for line in fh:
                if '"event": "server_update"' in line:
                    n += 1
    return n


def analyze(run_dir, cache):
    rid, path, cfg = slice_run(run_dir, cache)
    rounds = load_agg_rounds(run_dir)
    if not rounds:
        print(f"{rid}: no agg_round events", file=sys.stderr)
        return None
    T = count_commits(run_dir)
    vclock_final = max(r.get("vclock_now") or 0 for r in rounds)
    commits_per_vclock_h = T / (vclock_final / 3600) if vclock_final else float("nan")

    staleness_vals = [s for r in rounds for s in (r.get("staleness") or [])]
    stale_frac = (sum(1 for s in staleness_vals if s >= 1) / len(staleness_vals)
                  if staleness_vals else float("nan"))
    pastdated_max = max((r.get("pastdated_commits") or 0) for r in rounds)

    # measured round-trip time, THIS run's own vclock (sim mode) -- distinct
    # from the REAL-wall 8.9/12.3 s at K=30/50 quoted in P3 (a different clock,
    # not reproducible from this telemetry alone; reported for context only).
    n_rounds = len(rounds)
    tau_measured_vclock = vclock_final / n_rounds if n_rounds else float("nan")

    h = cfg.get("hyperparameters", {})
    K = int(h.get("aggGoal") or 10)
    C = cfg.get("selector", {}).get("kwargs", {}).get("c")
    rho_star = h.get("rho_star") or RHO_STAR_DEFAULT
    s = h.get("gate_safety_s") or GATE_SAFETY_S_DEFAULT
    rule = h.get("probe_combine") or PROBE_COMBINE_DEFAULT
    P = h.get("perturbation_count") or PERTURBATION_COUNT_DEFAULT
    p, p_src = meta(cfg, run_dir)["p"], meta(cfg, run_dir)["p_source"]
    G = g_rule(rule, P)
    n_req = p * (rho_star / s) ** 2 / G

    return dict(rid=rid, K=K, C=C, T=T, n_rounds=n_rounds, vclock_final=vclock_final,
                commits_per_vclock_h=commits_per_vclock_h, stale_frac=stale_frac,
                pastdated_max=pastdated_max, tau_measured_vclock=tau_measured_vclock,
                n_req=n_req, p=p, p_source=p_src)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--cache", default="/tmp/fwd_replay_cache")
    a = ap.parse_args()

    results = []
    for run in a.runs:
        r = analyze(run, a.cache)
        if r:
            results.append(r)
    if not results:
        return

    print(f"{'rid':9s}{'K':>4s}{'C':>4s}{'T':>6s}{'commits/vclk-h':>15s}"
          f"{'stale>=1':>9s}{'pastdated_max':>14s}{'tau_vclk/rnd':>13s}{'n_req':>8s}")
    for r in results:
        print(f"{r['rid']:9s}{r['K']:4d}{r['C']:4d}{r['T']:6d}"
              f"{r['commits_per_vclock_h']:15.1f}{r['stale_frac']:9.3f}"
              f"{r['pastdated_max']:14d}{r['tau_measured_vclock']:13.3f}{r['n_req']:8.1f}")

    if len(results) < 2:
        print("\nneed >=2 arms to compare the two models")
        return

    ref = results[0]
    print(f"\n--- model check, relative to {ref['rid']} (K={ref['K']}, C={ref['C']}) ---")
    print(f"{'rid':9s}{'observed x':>12s}{'C-model x (C/tau_meas)':>26s}{'K-model x (K/tau(K))':>24s}")
    for r in results:
        obs_x = r["commits_per_vclock_h"] / ref["commits_per_vclock_h"]
        # C-model: commit_rate ~ C / (n_req * tau); n_req is arm-invariant here
        # (same rho_star/s/p/G_rule), so it cancels in the ratio -- left in for
        # clarity, not because it moves the prediction.
        c_model_x = ((r["C"] / (r["n_req"] * r["tau_measured_vclock"])) /
                     (ref["C"] / (ref["n_req"] * ref["tau_measured_vclock"])))
        # K-model: commit_rate ~ K / tau(K), the POOLING law (buildplan 0.7,
        # unvalidated -- K-1 is the registered A/B).
        k_model_x = (r["K"] / tau_round_s(r["K"])) / (ref["K"] / tau_round_s(ref["K"]))
        print(f"{r['rid']:9s}{obs_x:12.3f}{c_model_x:26.3f}{k_model_x:24.3f}")


if __name__ == "__main__":
    main()
