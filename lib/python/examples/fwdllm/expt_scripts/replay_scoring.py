#!/usr/bin/env python3
"""Rung-1 replay: score any run in B / Lambda / Phi, and audit the cos probe.

Standing replacement for the one-off shell work every portfolio has re-done.
Reads only `server_update` and `agg_eval` from a run's aggregator telemetry.

    ./replay_scoring.py RUN_DIR [RUN_DIR ...]           # score arms
    ./replay_scoring.py --cos RUN_DIR [RUN_DIR ...]     # + cos-probe audit

Scoring (fl_fwd_ft_solution.md §4.1). B is summed over t = 0 .. T-2 -- exactly
the steps lying between tw[0] and tw[-1]. The t = 1 .. T-1 window drops commit
0's step and includes a last step that is not in the norm; it fits Phi_obs
worse (mean |err| 1.21% vs 0.77% over 25 arms).

The cos audit reports r = cos_measured / cos_closedform per 50-commit block and
n_dir = r^2 * N, the effective independent uploads implied if the shortfall were
client disagreement. n_dir < 1 is impossible, and falsifies that explanation.
Note that every cos_ground_truth logged before the B17 fix used a reference
batch that was one client's non-IID shard and is void -- see §3.9.
"""
import argparse
import glob
import json
import math
import os
import subprocess
import sys

E_SELECT = 2.988          # E[v_par^2] for coin-flip top-2 of P=10
P_PROBES = 10
P_BY_RF = {16: 450340, 32: 229012, 64: 118348}
BLOCK = 50
EVENTS = r'"event": "(server_update|agg_eval)"'


def slice_run(run_dir, cache):
    """grep the two events out of ~GBs of telemetry, once, into a cache."""
    rid = os.path.basename(run_dir.rstrip("/")).split("_")[2]
    out = os.path.join(cache, f"{rid}.jsonl")
    if not os.path.exists(out) or os.path.getsize(out) == 0:
        os.makedirs(cache, exist_ok=True)
        src = glob.glob(os.path.join(run_dir, "telemetry", "aggregator_*.jsonl"))
        if not src:
            return rid, None, None
        with open(out, "w") as fh:
            subprocess.run(["grep", "-hE", EVENTS] + src, stdout=fh, check=False)
    cfg = os.path.join(run_dir, "aggregator_config.json")
    return rid, out, (json.load(open(cfg)) if os.path.exists(cfg) else {})


def meta(cfg):
    h = cfg.get("hyperparameters", {})
    rf = int(h.get("adapter_reduction_factor", 16))
    rule = h.get("probe_combine", "select")
    pm = str(h.get("partition_method", ""))
    alpha = 1.0
    if "alpha=" in pm:
        try:
            alpha = float(pm.split("alpha=")[1].split("_")[0])
        except ValueError:
            pass
    return dict(rf=rf, p=P_BY_RF.get(rf, 450340), rule=rule, alpha=alpha,
                K=int(h.get("aggGoal", 10)),
                G_rule=(P_PROBES if rule == "mean" else E_SELECT))


def load(path):
    commits, evals = [], []
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            (commits if r.get("event") == "server_update" else evals).append(r)
    commits.sort(key=lambda r: r["ts"])
    evals.sort(key=lambda r: r["ts"])
    return commits, evals


def enrich(m, commits):
    B = Lam = 0.0
    out = []
    for i, r in enumerate(commits):
        # pool_size is absent when I == 1, and iteration_per_data_id can be null
        I = r.get("pool_size") or ((r.get("iteration_per_data_id") or 0) + 1)
        N = m["K"] * I
        rho = r["rho"]
        cos_pred = math.sqrt(m["G_rule"] * N / m["p"])
        Lam += rho * cos_pred
        out.append(dict(i=i, ts=r["ts"], rho=rho, N=N, Lam=Lam, cos_pred=cos_pred,
                        cos_meas=r.get("cos_ground_truth"),
                        tw=r.get("trainable_weight_norm")))
    for j, row in enumerate(out):           # B over t = 0 .. T-2
        B += 0.5 * math.log1p(row["rho"] ** 2) if j < len(out) - 1 else 0.0
        row["B"] = B
    return out


def score(rid, m, rows, evals):
    B = rows[-1]["B"]
    phi_obs = rows[-1]["tw"] / rows[0]["tw"] if rows[0].get("tw") else float("nan")
    acc = [e.get("test-accuracy") or 0 for e in evals]
    peak = max(acc) if acc else float("nan")
    final = acc[-1] if acc else float("nan")
    err = 100 * (math.exp(B) - phi_obs) / phi_obs if phi_obs == phi_obs else float("nan")
    print(f"{rid:9s}{m['rule']:7s}{m['rf']:4d}{m['alpha']:7.1f}{m['K']:4d}"
          f"{len(rows):6d}{B:9.4f}{math.exp(B):9.2f}{phi_obs:9.2f}{err:8.2f}"
          f"{rows[-1]['Lam']:8.3f}{peak:8.3f}{final:8.3f}")


def cos_audit(rid, m, rows, evals):
    have = [r for r in rows if r["cos_meas"] is not None]
    if not have:
        return
    print(f"\n  --- {rid} cos audit ({m['rule']}, p={m['p']}) ---")
    print(f"  {'commit':>7s}{'N':>6s}{'acc':>7s}{'cos_meas':>11s}{'cos_pred':>10s}"
          f"{'r':>9s}{'n_dir':>8s}")
    for s in range(0, len(rows), BLOCK):
        blk = [x for x in rows[s:s + BLOCK] if x["cos_meas"] is not None]
        if len(blk) < 5:
            continue
        cm = sum(x["cos_meas"] for x in blk) / len(blk)
        cp = sum(x["cos_pred"] for x in blk) / len(blk)
        N = sum(x["N"] for x in blk) / len(blk)
        r = cm / cp
        inb = [e for e in evals if blk[0]["ts"] <= e["ts"] <= blk[-1]["ts"]]
        a = (sum(e["test-accuracy"] for e in inb) / len(inb)) if inb else float("nan")
        print(f"  {blk[0]['i']:7d}{N:6.0f}{a:7.3f}{cm:11.5f}{cp:10.4f}{r:9.4f}"
              f"{r * r * N:8.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--cos", action="store_true", help="also audit the cos probe")
    ap.add_argument("--cache", default="/tmp/fwd_replay_cache")
    a = ap.parse_args()

    print(f"{'rid':9s}{'rule':7s}{'rf':>4s}{'alpha':>7s}{'K':>4s}{'T':>6s}"
          f"{'B':>9s}{'Phi_prd':>9s}{'Phi_obs':>9s}{'err%':>8s}{'Lam':>8s}"
          f"{'peak':>8s}{'final':>8s}")
    audits = []
    for run in a.runs:
        rid, path, cfg = slice_run(run, a.cache)
        if not path:
            print(f"{rid:9s}  no telemetry", file=sys.stderr)
            continue
        m = meta(cfg)
        commits, evals = load(path)
        if not commits:
            print(f"{rid:9s}  no server_update records", file=sys.stderr)
            continue
        rows = enrich(m, commits)
        score(rid, m, rows, evals)
        audits.append((rid, m, rows, evals))
    if a.cos:
        for args in audits:
            cos_audit(*args)


if __name__ == "__main__":
    main()
