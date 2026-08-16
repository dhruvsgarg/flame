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
import re
import statistics
import subprocess
import sys

P_PROBES = 10              # today's only shipped perturbation_count
E_SELECT_BY_P = {10: 2.988, 30: 4.744}   # measured, not derived -- do not interpolate
P_BY_RF = {16: 450340, 32: 229012, 64: 118348}
COS_BLOCK_FIRES = 10      # cos_probe_every=25 means a 50-commit BLOCK held only 2 fires
EVENTS = r'"event": "(server_update|agg_eval)"'


def slice_run(run_dir, cache):
    """grep the two events out of ~GBs of telemetry, once, into a cache.

    Keyed by run-id only, so a run still growing when first scored (still
    running, or hung post-completion before its process was reaped) leaves a
    truncated cache that a later re-score would silently keep serving forever.
    Invalidate on source mtime, not just presence.
    """
    rid = os.path.basename(run_dir.rstrip("/")).split("_")[2]
    out = os.path.join(cache, f"{rid}.jsonl")
    src = glob.glob(os.path.join(run_dir, "telemetry", "aggregator_*.jsonl"))
    stale = (
        not os.path.exists(out)
        or os.path.getsize(out) == 0
        or (src and os.path.getmtime(out) < max(os.path.getmtime(s) for s in src))
    )
    if stale:
        os.makedirs(cache, exist_ok=True)
        if not src:
            return rid, None, None
        with open(out, "w") as fh:
            subprocess.run(["grep", "-hE", EVENTS] + src, stdout=fh, check=False)
    cfg = os.path.join(run_dir, "aggregator_config.json")
    return rid, out, (json.load(open(cfg)) if os.path.exists(cfg) else {})


def resolve_p(cfg, rf, run_dir=None):
    """`p` for this run, most authoritative source first.

    P_BY_RF pins agnews' 4-label classifier, so it is silently 4,614 low on
    yahoo and 1,538 high on yelp-p -- and `p` is under a square root in every
    cos, so a wrong one biases D, L and S without ever looking wrong.

      1. `[ProbeDim]` in the trainer log -- the p the run actually probed
      2. derived: adapters(rf) + classifier(num_labels of the run's dataset)
      3. the rf table (agnews), for a run whose config predates the registry
    """
    if run_dir:
        for log in glob.glob(os.path.join(run_dir, "*trainers.log")):
            try:
                with open(log, errors="ignore") as fh:
                    for line in fh:
                        if "[ProbeDim]" in line:
                            m = re.search(r"trainable_p[= ]+(\d+)", line) or \
                                re.search(r"\bp[= ]+(\d+)", line)
                            if m:
                                return int(m.group(1)), "ProbeDim"
            except OSError:
                pass
    name = str(cfg.get("hyperparameters", {}).get("dataset", "") or "")
    try:
        sys.path.insert(0, os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
        from examples.fwdllm.expts.dataset_registry import get as _get, probe_dim
        return probe_dim(_get(name).num_labels, rf), f"registry({name})"
    except Exception:
        # Silently correct for agnews (every P4 arm); silently WRONG for any
        # other dataset a config just predates the `dataset` field for -- p
        # sits under a sqrt in every cos, so a wrong one biases D/L/S without
        # ever looking wrong. Never pass this tier silently.
        print(f"WARNING: p resolved via P_BY_RF(agnews) fallback (no [ProbeDim] "
              f"line, dataset={name!r} not in registry) -- correct only if this "
              f"run is agnews", file=sys.stderr)
        return P_BY_RF.get(rf, 450340), "P_BY_RF(agnews)"


def meta(cfg, run_dir=None):
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
    p, p_src = resolve_p(cfg, rf, run_dir)
    return dict(rf=rf, p=p, p_source=p_src, rule=rule, alpha=alpha,
                K=int(h.get("aggGoal", 10)),
                P_default=int(h.get("perturbation_count", P_PROBES)))


def g_rule_of(rule, p_t):
    """G_rule_t: P_t under `mean`, measured E_select(P_t) under `select`.

    E_select is measured per P, never derived -- an unmeasured P under
    `select` must refuse rather than interpolate (buildplan 0.4).
    """
    if rule == "mean":
        return p_t
    e = E_SELECT_BY_P.get(p_t)
    if e is None:
        raise SystemExit(
            f"E_select unmeasured for P={p_t} under `select` -- only "
            f"{sorted(E_SELECT_BY_P)} are measured; refusing to interpolate")
    return e


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
        # per-commit P: `p_probes` on the record once 3.4 lands, else the
        # run's constant perturbation_count -- a one-line switch either way
        P_t = r.get("p_probes") or m["P_default"]
        G_rule = g_rule_of(m["rule"], P_t)
        cos_pred = math.sqrt(G_rule * N / m["p"])
        Lam += rho * cos_pred
        out.append(dict(i=i, ts=r["ts"], rho=rho, N=N, Lam=Lam, cos_pred=cos_pred,
                        G_rule=G_rule, cos_meas=r.get("cos_ground_truth"),
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


def cos_audit(rid, m, rows, evals, block_fires=COS_BLOCK_FIRES):
    """Block by probe FIRES, not commits -- at cos_probe_every=25 a 50-commit
    BLOCK held only 2 fires and the old `len(blk) < 5` guard dropped every row.

    Per-block rows are for joining Phi/accuracy onto a commit range; the
    summary D is the mean +- SEM of the per-FIRE ratio (not block-averaged --
    per-fire sd is ~= the mean, SNR ~= 1, so only the arm-level mean is
    meaningful; see fl_fwd_ft_practice.md P4.2/D-2).
    """
    have = [r for r in rows if r["cos_meas"] is not None]
    if not have:
        print(f"\n  --- {rid} cos audit: no cos fires ---")
        return
    fires_r = [x["cos_meas"] / x["cos_pred"] for x in have]
    d_mean = statistics.mean(fires_r)
    d_sem = (statistics.stdev(fires_r) / math.sqrt(len(fires_r))
              if len(fires_r) > 1 else float("nan"))
    spacings = [b["i"] - a["i"] for a, b in zip(have, have[1:])]
    spacing = statistics.median(spacings) if spacings else float("nan")

    print(f"\n  --- {rid} cos audit ({m['rule']}, p={m['p']}, p_source={m['p_source']}) ---")
    print(f"  {'commit':>7s}{'N':>6s}{'acc':>7s}{'cos_meas':>11s}{'cos_pred':>10s}"
          f"{'r':>9s}{'n_dir':>8s}")
    for s in range(0, len(have), block_fires):
        blk = have[s:s + block_fires]
        if len(blk) < block_fires / 2:
            print(f"  ... dropped last partial block ({len(blk)} fires < "
                  f"half of {block_fires})")
            continue
        cm = sum(x["cos_meas"] for x in blk) / len(blk)
        cp = sum(x["cos_pred"] for x in blk) / len(blk)
        N = sum(x["N"] for x in blk) / len(blk)
        r = cm / cp
        inb = [e for e in evals if blk[0]["ts"] <= e["ts"] <= blk[-1]["ts"]]
        a = (sum(e["test-accuracy"] for e in inb) / len(inb)) if inb else float("nan")
        print(f"  {blk[0]['i']:4d}-{blk[-1]['i']:<2d}{N:6.0f}{a:7.3f}{cm:11.5f}"
              f"{cp:10.4f}{r:9.4f}{r * r * N:8.3f}")
    print(f"  D = {d_mean:.4f} +/- {d_sem:.4f}  (n={len(fires_r)} fires, "
          f"median spacing {spacing:g} commits)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--cos", action="store_true", help="also audit the cos probe")
    ap.add_argument("--cos-block-fires", type=int, default=COS_BLOCK_FIRES,
                     help="cos fires per reported block (default %(default)s)")
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
        m = meta(cfg, run)
        commits, evals = load(path)
        if not commits:
            print(f"{rid:9s}  no server_update records", file=sys.stderr)
            continue
        rows = enrich(m, commits)
        score(rid, m, rows, evals)
        audits.append((rid, m, rows, evals))
    if a.cos:
        for args in audits:
            cos_audit(*args, block_fires=a.cos_block_fires)


if __name__ == "__main__":
    main()
