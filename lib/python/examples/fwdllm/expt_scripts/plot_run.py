#!/usr/bin/env python3
# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Single-run extensive plots (EXPERIMENTS.md §4 — all five experiments as views
over ONE run's telemetry).

`compare_baselines.py` overlays the three baselines; this is its per-run twin:
it renders the full plot set for a SINGLE run dir so a stalled/converged run can
be inspected on its own before (or without) a cross-baseline comparison. Same
provenance discipline — every panel is a pure reducer over `telemetry/*.jsonl`.

    python plot_run.py --run-dir experiments/run_..._fluxtune_..._real \
        [--target-acc 0.82] [--out DIR]

Streams the aggregator JSONL in ONE pass (it can be >1 GB — never loaded whole)
and each trainer JSONL in one pass, accumulating only the per-experiment
aggregates. Writes PNG panels + a `summary.json` of the scalar metrics.
"""

from __future__ import annotations

import argparse
import json
import glob
import os
import sys


# --------------------------------------------------------------------------- #
# streaming loaders (bounded memory — aggregator file can be gigabytes)
# --------------------------------------------------------------------------- #
def _iter_jsonl(path):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue  # partial trailing line
    except OSError:
        return


def scan_aggregator(path):
    """One streaming pass → all aggregator-side aggregates for the 5 experiments."""
    acc_by_bin, loss_by_bin, ts_by_bin = {}, {}, {}          # E1 / E3
    agg_compute_s = agg_barrier_s = agg_drain_s = 0.0         # E2
    agg_wall_s = 0.0
    session_durs = []                                          # E5
    comm_sizes = {"agg_to_trainer": [], "trainer_to_agg": []} # E4 (agg side)
    comm_by_kind = {}                                          # E4 breakdown
    comm_cum = []  # (ts, direction, size) for cumulative-bytes-over-time
    t0 = None

    for e in _iter_jsonl(path):
        ev = e.get("event")
        ts = e.get("ts")
        if ts is not None:
            t0 = ts if t0 is None else min(t0, ts)

        if ev == "agg_eval":
            d = e.get("data_id")
            if d is None:
                continue
            d = int(d)
            a = e.get("test-accuracy")
            if a is not None:
                acc_by_bin[d] = float(a)
            if e.get("test-loss") is not None:
                loss_by_bin[d] = float(e["test-loss"])
            if ts is not None:
                ts_by_bin[d] = ts

        elif ev == "agg_round":
            agg_compute_s += (e.get("aggregate_fedavg_s") or 0.0) + (e.get("eval_s") or 0.0)
            agg_barrier_s += e.get("barrier_wait_s") or 0.0
            agg_drain_s += e.get("drain_tail_s") or 0.0
            we = e.get("wall_elapsed_s")
            if we is not None:
                agg_wall_s = max(agg_wall_s, we)
            for iv in (e.get("contributor_intervals") or []):
                d, c = iv.get("dispatch_ts"), iv.get("commit_ts")
                if d is not None and c is not None and c >= d:
                    session_durs.append(c - d)

        elif ev == "comm":
            direction = e.get("direction")
            sz = e.get("size_bytes")
            if direction in comm_sizes and sz is not None:
                comm_sizes[direction].append(sz)
                comm_cum.append((ts, direction, sz))
                key = (direction, e.get("payload_kind"))
                comm_by_kind[key] = comm_by_kind.get(key, 0) + sz

    return {
        "acc_by_bin": acc_by_bin, "loss_by_bin": loss_by_bin, "ts_by_bin": ts_by_bin,
        "agg_compute_s": agg_compute_s, "agg_barrier_s": agg_barrier_s,
        "agg_drain_s": agg_drain_s, "agg_wall_s": agg_wall_s,
        "session_durs": session_durs, "comm_sizes": comm_sizes,
        "comm_by_kind": comm_by_kind, "comm_cum": comm_cum, "t0": t0,
    }


def scan_trainers(tdir):
    """Per-trainer aggregates: busy fraction, compute, forward passes, participation."""
    busy_frac, gpu_secs = [], 0.0
    fwd_total = pert_total = 0
    part_rounds, part_bins, part_iters = [], [], []
    up_sizes = []  # trainer_to_agg message sizes (client-side comm)
    n_trainers = 0

    for f in glob.glob(os.path.join(tdir, "trainer_*.jsonl")):
        n_trainers += 1
        rounds_seen, bins_seen, n_iters = set(), set(), 0
        gpu = 0.0
        ts_lo = ts_hi = None
        fp_max = pt_max = 0
        for e in _iter_jsonl(f):
            ev = e.get("event")
            ts = e.get("ts")
            if ev == "trainer_round":
                gpu += e.get("gpu_compute_s") or 0.0
                n_iters += 1
                if e.get("round") is not None:
                    rounds_seen.add(e["round"])
                if e.get("data_id") is not None:
                    bins_seen.add(e["data_id"])
                if e.get("forward_passes_total") is not None:
                    fp_max = max(fp_max, e["forward_passes_total"])
                if e.get("perturbations_total") is not None:
                    pt_max = max(pt_max, e["perturbations_total"])
                if ts is not None:
                    ts_lo = ts if ts_lo is None else min(ts_lo, ts)
                    ts_hi = ts if ts_hi is None else max(ts_hi, ts)
            elif ev == "comm" and e.get("direction") == "trainer_to_agg":
                if e.get("size_bytes") is not None:
                    up_sizes.append(e["size_bytes"])
        gpu_secs += gpu
        fwd_total += fp_max
        pert_total += pt_max
        if n_iters:
            part_rounds.append(len(rounds_seen))
            part_bins.append(len(bins_seen))
            part_iters.append(n_iters)
        if ts_lo is not None and ts_hi is not None and ts_hi > ts_lo:
            busy_frac.append(min(1.0, gpu / (ts_hi - ts_lo)))

    return {
        "busy_frac": busy_frac, "gpu_secs": gpu_secs,
        "fwd_total": fwd_total, "pert_total": pert_total,
        "part_rounds": part_rounds, "part_bins": part_bins, "part_iters": part_iters,
        "up_sizes": up_sizes, "n_trainers": n_trainers,
    }


# --------------------------------------------------------------------------- #
# stats helpers
# --------------------------------------------------------------------------- #
def _pct(xs, q):
    import numpy as np
    xs = [x for x in xs if x is not None]
    return float(np.percentile(xs, q)) if xs else None


def _pctiles(xs):
    return {"p50": _pct(xs, 50), "p90": _pct(xs, 90), "p99": _pct(xs, 99)}


# --------------------------------------------------------------------------- #
# plotting
# --------------------------------------------------------------------------- #
def make_plots(A, T, meta, out_dir, target):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    baseline = meta.get("baseline", "run")
    t0 = A["t0"]

    def save(fig, name):
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(out_dir, f"{name}.{ext}"), dpi=140)
        plt.close(fig)

    # ---- Experiment 1: learning curves -------------------------------------
    bins = sorted(A["acc_by_bin"])
    accs = [A["acc_by_bin"][b] * 100 for b in bins]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    if bins:
        ax[0].plot(bins, accs, lw=1.6, color="#2b6cb0")
        ax[0].axhline(target * 100, ls="--", color="#c53030", lw=1,
                      label=f"target {target*100:.0f}%")
        peak_b = max(A["acc_by_bin"], key=A["acc_by_bin"].get)
        peak_a = A["acc_by_bin"][peak_b] * 100
        ax[0].scatter([peak_b], [peak_a], color="#dd6b20", zorder=5,
                      label=f"peak {peak_a:.2f}% @bin {peak_b}")
        ax[0].set(xlabel="data bin", ylabel="test accuracy (%)",
                  title="E1 — accuracy vs data bin")
        ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
        # accuracy vs wall clock
        if t0 is not None:
            hrs = [(A["ts_by_bin"][b] - t0) / 3600 for b in bins if b in A["ts_by_bin"]]
            av = [A["acc_by_bin"][b] * 100 for b in bins if b in A["ts_by_bin"]]
            ax[1].plot(hrs, av, lw=1.6, color="#2b6cb0")
            ax[1].axhline(target * 100, ls="--", color="#c53030", lw=1)
            ax[1].set(xlabel="wall clock (h)", ylabel="test accuracy (%)",
                      title="E1 — accuracy vs wall time")
            ax[1].grid(alpha=.3)
        lbins = sorted(A["loss_by_bin"])
        ax[2].plot(lbins, [A["loss_by_bin"][b] for b in lbins], lw=1.6, color="#805ad5")
        ax[2].set(xlabel="data bin", ylabel="test loss",
                  title="E1 — loss vs data bin")
        ax[2].grid(alpha=.3)
    save(fig, "e1_learning_curves")

    # ---- Experiment 2: utilization -----------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    bf = T["busy_frac"]
    if bf:
        ax[0].hist(np.array(bf) * 100, bins=25, color="#38a169", alpha=.75)
        for q, c in [(50, "#2b6cb0"), (90, "#dd6b20"), (99, "#c53030")]:
            v = _pct(bf, q) * 100
            ax[0].axvline(v, ls="--", color=c, lw=1, label=f"P{q}={v:.1f}%")
        ax[0].set(xlabel="trainer busy fraction (%)", ylabel="# trainers",
                  title=f"E2 — trainer busy fraction (n={len(bf)})")
        ax[0].legend(fontsize=8)
    # aggregator time decomposition
    w = A["agg_wall_s"] or 1.0
    comp, barr, drain = A["agg_compute_s"], A["agg_barrier_s"], A["agg_drain_s"]
    other = max(0.0, w - comp - barr - drain)
    parts = [("compute", comp), ("barrier-wait", barr), ("drain", drain), ("idle/other", other)]
    cols = ["#38a169", "#dd6b20", "#805ad5", "#a0aec0"]
    ax[1].bar([p[0] for p in parts], [p[1] / w * 100 for p in parts], color=cols)
    ax[1].set(ylabel="% of aggregator wall", title="E2 — aggregator time decomposition")
    for i, p in enumerate(parts):
        ax[1].text(i, p[1] / w * 100, f"{p[1]/w*100:.1f}%", ha="center", va="bottom", fontsize=8)
    save(fig, "e2_utilization")

    # ---- Experiment 3: compute productivity --------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    lbins = sorted(A["loss_by_bin"])
    if lbins:
        first_loss = A["loss_by_bin"][lbins[0]]
        dloss = [first_loss - A["loss_by_bin"][b] for b in lbins]
        ax[0].plot(lbins, dloss, lw=1.6, color="#2b6cb0")
        ax[0].set(xlabel="data bin", ylabel="Δloss (from first)",
                  title="E3 — cumulative loss reduction")
        ax[0].grid(alpha=.3)
    gpu = T["gpu_secs"] + A["agg_compute_s"]
    fwd = T["fwd_total"]
    dl = (A["loss_by_bin"][lbins[0]] - A["loss_by_bin"][lbins[-1]]) if lbins else None
    labels = ["Δloss / GPU-hour", "Δloss / Mfwd-pass"]
    vals = [
        (dl / (gpu / 3600)) if (dl and gpu) else 0,
        (dl / (fwd / 1e6)) if (dl and fwd) else 0,
    ]
    ax[1].bar(labels, vals, color=["#38a169", "#805ad5"])
    for i, v in enumerate(vals):
        ax[1].text(i, v, f"{v:.4g}", ha="center", va="bottom", fontsize=9)
    ax[1].set(title="E3 — learning per unit compute")
    save(fig, "e3_productivity")

    # ---- Experiment 4: network ---------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    down = sum(A["comm_sizes"]["agg_to_trainer"])
    up = sum(A["comm_sizes"]["trainer_to_agg"]) or sum(T["up_sizes"])
    ax[0].bar(["agg→trainer\n(down)", "trainer→agg\n(up)"], [down / 1e9, up / 1e9],
              color=["#2b6cb0", "#dd6b20"])
    for i, v in enumerate([down, up]):
        ax[0].text(i, v / 1e9, f"{v/1e9:.2f} GB", ha="center", va="bottom", fontsize=9)
    ax[0].set(ylabel="total bytes (GB)", title="E4 — data transmitted")
    # message-size distributions
    ds = A["comm_sizes"]["agg_to_trainer"]
    us = A["comm_sizes"]["trainer_to_agg"] or T["up_sizes"]
    if ds:
        ax[1].hist(np.array(ds) / 1e6, bins=40, color="#2b6cb0", alpha=.7)
        ax[1].set(xlabel="message size (MB)", ylabel="# messages",
                  title="E4 — agg→trainer msg sizes")
    if us:
        ax[2].hist(np.array(us) / 1e6, bins=40, color="#dd6b20", alpha=.7)
        ax[2].set(xlabel="message size (MB)", ylabel="# messages",
                  title="E4 — trainer→agg msg sizes")
    save(fig, "e4_network")

    # ---- Experiment 5: sessions & participation ----------------------------
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    sd = A["session_durs"]
    if sd:
        ax[0].hist(sd, bins=40, color="#805ad5", alpha=.75)
        for q, c in [(50, "#2b6cb0"), (90, "#dd6b20"), (99, "#c53030")]:
            v = _pct(sd, q)
            ax[0].axvline(v, ls="--", color=c, lw=1, label=f"P{q}={v:.1f}s")
        ax[0].set(xlabel="session duration (s)", ylabel="# sessions",
                  title=f"E5 — session durations (n={len(sd)})")
        ax[0].legend(fontsize=8)
        xs = np.sort(sd)
        ax[1].plot(xs, np.arange(1, len(xs) + 1) / len(xs) * 100, color="#805ad5")
        ax[1].set(xlabel="session duration (s)", ylabel="CDF (%)",
                  title="E5 — session-duration CDF")
        ax[1].grid(alpha=.3)
    if T["part_iters"]:
        ax[2].hist(T["part_iters"], bins=25, color="#38a169", alpha=.6, label="iterations")
        ax[2].hist(T["part_bins"], bins=25, color="#dd6b20", alpha=.6, label="data bins")
        ax[2].set(xlabel="count per client", ylabel="# clients",
                  title="E5 — participation per client")
        ax[2].legend(fontsize=8)
    save(fig, "e5_sessions")

    print(f"  [plot_run] wrote 5 panels (png+pdf) to {out_dir}")


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--target-acc", type=float, default=0.82)
    ap.add_argument("--out", default=None, help="default: <run-dir>/plots")
    args = ap.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    tdir = os.path.join(run_dir, "telemetry")
    agg_files = glob.glob(os.path.join(tdir, "aggregator_*.jsonl"))
    if not agg_files:
        print(f"  [plot_run] no aggregator telemetry under {tdir}", file=sys.stderr)
        return 1
    out_dir = os.path.abspath(args.out) if args.out else os.path.join(run_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)

    base = os.path.basename(run_dir)
    m = None
    import re
    mm = re.match(r"^run_\d{8}_\d{6}_(?P<b>.+)_n(?P<n>\d+)_smoke", base)
    meta = {"baseline": mm["b"] if mm else base, "run_dir": base}

    print(f"  [plot_run] streaming aggregator ({os.path.getsize(agg_files[0])/1e9:.2f} GB)…")
    A = scan_aggregator(agg_files[0])
    print(f"  [plot_run] streaming {len(glob.glob(os.path.join(tdir,'trainer_*.jsonl')))} trainers…")
    T = scan_trainers(tdir)

    # scalar summary
    lbins = sorted(A["loss_by_bin"])
    dl = (A["loss_by_bin"][lbins[0]] - A["loss_by_bin"][lbins[-1]]) if lbins else None
    gpu = T["gpu_secs"] + A["agg_compute_s"]
    down = sum(A["comm_sizes"]["agg_to_trainer"])
    up = sum(A["comm_sizes"]["trainer_to_agg"]) or sum(T["up_sizes"])
    w = A["agg_wall_s"] or None
    summary = {
        "run_dir": base, "baseline": meta["baseline"], "target_acc": args.target_acc,
        "e1_max_accuracy": max(A["acc_by_bin"].values()) if A["acc_by_bin"] else None,
        "e1_n_data_bins": len(A["acc_by_bin"]),
        "e1_wall_h": round((A["ts_by_bin"][max(A["ts_by_bin"])] - A["t0"]) / 3600, 3)
        if A["ts_by_bin"] and A["t0"] else None,
        "e2_trainer_busy_frac": _pctiles(T["busy_frac"]),
        "e2_agg_compute_frac": round(A["agg_compute_s"] / w, 4) if w else None,
        "e2_agg_barrier_frac": round(A["agg_barrier_s"] / w, 4) if w else None,
        "e2_agg_drain_frac": round(A["agg_drain_s"] / w, 4) if w else None,
        "e3_delta_loss": dl,
        "e3_gpu_s_total": round(gpu, 1),
        "e3_forward_passes_total": T["fwd_total"],
        "e3_perturbations_total": T["pert_total"],
        "e3_delta_loss_per_gpu_hour": round(dl / (gpu / 3600), 6) if (dl and gpu) else None,
        "e3_delta_loss_per_Mfwd": round(dl / (T["fwd_total"] / 1e6), 6)
        if (dl and T["fwd_total"]) else None,
        "e4_bytes_down": down, "e4_bytes_up": up, "e4_bytes_total": down + up,
        "e4_msgs_down": len(A["comm_sizes"]["agg_to_trainer"]),
        "e4_msgs_up": len(A["comm_sizes"]["trainer_to_agg"]) or len(T["up_sizes"]),
        "e4_bytes_by_kind": {f"{d}/{k}": v for (d, k), v in A["comm_by_kind"].items()},
        "e5_session_s": _pctiles(A["session_durs"]),
        "e5_n_sessions": len(A["session_durs"]),
        "e5_part_iters": _pctiles(T["part_iters"]),
        "e5_part_databins": _pctiles(T["part_bins"]),
        "n_trainers": T["n_trainers"],
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    make_plots(A, T, meta, out_dir, args.target_acc)

    # headline to stdout
    print(f"\n=== {base} ===")
    print(f"  E1  max acc         : {summary['e1_max_accuracy']*100:.2f}%  "
          f"over {summary['e1_n_data_bins']} bins / {summary['e1_wall_h']} h")
    bf = summary["e2_trainer_busy_frac"]
    print(f"  E2  trainer busy    : P50={bf['p50']*100:.1f}%  P90={bf['p90']*100:.1f}%  "
          f"P99={bf['p99']*100:.1f}%   agg compute={summary['e2_agg_compute_frac']}")
    print(f"  E3  Δloss           : {summary['e3_delta_loss']:.4f}  "
          f"| {summary['e3_gpu_s_total']} GPU-s  | {summary['e3_forward_passes_total']:,} fwd-passes")
    print(f"  E4  network         : {summary['e4_bytes_total']/1e9:.2f} GB total  "
          f"({summary['e4_msgs_down']:,}↓ / {summary['e4_msgs_up']:,}↑ msgs)")
    ss = summary["e5_session_s"]
    print(f"  E5  session dur     : P50={ss['p50']:.1f}s  P90={ss['p90']:.1f}s  P99={ss['p99']:.1f}s  "
          f"({summary['e5_n_sessions']:,} sessions)")
    print(f"\n  summary.json + plots → {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
