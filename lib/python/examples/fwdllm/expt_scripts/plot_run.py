#!/usr/bin/env python3
# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Single-run diagnostics — rich multi-panel PDFs for ONE run dir (per-run twin of
compare_baselines.py). Metrics come from `plotlib.reducers` (shared, streaming).

    python plot_run.py --run-dir experiments/run_..._fluxtune_..._real [--target-acc 0.84]

Writes vector-PDF composite panels + summary.json. Learning curves are over
wall-clock / eval order (not data_id, which cycles per round), round boundaries marked.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from plotlib import reducers as R          # noqa: E402  (single source of the metrics)
from plotlib import style as S             # noqa: E402


def _pct(xs, q):
    import numpy as np
    xs = [x for x in xs if x is not None]
    return float(np.percentile(xs, q)) if xs else None


def _pctiles(xs):
    return {"p50": _pct(xs, 50), "p90": _pct(xs, 90), "p99": _pct(xs, 99)}


# --------------------------------------------------------------------------- #
# plotting — composite per-run panels
# --------------------------------------------------------------------------- #
def make_plots(rr: R.RunResult, out_dir, target, smooth=0.0):
    import numpy as np
    import matplotlib.pyplot as plt
    S.use_paper_style()

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{name}.pdf"))
        plt.close(fig)

    hrs, acc_raw, loss_raw, _rnd = rr.learning_curve()
    acc = S.ema(acc_raw, smooth)                      # visual smoothing (scalars use raw)
    loss = S.ema(loss_raw, smooth)
    idx = list(range(len(rr.evals)))                 # cumulative eval index
    trans = rr.round_transition_indices()

    # ---- Experiment 1: learning curves (time / eval-index / loss) ----------
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.0))
    if hrs:
        ax[0].plot(hrs, acc, lw=1.6, color="#2b6cb0")
        ax[0].axhline(target * 100, ls="--", color="#c53030", lw=1,
                      label=f"target {target*100:.0f}%")
        for i in trans:
            ax[0].axvline(hrs[i], color="#dd6b20", ls=":", lw=0.9, alpha=0.7)
        pa = rr.max_accuracy() * 100
        pi = max(idx, key=lambda i: (acc_raw[i] if acc_raw[i] is not None else -1))
        ax[0].scatter([hrs[pi]], [acc_raw[pi]], color="#dd6b20", zorder=5,
                      label=f"peak {pa:.2f}%")
        ax[0].set(xlabel="wall-clock time (h)", ylabel="test accuracy (%)",
                  title="E1 — accuracy vs wall time")
        ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
        ax[1].plot(idx, acc, lw=1.4, color="#2b6cb0")
        for i in trans:
            ax[1].axvline(i, color="#dd6b20", ls=":", lw=0.9, alpha=0.7,
                          label="round start" if i == trans[0] else None)
        ax[1].set(xlabel="cumulative eval (data bin)", ylabel="test accuracy (%)",
                  title="E1 — accuracy vs eval index")
        if trans:
            ax[1].legend(fontsize=8)
        ax[1].grid(alpha=.3)
        ax[2].plot(idx, loss, lw=1.4, color="#805ad5")
        for i in trans:
            ax[2].axvline(i, color="#dd6b20", ls=":", lw=0.9, alpha=0.7)
        ax[2].set(xlabel="cumulative eval (data bin)", ylabel="test loss",
                  title="E1 — loss vs eval index")
        ax[2].grid(alpha=.3)
    save(fig, "e1_learning_curves")

    # ---- Experiment 2: utilization -----------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.0))
    bf = rr.busy_frac
    if bf:
        ax[0].hist(np.array(bf) * 100, bins=25, color="#38a169", alpha=.75)
        for q, c in [(50, "#2b6cb0"), (90, "#dd6b20"), (99, "#c53030")]:
            v = _pct(bf, q) * 100
            ax[0].axvline(v, ls="--", color=c, lw=1, label=f"P{q}={v:.1f}%")
        ax[0].set(xlabel="trainer busy fraction (%)", ylabel="# trainers",
                  title=f"E2 — trainer busy fraction (n={len(bf)})")
        ax[0].legend(fontsize=8)
    w = rr.agg_wall_s or 1.0
    comp, barr, drain = rr.agg_compute_s, rr.agg_barrier_s, rr.agg_drain_s
    other = max(0.0, w - comp - barr - drain)
    parts = [("compute", comp), ("barrier-wait", barr), ("drain", drain), ("idle/other", other)]
    cols = ["#38a169", "#dd6b20", "#805ad5", "#a0aec0"]
    ax[1].bar([p[0] for p in parts], [p[1] / w * 100 for p in parts], color=cols)
    ax[1].set(ylabel="% of aggregator wall", title="E2 — aggregator time decomposition")
    for i, p in enumerate(parts):
        ax[1].text(i, p[1] / w * 100, f"{p[1]/w*100:.1f}%", ha="center", va="bottom", fontsize=8)
    save(fig, "e2_utilization")

    # ---- Experiment 3: compute productivity --------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.0))
    if loss:
        dloss = [loss[0] - v for v in loss]
        ax[0].plot(idx, dloss, lw=1.6, color="#2b6cb0")
        for i in trans:
            ax[0].axvline(i, color="#dd6b20", ls=":", lw=0.9, alpha=0.7)
        ax[0].set(xlabel="cumulative eval (data bin)", ylabel="Δloss (from first)",
                  title="E3 — cumulative loss reduction")
        ax[0].grid(alpha=.3)
    gpu = rr.gpu_s_total()
    dl = rr.delta_loss()
    labels = ["Δloss / GPU-hour", "Δloss / Mfwd-pass"]
    vals = [
        (dl / (gpu / 3600)) if (dl and gpu) else 0,
        (dl / (rr.fwd_total / 1e6)) if (dl and rr.fwd_total) else 0,
    ]
    ax[1].bar(labels, vals, color=["#38a169", "#805ad5"])
    for i, v in enumerate(vals):
        ax[1].text(i, v, f"{v:.4g}", ha="center", va="bottom", fontsize=9)
    ax[1].set(title="E3 — learning per unit compute")
    save(fig, "e3_productivity")

    # ---- Experiment 4: network ---------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.0))
    down = sum(rr.down_sizes)
    up = sum(rr.up_sizes)
    ax[0].bar(["agg→trainer\n(down)", "trainer→agg\n(up)"], [down / 1e9, up / 1e9],
              color=["#2b6cb0", "#dd6b20"])
    for i, v in enumerate([down, up]):
        ax[0].text(i, v / 1e9, f"{v/1e9:.2f} GB", ha="center", va="bottom", fontsize=9)
    ax[0].set(ylabel="total bytes (GB)", title="E4 — data transmitted")
    if rr.down_sizes:
        ax[1].hist(np.array(rr.down_sizes) / 1e6, bins=40, color="#2b6cb0", alpha=.7)
        ax[1].set(xlabel="message size (MB)", ylabel="# messages",
                  title="E4 — agg→trainer msg sizes")
    if rr.up_sizes:
        ax[2].hist(np.array(rr.up_sizes) / 1e6, bins=40, color="#dd6b20", alpha=.7)
        ax[2].set(xlabel="message size (MB)", ylabel="# messages",
                  title="E4 — trainer→agg msg sizes")
    save(fig, "e4_network")

    # ---- Experiment 5: sessions & participation ----------------------------
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.0))
    sd = rr.session_durs
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
    if rr.part_iters:
        ax[2].hist(rr.part_iters, bins=25, color="#38a169", alpha=.6, label="iterations")
        ax[2].hist(rr.part_bins, bins=25, color="#dd6b20", alpha=.6, label="data bins")
        ax[2].set(xlabel="count per client", ylabel="# clients",
                  title="E5 — participation per client")
        ax[2].legend(fontsize=8)
    save(fig, "e5_sessions")

    print(f"  [plot_run] wrote 5 PDF panels to {out_dir}")


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--target-acc", type=float, default=0.84)
    ap.add_argument("--out", default=None, help="default: <run-dir>/plots")
    ap.add_argument("--smooth", type=float, default=0.7,
                    help="EMA line-smoothing factor ∈ [0,1) (visual only; 0 = raw)")
    ap.add_argument("--loss-plateau-rel", type=float, default=0.01,
                    help="cut at last cumulative test-loss drop of this relative size "
                         "(default 0.01 = 1%%; end of productive learning)")
    ap.add_argument("--post-peak-grace-min", type=float, default=0.0,
                    help="minutes to keep past the loss-plateau point (default 0)")
    ap.add_argument("--no-cutoff", action="store_true", help="use full telemetry")
    args = ap.parse_args()
    grace_s = None if args.no_cutoff else args.post_peak_grace_min * 60.0

    run_dir = os.path.abspath(args.run_dir)
    print(f"  [plot_run] streaming telemetry from {os.path.basename(run_dir)}…")
    rr = R.load_run(run_dir, key=os.path.basename(run_dir), post_peak_grace_s=grace_s,
                    loss_plateau_rel=args.loss_plateau_rel)
    if rr is None:
        print(f"  [plot_run] no aggregator telemetry under {run_dir}/telemetry", file=sys.stderr)
        return 1
    out_dir = os.path.abspath(args.out) if args.out else os.path.join(run_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)

    losses = [e["loss"] for e in rr.evals if e["loss"] is not None]
    dl = rr.delta_loss()
    gpu = rr.gpu_s_total()
    down, up = sum(rr.down_sizes), sum(rr.up_sizes)
    w = rr.agg_wall_s or None
    wall_h = round((rr.evals[-1]["ts"] - rr.t0) / 3600, 3) if (rr.evals and rr.t0) else None
    summary = {
        "run_dir": os.path.basename(run_dir), "key": rr.key, "target_acc": args.target_acc,
        "e1_max_accuracy": rr.max_accuracy(), "e1_final_accuracy": rr.final_accuracy(),
        "e1_n_evals": len(rr.evals), "e1_wall_h": wall_h,
        "e2_trainer_busy_frac": _pctiles(rr.busy_frac),
        "e2_agg_compute_frac": round(rr.agg_compute_s / w, 4) if w else None,
        "e2_agg_barrier_frac": round(rr.agg_barrier_s / w, 4) if w else None,
        "e2_agg_drain_frac": round(rr.agg_drain_s / w, 4) if w else None,
        "e3_delta_loss": dl, "e3_gpu_s_total": round(gpu, 1),
        "e3_forward_passes_total": rr.fwd_total if rr.have_fwd else None,
        "e3_perturbations_total": rr.pert_total if rr.have_fwd else None,
        "e3_delta_loss_per_gpu_hour": round(dl / (gpu / 3600), 6) if (dl and gpu) else None,
        "e3_delta_loss_per_Mfwd": round(dl / (rr.fwd_total / 1e6), 6)
        if (dl and rr.have_fwd and rr.fwd_total) else None,
        "e4_bytes_down": down, "e4_bytes_up": up, "e4_bytes_total": down + up,
        "e4_msgs_down": len(rr.down_sizes), "e4_msgs_up": len(rr.up_sizes),
        "e4_bytes_by_kind": {f"{d}/{k}": v for (d, k), v in rr.comm_by_kind.items()},
        "e5_session_s": _pctiles(rr.session_durs), "e5_n_sessions": len(rr.session_durs),
        "e5_part_iters": _pctiles(rr.part_iters), "e5_part_databins": _pctiles(rr.part_bins),
        "n_trainers": rr.n_trainers,
    }
    summary["cutoff_h"] = (None if rr.cutoff_ts is None
                           else round((rr.cutoff_ts - rr.t0) / 3600, 3))
    summary["loss_plateau_rel"] = None if args.no_cutoff else args.loss_plateau_rel
    summary["post_peak_grace_min"] = None if args.no_cutoff else args.post_peak_grace_min
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    make_plots(rr, out_dir, args.target_acc, args.smooth)

    print(f"\n=== {os.path.basename(run_dir)} ===")
    print(f"  E1  max acc         : {(summary['e1_max_accuracy'] or 0)*100:.2f}%  "
          f"over {summary['e1_n_evals']} evals / {summary['e1_wall_h']} h")
    bf = summary["e2_trainer_busy_frac"]
    if bf["p50"] is not None:
        print(f"  E2  trainer busy    : P50={bf['p50']*100:.1f}%  P90={bf['p90']*100:.1f}%  "
              f"P99={bf['p99']*100:.1f}%   agg compute={summary['e2_agg_compute_frac']}")
    print(f"  E3  Δloss           : {summary['e3_delta_loss']}  "
          f"| {summary['e3_gpu_s_total']} GPU-s  | {summary['e3_forward_passes_total']} fwd-passes")
    print(f"  E4  network         : {summary['e4_bytes_total']/1e9:.2f} GB total  "
          f"({summary['e4_msgs_down']:,}↓ / {summary['e4_msgs_up']:,}↑ msgs)")
    ss = summary["e5_session_s"]
    if ss["p50"] is not None:
        print(f"  E5  session dur     : P50={ss['p50']:.1f}s  P90={ss['p90']:.1f}s  P99={ss['p99']:.1f}s  "
              f"({summary['e5_n_sessions']:,} sessions)")
    print(f"\n  summary.json + plots → {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
