#!/usr/bin/env python3
"""F42-T2: what each stack knows of availability vs the truth (the trace), on the leg's own clock.

  avail_view_plot.py RUN... [--out PNG] [--lanes 12]

Views (R27): notify = the trace delayed by `notify_latency_s` (Felix); oracle = the trace (`*` variants); none = everyone assumed
available; checkin = only clients waiting in the check-in pool (F42-T1; busy picks are known busy). Top: share of clients truly
AVL_TRAIN vs believed available, per run. Bottom: per-client lanes (truth vs the first check-in run's view, else the first run's).
Table: belief ≠ truth seconds per client-hour, missed (AVL_TRAIN, idle, believed not) and false (believed available, is not).
"""
import argparse
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "async_cifar10", "scripts"))
from parity import event_invariants as ev  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]  # dataviz fixed order
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def label(run_dir):
    name = os.path.basename(run_dir.rstrip("/"))
    m = re.search(r"_(s\d+u?)_(\w+?)_dbg_(?:google_speech_)?(\w+?)_n\d+", name)
    return f"{m.group(3)}:{m.group(2)}" if m else name[-40:]


def _train(trace, t):
    from flame.availability.trace import deliverable, state_at
    return True if not trace else deliverable(state_at(trace, t), "train")


def views(run, grid):
    """client -> (truth bool[], belief array: 1 available, 0 not, -1 known busy)."""
    gt = ev._ground_truth(run) or {}
    hp = run["cfg"].get("hyperparameters", {}) or {}
    know = hp.get("avail_knowledge") or "none"
    lat = float(hp.get("notify_latency_s") or 0)
    clients = [c for c in run["trainers"] if c in gt] or list(gt)
    ck = {}
    for e in sorted((e for e in run["agg"] if e.get("event") == "checkin"), key=lambda e: e.get("t", 0)):
        ck.setdefault(e["end_id"], []).append((float(e["t"]), e["kind"]))
    out = {}
    for c in clients:
        truth = np.array([_train(gt.get(c), t) for t in grid])
        if know == "notify":
            bel = np.array([_train(gt.get(c), t - lat) for t in grid], dtype=float)
        elif know == "oracle":
            bel = truth.astype(float)
        elif know == "checkin":
            bel, state, ev_c, i = np.zeros(len(grid)), 0.0, ck.get(c, []), 0
            for j, t in enumerate(grid):
                while i < len(ev_c) and ev_c[i][0] <= t:
                    state = {"pooled": 1.0, "picked": -1.0}.get(ev_c[i][1], 0.0)
                    i += 1
                bel[j] = state
        else:
            bel = np.ones(len(grid))
        out[c] = (truth, bel)
    return know, out


def mismatch(v, step):
    miss = sum(((t) & (b == 0)).sum() for t, b in v.values()) * step
    false = sum(((~t) & (b == 1)).sum() for t, b in v.values()) * step
    ch = max(1, len(v)) * max(1, len(next(iter(v.values()))[0])) * step / 3600
    return miss / ch, false / ch


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default="avail_view.png")
    ap.add_argument("--lanes", type=int, default=12)
    ap.add_argument("--step", type=float, default=60.0)
    a = ap.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    loaded = []
    for r in a.runs:
        run = ev.load_run(r)
        span = max([e.get("vclock_now") or 0 for e in run["agg"] if e.get("event") in ("agg_round", "selection")] or [0])
        grid = np.arange(0, span, a.step)
        if len(grid) < 2:
            print(f"skip {r}: no clock span")
            continue
        know, v = views(run, grid)
        loaded.append((label(r), know, grid, v))
    if not loaded:
        return 1
    fig, (top, bot) = plt.subplots(2, 1, figsize=(11, 7.5), gridspec_kw=dict(height_ratios=[1, 1.3]))
    g0, v0 = loaded[0][2], loaded[0][3]
    top.plot(g0 / 3600, np.mean([t for t, _ in v0.values()], axis=0), color=INK, lw=2, label="truth (AVL_TRAIN)")
    print("| run | knowledge | missed s / client-h | false s / client-h |\n|---|---|---|---|")
    for i, (lab, know, grid, v) in enumerate(loaded):
        avail = np.mean([b == 1 for _, b in v.values()], axis=0)
        top.plot(grid / 3600, avail, color=SERIES[i % len(SERIES)], lw=2, label=f"{lab} ({know})")
        m, f = mismatch(v, a.step)
        print(f"| {lab} | {know} | {m:.0f} | {f:.0f} |")
    top.set_ylabel("share of clients", color=MUTED)
    top.set_ylim(0, 1.02)
    top.legend(frameon=False, fontsize=8, ncol=2)
    lab, know, grid, v = next((x for x in loaded if x[1] == "checkin"), loaded[0])
    for k, c in enumerate(sorted(v)[: a.lanes]):
        t, b = v[c]
        y = k * 1.0
        bot.fill_between(grid / 3600, y, y + 0.42, where=t, color=MUTED, alpha=0.35, lw=0, step="post")
        bot.fill_between(grid / 3600, y + 0.48, y + 0.9, where=b == 1, color=SERIES[0], lw=0, step="post")
        bot.fill_between(grid / 3600, y + 0.48, y + 0.9, where=b == -1, color=SERIES[0], alpha=0.35, lw=0, step="post")
    bot.set_yticks([k + 0.45 for k in range(min(a.lanes, len(v)))], [c[-4:] for c in sorted(v)[: a.lanes]], fontsize=7)
    bot.set_xlabel("leg clock (h)", color=MUTED)
    bot.set_title(f"per client: truth AVL_TRAIN (gray, lower) vs {lab} view (blue: believed available; light: known busy)",
                  fontsize=9, color=INK, loc="left")
    top.set_title("believed available vs truth", fontsize=10, color=INK, loc="left")
    for ax in (top, bot):
        ax.grid(axis="x", color=GRID, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(a.out, dpi=130)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
