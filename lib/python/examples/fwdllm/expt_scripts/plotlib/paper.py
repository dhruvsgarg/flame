# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""FluxTune paper figures (EXPERIMENTS.md §7): [A] end-to-end rows, [B] mechanism, [C] ablation.

An Arm is one configuration with one RunResult per seed. Curves draw the arm's
median-TTA seed; bars draw the seed mean with min-max whiskers. Time is each run's
native clock (sim: vclock). TTA = first eval ending `consecutive` evals >= target.
Arms with no runs are skipped, so a half-filled manifest still renders.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field

import numpy as np

from . import baselines as B
from . import style as S

INK = "#333333"      # labels/values: text ink, never the series color
MUTED = "#777777"
DNF = "DNF"          # seed never reached target

_ARM_MARKERS = ["o", "s", "D", "v", "P", "X", "h", "<", ">"]
_ARM_LS = ["--", ":", "-."]
# saturated categorical hues for non-registry arms; skip FwdLLM vermillion / FluxTune navy
_ARM_COLORS = ["#009E73", "#AA3377", "#E69F00", "#56B4E9", "#000000", "#999933"]


# --------------------------------------------------------------------------- #
# arms + per-seed metrics
# --------------------------------------------------------------------------- #
@dataclass
class Arm:
    key: str
    label: str
    style: B.Style
    runs: list = field(default_factory=list)      # RunResult per seed


def arm_style(key, label=None, idx=0, n=1, sync=False):
    """Registry style for a baseline key; else a distinct categorical hue."""
    if key in B.BASELINES:
        st = B.BASELINES[key]
        return dataclasses.replace(st, label=label or st.label)
    return B.Style(key=key, label=label or key, color=_ARM_COLORS[idx % len(_ARM_COLORS)],
                   linestyle=_ARM_LS[idx % len(_ARM_LS)],
                   marker=_ARM_MARKERS[idx % len(_ARM_MARKERS)],
                   emphasis="regular", order=50 + idx, sync=sync)


def seed_metrics(rr, target, consecutive=1):
    """One seed's headline numbers; bytes/commits are to-target, or whole-run if DNF."""
    e = rr.first_crossing(target, consecutive) if target is not None else None
    cut = e["ts"] if e else None
    down, up = rr.bytes_until(cut)
    peak = rr.max_accuracy()
    return {
        "reached": e is not None,
        "tta_h": (rr.native_s(cut) / 3600.0) if e else None,
        "down_gb": down / 1e9, "up_gb": up / 1e9,
        "commits": rr.commits_until(cut),
        "peak_acc": peak * 100 if peak is not None else None,
        "final_acc": (rr.final_accuracy() or 0) * 100,
    }


def representative(arm, target, consecutive=1):
    """Median-TTA seed (ties to the faster); if none reached, the highest-peak seed."""
    if not arm.runs:
        return None
    ms = [(rr, seed_metrics(rr, target, consecutive)) for rr in arm.runs]
    hit = sorted((m["tta_h"], i) for i, (_, m) in enumerate(ms) if m["reached"])
    if hit:
        return ms[hit[(len(hit) - 1) // 2][1]][0]
    return max(arm.runs, key=lambda rr: rr.max_accuracy() or 0)


def _agg(vals):
    """(mean, lo, hi) over non-None values; None if empty."""
    v = [x for x in vals if x is not None]
    if not v:
        return None
    return float(np.mean(v)), float(min(v)), float(max(v))


def arm_summary(arm, target, consecutive=1):
    ms = [seed_metrics(rr, target, consecutive) for rr in arm.runs]
    reached = [m for m in ms if m["reached"]]
    return {
        "n_seeds": len(ms), "n_reached": len(reached),
        "tta_h": _agg([m["tta_h"] for m in reached]),
        "down_gb": _agg([m["down_gb"] for m in (reached or ms)]),
        "up_gb": _agg([m["up_gb"] for m in (reached or ms)]),
        "total_gb": _agg([m["down_gb"] + m["up_gb"] for m in (reached or ms)]),
        "commits": _agg([m["commits"] for m in (reached or ms)]),
        "peak_acc": _agg([m["peak_acc"] for m in ms]),
    }


# --------------------------------------------------------------------------- #
# drawing helpers
# --------------------------------------------------------------------------- #
def _row(n_panels, height=2.05, width=S.TEXT_WIDTH_IN):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, n_panels, figsize=(width, height))
    return fig, list(np.atleast_1d(axes))


def _clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _line(ax, x, y, st, smooth=0.0, label=None):
    pts = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    if not pts:
        return False
    xs, ys = zip(*pts)
    if smooth:
        ax.plot(xs, ys, color=st.color, linestyle=st.linestyle, lw=0.8, alpha=0.3)
        ys = S.ema(list(ys), smooth)
    ax.plot(xs, ys, color=st.color, linestyle=st.linestyle, marker=st.marker,
            markevery=0.12, markersize=4.5, lw=2.0, label=label, zorder=3,
            **st.marker_fill_kwargs())
    return True


def _target_line(ax, target):
    if target is None:
        return
    ax.axhline(target * 100, ls=":", color=MUTED, lw=1.0, zorder=1)
    ax.annotate(f"target {target * 100:.0f}%", xy=(1, target * 100),
                xycoords=("axes fraction", "data"), xytext=(-2, -2),
                textcoords="offset points", ha="right", va="top",
                fontsize=6.5, color=MUTED)


def _acc_series(rr, xkind):
    """(x, acc%) for one run; xkind: 'time' (native hours) or 'commits'."""
    xs, ys = [], []
    for e in rr.evals:
        if e["acc"] is None:
            continue
        if xkind == "time":
            s = rr.native_s(e["ts"])
            xs.append(None if s is None else s / 3600.0)
        else:
            xs.append(rr.commits_until(e["ts"]))
        ys.append(e["acc"] * 100)
    return xs, ys


def _ratio_callout(ax, x0, x1, y, text, color):
    """Arrow between two x positions + an ink label (champion-vs-baseline gap)."""
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="<->", color=color, lw=1.6,
                                shrinkA=0, shrinkB=0), zorder=6)
    ax.annotate(text, xy=((x0 + x1) / 2, y), xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=7.5, fontweight="bold", color=INK)


def _shared_legend(fig, arms, ncol=None):
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=a.style.color, linestyle=a.style.linestyle,
                      marker=a.style.marker, markersize=4.5, lw=2.0,
                      **a.style.marker_fill_kwargs()) for a in arms]
    fig.tight_layout(pad=0.3)
    fig.legend(handles, [a.label for a in arms], loc="lower center",
               ncol=ncol or len(arms), bbox_to_anchor=(0.5, 1.0), frameon=False)


def watermark(fig):
    fig.text(0.5, 0.5, "SYNTHETIC DATA", ha="center", va="center", fontsize=28,
             color="#BBBBBB", alpha=0.35, rotation=20, zorder=100)


# --------------------------------------------------------------------------- #
# [A] end-to-end row: acc vs time · GB to target · acc vs commits
# --------------------------------------------------------------------------- #
def draw_e2e_row(axes, arms, cell, opts, title_prefix=True):
    target, cons, smooth = cell.get("target_acc"), opts.get("consecutive", 1), opts.get("smooth", 0.0)
    base_key, champ_key = opts.get("baseline", "fwdllm"), opts.get("champion", "fluxtune")
    ax_t, ax_b, ax_c = axes
    summ = {a.key: arm_summary(a, target, cons) for a in arms}

    for a in arms:                                       # (a) accuracy vs time
        rr = representative(a, target, cons)
        _line(ax_t, *_acc_series(rr, "time"), a.style, smooth, a.label)
    _target_line(ax_t, target)
    tb, tc = (summ.get(k, {}).get("tta_h") for k in (base_key, champ_key))
    if tb and tc and tc[0] > 0:
        _ratio_callout(ax_t, tb[0], tc[0], target * 100 + 1.5, f"{tb[0] / tc[0]:.1f}× faster",
                       B.style_for(champ_key).color)
    ax_t.set_xlabel("time (h)")
    ax_t.set_ylabel("test accuracy (%)")

    xs = np.arange(len(arms))                            # (b) GB to target
    for i, a in enumerate(arms):
        s = summ[a.key]
        down = s["down_gb"][0] if s["down_gb"] else 0.0
        up = s["up_gb"][0] if s["up_gb"] else 0.0
        dnf = s["n_reached"] == 0
        alpha = 0.6 if dnf else 1.0
        ax_b.bar(i, down, 0.6, color=a.style.color, edgecolor="white", lw=1.0,
                 alpha=alpha, zorder=3)
        ax_b.bar(i, up, 0.6, bottom=down, color=a.style.color, hatch="....",
                 edgecolor="white", lw=1.0, alpha=alpha, zorder=3)
        tot = s["total_gb"]
        if tot and tot[2] > tot[1]:
            ax_b.errorbar(i, down + up, yerr=[[down + up - tot[1]], [tot[2] - down - up]],
                          color=INK, lw=0.8, capsize=2, zorder=4)
        lab = f"{down + up:.1f}" + (f"\n{DNF}" if dnf else "")
        ax_b.annotate(lab, xy=(i, tot[2] if tot else down + up), xytext=(0, 2),
                      textcoords="offset points", ha="center", va="bottom",
                      fontsize=6.5, color=INK)
    ax_b.set_xticks(xs)
    ax_b.set_xticklabels([a.label for a in arms], fontsize=7)
    ax_b.set_ylabel("GB to target")
    from matplotlib.patches import Patch
    ax_b.legend([Patch(facecolor=MUTED, edgecolor="white"),
                 Patch(facecolor=MUTED, hatch="....", edgecolor="white")],
                ["downlink", "uplink"], loc="upper center", ncol=2, fontsize=6.5)
    top = max([s["total_gb"][2] for s in summ.values() if s["total_gb"]] or [1.0])
    ax_b.set_ylim(0, top * 1.6)
    gb, gc = (summ.get(k, {}).get("total_gb") for k in (base_key, champ_key))
    if gb and gc and gc[0] > 0 and summ[base_key]["n_reached"] and summ[champ_key]["n_reached"]:
        ib = [a.key for a in arms].index(base_key)
        ic = [a.key for a in arms].index(champ_key)
        _ratio_callout(ax_b, ib, ic, top * 1.22, f"{gb[0] / gc[0]:.1f}× less",
                       B.style_for(champ_key).color)

    for a in arms:                                       # (c) accuracy vs commits
        rr = representative(a, target, cons)
        _line(ax_c, *_acc_series(rr, "commits"), a.style, smooth, a.label)
    _target_line(ax_c, target)
    ax_c.set_xlabel("server model updates (commits)")
    ax_c.set_ylabel("test accuracy (%)")

    if title_prefix:
        ax_t.set_title(cell.get("label", cell.get("id", "")), loc="left", fontsize=8.5)
    for ax in axes:
        _clean(ax)


def fig_e2e_row(arms, cell, opts):
    if not any(a.runs for a in arms):
        return None
    arms = [a for a in arms if a.runs]
    fig, axes = _row(3)
    draw_e2e_row(axes, arms, cell, opts)
    _shared_legend(fig, arms)
    return fig


def fig_e2e_grid(cells_arms, opts):
    """All [A] rows stacked (one per model/dataset cell)."""
    rows = [(c, [a for a in arms if a.runs]) for c, arms in cells_arms]
    rows = [(c, arms) for c, arms in rows if arms]
    if not rows:
        return None
    import matplotlib.pyplot as plt
    fig, grid = plt.subplots(len(rows), 3, figsize=(S.TEXT_WIDTH_IN, 2.0 * len(rows)),
                             squeeze=False)
    for (cell, arms), axes in zip(rows, grid):
        draw_e2e_row(list(axes), arms, cell, opts)
    seen = {}
    for _, arms in rows:
        for a in arms:
            seen.setdefault(a.key, a)
    _shared_legend(fig, list(seen.values()))
    return fig


# --------------------------------------------------------------------------- #
# [B] mechanism: participation · diversity · progression
# --------------------------------------------------------------------------- #
def _session_minutes(rr):
    """Per-client engagement: async = dispatch->commit per upload; sync = one round span."""
    durs = rr.session_durs if rr.is_async or not rr.round_span_durs else rr.round_span_durs
    return [d / 60.0 for d in durs if d is not None and d >= 0]


def fig_b1_participation(arms, cell, opts):
    """CDF of how long a selected client must stay engaged."""
    target, cons = cell.get("target_acc"), opts.get("consecutive", 1)
    fig, (ax,) = _row(1, height=2.2, width=S.COL_WIDTH_IN)
    drawn = []
    for a in arms:
        rr = representative(a, target, cons)
        mins = sorted(_session_minutes(rr)) if rr else []
        mins = [m for m in mins if m > 0]
        if not mins:
            continue
        ys = np.arange(1, len(mins) + 1) / len(mins) * 100
        p50 = float(np.median(mins))
        ax.plot(mins, ys, color=a.style.color, linestyle=a.style.linestyle, lw=2.0,
                label=f"{a.label} (P50 {p50:.1f} min)")
        ax.plot([p50], [50], marker="|", markersize=9, color=a.style.color)
        drawn.append(a)
    if not drawn:
        return None
    ax.set_xscale("log")
    ax.set_xlabel("client engagement per selection (min, log)")
    ax.set_ylabel("CDF (%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="lower right", fontsize=6.5)
    _clean(ax)
    return fig


def _gini(counts):
    c = np.sort(np.asarray(counts, dtype=float))
    if c.size == 0 or c.sum() == 0:
        return None
    n = c.size
    return float((2 * np.arange(1, n + 1) - n - 1).dot(c) / (n * c.sum()))


def _contrib_counts(rr, n_clients):
    counts = {}
    for _, end in rr.contribs:
        counts[end] = counts.get(end, 0) + 1
    vals = list(counts.values())
    pad = max(0, (n_clients or rr.n_trainers or 0) - len(vals))
    return vals + [0] * pad


def fig_b2_diversity(arms, cell, opts):
    """(a) distinct clients that contributed vs time; (b) Lorenz curve of contributions."""
    target, cons = cell.get("target_acc"), opts.get("consecutive", 1)
    n = cell.get("n_clients")
    fig, (ax_a, ax_b) = _row(2, height=2.1, width=S.TEXT_WIDTH_IN * 0.66)
    drawn = []
    for a in arms:
        rr = representative(a, target, cons)
        if rr is None or not rr.contribs:
            continue
        seen, xs, ys = set(), [], []
        for ts, end in sorted(rr.contribs, key=lambda t: t[0] or 0):
            if end not in seen:
                seen.add(end)
                s = rr.native_s(ts)
                xs.append(None if s is None else s / 3600.0)
                ys.append(len(seen) / (n or rr.n_trainers or len(seen)) * 100)
        _line(ax_a, xs, ys, a.style, label=a.label)
        counts = _contrib_counts(rr, n)
        c = np.sort(np.asarray(counts, dtype=float))
        lx = np.linspace(0, 100, c.size + 1)
        ly = np.concatenate([[0], np.cumsum(c) / c.sum() * 100])
        g = _gini(counts)
        ax_b.plot(lx, ly, color=a.style.color, linestyle=a.style.linestyle, lw=2.0,
                  label=f"{a.label} (Gini {g:.2f})")
        drawn.append(a)
    if not drawn:
        return None
    ax_a.set_xlabel("time (h)")
    ax_a.set_ylabel("distinct clients contributed (%)")
    ax_a.legend(loc="lower right", fontsize=6.5)
    ax_b.plot([0, 100], [0, 100], color=MUTED, lw=0.8, ls=":", label="equal share")
    ax_b.set_xlabel("clients, fewest contributions first (%)")
    ax_b.set_ylabel("share of contributions (%)")
    ax_b.legend(loc="upper left", fontsize=6.5)
    for ax in (ax_a, ax_b):
        _clean(ax)
    return fig


def fig_b3_progression(arms, cell, opts):
    """(a) uploads pooled per commit; (b) commits vs time — how fast the model advances."""
    target, cons = cell.get("target_acc"), opts.get("consecutive", 1)
    fig, (ax_a, ax_b) = _row(2, height=2.1, width=S.TEXT_WIDTH_IN * 0.66)
    drawn = []
    for a in arms:
        rr = representative(a, target, cons)
        if rr is None or not rr.commit_ts:
            continue
        k = np.arange(1, len(rr.commit_uploads) + 1)
        med = float(np.median(rr.commit_uploads))
        _line(ax_a, list(k), list(rr.commit_uploads), a.style, smooth=0.8,
              label=f"{a.label} (median {med:.0f})")
        hrs = [(rr.native_s(t) or 0) / 3600.0 for t in rr.commit_ts]
        _line(ax_b, hrs, list(k), a.style, label=a.label)
        drawn.append(a)
    if not drawn:
        return None
    ax_a.set_xlabel("commit index")
    ax_a.set_ylabel("uploads per commit")
    ax_a.set_yscale("log")
    ax_a.legend(loc="best", fontsize=6.5)
    ax_b.set_xlabel("time (h)")
    ax_b.set_ylabel("cumulative commits")
    ax_b.legend(loc="upper left", fontsize=6.5)
    for ax in (ax_a, ax_b):
        _clean(ax)
    return fig


# --------------------------------------------------------------------------- #
# [C] ablation group: TTA bars · acc vs time · acc vs commits
# --------------------------------------------------------------------------- #
def fig_c_group(arms, cell, opts, title=None):
    arms = [a for a in arms if a.runs]
    if not arms:
        return None
    target, cons, smooth = cell.get("target_acc"), opts.get("consecutive", 1), opts.get("smooth", 0.0)
    fig, (ax_bar, ax_t, ax_c) = _row(3, height=2.2)
    summ = [arm_summary(a, target, cons) for a in arms]
    ref = summ[0]["tta_h"][0] if summ[0]["tta_h"] else None
    tops = [s["tta_h"][2] for s in summ if s["tta_h"]]
    ymax = (max(tops) if tops else 1.0) * 1.25
    for i, (a, s) in enumerate(zip(arms, summ)):
        if s["tta_h"]:
            m, lo, hi = s["tta_h"]
            ax_bar.bar(i, m, 0.6, color=a.style.color, edgecolor="white", lw=1.0, zorder=3)
            if hi > lo:
                ax_bar.errorbar(i, m, yerr=[[m - lo], [hi - m]], color=INK, lw=0.8,
                                capsize=2, zorder=4)
            lab = f"{m:.1f}h"
            if ref and i:
                lab += f"\n{ref / m:.1f}×"
            if s["n_reached"] < s["n_seeds"]:
                lab += f"\n{s['n_reached']}/{s['n_seeds']} seeds"
            ax_bar.annotate(lab, xy=(i, hi), xytext=(0, 2), textcoords="offset points",
                            ha="center", va="bottom", fontsize=6, color=INK)
        else:
            ax_bar.bar(i, ymax * 0.9, 0.6, color="white", edgecolor=a.style.color,
                       hatch="////", lw=0.8, zorder=3)
            ax_bar.text(i, ymax * 0.9, DNF, ha="center", va="bottom", fontsize=6.5, color=INK)
    ax_bar.set_ylim(0, ymax * 1.15)
    ax_bar.set_xticks(range(len(arms)))
    ax_bar.set_xticklabels([a.label for a in arms], rotation=30, ha="right", fontsize=6.5)
    ax_bar.set_ylabel(f"time to {target * 100:.0f}% (h)" if target else "time to target (h)")

    for a in arms:
        rr = representative(a, target, cons)
        _line(ax_t, *_acc_series(rr, "time"), a.style, smooth, a.label)
        _line(ax_c, *_acc_series(rr, "commits"), a.style, smooth, a.label)
    for ax, xl in ((ax_t, "time (h)"), (ax_c, "server model updates (commits)")):
        _target_line(ax, target)
        ax.set_xlabel(xl)
        ax.set_ylabel("test accuracy (%)")
    if title:
        ax_bar.set_title(title, loc="left", fontsize=8.5)
    for ax in (ax_bar, ax_t, ax_c):
        _clean(ax)
    _shared_legend(fig, arms, ncol=min(len(arms), 4))
    return fig


def fig_c_step_size(arms, cell, opts):
    """Server step over the run: (a) ρ = ‖Δθ‖/‖θ‖ per commit; (b) Φ (weight-norm growth).
    Needs `server_update_audit: true` on the run."""
    fig, (ax_r, ax_p) = _row(2, height=2.1, width=S.TEXT_WIDTH_IN * 0.66)
    drawn = []
    for a in arms:
        rr = representative(a, cell.get("target_acc"), opts.get("consecutive", 1))
        ups = [u for u in (rr.server_updates if rr else []) if u.get("rho") is not None]
        if not ups:
            continue
        k = list(range(1, len(ups) + 1))
        _line(ax_r, k, [u["rho"] for u in ups], a.style, smooth=0.6, label=a.label)
        phi = [u.get("phi") for u in ups]
        if all(p is None for p in phi):                 # Φ from ρ: ‖θ_t‖² = ‖θ_{t−1}‖²(1+ρ²)
            phi = list(np.sqrt(np.cumprod([1 + u["rho"] ** 2 for u in ups])))
        _line(ax_p, k, phi, a.style, label=a.label)
        drawn.append(a)
    if not drawn:
        return None
    ax_r.set_xlabel("commit index")
    ax_r.set_ylabel("step size ρ")
    ax_r.legend(loc="best", fontsize=6.5)
    ax_p.set_xlabel("commit index")
    ax_p.set_ylabel("trainable-norm growth Φ")
    ax_p.legend(loc="upper left", fontsize=6.5)
    for ax in (ax_r, ax_p):
        _clean(ax)
    return fig


# --------------------------------------------------------------------------- #
# summary rows (paper numbers)
# --------------------------------------------------------------------------- #
def summary_rows(set_id, cell, arms, opts):
    target, cons = cell.get("target_acc"), opts.get("consecutive", 1)
    rows = []
    for a in arms:
        for seed, rr in enumerate(a.runs):
            m = seed_metrics(rr, target, cons)
            rows.append({"set": set_id, "cell": cell.get("id"), "arm": a.key,
                         "label": a.label, "seed": seed, "run_dir": rr.run_dir,
                         "target_acc": target, **{k: (round(v, 4) if isinstance(v, float)
                                                     and not math.isnan(v) else v)
                                                  for k, v in m.items()}})
    return rows
