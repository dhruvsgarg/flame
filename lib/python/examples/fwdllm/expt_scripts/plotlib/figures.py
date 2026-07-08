# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Paper figure builders — one single-panel figure per experiment.

Each builder takes an ordered list of RunResult and returns a column-sized Figure;
color/label/linestyle/marker/emphasis come from `baselines.style_for()` (figures
make no independent style decisions). Baselines with no data are skipped, so partial
run-sets just plot what exists. FIG_BUILDERS maps a stable name -> builder; the name
is also the PDF basename, so adding a figure = one entry.
"""

from __future__ import annotations

import numpy as np

from . import baselines as B
from . import style as S


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _new_ax(fraction=1.0, aspect=S.DEFAULT_ASPECT):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=S.column_figsize(fraction, aspect))
    return fig, ax


def _finish(fig, ax, order_keys):
    """Legend (registry order + emphasis) + light frame cleanup."""
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        leg = ax.legend(loc="best")
        B.apply_legend_emphasis(leg)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig


def _cdf_xy(values):
    xs = np.sort(np.asarray([v for v in values if v is not None], dtype=float))
    if xs.size == 0:
        return None, None
    ys = np.arange(1, xs.size + 1) / xs.size * 100.0
    return xs, ys


def _plot_curve(ax, x, y_raw, st, smooth, label):
    """Smoothed line + a faint RAW envelope underneath — the envelope preserves the
    true peaks (e.g. the accuracy actually reached) that EMA smoothing pulls down."""
    if smooth and smooth > 0:
        ax.plot(x, y_raw, color=st.color, linestyle=st.linestyle, lw=0.7,
                alpha=0.22, zorder=2 + st.order)
        y = S.ema(y_raw, smooth)
    else:
        y = y_raw
    ax.plot(x, y, color=st.color, linestyle=st.linestyle, marker=st.marker,
            markevery=0.1, markersize=4.5, markeredgecolor="white",
            markeredgewidth=0.5, label=label, zorder=3 + st.order)


def _round_boundaries(ax, rr, hrs):
    for i in rr.round_transition_indices():
        ax.axvline(hrs[i], color=B.style_for(rr.key).color, ls=(0, (1, 2)),
                   lw=0.9, alpha=0.55, zorder=2)


def _mark_peak(ax, hrs, acc, st):
    """Star at a run's highest accuracy (over the RAW series, so it survives EMA
    smoothing) + return the peak value so the caller can print it in the legend —
    lets the reader read each baseline's best acc and its gap to the target line."""
    pts = [(h, a) for h, a in zip(hrs, acc) if a is not None]
    if not pts:
        return None
    hp, ap = max(pts, key=lambda p: p[1])
    ax.scatter([hp], [ap], marker="*", s=95, color=st.color, edgecolor="white",
               linewidth=0.6, zorder=8 + st.order)
    return ap


# --------------------------------------------------------------------------- #
# Experiment 1 — accuracy / loss vs wall-clock time (adjacent paper figures)
# --------------------------------------------------------------------------- #
def fig_e1_acc_vs_time(runs, target=None, smooth=0.0, **_):
    fig, ax = _new_ax(aspect=0.68)
    for rr in runs:
        hrs, acc, _loss, _rnd = rr.learning_curve()
        if not hrs:
            continue
        st = B.style_for(rr.key)
        peak = _mark_peak(ax, hrs, acc, st)     # ★ + peak value in the legend
        label = st.label if peak is None else f"{st.label} — peak {peak:.1f}%"
        _plot_curve(ax, hrs, acc, st, smooth, label)
        _round_boundaries(ax, rr, hrs)          # data_id cycles per round → mark epochs
    if target is not None:
        ax.axhline(target * 100, ls=":", color="#555555", lw=1.0, zorder=2)
        ax.text(ax.get_xlim()[1], target * 100, f" target {target*100:.0f}%",
                va="center", ha="left", fontsize=7, color="#555555")
    ax.set_xlabel("wall-clock time (h)")
    ax.set_ylabel("test accuracy (%)")
    ax.plot([], [], color="#888888", ls=(0, (1, 2)), lw=0.9, label="round boundary")
    ax.scatter([], [], marker="*", s=95, color="#555555", edgecolor="white",
               linewidth=0.6, label="peak accuracy")
    return _finish(fig, ax, [r.key for r in runs])


def fig_e1_loss_vs_time(runs, smooth=0.0, **_):
    """Test-loss vs wall time — the grounded companion to the accuracy curve."""
    fig, ax = _new_ax(aspect=0.68)
    any_data = False
    for rr in runs:
        hrs, _acc, loss, _rnd = rr.learning_curve()
        if not hrs or all(v is None for v in loss):
            continue
        _plot_curve(ax, hrs, loss, B.style_for(rr.key), smooth, B.style_for(rr.key).label)
        _round_boundaries(ax, rr, hrs)
        any_data = True
    if not any_data:
        return None
    ax.set_xlabel("wall-clock time (h)")
    ax.set_ylabel("test loss")
    ax.plot([], [], color="#888888", ls=(0, (1, 2)), lw=0.9, label="round boundary")
    return _finish(fig, ax, [r.key for r in runs])


# --------------------------------------------------------------------------- #
# Experiment 2 — trainer busy-fraction CDF across clients
# --------------------------------------------------------------------------- #
def fig_e2_trainer_busy_cdf(runs, **_):
    fig, ax = _new_ax()
    any_data = False
    for rr in runs:
        st = B.style_for(rr.key)
        xs, ys = _cdf_xy([b * 100 for b in rr.busy_frac])
        if xs is None:
            continue
        ax.plot(xs, ys, color=st.color, linestyle=st.linestyle,
                marker=st.marker, markevery=0.12, markersize=4.5,
                markeredgecolor="white", markeredgewidth=0.5,
                label=st.label, zorder=3 + st.order)
        any_data = True
    if not any_data:
        return None
    ax.set_xlabel("trainer busy fraction (%)")
    ax.set_ylabel("CDF of clients (%)")
    ax.set_ylim(0, 100)
    return _finish(fig, ax, [r.key for r in runs])


# --------------------------------------------------------------------------- #
# Experiment 3 — learning per unit compute (ONE figure per denominator: the two
# metrics differ ~100x in scale, so a shared axis makes one set of bars invisible)
# --------------------------------------------------------------------------- #
def _bar_by_baseline(runs, valfn, ylabel, fmt="{:.3g}"):
    fig, ax = _new_ax(aspect=0.72)
    keys = [r.key for r in runs]
    if not keys:
        return None
    xs = np.arange(len(keys))
    vals = [(valfn(rr) or 0.0) for rr in runs]
    ax.bar(xs, vals, 0.62, color=[B.style_for(k).color for k in keys], zorder=3)
    ax.axhline(0, color="#888888", lw=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels([B.style_for(k).label for k in keys])
    for t, k in zip(ax.get_xticklabels(), keys):        # bold/italic per registry
        w, s = B.style_for(k).legend_font()
        t.set_fontweight(w); t.set_fontstyle(s)
    for xi, v in zip(xs, vals):
        ax.text(xi, v, fmt.format(v), ha="center",
                va="bottom" if v >= 0 else "top", fontsize=7)
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig


def fig_e3_dloss_per_gpu_hour(runs, **_):
    return _bar_by_baseline(
        runs, lambda rr: (rr.delta_loss() / (rr.gpu_s_total() / 3600.0))
        if (rr.delta_loss() and rr.gpu_s_total()) else 0.0, "Δloss per GPU-hour")


def fig_e3_dloss_per_mfwd(runs, **_):
    return _bar_by_baseline(
        runs, lambda rr: (rr.delta_loss() / (rr.fwd_total / 1e6))
        if (rr.delta_loss() and rr.have_fwd and rr.fwd_total) else 0.0,
        "Δloss per M forward-pass")


# --------------------------------------------------------------------------- #
# Experiment 4 — data transmitted (grouped bar, up vs down)
# --------------------------------------------------------------------------- #
def fig_e4_network_bytes(runs, **_):
    fig, ax = _new_ax(aspect=0.7)
    runs = [r for r in runs if r.have_comm or r.up_sizes or r.down_sizes]
    if not runs:
        return None
    labels = ["agg→trainer\n(down)", "trainer→agg\n(up)"]
    x = np.arange(len(labels))
    n = len(runs)
    width = 0.8 / n
    for j, rr in enumerate(runs):
        st = B.style_for(rr.key)
        down = sum(rr.down_sizes) / 1e9
        up = sum(rr.up_sizes) / 1e9
        ax.bar(x + (j - (n - 1) / 2) * width, [down, up], width * 0.92,
               color=st.color, label=st.label, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("data transmitted (GB)")
    return _finish(fig, ax, [r.key for r in runs])


# --------------------------------------------------------------------------- #
# Experiment 5 — client session-duration CDF (the E5 money plot)
# --------------------------------------------------------------------------- #
def fig_e5_session_cdf(runs, **_):
    fig, ax = _new_ax()
    any_data = False
    for rr in runs:
        st = B.style_for(rr.key)
        xs, ys = _cdf_xy(rr.session_durs)
        if xs is None:
            continue
        ax.plot(xs, ys, color=st.color, linestyle=st.linestyle,
                marker=st.marker, markevery=0.12, markersize=4.5,
                markeredgecolor="white", markeredgewidth=0.5,
                label=st.label, zorder=3 + st.order)
        any_data = True
    if not any_data:
        return None
    ax.set_xlabel("client session duration (s)")
    ax.set_ylabel("CDF of sessions (%)")
    ax.set_ylim(0, 100)
    return _finish(fig, ax, [r.key for r in runs])


# stable name -> builder (name is also the PDF basename)
FIG_BUILDERS = {
    "e1_acc_vs_time": fig_e1_acc_vs_time,
    "e1_loss_vs_time": fig_e1_loss_vs_time,
    "e2_trainer_busy_cdf": fig_e2_trainer_busy_cdf,
    "e3_dloss_per_gpu_hour": fig_e3_dloss_per_gpu_hour,
    "e3_dloss_per_mfwd": fig_e3_dloss_per_mfwd,
    "e4_network_bytes": fig_e4_network_bytes,
    "e5_session_cdf": fig_e5_session_cdf,
}
