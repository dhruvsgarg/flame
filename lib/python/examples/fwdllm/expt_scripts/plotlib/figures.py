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


# --------------------------------------------------------------------------- #
# Experiment 1 — accuracy vs wall-clock time (the time-to-target money plot)
# --------------------------------------------------------------------------- #
def fig_e1_acc_vs_time(runs, target=None, smooth=0.0, **_):
    fig, ax = _new_ax(aspect=0.68)
    for rr in runs:
        st = B.style_for(rr.key)
        hrs, acc, _loss, _rnd = rr.learning_curve()
        if not hrs:
            continue
        acc = S.ema(acc, smooth)                 # visual smoothing only (scalars use raw)
        ax.plot(hrs, acc, color=st.color, linestyle=st.linestyle,
                marker=st.marker, markevery=0.1, markersize=4.5,
                markeredgecolor="white", markeredgewidth=0.5,
                label=st.label, zorder=3 + st.order)
        # mark round (epoch) boundaries — data_id cycles per round, so a faint
        # vertical rule at each transition shows where a new pass over the data
        # begins (kept off the line so it never collides with the data markers).
        for i in rr.round_transition_indices():
            ax.axvline(hrs[i], color=st.color, ls=(0, (1, 2)), lw=0.9,
                       alpha=0.55, zorder=2)
    if target is not None:
        ax.axhline(target * 100, ls=":", color="#555555", lw=1.0, zorder=2)
        ax.text(ax.get_xlim()[1], target * 100, f" target {target*100:.0f}%",
                va="center", ha="left", fontsize=7, color="#555555")
    ax.set_xlabel("wall-clock time (h)")
    ax.set_ylabel("test accuracy (%)")
    # note the round-boundary semantics once, unobtrusively
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
# Experiment 3 — learning per unit compute (grouped bar, two denominators)
# --------------------------------------------------------------------------- #
def fig_e3_productivity(runs, **_):
    fig, ax = _new_ax(aspect=0.7)
    groups = ["Δloss / GPU-hour", "Δloss / M fwd-pass"]
    keys = [r.key for r in runs]
    n = len(keys)
    if n == 0:
        return None
    width = 0.8 / n
    x = np.arange(len(groups))
    plotted = False
    for j, rr in enumerate(runs):
        st = B.style_for(rr.key)
        dl = rr.delta_loss()
        gpu_h = rr.gpu_s_total() / 3600.0
        v_gpu = (dl / gpu_h) if (dl is not None and gpu_h) else 0.0
        v_fwd = (dl / (rr.fwd_total / 1e6)) if (dl is not None and rr.have_fwd and rr.fwd_total) else 0.0
        ax.bar(x + (j - (n - 1) / 2) * width, [v_gpu, v_fwd], width * 0.92,
               color=st.color, label=st.label, zorder=3)
        plotted = True
    if not plotted:
        return None
    ax.axhline(0, color="#888888", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel("loss reduction per unit compute")
    return _finish(fig, ax, keys)


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
    "e2_trainer_busy_cdf": fig_e2_trainer_busy_cdf,
    "e3_productivity": fig_e3_productivity,
    "e4_network_bytes": fig_e4_network_bytes,
    "e5_session_cdf": fig_e5_session_cdf,
}
