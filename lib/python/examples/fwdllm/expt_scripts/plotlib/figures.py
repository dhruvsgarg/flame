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


def _annotate_cdf_percentiles(ax, xs, ys, color, which=(50, 90)):
    """Mark P50/P90 on one CDF curve: a short color-matched tick at the curve
    + a small label, so each percentile is readable per-line without eyeballing
    it off the axes. Silently skipped for a percentile the curve never reaches
    (i.e. the run has too few samples to reach that far up the CDF)."""
    for i, p in enumerate(which):
        if ys[-1] < p:
            continue
        xp = np.interp(p, ys, xs)
        ax.plot([xp], [p], marker="|", markersize=9, markeredgewidth=1.6,
                color=color, zorder=6)
        ax.annotate(f"P{p}", xy=(xp, p), xytext=(3, -7 if i == 0 else 4),
                    textcoords="offset points", fontsize=6, color=color,
                    fontweight="bold", zorder=6)


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
            markevery=0.1, markersize=4.5, label=label, zorder=3 + st.order,
            **st.marker_fill_kwargs())


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
# shared: native time-axis label + the "N× faster" speedup callout (E1 pair)
# --------------------------------------------------------------------------- #
def _time_axis_label(runs):
    """'simulated time (h)' if every plotted run is sim-mode, 'wall-clock time (h)'
    if every run is real-mode (the common case — a run-set is one variant per
    EXPERIMENTS.md §0), else a variant-agnostic label for a mixed set."""
    sims = [r.is_sim for r in runs if r.evals]
    if sims and all(sims):
        return "simulated time (h)"
    if sims and not any(sims):
        return "wall-clock time (h)"
    return "time (h) — native clock per run (sim: vclock, real: wall)"


def _target_crossing_hr(rr, target):
    """First native-clock hour (see learning_curve) at which `rr` first reaches
    `target` accuracy — the same 'crosses the X% target in Yh' event the paper
    prose already reports, looser than the W-consecutive-bins CONVERGED verdict
    so it stays defined even without a converge.json. None if never reached."""
    hrs, acc, _loss, _rnd = rr.learning_curve()
    for h, a in zip(hrs, acc):
        if a is not None and a >= target * 100:
            return h
    return None


def _annotate_speedup(ax, runs, target, y, baseline_key="fwdllm",
                       champion_key="fluxtune", fmt_unit="h"):
    """Double-headed arrow + boxed, high-contrast label between `baseline_key`'s
    and `champion_key`'s time-to-target crossings — the 'how much faster' callout
    the reader should not be able to miss. Silently omitted if either baseline is
    absent from `runs` or either never reaches `target` (nothing to compare)."""
    by_key = {r.key: r for r in runs}
    base, champ = by_key.get(baseline_key), by_key.get(champion_key)
    if base is None or champ is None:
        return
    t_base = _target_crossing_hr(base, target)
    t_champ = _target_crossing_hr(champ, target)
    if t_base is None or t_champ is None or t_champ <= 0:
        return
    speedup = t_base / t_champ
    champ_color = B.style_for(champion_key).color
    ax.annotate("", xy=(t_champ, y), xytext=(t_base, y),
                arrowprops=dict(arrowstyle="<->", color=champ_color, lw=2.2,
                                shrinkA=0, shrinkB=0), zorder=20)
    xm = (t_base + t_champ) / 2
    ax.annotate(f"{speedup:.1f}× faster",
                xy=(xm, y), xytext=(0, 9), textcoords="offset points",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold",
                color="white", zorder=21,
                bbox=dict(boxstyle="round,pad=0.28", facecolor=champ_color,
                          edgecolor="none", alpha=0.95))


# --------------------------------------------------------------------------- #
# auto broken y-axis (PLOT_TRACKER open-work #3) — data-driven, not a hardcoded
# break point. Wired into e1_acc_vs_time only (the doc's named motivating
# case); x-axis breaking is a deferred follow-up, see PLOT_TRACKER.
# --------------------------------------------------------------------------- #
def _detect_axis_break(values, min_gap_frac=0.30, min_side_point_frac=0.10):
    """Largest gap between consecutive sorted `values` that's >= `min_gap_frac`
    of the total span, keeping >= `min_side_point_frac` of the *points* (not
    range) on each side -- a range-based guard would reject the motivating
    case itself (two tight clusters far apart each span almost no range).
    None if no gap qualifies. The only automatic decision point: looks at
    what's actually plotted, never a hardcoded per-baseline value."""
    vs = sorted(float(v) for v in values if v is not None)
    n = len(vs)
    if n < 4:
        return None
    total = vs[-1] - vs[0]
    if total <= 0:
        return None
    min_side_n = max(1, int(min_side_point_frac * n))
    best = None
    for i in range(min_side_n - 1, n - min_side_n - 1):
        a, b = vs[i], vs[i + 1]
        gap = b - a
        if gap < min_gap_frac * total:
            continue
        if best is None or gap > best[1] - best[0]:
            best = (a, b)
    return best


def _broken_y_axes(aspect):
    """Two vertically-stacked Axes sharing x: `ax_top` (most of the height) for
    the high-value band, `ax_bot` (a compressed sliver) for the low-value band
    -- the standard broken-axis idiom (matplotlib gallery "Broken Axis"), with
    diagonal break marks on the shared border."""
    import matplotlib.pyplot as plt
    w, h = S.column_figsize(1.0, aspect)
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, sharex=True, figsize=(w, h),
        gridspec_kw={"height_ratios": [4, 1], "hspace": 0.08})
    ax_top.spines["bottom"].set_visible(False)
    ax_bot.spines["top"].set_visible(False)
    ax_top.tick_params(labelbottom=False, bottom=False)
    d = 0.6
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=8, linestyle="none",
                  color="k", mec="k", mew=1, clip_on=False)
    ax_top.plot([0, 1], [0, 0], transform=ax_top.transAxes, **kwargs)
    ax_bot.plot([0, 1], [1, 1], transform=ax_bot.transAxes, **kwargs)
    return fig, ax_top, ax_bot


def _draw_e1_curves(ax, curves, smooth):
    """Draw every run's curve (+ peak star + round-boundary lines) on `ax`.
    Relies on the default `clip_on=True` so calling this once per panel of a
    broken-axis figure naturally splits the visible output by each panel's own
    y-limits -- no manual per-panel data slicing needed."""
    for rr, hrs, acc, st in curves:
        peak = _mark_peak(ax, hrs, acc, st)     # ★ + peak value in the legend
        label = st.label if peak is None else f"{st.label} — peak {peak:.1f}%"
        _plot_curve(ax, hrs, acc, st, smooth, label)
        _round_boundaries(ax, rr, hrs)          # data_id cycles per round → mark epochs


def _e1_target_and_speedup(ax, runs, target, baseline_key, champion_key):
    if target is None:
        return
    ax.axhline(target * 100, ls=":", color="#555555", lw=1.0, zorder=2)
    ax.text(ax.get_xlim()[1], target * 100, f" target {target*100:.0f}%",
            va="center", ha="left", fontsize=7, color="#555555")
    y_annot = target * 100 + 2.0
    top = max(ax.get_ylim()[1], y_annot + 5.0)
    ax.set_ylim(top=top)
    _annotate_speedup(ax, runs, target, y_annot,
                      baseline_key=baseline_key, champion_key=champion_key)


def _e1_legend_proxies(ax):
    ax.plot([], [], color="#888888", ls=(0, (1, 2)), lw=0.9, label="round boundary")
    ax.scatter([], [], marker="*", s=95, color="#555555", edgecolor="white",
               linewidth=0.6, label="peak accuracy")


# --------------------------------------------------------------------------- #
# Experiment 1 — accuracy / loss vs wall-clock time (adjacent paper figures)
# --------------------------------------------------------------------------- #
def fig_e1_acc_vs_time(runs, target=None, smooth=0.0,
                       baseline_key="fwdllm", champion_key="fluxtune", **_):
    curves, all_acc = [], []
    for rr in runs:
        hrs, acc, _loss, _rnd = rr.learning_curve()
        if not hrs:
            continue
        curves.append((rr, hrs, acc, B.style_for(rr.key)))
        all_acc.extend(a for a in acc if a is not None)

    y_break = _detect_axis_break(all_acc)

    if y_break is None:
        fig, ax = _new_ax(aspect=0.68)
        _draw_e1_curves(ax, curves, smooth)
        _e1_target_and_speedup(ax, runs, target, baseline_key, champion_key)
        ax.set_xlabel(_time_axis_label(runs))
        ax.set_ylabel("test accuracy (%)")
        _e1_legend_proxies(ax)
        return _finish(fig, ax, [r.key for r in runs])

    # broken y-axis: give more canvas to the dense high-accuracy band, compress
    # the gap the data itself never occupies (PLOT_TRACKER open-work #3)
    gap_lo, gap_hi = y_break
    fig, ax_top, ax_bot = _broken_y_axes(aspect=0.78)
    _draw_e1_curves(ax_top, curves, smooth)
    _draw_e1_curves(ax_bot, curves, smooth)
    y_min, y_max = min(all_acc), max(all_acc)
    pad = 0.05 * max(y_max - gap_hi, gap_lo - y_min, 1.0)
    ax_top.set_ylim(gap_hi - pad, y_max + pad * 2)
    ax_bot.set_ylim(y_min - pad, gap_lo + pad)
    # target/speedup callout goes on whichever panel actually contains it
    target_ax = ax_bot if (target is not None and target * 100 <= gap_lo) else ax_top
    _e1_target_and_speedup(target_ax, runs, target, baseline_key, champion_key)
    ax_bot.set_xlabel(_time_axis_label(runs))
    fig.supylabel("test accuracy (%)", fontsize=9)
    _e1_legend_proxies(target_ax)
    handles, _labels = target_ax.get_legend_handles_labels()
    if handles:
        leg = target_ax.legend(loc="best")
        B.apply_legend_emphasis(leg)
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)
    ax_bot.spines["right"].set_visible(False)
    return fig


def fig_e1_loss_vs_time(runs, target=None, smooth=0.0,
                        baseline_key="fwdllm", champion_key="fluxtune", **_):
    """Test-loss vs native time — the grounded companion to the accuracy curve.
    Carries the SAME time-to-target-accuracy speedup callout as e1_acc_vs_time
    (anchored near the top of the loss axis) so a reader skimming either figure
    sees the same headline number."""
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
    if target is not None:
        lo, hi = ax.get_ylim()
        y_annot = hi - 0.08 * (hi - lo)
        _annotate_speedup(ax, runs, target, y_annot,
                          baseline_key=baseline_key, champion_key=champion_key)
    ax.set_xlabel(_time_axis_label(runs))
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
                label=st.label, zorder=3 + st.order, **st.marker_fill_kwargs())
        _annotate_cdf_percentiles(ax, xs, ys, st.color)
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
def _annotate_bar_delta(ax, xs, vals, keys, base_key="fwdllm", champ_key="fluxtune",
                        better="higher", unit=""):
    """Bracket + boxed delta label between two named bars: how much MORE/LESS
    `champ_key` has vs `base_key`, colored green when that direction is the
    favorable one (`better`: 'higher' or 'lower' is-better) and vermillion when
    not — so the reader reads the comparison directly instead of subtracting the
    two printed bar values themselves. Silently omitted if either bar is absent."""
    if base_key not in keys or champ_key not in keys:
        return
    i_base, i_champ = keys.index(base_key), keys.index(champ_key)
    v_base, v_champ = vals[i_base], vals[i_champ]
    delta = v_champ - v_base
    favorable = (delta >= 0) if better == "higher" else (delta <= 0)
    color = "#1B7837" if favorable else "#D55E00"          # green good / vermillion bad
    span = (max(vals) - min(vals)) or (abs(v_base) + abs(v_champ)) or 1.0
    y = max(v_base, v_champ, 0) + 0.14 * span
    x_base, x_champ = xs[i_base], xs[i_champ]
    ax.set_ylim(top=max(ax.get_ylim()[1], y + 0.12 * span))
    ax.annotate("", xy=(x_champ, y), xytext=(x_base, y),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=2.0,
                                shrinkA=0, shrinkB=0), zorder=20)
    word = "more" if delta >= 0 else "less"
    xm = (x_base + x_champ) / 2
    ax.annotate(f"{abs(delta):.3g}{unit} {word}", xy=(xm, y), xytext=(0, 6),
                textcoords="offset points", ha="center", va="bottom",
                fontsize=8, fontweight="bold", color="white", zorder=21,
                bbox=dict(boxstyle="round,pad=0.25", facecolor=color,
                          edgecolor="none", alpha=0.95))


def _bar_by_baseline(runs, valfn, ylabel, fmt="{:.3g}", delta=None):
    """`delta`: optional (base_key, champ_key, better) to also draw the
    _annotate_bar_delta callout between those two bars."""
    fig, ax = _new_ax(aspect=0.72)
    keys = [r.key for r in runs]
    if not keys:
        return None
    xs = np.arange(len(keys))
    vals = [(valfn(rr) or 0.0) for rr in runs]
    styles = [B.style_for(k) for k in keys]
    ax.bar(xs, vals, 0.62, color=[s.color for s in styles],
           hatch=[s.bar_hatch() for s in styles],
           edgecolor="white", linewidth=0.6, zorder=3)
    ax.axhline(0, color="#888888", lw=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels([B.style_for(k).label for k in keys])
    for t, k in zip(ax.get_xticklabels(), keys):        # bold/italic per registry
        w, s = B.style_for(k).legend_font()
        t.set_fontweight(w); t.set_fontstyle(s)
    for xi, v in zip(xs, vals):
        ax.text(xi, v, fmt.format(v), ha="center",
                va="bottom" if v >= 0 else "top", fontsize=7)
    if delta is not None:
        base_key, champ_key, better = delta
        _annotate_bar_delta(ax, list(xs), vals, keys, base_key, champ_key, better)
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig


def fig_e3_dloss_per_gpu_hour(runs, baseline_key="fwdllm", champion_key="fluxtune", **_):
    # more Δloss/GPU-hour is better -- champion "generates more" is the favorable read.
    return _bar_by_baseline(
        runs, lambda rr: (rr.delta_loss() / (rr.gpu_s_total() / 3600.0))
        if (rr.delta_loss() and rr.gpu_s_total()) else 0.0, "Δloss per GPU-hour",
        delta=(baseline_key, champion_key, "higher"))


def fig_e3_dloss_per_mfwd(runs, baseline_key="fwdllm", champion_key="fluxtune", **_):
    # more Δloss/M-fwd is better too -- here the champion typically trails (the
    # guidance-surcharge caveat, EXPTS_CHARTER.md), so the callout usually reads
    # "less" in vermillion, not "more" in green -- an honest deficit, not hidden.
    return _bar_by_baseline(
        runs, lambda rr: (rr.delta_loss() / (rr.fwd_total / 1e6))
        if (rr.delta_loss() and rr.have_fwd and rr.fwd_total) else 0.0,
        "Δloss per M forward-pass", delta=(baseline_key, champion_key, "higher"))


# --------------------------------------------------------------------------- #
# Experiment 4 — data transmitted (grouped bar, up vs down)
# --------------------------------------------------------------------------- #
def fig_e4_network_bytes(runs, savings_from="fwdllm", savings_to="felix_round", **_):
    """Down / up / total grouped bars, each bar labeled (in its baseline's own
    color, nearest whole GB) with the value it represents. The `total` group's
    `savings_from`->`savings_to` pair gets an extra "Nx less bandwidth" callout
    (default FwdLLM->Felix(P): how much the smart-selection anchor alone saves)."""
    fig, ax = _new_ax(aspect=0.7)
    runs = [r for r in runs if r.have_comm or r.up_sizes or r.down_sizes]
    if not runs:
        return None
    labels = ["agg→trainer\n(down)", "trainer→agg\n(up)", "total"]
    x = np.arange(len(labels))
    n = len(runs)
    width = 0.8 / n
    total_x_by_key, total_val_by_key = {}, {}
    max_h = 0.0
    for j, rr in enumerate(runs):
        st = B.style_for(rr.key)
        down = sum(rr.down_sizes) / 1e9
        up = sum(rr.up_sizes) / 1e9
        total = down + up
        xj = x + (j - (n - 1) / 2) * width
        ax.bar(xj, [down, up, total], width * 0.92, color=st.color,
               hatch=st.bar_hatch(), edgecolor="white", linewidth=0.6,
               label=st.label, zorder=3)
        for xi, v in zip(xj, (down, up, total)):
            ax.text(xi, v, f"{v:.0f}", ha="center", va="bottom", fontsize=6.5,
                    color=st.color, fontweight="bold", zorder=4)
        max_h = max(max_h, total)
        total_x_by_key[rr.key] = xj[2]
        total_val_by_key[rr.key] = total
    ax.set_ylim(top=max_h * 1.22)
    if savings_from in total_x_by_key and savings_to in total_x_by_key:
        v_from, v_to = total_val_by_key[savings_from], total_val_by_key[savings_to]
        if v_from > 0 and v_to > 0:
            # direction-aware: `to` might use MORE than `from`, not less -- say so
            # rather than forcing a "savings" framing the data doesn't support.
            savings = v_to <= v_from
            factor = (v_from / v_to) if savings else (v_to / v_from)
            word = "less" if savings else "more"
            color = B.style_for(savings_to).color if savings else "#D55E00"
            xa, xb = total_x_by_key[savings_from], total_x_by_key[savings_to]
            y = max_h * 1.12
            ax.annotate("", xy=(xb, y), xytext=(xa, y),
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=2.2,
                                        shrinkA=0, shrinkB=0), zorder=20)
            ax.annotate(f"{factor:.2f}× {word} bandwidth",
                        xy=((xa + xb) / 2, y), xytext=(0, 5),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=8.5, fontweight="bold", color="white", zorder=21,
                        bbox=dict(boxstyle="round,pad=0.28", facecolor=color,
                                  edgecolor="none", alpha=0.95))
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
                label=st.label, zorder=3 + st.order, **st.marker_fill_kwargs())
        _annotate_cdf_percentiles(ax, xs, ys, st.color)
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
