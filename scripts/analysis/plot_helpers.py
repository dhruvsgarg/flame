# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Reusable matplotlib helpers for per-run telemetry analysis.

Outputs vector PDF with paper-style fonts. Every figure carries a two-line title:
line 1 = what it shows; line 2 (smaller, gray) = a compact experiment-config stamp
so a plot is self-describing. Selectors get fixed colors for visual consistency.
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Quiet the very verbose PDF font-subsetting / matplotlib INFO logs.
for _n in ("matplotlib", "matplotlib.font_manager", "fontTools",
           "fontTools.subset", "fontTools.ttLib"):
    logging.getLogger(_n).setLevel(logging.WARNING)

# Paper-style rcParams (fonttype 42 => editable text in PDF/PS).
plt.rcParams.update({
    "axes.labelsize": 20,
    "axes.titlesize": 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "figure.figsize": [6, 3],
    "legend.fontsize": 18,
    "legend.columnspacing": 2,
    "legend.handletextpad": 0.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Fixed per-selector colors (control arms reuse the hue, dashed) so the same
# selector always reads the same across plots/runs.
SELECTOR_COLORS = {
    "felix": "#1f77b4", "async_oort": "#1f77b4",
    "oort": "#ff7f0e",
    "refl": "#2ca02c", "refl_oort": "#2ca02c",
    "feddance": "#d62728",
    "fedbuff": "#9467bd", "fedavg": "#8c564b", "random": "#7f7f7f",
}


def color_for(label: str) -> Optional[str]:
    key = str(label).lower()
    for name, c in SELECTOR_COLORS.items():
        if name in key:
            return c
    return None


# --- config stamp -----------------------------------------------------------


def config_stamp(run_dir: str) -> str:
    """Compact one-line experiment-config string from snapshot/exec config."""
    import glob
    import json
    try:
        import yaml
    except Exception:
        yaml = None
    info = {}
    # prefer execution_config.yaml (has selector/agg_goal), fall back to snapshot
    for fn in ("execution_config.yaml", "snapshot.yaml"):
        p = os.path.join(run_dir, fn)
        if os.path.exists(p) and yaml is not None:
            try:
                info = yaml.safe_load(open(p)) or {}
                break
            except Exception:
                continue
    exp = (info.get("experiment") or {})
    tr = exp.get("trainer", {})
    ds = tr.get("dataset", {})
    agg = info.get("aggregator", {}) or exp.get("aggregator", {})
    ds_cfg = ((tr.get("config_overrides") or {}).get("hyperparameters") or {}).get(
        "data_streaming", {}
    )
    name = exp.get("name", os.path.basename(run_dir))
    sel = agg.get("selector", "?")
    alpha = ds.get("alpha", ds.get("dirichlet_alpha", "?"))
    n = tr.get("num_trainers", "?")
    avail = (tr.get("availability", {}) or {}).get("mode", "?")
    tm = tr.get("time_mode", "?")
    ag = agg.get("agg_goal", "?")
    stream = "off"
    if str(ds_cfg.get("enabled", "False")) == "True":
        stream = f"T={ds_cfg.get('full_data_available_after_s','?')}s"
    return (f"{name} | {sel} | n={n} α={alpha} | {avail} | "
            f"stream:{stream} | aggGoal={ag} | {tm}")


def _save(fig, out_dir: str, file_name: str, stamp: Optional[str]) -> str:
    if not file_name.endswith(".pdf"):
        file_name = os.path.splitext(file_name)[0] + ".pdf"
    os.makedirs(out_dir, exist_ok=True)
    if stamp:
        fig.text(0.5, -0.02, stamp, ha="center", va="top", fontsize=8,
                 color="0.4", wrap=True)
    fig.tight_layout()
    path = os.path.join(out_dir, file_name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _legend(ax, n_series):
    if n_series > 1:
        ax.legend(fontsize=12, frameon=True)


# --- core figures (names preserved for existing callers) --------------------


def line_plot(series, x_label, y_label, title, out_dir, file_name,
              stamp=None, logy=False, target=None):
    if not series:
        return None
    fig, ax = plt.subplots()
    plotted = False
    for label, (xs, ys) in series.items():
        if xs is None or ys is None or len(xs) == 0:
            continue
        ax.plot(xs, ys, marker=".", markersize=3, linewidth=1.5, label=label,
                color=color_for(label))
        plotted = True
    if not plotted:
        plt.close(fig)
        return None
    if target is not None:
        ax.axhline(target, ls="--", color="0.5", lw=1)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    _legend(ax, len(series))
    return _save(fig, out_dir, file_name, stamp)


def banded_line(x, mean, lo, hi, x_label, y_label, title, out_dir, file_name,
                stamp=None, marker_x=None, color="#1f77b4"):
    """Mean line with min/max (or P-band) shaded region."""
    if not len(x):
        return None
    fig, ax = plt.subplots()
    ax.plot(x, mean, color=color, lw=2, label="mean")
    ax.fill_between(x, lo, hi, color=color, alpha=0.2, label="min-max")
    if marker_x is not None:
        ax.axvline(marker_x, ls="--", color="0.5", lw=1, label="config horizon")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    return _save(fig, out_dir, file_name, stamp)


def scatter_diag(x, y, x_label, y_label, title, out_dir, file_name, stamp=None,
                 groups=None, group_labels=None):
    """Scatter with y=x reference diagonal (expected-vs-actual style)."""
    if not len(x):
        return None
    x = np.asarray(x, float); y = np.asarray(y, float)
    fig, ax = plt.subplots(figsize=(5, 5))
    if groups is not None:
        g = np.asarray(groups)
        for gv, gl, c in group_labels:
            m = g == gv
            if m.any():
                ax.scatter(x[m], y[m], s=14, alpha=0.5, color=c, label=gl)
        ax.legend(fontsize=12)
    else:
        ax.scatter(x, y, s=14, alpha=0.5, color="#1f77b4")
    lim = [min(x.min(), y.min()), max(x.max(), y.max())]
    ax.plot(lim, lim, ls="--", color="0.4", lw=1)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return _save(fig, out_dir, file_name, stamp)


def hist_plot(values, x_label, title, out_dir, file_name, stamp=None, vline=0.0):
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    fig, ax = plt.subplots()
    arr = np.asarray(vals, float)
    ax.hist(arr, bins=min(60, max(10, len(arr) // 5)), color="#1f77b4", alpha=0.8)
    if vline is not None:
        ax.axvline(vline, color="red", ls="--", lw=1.5)
    for q, lab in [(0.5, "P50"), (0.9, "P90")]:
        xv = float(np.quantile(arr, q))
        ax.axvline(xv, color="0.3", ls=":", lw=1)
        ax.annotate(f"{lab}={xv:.3g}", (xv, 0), textcoords="offset points",
                    xytext=(3, 12), fontsize=10, rotation=90, color="0.3")
    ax.set_xlabel(x_label)
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    return _save(fig, out_dir, file_name, stamp)


def signed_bar(categories, values, x_label, y_label, title, out_dir, file_name,
               stamp=None):
    if not len(values):
        return None
    fig, ax = plt.subplots()
    v = np.asarray(values, float)
    colors = ["#2ca02c" if x >= 0 else "#d62728" for x in v]
    ax.bar(range(len(v)), v, color=colors, alpha=0.85)
    ax.axhline(0, color="0.3", lw=1)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    return _save(fig, out_dir, file_name, stamp)


def cdf_plot(values, x_label, title, out_dir, file_name, stamp=None):
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    arr = np.sort(np.asarray(vals, dtype=float))
    y = np.arange(1, len(arr) + 1) / len(arr)
    fig, ax = plt.subplots()
    ax.plot(arr, y, color="tab:red", linewidth=2)
    for q, label in [(0.5, "P50"), (0.9, "P90"), (0.99, "P99")]:
        xv = float(np.quantile(arr, q))
        ax.axvline(xv, color="0.4", ls=":", lw=1)
        ax.annotate(f"{label}={xv:.3g}", (xv, q), textcoords="offset points",
                    xytext=(5, -12), fontsize=10)
    ax.set_xlabel(x_label)
    ax.set_ylabel("CDF")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return _save(fig, out_dir, file_name, stamp)


def scatter_plot(x, y, groups, x_label, y_label, title, out_dir, file_name, stamp=None):
    if len(x) == 0:
        return None
    fig, ax = plt.subplots()
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    if groups is not None:
        g = np.asarray(groups, dtype=bool)
        ax.scatter(x[~g], y[~g], s=16, alpha=0.5, color="tab:gray", label="eligible")
        ax.scatter(x[g], y[g], s=24, alpha=0.8, color="tab:red", label="selected")
        ax.legend(fontsize=12)
    else:
        ax.scatter(x, y, s=16, alpha=0.6, color="tab:blue")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return _save(fig, out_dir, file_name, stamp)


def stacked_area(x, series, x_label, y_label, title, out_dir, file_name, stamp=None):
    if not series or len(x) == 0:
        return None
    labels = list(series.keys())
    ys = [np.asarray(series[k], dtype=float) for k in labels]
    fig, ax = plt.subplots()
    ax.stackplot(x, *ys, labels=labels, alpha=0.85)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(True, alpha=0.3)
    return _save(fig, out_dir, file_name, stamp)


def stacked_bar(categories, segments, y_label, title, out_dir, file_name, stamp=None,
                horizontal=False):
    if not segments or len(categories) == 0:
        return None
    n = len(categories)
    fig, ax = plt.subplots(figsize=(max(6, n * 0.35), 4) if not horizontal
                           else (7, max(3, n * 0.25)))
    bottom = np.zeros(n, dtype=float)
    for label, vals in segments.items():
        v = np.asarray(vals, dtype=float)
        if horizontal:
            ax.barh(range(n), v, left=bottom, label=label)
        else:
            ax.bar(range(n), v, bottom=bottom, label=label)
        bottom += v
    if horizontal:
        ax.set_yticks(range(n)); ax.set_yticklabels([str(c) for c in categories], fontsize=7)
        ax.set_xlabel(y_label)
    else:
        ax.set_xticks(range(n)); ax.set_xticklabels([str(c) for c in categories],
                                                    rotation=60, ha="right", fontsize=7)
        ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(fontsize=11)
    ax.grid(True, axis=("x" if horizontal else "y"), alpha=0.3)
    return _save(fig, out_dir, file_name, stamp)


def bar_plot(categories, values, y_label, title, out_dir, file_name, stamp=None):
    if len(categories) == 0:
        return None
    fig, ax = plt.subplots(figsize=(max(6, len(categories) * 0.3), 3.2))
    ax.bar(range(len(categories)), values, color="tab:blue", alpha=0.85)
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels([str(c) for c in categories], rotation=60, ha="right", fontsize=7)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    return _save(fig, out_dir, file_name, stamp)


def heatmap(matrix, x_label, y_label, title, out_dir, file_name, stamp=None,
            cmap="viridis", cbar_label=None, yticklabels=None):
    """Trainer x round style heatmap (matrix: rows=y, cols=x)."""
    m = np.asarray(matrix, dtype=float)
    if m.size == 0:
        return None
    fig, ax = plt.subplots(figsize=(8, max(3, m.shape[0] * 0.12)))
    im = ax.imshow(m, aspect="auto", interpolation="nearest", cmap=cmap)
    cb = fig.colorbar(im, ax=ax)
    if cbar_label:
        cb.set_label(cbar_label, fontsize=12)
    if yticklabels is not None and len(yticklabels) <= 40:
        ax.set_yticks(range(len(yticklabels)))
        ax.set_yticklabels(yticklabels, fontsize=6)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    return _save(fig, out_dir, file_name, stamp)


def dual_axis_line(x, y1, y2, x_label, y1_label, y2_label, title, out_dir,
                   file_name, stamp=None):
    if not len(x):
        return None
    fig, ax1 = plt.subplots()
    ax1.plot(x, y1, color="#1f77b4", lw=2, label=y1_label)
    ax1.set_xlabel(x_label); ax1.set_ylabel(y1_label, color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax2 = ax1.twinx()
    ax2.plot(x, y2, color="#d62728", lw=2, ls="--", label=y2_label)
    ax2.set_ylabel(y2_label, color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax1.set_title(title)
    ax1.grid(True, alpha=0.3)
    return _save(fig, out_dir, file_name, stamp)
