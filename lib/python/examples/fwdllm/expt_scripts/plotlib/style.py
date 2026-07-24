# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""SOCC-2026 figure style — the single source of every "look" decision: rcParams,
column geometry, PDF export, EMA smoothing and the output-dir convention.

`pdf.fonttype=42` embeds TrueType glyphs (camera-ready checkers reject Type-3).
`render_outdir()` writes straight into `<root>/` -- stable basenames, so each
render overwrites in place; copy `<root>/*.pdf` into Overleaf directly.
"""

from __future__ import annotations

import os


# single-column width for a 2-column ACM/IEEE-style layout (inches)
COL_WIDTH_IN = 3.35
# default aspect (height / width) for a line/CDF panel
DEFAULT_ASPECT = 0.66


# --------------------------------------------------------------------------- #
# rcParams — applied once via use_paper_style()
# --------------------------------------------------------------------------- #
_RC = {
    # fonts: embed real glyphs; serif body to match a LaTeX column
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "serif",
    # DejaVu Serif is always present; Times/STIX used if the system has them
    "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    # thin, recessive frame + grid (dataviz: recessive grid/axes)
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "lines.linewidth": 1.8,
    "lines.markersize": 5,
    "legend.frameon": False,
    "legend.handlelength": 1.8,
    "legend.columnspacing": 1.2,
    "legend.labelspacing": 0.35,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
}


def use_paper_style():
    """Idempotently apply the SOCC-2026 rcParams to the active matplotlib."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update(_RC)


def column_figsize(fraction: float = 1.0, aspect: float = DEFAULT_ASPECT):
    """(w, h) in inches for `fraction` of a column width at the given aspect."""
    w = COL_WIDTH_IN * fraction
    return (w, w * aspect)


# --------------------------------------------------------------------------- #
# output convention — flat dir, stable basenames, overwritten every render
# --------------------------------------------------------------------------- #
def render_outdir(root: str) -> str:
    """Create (if needed) and return `root` itself. No timestamped subdir, no
    `latest` symlink -- every figure name is already stable and unique, so a
    rerun just overwrites its own PDF in place."""
    out = os.path.abspath(root)
    os.makedirs(out, exist_ok=True)
    return out


# --------------------------------------------------------------------------- #
# line smoothing — VISUAL ONLY (never feeds a scalar metric)
# --------------------------------------------------------------------------- #
def ema(values, factor: float):
    """TensorBoard-style exponential-moving-average smoothing of a noisy curve.

    `factor` ∈ [0, 1): 0 = no smoothing (returns input), higher = smoother. None
    entries pass through untouched. Applied ONLY to plotted lines — the underlying
    accuracy/loss scalars (max-acc, Δloss) always use the raw series.
    """
    if not factor or factor <= 0:
        return list(values)
    out, last = [], None
    for v in values:
        if v is None:
            out.append(None)
            continue
        last = v if last is None else last * factor + v * (1 - factor)
        out.append(last)
    return out


def save_pdf(fig, out_dir: str, name: str):
    """Write `<out_dir>/<name>.pdf` — the only artifact. Figures are vector PDF
    (resolution-independent); `savefig.dpi=300` covers any rasterized element."""
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(out_dir, f"{name}.pdf"))
    import matplotlib.pyplot as plt
    plt.close(fig)
