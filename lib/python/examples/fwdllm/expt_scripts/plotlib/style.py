# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""SOCC-2026 figure style — the single source of every "look" decision: rcParams,
column geometry, PDF export, EMA smoothing and the timestamped-output convention.

`pdf.fonttype=42` embeds TrueType glyphs (camera-ready checkers reject Type-3).
`timestamped_outdir()` writes to `<root>/<ts>/` with stable basenames + a `latest`
symlink, so renders never clobber each other yet copying `latest/` into Overleaf
overwrites the draft's figures by name.
"""

from __future__ import annotations

import os
from datetime import datetime


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
# output convention — timestamped dir + stable basenames + `latest` symlink
# --------------------------------------------------------------------------- #
def timestamped_outdir(root: str, stamp: str | None = None) -> str:
    """Create `<root>/<stamp>/`, refresh a `<root>/latest` symlink, return the dir.

    `stamp` defaults to now (YYYYMMDD_HHMMSS). Basenames written inside are stable,
    so old renders are preserved while `latest/` always holds the newest set to
    copy into the paper.
    """
    stamp = stamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(os.path.abspath(root), stamp)
    os.makedirs(out, exist_ok=True)
    link = os.path.join(os.path.abspath(root), "latest")
    try:
        if os.path.islink(link) or os.path.exists(link):
            os.remove(link)
        os.symlink(stamp, link)  # relative target -> portable if root moves
    except OSError:
        pass  # symlink is a convenience, never fatal
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
