"""Master plot config + helpers for every FluxTune paper figure (_metadata/plot_guideline_fluxtune.md): edit here, never in a
figure script. Importing it sets matplotlib's rcParams.

Look (§1a): sans font, Paul Tol palette with a wine hero, `D` marker for ours, y-only dotted grid, inward ticks, round caps.
Function (shared with Felix): print-size figures, black text / axes / callouts, Type 42 fonts, overlap check on every save.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SYS_NAME = "SwiFT"


def label(name):
    """Display name: internal key "fluxtune" -> SYS_NAME."""
    return name.replace("fluxtune", SYS_NAME)


# ---- sizes (inches, points): drawn at print size, never scaled in LaTeX ----
COL_W, FULL_W = 3.25, 6.75
FONT = {"base": 8, "label": 8, "tick": 7, "legend": 7, "annot": 7, "status": 6}
LINE = {"width": 1.1, "hero": 1.8, "muted": 0.9, "ref": 0.8, "grid": 0.4, "axis": 0.6}
MARKER = {"size": 3.5}
DPI, FORMATS = 300, ("pdf", "png")

# ---- colors (Paul Tol muted / high-contrast; no hue shared with Felix's Okabe-Ito) ----
INK, GRID, SURFACE, FILL = "#000000", "#d9d9d9", "#ffffff", "#ebe3d0"
STACK = {  # key: (color, linestyle, marker)
    SYS_NAME: ("#882255", "-", "D"),
    "FwdLLM": ("#44AA99", "-", "o"),
    "FwdLLM async": ("#EE7733", "-.", "o"),
    "FedBuff(P)": ("#332288", "-", "s"),
    "Felix(P)": ("#DDAA33", "-", "^"),
}
ARM = [("#117733", "v"), ("#4477AA", "s"), ("#999933", "^"), ("#88CCEE", "P")]  # FluxTune variants (ablation / tuning arms)
STATUS = {"PRELIM": "#b8860b", "MOCK": "#cc3311"}

Z = {"fill": 1, "bar": 2, "line": 3, "hero": 4, "annot": 6, "status": 10}

RC = {
    "font.family": "sans-serif", "font.sans-serif": ["TeX Gyre Heros", "Liberation Sans", "DejaVu Sans"],
    "mathtext.fontset": "custom", "mathtext.rm": "DejaVu Sans", "mathtext.it": "DejaVu Sans:italic",
    "mathtext.bf": "DejaVu Sans:bold",
    "font.size": FONT["base"], "axes.labelsize": FONT["label"], "xtick.labelsize": FONT["tick"], "ytick.labelsize": FONT["tick"],
    "legend.fontsize": FONT["legend"],
    "axes.edgecolor": INK, "axes.labelcolor": INK, "axes.linewidth": LINE["axis"], "text.color": INK,
    "xtick.color": INK, "ytick.color": INK, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.minor.size": 1.5, "ytick.minor.size": 1.5, "xtick.major.pad": 2.5, "ytick.major.pad": 2.5,
    "axes.grid": True, "axes.grid.axis": "y", "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": LINE["grid"],
    "grid.linestyle": ":", "axes.spines.top": False, "axes.spines.right": False,
    "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
    "lines.linewidth": LINE["width"], "lines.markersize": MARKER["size"],
    "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round", "lines.dash_capstyle": "round",
    "patch.edgecolor": INK, "patch.linewidth": 0.5, "hatch.linewidth": 0.5,
    "legend.frameon": False, "legend.handlelength": 2.2, "legend.columnspacing": 1.0, "legend.handletextpad": 0.4,
    "legend.borderaxespad": 0.2, "legend.labelspacing": 0.25,
    "figure.constrained_layout.use": True, "figure.constrained_layout.h_pad": 0.01, "figure.constrained_layout.w_pad": 0.01,
    "figure.constrained_layout.hspace": 0.02, "figure.constrained_layout.wspace": 0.02,
    "savefig.dpi": DPI, "savefig.bbox": None, "savefig.pad_inches": 0.01, "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.labelpad": 2, "axes.xmargin": 0.01,
}
plt.rcParams.update(RC)


def status(ax, kind, note=""):
    """PRELIM / MOCK tag above a panel's top-left corner; final panels carry none."""
    ax.text(0.0, 1.02, f"{kind}: {note}" if note else kind, transform=ax.transAxes, fontsize=FONT["status"], color=STATUS[kind],
            ha="left", va="bottom", weight="bold", zorder=Z["status"], gid="status")


def _hits(ax, bb, r, skip=None):
    """What a display bbox touches in ax: lines / markers, arrows, bars, other texts, or the axes edge (Felix figlib port)."""
    from matplotlib.patches import FancyArrowPatch
    from matplotlib.path import Path
    from matplotlib.transforms import Bbox
    out, ab, raw = [], ax.get_window_extent(r), bb.expanded(1 / 1.03, 1 / 1.08)
    if not (ab.contains(raw.x0, raw.y0) and ab.contains(raw.x1, raw.y1)):
        out.append("axes edge")
    px = ax.figure.dpi / 72
    for ln in ax.lines:
        xy = ln.get_transform().transform(np.asarray(ln.get_xydata(), float))
        xy = xy[np.isfinite(xy).all(axis=1)]
        if not len(xy):
            continue
        pad = ln.get_linewidth() / 2 * px
        hit = ln.get_linestyle() not in ("None", "", " ") and \
            Path(xy).intersects_bbox(Bbox.from_extents(bb.x0 - pad, bb.y0 - pad, bb.x1 + pad, bb.y1 + pad), filled=False)  # marker-only: no path
        if ln.get_marker() not in (None, "None", ""):
            me, mk = ln.get_markevery(), xy
            if isinstance(me, float):
                d = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))])
                diag = np.hypot(*ab.size)
                mk = xy[np.unique(np.searchsorted(d, np.arange(0, d[-1], me * diag)).clip(0, len(xy) - 1))]
            elif me is not None:
                mk = xy[me]
            m = (ln.get_markersize() / 2 + ln.get_markeredgewidth()) * px
            hit |= bool(((mk[:, 0] > bb.x0 - m) & (mk[:, 0] < bb.x1 + m) & (mk[:, 1] > bb.y0 - m) & (mk[:, 1] < bb.y1 + m)).any())
        if hit:
            out.append(ln.get_label())
    for pa in ax.patches:
        if not pa.get_visible() or pa.get_gid() == "region":
            continue
        if isinstance(pa, FancyArrowPatch):
            v = pa.get_path().transformed(pa.get_transform()).interpolated(20).vertices
            if ((v[:, 0] > bb.x0) & (v[:, 0] < bb.x1) & (v[:, 1] > bb.y0) & (v[:, 1] < bb.y1)).any():
                out.append("arrow")
            continue
        pb = pa.get_window_extent(r)
        inside = pb.x0 <= raw.x0 and pb.y0 <= raw.y0 and pb.x1 >= raw.x1 and pb.y1 >= raw.y1
        if pb.overlaps(bb) and not inside:
            out.append("bar")
    for t in ax.texts:
        if t is skip or not t.get_visible() or t.get_gid() == "status":
            continue
        if getattr(t, "arrow_patch", None) is not None:
            v = t.arrow_patch.get_path().transformed(t.arrow_patch.get_transform()).interpolated(20).vertices
            if ((v[:, 0] > bb.x0) & (v[:, 0] < bb.x1) & (v[:, 1] > bb.y0) & (v[:, 1] < bb.y1)).any():
                out.append("arrow")
        if t.get_text().strip() and t.get_window_extent(r).overlaps(bb):
            out.append(t.get_text())
    return out


def overlaps(fig):
    """Every in-axes text touching data, an arrow, another text or the axes edge: [(text, what)]. Empty = clean."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    return [(t.get_text(), w) for ax in fig.axes for t in ax.texts if t.get_visible() and t.get_text().strip() and t.get_gid() != "status"
            for w in _hits(ax, t.get_window_extent(r).expanded(1.03, 1.08), r, skip=t)]


def freeze(fig):
    """Run the layout once, then stop it, so placed text keeps its spot."""
    fig.canvas.draw()
    fig.set_layout_engine("none")


def place(ax, s, spots, **kw):
    """Text s at the first free (x, y, ha, va) spot in axes coords; else the free 40 x 40 grid point nearest the first spot."""
    r = ax.figure.canvas.get_renderer()
    x0, y0, ha0, va0 = spots[0]
    t = ax.text(x0, y0, s, ha=ha0, va=va0, transform=ax.transAxes, fontsize=FONT["annot"], zorder=Z["annot"], **kw)
    free = lambda: not _hits(ax, t.get_window_extent(r).expanded(1.03, 1.08), r, skip=t)  # noqa: E731
    for x, y, ha, va in spots:
        t.set(x=x, y=y, ha=ha, va=va)
        if free():
            return t
    t.set(ha=ha0, va=va0)
    for p in sorted(((i / 40, j / 40) for i in range(41) for j in range(41)), key=lambda p: (p[0] - x0) ** 2 + (p[1] - y0) ** 2):
        t.set_position(p)
        if free():
            return t
    t.set_position((x0, y0))
    print(f"  NO FREE SPOT for {s!r}", flush=True)
    return t


def save(fig, out_dir, name):
    for t, what in overlaps(fig):
        print(f"  OVERLAP {name}: {t!r} touches {what!r}", flush=True)
    os.makedirs(out_dir, exist_ok=True)
    for ext in FORMATS:
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"))
    plt.close(fig)
    return os.path.join(out_dir, f"{name}.pdf")


def smooth(ys, k):
    """Centered moving average, window shrinking symmetrically at the ends so endpoints stay raw (display only)."""
    ys, n = np.asarray(ys, float), len(ys)
    return np.array([ys[i - h:i + h + 1].mean() for i in range(n) for h in [min(k // 2, i, n - 1 - i)]])
