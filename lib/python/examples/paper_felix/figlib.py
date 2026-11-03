"""Figure helpers on top of plot_config (the one place for style): status labels, save, smoothing."""
import os

from plot_config import (COL_W, FONT, FORMATS, FULL_W, GRID, INK, LINE, MARK, MUTED, PRED, ROW_H, STACK, STATUS, TRACE, Z, label,  # noqa: F401
                         plt)

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figs", "paper")


def status(target, kind, note=""):
    """Label a figure (whole) or one axes (panel) MOCK / PRELIM / DRAFT; real figures and panels get nothing.
    MOCK = placeholder numbers (layout only); PRELIM = real data, not final; DRAFT = schematic."""
    color = STATUS[kind]
    text = f"{kind}: {note}" if note else kind
    if hasattr(target, "transAxes"):  # a panel: faint watermark + tag above its top-left corner
        target.text(0.5, 0.5, kind, transform=target.transAxes, fontsize=14, color=color, alpha=0.18, ha="center", va="center",
                    rotation=20, weight="bold", zorder=Z["status"])
        target.text(0.0, 1.13, text, transform=target.transAxes, fontsize=FONT["status"], color=color, ha="left", va="bottom",
                    weight="bold", zorder=Z["status"])
    else:
        target.text(0.5, 0.5, kind, fontsize=40, color=color, alpha=0.12, ha="center", va="center", rotation=20, weight="bold",
                    zorder=Z["status"])
        target.text(0.995, 0.995, text, fontsize=FONT["status"], color=color, ha="right", va="top", weight="bold")


def _hits(ax, bb, r, skip=None):
    """What a display bbox touches in ax: data lines / markers, annotation arrows, other texts, or the axes edge."""
    import numpy as np
    from matplotlib.path import Path
    from matplotlib.transforms import Bbox
    out, ab, raw = [], ax.get_window_extent(r), bb.expanded(1 / 1.03, 1 / 1.08)  # edge test uses the unpadded box
    if not (ab.contains(raw.x0, raw.y0) and ab.contains(raw.x1, raw.y1)):
        out.append("axes edge")
    for ln in ax.lines:
        xy = ln.get_transform().transform(np.asarray(ln.get_xydata(), float))
        xy = xy[np.isfinite(xy).all(axis=1)]
        if not len(xy):
            continue
        px = ax.figure.dpi / 72
        pad = ln.get_linewidth() / 2 * px
        hit = Path(xy).intersects_bbox(Bbox.from_extents(bb.x0 - pad, bb.y0 - pad, bb.x1 + pad, bb.y1 + pad), filled=False)
        if ln.get_marker() not in (None, "None", ""):
            me, mk = ln.get_markevery(), xy
            if isinstance(me, float):  # matplotlib's rule: one marker per `me` × axes diagonal of path length
                d = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))])
                diag = np.hypot(*ax.get_window_extent(r).size)
                mk = xy[np.unique(np.searchsorted(d, np.arange(0, d[-1], me * diag)).clip(0, len(xy) - 1))]
            elif me is not None:
                mk = xy[me]
            m = (ln.get_markersize() / 2 + ln.get_markeredgewidth()) * px
            hit |= bool(((mk[:, 0] > bb.x0 - m) & (mk[:, 0] < bb.x1 + m) & (mk[:, 1] > bb.y0 - m) & (mk[:, 1] < bb.y1 + m)).any())
        if hit:
            out.append(ln.get_label())
    from matplotlib.patches import FancyArrowPatch
    for pa in ax.patches:  # bars, fills, boxes, arrows
        if not pa.get_visible() or pa.get_gid() == "region":  # background regions hold text by design
            continue
        if isinstance(pa, FancyArrowPatch):
            v = pa.get_path().transformed(pa.get_transform()).interpolated(20).vertices
            if ((v[:, 0] > bb.x0) & (v[:, 0] < bb.x1) & (v[:, 1] > bb.y0) & (v[:, 1] < bb.y1)).any():
                out.append("arrow")
            continue
        pb = pa.get_window_extent(r)
        inside = pb.x0 <= raw.x0 and pb.y0 <= raw.y0 and pb.x1 >= raw.x1 and pb.y1 >= raw.y1  # a label inside its own box
        if pb.overlaps(bb) and not inside:
            out.append("bar")
    for t in ax.texts:
        if t is skip or not t.get_visible():
            continue
        if getattr(t, "arrow_patch", None) is not None:
            v = t.arrow_patch.get_path().transformed(t.arrow_patch.get_transform()).interpolated(20).vertices
            if ((v[:, 0] > bb.x0) & (v[:, 0] < bb.x1) & (v[:, 1] > bb.y0) & (v[:, 1] < bb.y1)).any():
                out.append("arrow")
        if t.get_text().strip() and t.get_window_extent(r).overlaps(bb):
            out.append(t.get_text())
    return out


def overlaps(fig):
    """Every in-axes text that touches data, an arrow, another text or the axes edge: [(text, what)]. Empty = clean."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    return [(t.get_text(), w) for ax in fig.axes for t in ax.texts if t.get_visible() and t.get_text().strip()
            for w in _hits(ax, t.get_window_extent(r).expanded(1.03, 1.08), r, skip=t)]


def place(ax, s, spots, **kw):
    """Put text s at the first (x, y, ha, va) spot (data coords) where it touches nothing; else the free grid point nearest the
    first spot; else the first spot with a warning (overlaps() then reports it)."""
    fig = ax.figure
    fig.canvas.draw()
    fig.set_layout_engine("none")  # freeze the layout, so later draws don't move the axes under placed text
    r = fig.canvas.get_renderer()
    x0, y0, ha0, va0 = spots[0]
    t = ax.text(x0, y0, s, ha=ha0, va=va0, **kw)
    free = lambda: not _hits(ax, t.get_window_extent(r).expanded(1.03, 1.08), r, skip=t)
    for x, y, ha, va in spots:
        t.set(x=x, y=y, ha=ha, va=va)
        if free():
            return t
    f0 = ax.transAxes.inverted().transform(ax.transData.transform((x0, y0)))
    grid = sorted(((i / 40, j / 40) for i in range(41) for j in range(41)), key=lambda p: (p[0] - f0[0]) ** 2 + (p[1] - f0[1]) ** 2)
    t.set(ha=ha0, va=va0, transform=ax.transAxes)
    for p in grid:
        t.set_position(p)
        if free():
            return t
    t.set(x=x0, y=y0, transform=ax.transData)
    print(f"  NO FREE SPOT for {s!r}", flush=True)
    return t


def save(fig, section, name):
    for t, what in overlaps(fig):
        print(f"  OVERLAP {name}: {t!r} touches {what!r}", flush=True)
    d = os.path.join(FIGS, section)
    os.makedirs(d, exist_ok=True)
    for ext in FORMATS:
        fig.savefig(os.path.join(d, f"{name}.{ext}"))
    plt.close(fig)
    return os.path.join(d, f"{name}.png")


def smooth(ys, k):
    return [sum(ys[max(0, i - k + 1):i + 1]) / len(ys[max(0, i - k + 1):i + 1]) for i in range(len(ys))]
