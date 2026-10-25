"""Figure helpers on top of plot_config (the one place for style): status labels, save, smoothing."""
import os

from plot_config import (COL_W, FONT, FORMATS, FULL_W, GRID, INK, LINE, MUTED, PRED, ROW_H, STACK, STATUS, TRACE, Z,  # noqa: F401
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


def save(fig, section, name):
    d = os.path.join(FIGS, section)
    os.makedirs(d, exist_ok=True)
    for ext in FORMATS:
        fig.savefig(os.path.join(d, f"{name}.{ext}"))
    plt.close(fig)
    return os.path.join(d, f"{name}.png")


def smooth(ys, k):
    return [sum(ys[max(0, i - k + 1):i + 1]) / len(ys[max(0, i - k + 1):i + 1]) for i in range(len(ys))]
