"""Shared style for `fl_fwd_ft_writeup.md`'s figures.

Palette is the validated three-slot categorical set (blue/orange/aqua) plus the
reserved status colours. Three slots is the documented cap for scatter, where
every pair is on screen at once; the aqua slot sits below 3:1 on the light
surface, so every series carries a **visible direct label** -- that is the
relief rule, not a preference.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Categorical slots 1-3 (all-pairs validated: CVD dE 9.2, normal-vision 24.0).
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
# Status -- reserved, never a series colour.
GOOD, CRITICAL, WARNING = "#0ca30c", "#d03b3b", "#fab219"
# Chrome.
SURFACE = "#fcfcfb"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8880"
GRID = "#e4e3df"

DATASET_COLOR = {"agnews": BLUE, "yahoo": ORANGE, "yelp-p": AQUA}


def use_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": ["DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK,
        "axes.labelsize": 9,
        "axes.labelcolor": INK2,
        "axes.edgecolor": GRID,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "lines.linewidth": 2.0,
        "figure.dpi": 160,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.25,
    })


def despine(ax, keep=("left", "bottom")):
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)


def note(ax, text, xy, xytext, color=INK2, **kw):
    """A direct label with a hairline leader -- the relief rule's mitigation."""
    ax.annotate(text, xy=xy, xytext=xytext, color=color, fontsize=8,
                arrowprops=dict(arrowstyle="-", color=color, lw=0.7,
                                shrinkA=0, shrinkB=3), **kw)


def caption(fig, text, width=None):
    """Figure caption, hard-wrapped to the figure width (mpl `wrap` is unreliable
    for `figure.text` anchored outside the axes)."""
    import textwrap
    if width is None:
        width = int(fig.get_size_inches()[0] * 15)
    fig.text(0.0, -0.03, "\n".join(textwrap.wrap(text, width)), ha="left",
             va="top", fontsize=7.5, color=INK3)
