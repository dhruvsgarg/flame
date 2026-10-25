"""Master plot config for every Felix paper and evaluation figure: edit here, never in a figure script.

Import it (directly or via figlib) before plotting; `apply()` runs on import and sets matplotlib's rcParams.
Colors: palette.md categorical slots (validated for CVD); one color per stack across the paper, `*` oracle arms dashed in the
stack's hue (B.18). Grid and axes sit below data (`axes.axisbelow`), so grid lines never cross bars.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ---- sizes (inches, points) ----
COL_W, FULL_W, ROW_H = 3.3, 6.9, 1.45  # MLSys column, full width, one row of subfigures
FONT = {"base": 7, "title": 7, "label": 7, "tick": 6, "legend": 6, "annot": 5.5, "status": 5.5}
LINE = {"width": 1.4, "hero": 1.8, "muted": 1.0, "ref": 0.8, "grid": 0.5, "axis": 0.6}
MARKER = {"size": 3.5, "end": 2.5}
TICK = {"length": 2.5, "width": 0.5, "pad": 2, "direction": "out"}
BAR = {"width": 0.34, "edge": "white", "edge_width": 0.5}
DPI, FORMATS = 300, ("pdf", "png")

# ---- colors ----
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a64", "#e4e3de", "#ffffff"
STACK = {"Felix": ("#2a78d6", "-"), "FedBuff": ("#1baf7a", "-"), "REFL": ("#eda100", "-"), "FedDance": ("#e87ba4", "-"),
         "Oort": ("#008300", "-"), "FedAvg": ("#9c6b30", "-"), "REFL*": ("#eda100", "--"), "FedDance*": ("#e87ba4", "--"),
         "Oort*": ("#008300", "--"), "FedBuff*": ("#1baf7a", "--")}
PRED = {"naive: eligible now": ("#2a78d6", "-", "o"), "REFL Prophet": ("#eda100", "-", "s"), "FedDance $V_m$": ("#e87ba4", "-", "^"),
        "time-of-day vote": ("#6250d6", "--", "D")}  # availability predictors (FX-N147)
TRACE = {"MobiPerf": (INK, "-"), "syn_50": (MUTED, "--"), "syn_0": ("#b1b0a9", ":")}
STATUS = {"MOCK": "#e34948", "PRELIM": "#c98500", "DRAFT": "#6250d6"}  # real figures carry no status

# ---- z-order: data above grid; lines and markers above bars and fills ----
Z = {"fill": 1, "bar": 2, "line": 3, "hero": 4, "marker": 5, "annot": 6, "status": 10}

RC = {
    "font.family": "DejaVu Sans", "font.size": FONT["base"], "axes.titlesize": FONT["title"], "axes.labelsize": FONT["label"],
    "xtick.labelsize": FONT["tick"], "ytick.labelsize": FONT["tick"], "legend.fontsize": FONT["legend"],
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.linewidth": LINE["axis"], "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.major.size": TICK["length"], "ytick.major.size": TICK["length"], "xtick.major.width": TICK["width"],
    "ytick.major.width": TICK["width"], "xtick.major.pad": TICK["pad"], "ytick.major.pad": TICK["pad"],
    "xtick.direction": TICK["direction"], "ytick.direction": TICK["direction"],
    "xtick.minor.size": TICK["length"] * 0.6, "ytick.minor.size": TICK["length"] * 0.6,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": LINE["grid"],
    "axes.spines.top": False, "axes.spines.right": False, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
    "lines.linewidth": LINE["width"], "lines.markersize": MARKER["size"],
    "patch.edgecolor": BAR["edge"], "patch.linewidth": BAR["edge_width"],
    "legend.frameon": False, "legend.handlelength": 1.8, "legend.columnspacing": 1.0, "legend.borderaxespad": 0.3,
    "savefig.dpi": DPI, "savefig.bbox": "tight", "pdf.fonttype": 42, "ps.fonttype": 42,
}


def apply():
    plt.rcParams.update(RC)


apply()
