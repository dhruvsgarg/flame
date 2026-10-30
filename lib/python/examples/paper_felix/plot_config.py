"""Master plot config for every Felix paper and evaluation figure: edit here, never in a figure script.

Import it (directly or via figlib) before plotting; `apply()` runs on import and sets matplotlib's rcParams.
Colors: palette.md categorical slots (validated for CVD); one color per stack across the paper, `*` oracle arms dashed in the
stack's hue (B.18). Grid and axes sit below data (`axes.axisbelow`), so grid lines never cross bars.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ---- system name: the only place it is set; every label goes through label() ----
SYS_NAME = "FLuX"


def label(name):
    """Display name: internal key "Felix" (stacks, mechanisms "Felix-X") -> SYS_NAME."""
    return name.replace("Felix", SYS_NAME)


# ---- sizes (inches, points): figures are drawn at print size, never scaled in LaTeX ----
COL_W, FULL_W, ROW_H = 3.25, 6.75, 1.45  # MLSys (ICML-style) \columnwidth, \textwidth, one row of subfigures
FONT = {"base": 8, "title": 8, "label": 8, "tick": 7, "legend": 7, "annot": 7, "status": 6}  # ≥ 7 pt in print
LINE = {"width": 1.2, "hero": 1.8, "muted": 0.9, "ref": 0.8, "grid": 0.4, "axis": 0.6}
MARKER = {"size": 3.5, "end": 2.5}
TICK = {"length": 2.5, "width": 0.5, "pad": 2, "direction": "out"}
BAR = {"width": 0.34, "edge": "white", "edge_width": 0.5}
DPI, FORMATS = 300, ("pdf", "png")

# ---- colors ----
INK, MUTED, GRID, SURFACE = "#000000", "#6b6a64", "#e4e3de", "#ffffff"  # MUTED only for what must not draw the eye
# CVD-safe, one distinct hue per stack; markers + widths separate them in grayscale. No grey stack.
OI = {"blue": "#0072B2", "vermillion": "#D55E00", "green": "#009E73", "gold": "#DDAA33", "purple": "#882E72", "sky": "#56B4E9"}
STACK = {"Felix": (OI["blue"], "-"), "FedBuff": (OI["vermillion"], "-"), "Oort": (OI["green"], "-"), "REFL": (OI["gold"], "-"),
         "FedDance": (OI["purple"], "-"), "FedAvg": (OI["sky"], "-"), "FedBuff*": (OI["vermillion"], "--"),
         "Oort*": (OI["green"], "--"), "REFL*": (OI["gold"], "--"), "FedDance*": (OI["purple"], "--"), "FedAvg*": (OI["sky"], "--")}
MARK = {"Felix": "o", "FedBuff": "s", "Oort": "^", "REFL": "v", "FedDance": "D", "FedAvg": "X"}  # oracle `*` = its stack's marker
PRED = {"Persistence": (INK, "-", "o"), "REFL Prophet": (OI["gold"], "-", "v"), "FedDance forecast": (OI["purple"], "-", "D"),
        "Time-of-day vote": (INK, "--", "s")}  # FX-N147 predictors: forecasters in their stack's color, naive guesses black
TRACE = {"MobiPerf": (INK, "-"), "Syn-50%": ("#8c8c8c", "--"), "Syn-0%": ("#b1b0a9", ":")}  # syn = memoryless control: grey on purpose
STATUS = {"MOCK": "#e34948", "PRELIM": "#c98500", "DRAFT": "#6250d6", "MISSING": "#6b6a64"}  # real figures carry no status

# ---- z-order: data above grid; lines and markers above bars and fills ----
Z = {"fill": 1, "bar": 2, "line": 3, "hero": 4, "marker": 5, "annot": 6, "status": 10}

RC = {
    "font.family": "STIXGeneral", "mathtext.fontset": "stix",  # Times-like, matches the MLSys body; Type 42 embedded
    "font.size": FONT["base"], "axes.titlesize": FONT["title"], "axes.labelsize": FONT["label"],
    "xtick.labelsize": FONT["tick"], "ytick.labelsize": FONT["tick"], "legend.fontsize": FONT["legend"],
    "axes.edgecolor": INK, "axes.labelcolor": INK, "axes.linewidth": LINE["axis"], "text.color": INK,
    "xtick.color": INK, "ytick.color": INK,
    "xtick.major.size": TICK["length"], "ytick.major.size": TICK["length"], "xtick.major.width": TICK["width"],
    "ytick.major.width": TICK["width"], "xtick.major.pad": TICK["pad"], "ytick.major.pad": TICK["pad"],
    "xtick.direction": TICK["direction"], "ytick.direction": TICK["direction"],
    "xtick.minor.size": TICK["length"] * 0.6, "ytick.minor.size": TICK["length"] * 0.6,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": LINE["grid"],
    "axes.spines.top": False, "axes.spines.right": False, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
    "lines.linewidth": LINE["width"], "lines.markersize": MARKER["size"],
    "patch.edgecolor": BAR["edge"], "patch.linewidth": BAR["edge_width"],
    "legend.frameon": False, "legend.handlelength": 1.8, "legend.columnspacing": 1.0, "legend.borderaxespad": 0.3,
    # no bbox cropping: PDF = figsize, so LaTeX never rescales text
    "figure.constrained_layout.use": True, "figure.constrained_layout.h_pad": 0.01, "figure.constrained_layout.w_pad": 0.01,
    "figure.constrained_layout.hspace": 0.02, "figure.constrained_layout.wspace": 0.02,
    "savefig.dpi": DPI, "savefig.bbox": None, "savefig.pad_inches": 0.01, "pdf.fonttype": 42, "ps.fonttype": 42,
    "legend.handletextpad": 0.4, "legend.labelspacing": 0.25, "axes.labelpad": 2, "axes.xmargin": 0.01,
}


def apply():
    plt.rcParams.update(RC)


apply()
