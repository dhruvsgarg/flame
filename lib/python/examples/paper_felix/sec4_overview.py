"""§4 Overview: architecture (P10, numbered steps, client vs server) and overview-by-example timeline (P11). Schematics, no data."""
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from figlib import COL_W, FONT, INK, LINE, STACK, label, plt, save

SEC = "s4_overview"
FELIX = STACK["Felix"][0]


def _box(ax, x, y, w, h, text, ec=INK, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.005,rounding_size=0.015", fc="white", ec=ec, lw=LINE["width"] * 0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=FONT["annot"], weight=weight, linespacing=1.1)


def _arrow(ax, a, b, n=None, rad=0.0, at=None, va="bottom"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=7, color=INK, lw=LINE["ref"], connectionstyle=f"arc3,rad={rad}"))
    if n:
        x, y = at or ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 0.035)
        ax.text(x, y, n, fontsize=FONT["annot"], weight="bold", ha="center", va=va)


def fig_architecture():
    """Client vs server components, numbered steps (legend below the diagram)."""
    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    ax.set(xlim=(0, 1), ylim=(-0.56, 1))
    ax.axis("off")
    S = label("Felix")
    ax.add_patch(Rectangle((0.0, 0.04), 0.3, 0.92, fc="#f2f2f2", ec="none", gid="region"))
    ax.add_patch(Rectangle((0.34, 0.04), 0.66, 0.92, fc="#e8f1f8", ec="none", gid="region"))
    ax.text(0.15, 0.93, "Client (device)", ha="center", va="top", fontsize=FONT["label"], weight="bold")
    ax.text(0.67, 0.93, "Server", ha="center", va="top", fontsize=FONT["label"], weight="bold")
    _box(ax, 0.02, 0.52, 0.26, 0.2, "Eligibility\nmonitor")
    _box(ax, 0.02, 0.1, 0.26, 0.2, "Local training\n(1 epoch)")
    _box(ax, 0.36, 0.52, 0.19, 0.2, "Eligibility\nledger", ec=STACK["Felix"][0])
    _box(ax, 0.585, 0.52, 0.19, 0.2, f"{S}-Pace", ec=STACK["Felix"][0], weight="bold")
    _box(ax, 0.805, 0.52, 0.185, 0.2, f"{S}-Select", ec=STACK["Felix"][0], weight="bold")
    _box(ax, 0.36, 0.1, 0.19, 0.2, f"{S}-Notify", ec=STACK["Felix"][0], weight="bold")
    _box(ax, 0.585, 0.1, 0.19, 0.2, f"{S}-Agg", ec=STACK["Felix"][0], weight="bold")
    _box(ax, 0.805, 0.1, 0.185, 0.2, "Averaged\nmodel")
    _arrow(ax, (0.28, 0.62), (0.36, 0.62), "1")
    _arrow(ax, (0.55, 0.62), (0.585, 0.62), "2")
    _arrow(ax, (0.775, 0.62), (0.805, 0.62), "3")
    _arrow(ax, (0.86, 0.52), (0.5, 0.3), "4", rad=-0.1, at=(0.7, 0.42))
    _arrow(ax, (0.36, 0.22), (0.28, 0.22), "5", at=(0.32, 0.235))
    _arrow(ax, (0.2, 0.1), (0.66, 0.1), "6", rad=0.25, at=(0.43, -0.02), va="top")
    _arrow(ax, (0.775, 0.2), (0.805, 0.2), "7", at=(0.79, 0.3))
    steps = ("Client reports each eligibility change", "Ledger tracks the eligible pool", "Pace sets concurrency c",
             "Select picks a client per free slot", "Dispatch; evict on departure", "Update and loss return",
             "Commit a version every K updates")
    for i, t in enumerate(steps):
        ax.text(0.01 if i < 4 else 0.53, -0.15 - 0.1 * (i % 4), f"{i + 1}  {t}", fontsize=FONT["annot"], va="top")
    return save(fig, SEC, "fig_architecture")


# Toy timeline: 6 clients, eligible windows (gray), tasks (colored); FedBuff learns at check-in and departures at timeout.
ELIG = {0: [(0, 10)], 1: [(0, 4), (7, 10)], 2: [(2, 10)], 3: [(0, 6)], 4: [(3, 7)], 5: [(5, 10)]}
FEDBUFF = {0: [(0.0, 2.5, "ok"), (3.5, 6.0, "ok"), (7.0, 9.5, "ok")], 1: [(0.0, 2.5, "ok"), (2.5, 6.5, "held")],
           2: [(3.0, 5.5, "ok"), (6.5, 9.0, "ok")], 3: [(1.0, 3.5, "ok"), (4.5, 8.5, "held")], 4: [(4.0, 6.5, "ok")],
           5: [(6.0, 8.5, "ok")]}
FELIXT = {0: [(0.0, 2.5, "ok"), (2.5, 5.0, "ok"), (5.0, 7.5, "ok"), (7.5, 10, "ok")], 1: [(0.0, 2.5, "ok"), (2.5, 4.0, "evict"),
          (7.0, 9.5, "ok")], 2: [(2.0, 4.5, "ok"), (4.5, 7.0, "ok"), (7.0, 9.5, "ok")], 3: [(0.0, 2.5, "ok"), (2.5, 5.0, "ok")],
          4: [(3.0, 5.5, "ok")], 5: [(5.0, 7.5, "ok"), (7.5, 10, "ok")]}


def fig_toy_timeline():
    fig, axs = plt.subplots(2, 1, figsize=(COL_W, 2.3), sharex=True)
    for ax, (name, sched, color) in zip(axs, (("FedBuff", FEDBUFF, STACK["FedBuff"][0]), (label("Felix"), FELIXT, FELIX))):
        for c, wins in ELIG.items():
            for a, b in wins:
                ax.add_patch(Rectangle((a, c - 0.4), b - a, 0.8, fc="#ecebe6", ec="none"))
            for a, b, kind in sched.get(c, []):
                if kind == "ok":
                    ax.add_patch(Rectangle((a + 0.05, c - 0.25), b - a - 0.1, 0.5, fc=color, ec="none"))
                elif kind == "held":
                    ax.add_patch(Rectangle((a + 0.05, c - 0.25), b - a - 0.1, 0.5, fc="white", ec=color, hatch="////", lw=LINE["axis"]))
                else:
                    ax.add_patch(Rectangle((a + 0.05, c - 0.25), b - a - 0.1, 0.5, fc=color, ec="none", alpha=0.35))
                    ax.plot([b], [c], "x", color=INK, ms=4)
        ax.set(xlim=(0, 10), ylim=(5.6, -0.6), yticks=range(6), yticklabels=[f"Client {i + 1}" for i in range(6)], ylabel=name)
        ax.grid(False)
    for v in (2.5, 5, 7.5):
        for ax in axs:
            ax.axvline(v, color=INK, lw=LINE["grid"], ls=":")
    axs[1].set_xlabel("Time (a version commits at each dotted line)")
    axs[1].set_xticks([])
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(fc="#ecebe6", label="Eligible"), Patch(fc=STACK["FedBuff"][0], label="FedBuff task"),
                        Patch(fc=FELIX, label=f"{label('Felix')} task"), Patch(fc="white", ec=STACK["FedBuff"][0], hatch="////",
                        label="Held to timeout"), Patch(fc=FELIX, alpha=0.35, label="Evicted (×)")],
               loc="outside upper center", ncol=3, handlelength=1.4, columnspacing=0.8)
    return save(fig, SEC, "fig_toy_timeline")


FIGS = (fig_architecture, fig_toy_timeline)
