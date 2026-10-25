"""§4 Overview: architecture (P10, numbered steps, client vs server) and overview-by-example timeline (P11). Schematics, no data."""
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from figlib import COL_W, FONT, FULL_W, INK, LINE, MUTED, STACK, plt, save, status

SEC = "s4_overview"
FELIX = STACK["Felix"][0]


def _box(ax, x, y, w, h, text, fc="white", ec=INK, fs=6, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.02", fc=fc, ec=ec, lw=LINE["ref"]))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK, weight=weight)


def _arrow(ax, a, b, n=None, color=INK, rad=0.0, at=None):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=7, color=color, lw=LINE["ref"], connectionstyle=f"arc3,rad={rad}"))
    if n:
        x, y = at or ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 0.03)
        ax.text(x, y, n, fontsize=FONT["tick"], color=FELIX, weight="bold", ha="center")


def fig_architecture():
    fig, ax = plt.subplots(figsize=(FULL_W, 2.1))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    ax.add_patch(Rectangle((0.01, 0.05), 0.27, 0.9, fc="#f4f3ef", ec="none"))
    ax.add_patch(Rectangle((0.33, 0.05), 0.66, 0.9, fc="#eef4fc", ec="none"))
    ax.text(0.145, 0.9, "Client (device)", ha="center", fontsize=FONT["title"], weight="bold")
    ax.text(0.66, 0.9, "Server (aggregator)", ha="center", fontsize=FONT["title"], weight="bold")
    _box(ax, 0.03, 0.62, 0.23, 0.16, "eligibility monitor\n(idle · charging · unmetered)")
    _box(ax, 0.03, 0.2, 0.23, 0.2, "local training\n1 epoch → update,\nmean loss, |D|")
    _box(ax, 0.36, 0.62, 0.17, 0.16, "eligibility ledger\n(eligible / not)", fc="white", ec=FELIX)
    _box(ax, 0.58, 0.62, 0.17, 0.16, "Felix-Pace\nconcurrency c", ec=FELIX, weight="bold")
    _box(ax, 0.8, 0.62, 0.17, 0.16, "Felix-Select\n|D| × EMA loss", ec=FELIX, weight="bold")
    _box(ax, 0.58, 0.2, 0.17, 0.2, "Felix-Agg\nbuffer K, large step,\nmild staleness discount", ec=FELIX, weight="bold")
    _box(ax, 0.8, 0.2, 0.17, 0.2, "averaged model\nserved")
    _box(ax, 0.36, 0.2, 0.17, 0.2, "Felix-Notify\ndispatch · evict\non departure", ec=FELIX, weight="bold")
    _arrow(ax, (0.26, 0.7), (0.36, 0.7), "①")
    _arrow(ax, (0.53, 0.7), (0.58, 0.7), "②")
    _arrow(ax, (0.75, 0.7), (0.8, 0.7), "③")
    _arrow(ax, (0.885, 0.62), (0.47, 0.4), "④", rad=-0.15, at=(0.72, 0.5))
    _arrow(ax, (0.36, 0.3), (0.26, 0.3), "⑤")
    _arrow(ax, (0.26, 0.24), (0.6, 0.2), "⑥", rad=0.3, at=(0.43, 0.1))
    _arrow(ax, (0.75, 0.3), (0.8, 0.3), "⑦")
    ax.text(0.5, 0.0, "① client pushes every eligibility change  ② ledger sizes the eligible pool  ③ c from the pool + training phase  "
            "④ pick an eligible client per free slot  ⑤ dispatch (evict if it stops being eligible)  ⑥ update + loss  ⑦ commit every K",
            ha="center", fontsize=FONT["annot"], color=MUTED)
    status(fig, "DRAFT", "schematic; redraw in the paper's figure tool")
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
    for ax, (name, sched, color) in zip(axs, (("FedBuff: check-in, departure at timeout", FEDBUFF, STACK["FedBuff"][0]),
                                              ("Felix: pushed eligibility, evict at departure", FELIXT, FELIX))):
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
        ax.set(xlim=(0, 10), ylim=(-0.6, 5.6), yticks=range(6), yticklabels=[f"c{i}" for i in range(6)], title=name)
        ax.grid(False)
    for v in (2.5, 5, 7.5):
        for ax in axs:
            ax.axvline(v, color=MUTED, lw=LINE["axis"], ls=":")
    axs[1].set_xlabel("time (versions committed at dotted lines)")
    axs[0].text(8.55, 3, "held to\ntimeout", fontsize=FONT["annot"], color=INK, va="center")
    axs[1].text(4.2, 1, "evicted", fontsize=FONT["annot"], color=INK, va="center")
    fig.tight_layout()
    status(fig, "DRAFT", "toy schedule; gray = eligible")
    return save(fig, SEC, "fig_toy_timeline")


FIGS = (fig_architecture, fig_toy_timeline)
