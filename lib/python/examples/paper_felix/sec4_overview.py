"""§4 Overview: architecture (P10, numbered steps, client vs server) and overview-by-example timeline (P11). Schematics, no data."""
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from figlib import COL_W, FONT, FULL_W, INK, LINE, MUTED as MUTEDK, STACK, label, plt, save

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


def _num(ax, x, y, n):
    ax.text(x, y, n, ha="center", va="center", fontsize=FONT["status"], weight="bold", color="white", zorder=8,
            bbox=dict(boxstyle="circle,pad=0.12", fc=INK, ec="none"))


def _link(ax, a, b, kind, rad=0.0, color=INK):
    """kind: 'ctl' = thin dashed control message (bytes); 'data' = thick model / update (MB)."""
    st = dict(lw=LINE["ref"], ls=(0, (3, 2)), mutation_scale=6) if kind == "ctl" else dict(lw=LINE["hero"], mutation_scale=9)
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", color=color, connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0, zorder=5, **st))


def fig_architecture():
    """F6, figure*: one live signal (the eligible pool) drives three coupled controllers; FLuX parts blue, inherited FedBuff engine grey."""
    fig, ax = plt.subplots(figsize=(FULL_W, 2.55))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    S, kw, sm = label("Felix"), dict(fontsize=FONT["annot"]), dict(fontsize=FONT["status"])
    ax.add_patch(Rectangle((0.0, 0.02), 0.175, 0.96, fc="#f2f2f2", ec="none", gid="region"))
    ax.add_patch(Rectangle((0.2, 0.53), 0.555, 0.45, fc="#e8f1f8", ec=FELIX, lw=LINE["axis"], gid="region"))
    ax.add_patch(Rectangle((0.2, 0.02), 0.555, 0.47, fc="#f2f2f2", ec="none", gid="region"))
    ax.text(0.0875, 0.955, "Devices", ha="center", va="top", fontsize=FONT["label"], weight="bold")
    ax.text(0.207, 0.955, f"{S} control plane (new)", ha="left", va="top", fontsize=FONT["label"], weight="bold", color=FELIX)
    ax.text(0.207, 0.035, "Data plane (FedBuff engine, inherited)", ha="left", va="bottom", fontsize=FONT["label"], weight="bold")
    # devices
    phones = (("Eligible", True, ""), ("Not eligible", False, ""), ("Left mid-task", False, "x"), ("Training", True, "task"),
              ("Not eligible", False, ""))
    py = [0.80 - 0.135 * k for k in range(5)]
    for y, (txt, el, mk) in zip(py, phones):
        ax.add_patch(FancyBboxPatch((0.012, y - 0.05), 0.022, 0.1, boxstyle="round,pad=0.002,rounding_size=0.006", fc="white" if el else "#cfcfcf",
                                    ec=INK, lw=LINE["axis"], ls="-" if el else (0, (2, 1.5))))
        if mk == "task":
            ax.add_patch(Rectangle((0.016, y - 0.03), 0.014, 0.035, fc=FELIX, ec="none"))
        elif mk == "x":
            ax.plot([0.023], [y], "x", color=INK, ms=4, mew=1.0)
        ax.text(0.042, y, txt, ha="left", va="center", **kw)
    ax.text(0.0875, 0.1, "Eligible = idle, charging,\nunmetered (OS signals);\ndata arrives while in use", ha="center", va="center",
            linespacing=1.1, **sm)
    # control plane: ledger -> bus -> three controllers side by side
    _box(ax, 0.215, 0.6, 0.15, 0.3, "Eligibility ledger\nlive pool $P(t)$\n1 bit + $|D|$, $q$\nper client", ec=FELIX)
    ctl = ((0.405, f"{S}-Pace", "how many:\n$c = f(|P|, \\varphi)$"), (0.525, f"{S}-Select", "whom:\n$p \\propto |D|\\,q$"),
           (0.645, f"{S}-Evict", "free the\nslot now"))
    ax.plot([0.365, 0.695], [0.87, 0.87], color=FELIX, lw=LINE["ref"], ls=(0, (3, 2)), zorder=4)
    for x, name, body in ctl:
        _box(ax, x, 0.6, 0.1, 0.22, "", ec=FELIX)
        ax.text(x + 0.05, 0.8, name, ha="center", va="top", weight="bold", **kw)
        ax.text(x + 0.05, 0.67, body, ha="center", va="center", linespacing=1.1, **kw)
        _link(ax, (x + 0.05, 0.87), (x + 0.05, 0.82), "ctl", color=FELIX)
    # data plane: slot row under the controllers; buffer -> Agg -> version below
    ax.add_patch(Rectangle((0.45, 0.29), 0.295, 0.11, fc="white", ec=INK, lw=LINE["axis"]))
    for k in range(5):
        ax.add_patch(Rectangle((0.535 + 0.042 * k, 0.31), 0.034, 0.07, fc=FELIX if k != 3 else "white", ec=INK, lw=0.4))
    ax.text(0.457, 0.345, "$c$ slots\nin flight", ha="left", va="center", linespacing=1.0, **sm)
    for x, w, t, ec in ((0.215, 0.11, "Buffer\n$K$ updates", INK), (0.36, 0.16, f"{S}-Agg\nweight $(1+\\tau)^{{-1/4}}$", FELIX),
                        (0.555, 0.19, "Version $v{+}1$ → served model", INK)):
        _box(ax, x, 0.09, w, 0.13, t, ec=ec)
    _link(ax, (0.325, 0.155), (0.36, 0.155), "data")
    _link(ax, (0.52, 0.155), (0.555, 0.155), "data"); _num(ax, 0.5375, 0.2, "7")
    # scenario
    _link(ax, (0.13, py[0]), (0.215, py[0]), "ctl"); _num(ax, 0.17, py[0] + 0.035, "1")
    _link(ax, (0.125, py[2]), (0.215, 0.64), "ctl"); _num(ax, 0.17, 0.62, "5")
    for x, n in ((0.455, "2"), (0.575, "3"), (0.695, "5")):
        _link(ax, (x, 0.6), (x, 0.4), "ctl", color=FELIX); _num(ax, x + 0.014, 0.5, n)
    _link(ax, (0.45, 0.36), (0.105, py[3] + 0.015), "data"); _num(ax, 0.3, 0.405, "4")
    _link(ax, (0.07, py[3] - 0.035), (0.215, 0.17), "data"); _num(ax, 0.155, 0.31, "6")
    # feedback (8): update loss -> q in the ledger; commit loss -> φ in Pace; c -> τ -> Agg weight
    _link(ax, (0.26, 0.22), (0.26, 0.6), "ctl", color=FELIX); _num(ax, 0.245, 0.5, "8")
    ax.text(0.268, 0.47, "loss → $q$", ha="left", va="center", **sm)
    _link(ax, (0.42, 0.22), (0.42, 0.6), "ctl", color=FELIX); _num(ax, 0.405, 0.5, "8")
    ax.text(0.405, 0.27, "loss → $\\varphi$", ha="right", va="center", **sm)
    _link(ax, (0.48, 0.29), (0.48, 0.22), "ctl", color=FELIX)
    ax.text(0.487, 0.255, "$c$ → staleness $\\tau$", ha="left", va="center", **sm)
    # step list + key, inside the figure
    steps = ("Client becomes eligible, reports it", "Pace sets $c$ from pool, loss phase", "Select fills a free slot",
             "The model is dispatched", "Client leaves mid-task, reports;\nEvict frees the slot, refilled at once",
             "Update + loss return to the buffer", "Agg commits a version every $K$", "Losses update $q$ and $\\varphi$")
    y = 0.95
    for k, t in enumerate(steps, 1):
        _num(ax, 0.775, y - 0.025, str(k))
        ax.text(0.79, y, t, ha="left", va="top", linespacing=1.05, **kw)
        y -= 0.1 if "\n" in t else 0.065
    ax.add_patch(FancyArrowPatch((0.77, 0.12), (0.81, 0.12), arrowstyle="-|>", lw=LINE["ref"], ls=(0, (3, 2)), mutation_scale=6, color=INK))
    ax.text(0.815, 0.12, "Control message (bytes)", ha="left", va="center", **kw)
    ax.add_patch(FancyArrowPatch((0.77, 0.05), (0.81, 0.05), arrowstyle="-|>", lw=LINE["hero"], mutation_scale=9, color=INK))
    ax.text(0.815, 0.05, "Model / update (MB)", ha="left", va="center", **kw)
    return save(fig, SEC, "fig_architecture")


# Toy timeline (F7): 6 clients, eligible windows (grey), tasks; FedBuff (c = 3) learns eligibility at check-in and departures at
# timeout; FLuX dispatches on the client's report, evicts on departure and runs c = ⌈pool / 2⌉ tasks. Versions commit at the dotted lines.
ELIG = {0: [(0, 10)], 1: [(0, 4)], 2: [(1.5, 10)], 3: [(0, 6.5)], 4: [(2.5, 7)], 5: [(4, 9)]}
FEDBUFF = {0: [(0, 2.5, "ok"), (2.5, 5, "ok"), (5, 7.5, "ok"), (7.5, 10, "ok")], 1: [(0, 2.5, "ok"), (2.5, 6.25, "held")],
           2: [(2.5, 3.5, "wait"), (3.5, 6.0, "ok"), (6.0, 8.5, "ok"), (8.5, 10, "ok")], 3: [(0, 2.5, "ok")], 5: [(6.25, 8.75, "ok")]}
FELIXT = {0: [(0, 2.5, "ok"), (2.5, 5, "ok"), (5, 7.5, "ok"), (7.5, 10, "ok")], 1: [(2.5, 4, "evict")],
          2: [(2.5, 5, "ok"), (5, 7.5, "ok")], 3: [(0, 2.5, "ok")], 4: [(4, 6.5, "ok")], 5: [(7.5, 9, "evict")]}
WAIT = "#cfcfcf"


def _steps(f, ts):
    return [f(t + 1e-6) for t in ts]


def fig_toy_timeline():
    from matplotlib.patches import Patch
    fig, axs = plt.subplots(4, 1, figsize=(COL_W, 3.3), sharex=True, height_ratios=(3, 1, 3, 1))
    ts = [x / 20 for x in range(201)]
    pool = lambda t: sum(a <= t < b for w in ELIG.values() for a, b in w)
    for k, (name, sched, color) in enumerate((("FedBuff", FEDBUFF, STACK["FedBuff"][0]), (label("Felix"), FELIXT, FELIX))):
        ax, sx = axs[2 * k], axs[2 * k + 1]
        for c, wins in ELIG.items():
            for a, b in wins:
                ax.add_patch(Rectangle((a, c - 0.4), b - a, 0.8, fc="#ecebe6", ec="none", gid="region"))
            for a, b, kind in sched.get(c, []):
                box = (a + 0.05, c - 0.25), b - a - 0.1, 0.5
                if kind == "ok":
                    ax.add_patch(Rectangle(*box, fc=color, ec="none"))
                elif kind == "held":
                    ax.add_patch(Rectangle(*box, fc="white", ec=color, hatch="////", lw=LINE["axis"]))
                elif kind == "wait":
                    ax.add_patch(Rectangle(*box, fc=WAIT, ec=INK, hatch="....", lw=LINE["axis"]))
                else:
                    ax.add_patch(Rectangle(*box, fc=color, ec="none"))
                    ax.plot([b], [c], "x", color=INK, ms=4, mew=1.0)
        ax.set(xlim=(0, 10), ylim=(5.6, -0.6 - 0.9 * (k == 0)), yticks=range(6), yticklabels=[f"Client {i + 1}" for i in range(6)])
        ax.grid(False)
        ax.tick_params(axis="y", length=0)
        fly = lambda t: sum(a <= t < b for v in sched.values() for a, b, kind in v if kind != "wait")
        sx.step(ts, _steps(fly, ts), where="post", color=color, lw=LINE["hero"], label="Tasks in flight")
        sx.step(ts, _steps(pool, ts), where="post", color=INK, lw=LINE["muted"], label="Eligible pool")
        sx.set(ylim=(0, 6.2), yticks=[0, 3, 6])
        sx.set_ylabel(name, rotation=0, ha="right", va="center", weight="bold")
        sx.grid(axis="x", visible=False)
        for v in (2.5, 5, 7.5):
            ax.axvline(v, color=INK, lw=LINE["grid"], ls=":", zorder=0)
    axs[0].set_title("", pad=0)
    kw = dict(fontsize=FONT["annot"])
    axs[0].text(2.3, 2.0, "(i)", ha="right", va="center", weight="bold", **kw)
    axs[0].text(6.4, 1.0, "(ii)", ha="left", va="center", weight="bold", **kw)
    axs[0].annotate("(iii)", (7.55, -0.33), (8.1, -1.0), ha="left", va="center", weight="bold", **kw,
                    arrowprops=dict(arrowstyle="-", color=INK, lw=LINE["ref"], shrinkA=1, shrinkB=1))
    axs[3].set_xlabel("Time (a version commits at each dotted line)")
    axs[3].set_xticks([])
    fig.legend(handles=[Patch(fc="#ecebe6", label="Eligible"), Patch(fc=STACK["FedBuff"][0], label="FedBuff task"),
                        Patch(fc=FELIX, label=f"{label('Felix')} task"), Patch(fc=WAIT, ec=INK, hatch="....", label="(i) Slot idle: no check-in yet"),
                        Patch(fc="white", ec=STACK["FedBuff"][0], hatch="////", label="(ii) Held to timeout"),
                        plt.Line2D([], [], color=INK, marker="x", ls="none", ms=4, label="Client left, slot freed"),
                        plt.Line2D([], [], color=INK, lw=LINE["width"], label="Eligible pool"),
                        plt.Line2D([], [], color=MUTEDK, lw=LINE["hero"], label="Tasks in flight (task color)"),
                        Patch(fc="none", ec="none", label="(iii) Pick with pool ≤ c: no choice")],
               loc="outside upper center", ncol=2, handlelength=1.4, columnspacing=0.8)
    return save(fig, SEC, "fig_toy_timeline")


FIGS = (fig_architecture, fig_toy_timeline)
