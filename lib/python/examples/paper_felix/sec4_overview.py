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


# F6 palette: one accent for FLuX (new), one for the learning feedback; the inherited async engine stays grey.
NEW, FB = "#1B7F79", "#7B1E6B"
ENGINE, BUSY = "#f1f1ef", "#d6d6d3"


def _rbox(ax, x, y, w, h, ec=INK, fc="white", lw=1.0, ls="-", r=0.05, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))


def _badge(ax, x, y, n, c=INK):
    ax.text(x, y, n, ha="center", va="center", fontsize=10, weight="bold", color="white", zorder=9,
            bbox=dict(boxstyle="circle,pad=0.2", fc=c, ec="white", lw=0.8))


def _flow(ax, a, b, kind, c=INK, rad=0.0, z=5):
    """kind: 'ctl' = dashed control message (bytes); 'data' = thick model / update (MB)."""
    st = dict(lw=1.0, ls=(0, (3, 1.8)), mutation_scale=8) if kind == "ctl" else dict(lw=2.4, mutation_scale=11)
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", color=c, connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0, zorder=z, **st))


def fig_architecture():
    """F6, figure*: the live eligible pool, pushed by clients, drives Pace, Select and (through c -> τ) Agg; FLuX parts in one
    accent, feedback in a second, inherited FedBuff engine grey. Badges 1-8 = §4 ¶2 steps; the prose, not the figure, explains them."""
    W, H = FULL_W, 3.3
    fig = plt.figure(figsize=(W, H), layout="none")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(0, W), ylim=(0, H))
    ax.axis("off")
    S, F, TT = label("Felix"), 8.5, 10  # one body size, one title size

    # regions: devices | control plane (new, white) over data plane (inherited, grey)
    ax.add_patch(Rectangle((0.02, 0.02), 1.42, H - 0.04, fc="#f6f5f1", ec="none", zorder=0))
    ax.add_patch(Rectangle((1.54, 1.66), W - 1.56, H - 1.68, fc="white", ec=MUTEDK, lw=0.6, zorder=0))
    ax.add_patch(Rectangle((1.54, 0.02), W - 1.56, 1.56, fc=ENGINE, ec="none", zorder=0))
    ax.text(0.73, H - 0.07, "Devices", ha="center", va="top", fontsize=TT, weight="bold")
    ax.text(1.62, H - 0.07, f"{S} control plane", ha="left", va="top", fontsize=TT, weight="bold")
    ax.text(W - 0.08, 0.06, "Async training and aggregation", ha="right", va="bottom", fontsize=TT, weight="bold", color=MUTEDK)

    # devices: four states, each a phone + state + why
    dev = {"A": 2.72, "D": 2.12, "C": 1.42, "B": 0.78}
    def phone(y, ec, fc, ls="-"):
        _rbox(ax, 0.1, y - 0.21, 0.22, 0.42, ec=ec, fc=fc, lw=1.2, ls=ls, r=0.04)
    phone(dev["A"], NEW, "white")
    ax.text(0.21, dev["A"] + 0.02, "$\\bf{A}$", ha="center", va="center", fontsize=F, color=NEW)
    phone(dev["D"], INK, BUSY, ls=(0, (2, 1.5)))
    ax.plot([0.21], [dev["D"]], "x", color=INK, ms=6, mew=1.6, zorder=4)
    phone(dev["C"], INK, "white")
    ax.add_patch(Rectangle((0.14, dev["C"] - 0.12), 0.14, 0.2, fc=MUTEDK, ec="none", zorder=3))
    ax.text(0.21, dev["C"] - 0.02, "C", ha="center", va="center", fontsize=F, color="white", weight="bold", zorder=4)
    phone(dev["B"], INK, BUSY, ls=(0, (2, 1.5)))
    for k, hgt in enumerate((0.06, 0.1, 0.15)):
        ax.add_patch(Rectangle((0.135 + 0.05 * k, dev["B"] - 0.15), 0.035, hgt, fc=INK, ec="none", zorder=3))
    rows = (("A", "Eligible", "Idle, charging, Wi-Fi", NEW), ("D", "Leaves mid-task", "Unplugged: reports it", INK),
            ("C", "Training", "1 epoch on local data", INK), ("B", "In use: not eligible", "New data arrives, $|D|\\uparrow$", INK))
    for key, st, why, c in rows:
        ax.text(0.38, dev[key] + 0.07, st, ha="left", va="center", fontsize=F, weight="bold", color=c)
        ax.text(0.38, dev[key] - 0.09, why, ha="left", va="center", fontsize=F)
    _flow(ax, (0.12, 0.33), (0.42, 0.33), "ctl")
    ax.text(0.47, 0.33, "Control (bytes)", ha="left", va="center", fontsize=F)
    _flow(ax, (0.12, 0.15), (0.42, 0.15), "data")
    ax.text(0.47, 0.15, "Model / update (MB)", ha="left", va="center", fontsize=F)

    # Notify: eligibility ledger, one row per client
    lx, ly, lw_, lh = 1.62, 1.74, 1.16, 1.3
    _rbox(ax, lx, ly, lw_, lh, ec=NEW, fc="white", lw=1.2)
    ax.text(lx + lw_ / 2, ly + lh - 0.06, f"{S}-Notify", ha="center", va="top", fontsize=TT, weight="bold", color=NEW)
    ax.text(lx + lw_ / 2, ly + lh - 0.24, "Eligibility ledger", ha="center", va="top", fontsize=F)
    for x, h in ((lx + 0.36, "Elig."), (lx + 0.66, "$|D|$"), (lx + 0.98, "$q$")):
        ax.text(x, ly + 0.86, h, ha="center", va="center", fontsize=F, color=MUTEDK)
    led = (("A", 1, 0.55, ".7"), ("D", "x", 0.3, ".4"), ("C", 1, 0.4, ".5"), ("B", 0, 0.7, ".9"))
    for k, (n, el, d, q) in enumerate(led):
        y = ly + 0.7 - 0.15 * k
        ax.text(lx + 0.12, y, n, ha="center", va="center", fontsize=F, weight="bold")
        ax.add_patch(Rectangle((lx + 0.31, y - 0.05), 0.1, 0.1, fc=NEW if el == 1 else "white", ec=NEW, lw=0.8, zorder=3))
        if el == "x":
            ax.text(lx + 0.36, y, "1→0", ha="center", va="center", fontsize=F, color=NEW, weight="bold", zorder=4,
                    bbox=dict(boxstyle="round,pad=0.05", fc="white", ec="none"))
        ax.add_patch(Rectangle((lx + 0.5, y - 0.035), 0.32 * d / 0.7, 0.07, fc=INK, ec="none", zorder=3))
        ax.text(lx + 0.98, y, q, ha="center", va="center", fontsize=F)
    ax.text(lx + lw_ / 2, ly + 0.08, "No forecast", ha="center", va="bottom", fontsize=F, style="italic", color=NEW)

    # the live pool: one signal fanned out to the controllers
    by = 2.82
    ax.plot([lx + lw_, 6.3], [by, by], color=NEW, lw=2.6, solid_capstyle="round", zorder=4)
    ax.text(2.88, by + 0.06, "Live eligible pool $P(t)$, pushed by clients, drives every controller",
            ha="left", va="bottom", fontsize=F, weight="bold", color=NEW)
    cy, ch = 1.74, 0.92
    boxes = ((2.9, 1.22, NEW, f"{S}-Pace"), (4.24, 1.22, NEW, f"{S}-Select"), (5.58, 1.08, NEW, f"{S}-Notify"))
    for x, w, c, name in boxes:
        _rbox(ax, x, cy, w, ch, ec=c, fc="white", lw=1.2)
        ax.text(x + w / 2, cy + ch - 0.05, name, ha="center", va="top", fontsize=TT, weight="bold", color=c)
        _flow(ax, (x + w / 2, by), (x + w / 2, cy + ch), "ctl", c=NEW)

    # Pace mini-plot: c starts at c_max, falls toward ½ pool as φ falls, then follows the pool
    px, py, pw, ph = 3.0, 1.95, 0.8, 0.4
    pool = [0.8, 0.78, 0.74, 0.66, 0.68, 0.58, 0.46, 0.42, 0.5, 0.58, 0.56]
    ts = [px + pw * i / 10 for i in range(11)]
    phi = [max(0.0, 1 - i / 5) for i in range(11)]
    cc = [f * 1.0 + (1 - f) * 0.5 * p for f, p in zip(phi, pool)]
    ax.step(ts, [py + ph * p for p in pool], where="post", color=MUTEDK, lw=1.2, zorder=4)
    ax.plot(ts, [py + ph * v for v in cc], color=NEW, lw=2.0, zorder=5)
    ax.plot([px, px + pw], [py, py], color=INK, lw=0.6)
    ax.text(px + pw + 0.02, py + ph * 0.7, "Pool", ha="left", va="center", fontsize=F, color=MUTEDK)
    ax.text(px + pw + 0.02, py + ph * 0.28, "$c$", ha="left", va="center", fontsize=F, color=NEW, weight="bold")
    ax.text(px + 0.05, py + ph * 1.0, "$c_{\\rm max}$", ha="left", va="bottom", fontsize=F, color=NEW)
    ax.text(2.9 + 0.61, cy + 0.04, "How many: $c \\to$ pool / 2", ha="center", va="bottom", fontsize=F)

    # Select mini-plot: pick probability over eligible clients
    sx, sy = 4.4, 1.98
    for k, v in enumerate((0.35, 0.9, 0.55, 0.2, 0.7, 0.4)):
        ax.add_patch(Rectangle((sx + 0.15 * k, sy), 0.1, 0.4 * v, fc=NEW if k == 1 else BUSY, ec="none", zorder=3))
    ax.plot([sx - 0.03, sx + 0.9], [sy, sy], color=INK, lw=0.6)
    ax.text(sx + 0.27, sy + 0.33, "Pick", ha="left", va="center", fontsize=F, color=NEW, weight="bold")
    ax.text(4.24 + 0.61, cy + 0.04, "Whom: $p \\propto |D|\\cdot q$", ha="center", va="bottom", fontsize=F)

    # Evict mini-plot: slot time lost per departure
    ex = 5.66
    ax.text(ex, 2.37, "Lost per departure", ha="left", va="center", fontsize=F, color=MUTEDK)
    ax.add_patch(Rectangle((ex - 0.02, 2.12), 0.98, 0.12, fc=BUSY, ec="none", zorder=3))
    ax.text(ex + 0.01, 2.18, "Timeout: 6-21 min", ha="left", va="center", fontsize=F, zorder=4)
    ax.add_patch(Rectangle((ex, 1.94), 0.04, 0.12, fc=NEW, ec="none", zorder=3))
    ax.text(ex + 0.07, 2.0, f"{S}: $\\leq$49 s", ha="left", va="center", fontsize=F, color=NEW, weight="bold")
    ax.text(5.58 + 0.54, cy + 0.04, "Evict on report", ha="center", va="bottom", fontsize=F)

    # data plane: slots in flight
    sy0, sh = 1.12, 0.3
    _rbox(ax, 3.5, sy0 - 0.06, 3.14, sh + 0.12, ec=INK, fc="white", lw=0.8, r=0.03)
    ax.text(3.58, sy0 + sh / 2, "$c$ slots\nin flight", ha="left", va="center", fontsize=F, linespacing=1.0)
    slots = ("C", "", "E", "F", "D", "G")
    sxs = [4.18 + 0.4 * k for k in range(6)]
    for x, n in zip(sxs, slots):
        free = n in ("", "D")
        ax.add_patch(Rectangle((x, sy0), 0.32, sh, fc="white" if free else BUSY, ec=INK, lw=0.6, ls=(0, (2, 1.2)) if free else "-", zorder=3))
        if n == "D":
            ax.plot([x + 0.16], [sy0 + sh / 2], "x", color=NEW, ms=7, mew=1.8, zorder=4)
        elif n:
            ax.text(x + 0.16, sy0 + sh / 2, n, ha="center", va="center", fontsize=F, weight="bold", zorder=4)

    # data plane: buffer -> Agg -> version
    _rbox(ax, 1.62, 0.2, 0.9, 0.56, ec=INK, fc="white", lw=0.8)
    ax.text(2.07, 0.715, "Buffer", ha="center", va="top", fontsize=F, weight="bold")
    for k in range(5):
        ax.add_patch(Rectangle((1.72 + 0.14 * k, 0.25), 0.1, 0.14, fc=INK if k < 3 else "white", ec=INK, lw=0.5, zorder=3))
    ax.text(2.07, 0.4, "$K$ updates", ha="center", va="bottom", fontsize=F)
    ax_, aw = 2.82, 1.62
    _rbox(ax, ax_, 0.2, aw, 0.62, ec=NEW, fc="white", lw=1.2)
    ax.text(ax_ + 0.06, 0.77, f"{S}-Agg", ha="left", va="top", fontsize=TT, weight="bold", color=NEW)
    ax.text(ax_ + 0.06, 0.58, "Weight\n$(1{+}\\tau)^{-1/4}$", ha="left", va="top", fontsize=F, linespacing=1.1)
    gx, gy, gw, gh = 3.66, 0.34, 0.4, 0.34
    taus = [i / 2 for i in range(41)]
    for e, c, ls_, lw_ in ((0.25, NEW, "-", 2.0), (0.5, MUTEDK, (0, (2, 1.2)), 1.0)):
        ax.plot([gx + gw * t / 20 for t in taus], [gy + gh * (1 + t) ** -e for t in taus], color=c, ls=ls_, lw=lw_, zorder=4)
    ax.plot([gx, gx + gw], [gy, gy], color=INK, lw=0.6)
    ax.plot([gx, gx], [gy, gy + gh], color=INK, lw=0.6)
    ax.text(gx + gw + 0.03, gy + gh * 0.75, "Mild", ha="left", va="center", fontsize=F, color=NEW, weight="bold")
    ax.text(gx + gw + 0.03, gy + gh * 0.3, "Steep", ha="left", va="center", fontsize=F, color=MUTEDK)
    ax.text(gx + gw / 2, gy - 0.02, "Staleness $\\tau$", ha="center", va="top", fontsize=F)
    _rbox(ax, 4.74, 0.3, 1.9, 0.46, ec=INK, fc="white", lw=0.8)
    ax.text(5.69, 0.53, "Version $v{+}1$ → served model", ha="center", va="center", fontsize=F)
    _flow(ax, (2.52, 0.48), (2.82, 0.48), "data")
    _flow(ax, (4.44, 0.53), (4.74, 0.53), "data")

    # the scenario: badges colored by the mechanism that acts (black = inherited engine)
    _flow(ax, (1.36, dev["A"]), (lx, dev["A"]), "ctl", c=NEW); _badge(ax, 1.49, dev["A"] + 0.12, "1")
    _flow(ax, (1.36, dev["D"]), (lx, dev["D"]), "ctl", c=NEW); _badge(ax, 1.49, dev["D"] + 0.12, "5")
    for x, n, c in ((3.95, "2", NEW), (sxs[1] + 0.16, "3", NEW), (sxs[4] + 0.16, "5", NEW)):
        _flow(ax, (x, cy), (x, sy0 + sh), "ctl", c=c); _badge(ax, x + 0.13, 1.56, n)
    _flow(ax, (3.5, 1.3), (1.42, 1.33), "data"); _badge(ax, 1.9, 1.32, "4")
    _flow(ax, (1.2, dev["C"] - 0.2), (1.8, 0.76), "data"); _badge(ax, 1.6, 0.97, "6")
    _badge(ax, 4.59, 0.67, "7")
    # feedback: losses to q (Select's signal, kept in the ledger) and φ (Pace); c sets τ (Agg)
    _flow(ax, (2.3, 0.76), (2.3, ly), "ctl", c=FB); _badge(ax, 2.3, 1.56, "8")
    ax.text(2.36, 1.15, "Loss → Utility $q$", ha="left", va="center", fontsize=F, color=FB, weight="bold")
    _flow(ax, (3.28, 0.82), (3.28, cy), "ctl", c=FB); _badge(ax, 3.28, 1.56, "8")
    ax.text(3.22, 0.93, "Loss → Phase $\\varphi$", ha="right", va="center", fontsize=F, color=FB, weight="bold")
    _flow(ax, (4.0, sy0 - 0.06), (4.0, 0.82), "ctl", c=FB)
    ax.text(4.07, 0.95, "$c$ → staleness $\\tau \\approx c/K$", ha="left", va="center", fontsize=F, color=FB, weight="bold")
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
