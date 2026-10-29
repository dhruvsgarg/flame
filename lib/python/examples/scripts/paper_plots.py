#!/usr/bin/env python3
"""FX-N153: the paper's evaluation figures and tables (felix_paper_outline Table 2), one function per row, from leg digests only.

  paper_plots.py [--root R] [--seeds 1] [--smoke] [--only teaser headline ...] [--out DIR]

Legs come from the manifest (paper_runs.py); a panel with missing legs is drawn and labelled MISSING (count), never dropped; any
figure from smoke data is labelled PRELIM (seed 1 = first version, final until seeds 2-3 land). Figures -> paper_felix/figs/expts/ (PDF + PNG), tables -> .md.
"""
import argparse
import collections
import os
import statistics as st
import sys
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "paper_felix")]
import paper_runs as pr  # noqa: E402
from figlib import COL_W, FONT, FULL_W, INK, LINE, MARK, STACK, Z, label, overlaps, place, plt, status  # noqa: E402

OUT = os.path.join(HERE, "..", "paper_felix", "figs", "expts")
STACKS = {("fx", "felix"): "Felix", ("ctl_srv", "fedbuff"): "FedBuff", ("fbstar_srv", "fedbuff"): "FedBuff*",
          ("sync_srv", "refl"): "REFL", ("sync_srv", "feddance"): "FedDance", ("sync_srv", "fedavg"): "FedAvg",
          ("sync_srv", "oort"): "Oort", ("sync_srv", "oort_star"): "Oort*", ("rstar_srv", "refl"): "REFL*",
          ("fdstar_srv", "feddance"): "FedDance*", ("favstar_srv", "fedavg"): "FedAvg*"}
DEPLOY, ORACLE = ("FedBuff", "Oort", "REFL", "FedDance", "FedAvg"), ("FedBuff*", "Oort*", "REFL*", "FedDance*", "FedAvg*")
SYNC = ("Oort", "REFL", "FedDance", "FedAvg", "Oort*", "REFL*", "FedDance*", "FedAvg*")
ARMS = {"b1": "+ Notify", "uo_o": "+ Select", "fn": "+ Agg (c = 60)", "fx_cfix": "Fixed c = 30", "fx_ca": "c ∝ eligible",
        "fx_ck": "Check-in", "fx_or": "Oracle", "fx_rold": "1/√(1+τ)", "fx_r1": "1/(1+τ)", "fx_rnd": "Uniform", "fx_oort": "Oort's selector",
        "fx_ucb": "+ UCB", "fx_e2": "2 epochs"}
CELL = {"s0": "Syn-0% static", "s0u": "Syn-0%", "s20u": "Syn-20%", "s50u": "Syn-50%", "smu": "MobiPerf"}  # as table_headline.tex
DSN = {"cifar10": "CIFAR-10", "google_speech": "Speech"}


class Data:
    """Digests by (dataset, cell, run-set arm, baseline) for one root and seeds; tracks what each figure lacked."""

    def __init__(self, root, seeds, smoke):
        self.smoke, self.seeds = smoke, seeds
        self.states = pr.status(root, seeds, smoke)
        self.by = collections.defaultdict(list)
        for (rs, ds, lg, s), (st_, _, d) in self.states.items():
            if d is not None and not pr.leg_digest.check(d):
                self.by[(ds, lg[1], lg[2], lg[3])].append(d)
        self.prelim = smoke

    def cells(self, cells):
        return [pr.SMOKE_CELL.get("E2E", "smu")] if self.smoke else list(cells)

    def get(self, ds, cell, arm, b):
        if self.smoke and cell != "s0":
            cell = "smu"
        return self.by.get((ds, cell, arm, b), [])

    def stack(self, ds, cell, name):
        arm, b = next(k for k, v in STACKS.items() if v == name)
        return self.get(ds, cell, arm, b)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return st.mean(xs) if xs else None


def at_hit(ds_, key):
    """Mean over seeds of hit[key] (None when any seed missed: report end instead)."""
    if not ds_ or any(d["hit"] is None for d in ds_):
        return None
    return mean(d["hit"][key] for d in ds_)


def style(name):
    c, ls = STACK.get(name, ("#6b6a64", "-"))
    return dict(color=c, ls=ls, lw=LINE["hero"] if name == "Felix" else LINE["width"])


def mark(ax, missing, prelim):
    if missing:
        status(ax, "MISSING", f"{missing} legs")
    elif prelim:
        status(ax, "PRELIM")


def save(fig, name):
    for t, what in overlaps(fig):
        print(f"  OVERLAP {name}: {t!r} touches {what!r}", flush=True)
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"))
    plt.close(fig)
    return os.path.join(OUT, f"{name}.png")


def table(name, header, rows, note=""):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, f"{name}.md")
    with open(p, "w") as f:
        f.write((f"{note}\n\n" if note else "") + "| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n")
        f.writelines("| " + " | ".join(map(str, r)) + " |\n" for r in rows)
    return p


def fmt_hit(ds_):
    if not ds_:
        return "missing"
    h, gb = at_hit(ds_, "h"), at_hit(ds_, "gb")
    if h is None:
        return f"✗ {100 * mean(d['end']['acc_last'] for d in ds_):.1f}%"
    return f"{h:.1f} h · {gb:.0f} GB"


def curve_panel(ax, D, ds, cell, names, x="t"):
    miss = 0
    for name in names:
        legs = D.stack(ds, cell, name)
        if not legs:
            miss += 1
            continue
        d = legs[0]  # seed 1 curve; seeds add a band later
        xs = [t / 3600 for t in d["acc"]["t"]] if x == "t" else d["acc"]["gb"]
        ax.plot(xs, [100 * a for a in d["acc"]["acc"]], label=name, **style(name))
    tgt = next((d["config"].get("report_target") or d["config"]["target"] for n in names for d in D.stack(ds, cell, n)), None)
    if tgt:
        ax.axhline(100 * float(tgt), color="#6b6a64", lw=LINE["ref"], ls=":")
    ax.set_xlabel("hours" if x == "t" else "GB")
    ax.set_ylabel("accuracy %")
    ax.set_title(f"{DSN[ds]} {CELL[cell]}")
    mark(ax, miss, D.prelim)


# ---------------------------------------------------------------- End to end
def roll(ys, k=5):
    """k-eval trailing mean: the curve the target is scored on (5-eval mean)."""
    return [sum(ys[max(0, i - k + 1):i + 1]) / (i + 1 - max(0, i - k + 1)) for i in range(len(ys))]


def smooth_c(ys, k=31):
    """Centered k-point mean (window shrinks at the ends): display smoothing only, never used for numbers."""
    h = k // 2
    return [sum(ys[max(0, i - h):i + h + 1]) / len(ys[max(0, i - h):i + h + 1]) for i in range(len(ys))]


def first_at(d, level):
    """(h, GB) when d's 5-eval mean first reaches level (fraction), else None."""
    a = d["acc"]
    return next(((t / 3600, g) for t, g, v in zip(a["t"], a["gb"], roll(a["acc"])) if v >= level - 1e-9), None)


def plot_stack(ax, d, n, x, **kw):
    a = d["acc"]
    xs = [t / 3600 for t in a["t"]] if x == "h" else a["gb"]
    hero = n == "Felix"
    ax.plot(xs, smooth_c([100 * v for v in a["acc"]]), color=STACK[n][0], ls=STACK[n][1], label=label(n),
            marker=MARK.get(n.rstrip("*")), markevery=0.14, ms=3.2 if hero else 2.8, mfc="white" if not hero else STACK[n][0],
            mew=0.8, zorder=Z["hero" if hero else "line"], lw=LINE["hero"] if hero else LINE["width"], **kw)


def bf(v, kind):
    """Bold callout number: 'x' -> 1.4×, '%' -> +1.8% (a delta), 'pct' -> 62% (a share); mathtext, so words around stay regular."""
    return {"x": rf"$\mathbf{{{v:.0f}\times}}$" if v >= 10 else rf"$\mathbf{{{v:.1f}\times}}$", "%": rf"$\mathbf{{{v:+.1f}\%}}$", "pct": rf"$\mathbf{{{v:.0f}\%}}$"}[kind]


def gap_arrow(ax, x0, x1, y):
    """Horizontal ours -> baseline arrow at one accuracy level (the ratio is called out in text elsewhere)."""
    ax.annotate("", (x1, y), (x0, y), arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.9, shrinkA=0, shrinkB=0,
                                                       mutation_scale=7), zorder=Z["annot"])


def teaser(D):
    """Fig 1: cifar MobiPerf, accuracy vs time and vs GB; ours vs each baseline at that baseline's peak accuracy."""
    ds, c = "cifar10", D.cells(["smu"])[0]
    S = {n: D.stack(ds, c, n)[0] for n in ("Felix", *DEPLOY) if D.stack(ds, c, n)}
    fig, axs = plt.subplots(1, 2, figsize=(COL_W, 1.6), sharey=True)
    tgt = 100 * float(S["Felix"]["config"]["report_target"])
    for ax, x in zip(axs, ("h", "gb")):
        for n in list(S)[::-1]:  # ours drawn last, on top
            plot_stack(ax, S[n], n, x)
        ax.axhline(tgt, color=INK, lw=LINE["ref"], ls=(0, (3, 2)), zorder=Z["fill"])
    peak = {n: max(roll(d["acc"]["acc"])) for n, d in S.items()}
    sync_best = max((n for n in S if n in SYNC), key=peak.get)
    fx = {n: first_at(S["Felix"], peak[n]) for n in S}  # ours reaching each baseline's peak
    own = {n: first_at(S[n], peak[n]) for n in S}
    kw = dict(fontsize=FONT["annot"])
    a0, a1 = axs
    a0.set(xlim=(0, 120), ylim=(60, 82), xticks=[0, 24, 48, 72, 96, 120], yticks=[60, 65, 70, 75], xlabel="Time (hours)",
           ylabel="Test Accuracy (%)")
    for n in ("FedBuff", sync_best):
        gap_arrow(a0, fx[n][0], own[n][0], 100 * peak[n])
    sp = {n: own[n][0] / fx[n][0] for n in S}  # speedup to n's peak
    gb = own["FedBuff"][1] / fx["FedBuff"][1]
    pp = {n: 100 * (peak["Felix"] - peak[n]) for n in S}
    a1.set_xscale("log")
    a1.set(xlim=(30, 9000), xlabel="Data Moved (GB, log scale)")
    gap_arrow(a1, fx["FedBuff"][1], own["FedBuff"][1], 100 * peak["FedBuff"])
    h, l = a0.get_legend_handles_labels()
    fig.legend(h[::-1], l[::-1], loc="outside upper center", ncol=6, handlelength=1.6, columnspacing=0.6, borderaxespad=0)
    # callouts after the legend: it shrinks the axes
    # callout strip above the target line
    ys = [81.9 - 2.85 * i for i in range(2)]
    place(a0, f"{bf(pp['FedBuff'], '%')}, {bf(sp['FedBuff'], 'x')} faster vs FedBuff", [(0.8, ys[0], "left", "top")], **kw)
    place(a0, f"{bf(pp[sync_best], '%')}, {bf(sp[sync_best], 'x')} faster vs {sync_best}",
          [(0.8, ys[1], "left", "top")], **kw)
    place(a1, f"{bf(gb, 'x')} fewer GB vs FedBuff", [(35, ys[0], "left", "top")], **kw)
    place(a1, f"Target {tgt:.0f}% (only {label('Felix')})", [(35, ys[1], "left", "top")], **kw)
    return save(fig, "teaser")


def headline(D, static=False):
    cells = D.cells(["s0"] if static else pr.ALL) if not static else ["s0"]
    names = ("Felix", *DEPLOY) if static else ("Felix", *DEPLOY, *ORACLE)
    rows = [[DSN[ds], CELL[c], *(fmt_hit(D.stack(ds, c, n)) for n in names)] for ds in pr.DS for c in cells]
    return table("static" if static else "headline", ["dataset", "trace", *names], rows,
                 "time · GB to target (5-eval mean), ✗ = end accuracy; deployable then oracle (*)" + (" PRELIM" if D.prelim else ""))


def static(D):
    return headline(D, static=True)


def curves(D):
    """Appendix: accuracy vs time per cell (full width: 8 cells read together); deployable solid, oracle dashed."""
    cells = D.cells(pr.ALL)
    fig, axs = plt.subplots(len(pr.DS), len(cells), figsize=(FULL_W, 3.1), sharey="row", sharex=True, squeeze=False)
    names = ("Felix", *DEPLOY, *ORACLE)
    for i, ds in enumerate(pr.DS):
        for j, c in enumerate(cells):
            ax = axs[i][j]
            for n in names[::-1]:
                legs = D.stack(ds, c, n)
                if legs:
                    plot_stack(ax, legs[0], n, "h")
            tgt = next((100 * float(d["config"]["report_target"]) for n in names for d in D.stack(ds, c, n)), None)
            if tgt:
                ax.axhline(tgt, color=INK, lw=LINE["ref"], ls=(0, (3, 2)), zorder=Z["fill"])
            ax.set(xlim=(0, 120), xticks=[0, 24, 48, 72, 96, 120])
        axs[i][0].set_ylabel(f"{DSN[ds]}\nTest Accuracy (%)")
        axs[i][0].set_ylim(*((55, 82) if ds == "cifar10" else (30, 92)))
    for ax in axs[-1]:
        ax.set_xlabel("Time (hours)")
    h, l = axs[0][0].get_legend_handles_labels()
    order = [l.index(label(n)) for n in names if label(n) in l]
    fig.legend([h[k] for k in order], [l[k] for k in order], loc="outside upper center", ncol=6, handlelength=1.8, columnspacing=0.8)
    for i, ds in enumerate(pr.DS):
        for j, c in enumerate(cells):
            y1 = axs[i][j].get_ylim()[1]
            place(axs[i][j], CELL[c], [(118, y1 - 0.3, "right", "top"), (118, y1 - 1, "right", "top")], **TXT)
    return save(fig, "curves")


def efficiency(D):
    rows = []
    for ds in pr.DS:
        for c in D.cells(pr.ALL):
            fx = D.stack(ds, c, "Felix")
            fu, fr = at_hit(fx, "updates"), mean(d["updates"]["rate_h"] for d in fx)
            for n in ("Felix", *DEPLOY, *ORACLE):
                legs = D.stack(ds, c, n)
                u, r = at_hit(legs, "updates"), mean(d["updates"]["rate_h"] for d in legs)
                rows.append([DSN[ds], CELL[c], n, "–" if u is None else f"{u:.0f}", "–" if r is None else f"{r:.0f}",
                             "–" if None in (u, fu) else f"{u / fu:.2f}", "–" if None in (r, fr) else f"{fr / r:.2f}"])
    return table("efficiency", ["dataset", "trace", "stack", "updates to target", "updates / h", "× Felix updates", "Felix rate ×"], rows,
                 "speedup = (updates to target ratio) × (updates / h ratio)")


CELLS4 = (("cifar10", "s50u"), ("cifar10", "smu"), ("google_speech", "s50u"), ("google_speech", "smu"))
TXT = dict(fontsize=FONT["annot"])
SOLID, HATCH = dict(lw=0.5, edgecolor=INK), dict(lw=0.5, edgecolor=INK, color="white")
LIGHT = "#cfcfcf"  # de-emphasized fill (context, never a method)


def tag(ds, c):
    return f"{DSN[ds]}, {CELL[c]}"


def rowlabels(ax, names):
    """Stack names as y tick labels, top to bottom; ours bold."""
    ax.set_yticks(range(len(names)), [label(n) for n in names])
    for t, n in zip(ax.get_yticklabels(), names):
        t.set_weight("bold" if n == "Felix" else "normal")
    ax.set_ylim(len(names) - 0.5, -0.5)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)


def hstack(ax, names, parts, styles, labels):
    """Horizontal stacked bars (one row per name); returns each row's total."""
    left = [0.0] * len(names)
    for vals, st_, lab in zip(parts, styles, labels):
        ax.barh(range(len(names)), vals, left=left, height=0.68, label=lab, zorder=Z["bar"], **st_)
        left = [a + b for a, b in zip(left, vals)]
    rowlabels(ax, names)
    return left


def bar_ends(ax, names, totals, texts, pad):
    """Label each bar at its end; widen the x range until every label fits inside the axes."""
    ts = [ax.text(tot + pad, y, t, va="center", ha="left", weight="bold" if n == "Felix" else "normal", **TXT)
          for y, (n, tot, t) in enumerate(zip(names, totals, texts))]
    fig = ax.figure
    for _ in range(6):
        fig.canvas.draw()
        r, inv = fig.canvas.get_renderer(), ax.transData.inverted()
        x1 = max((inv.transform((t.get_window_extent(r).x1, 0))[0] for t in ts if t.get_text()), default=0)
        if x1 <= ax.get_xlim()[1] * 0.98:
            break
        ax.set_xlim(ax.get_xlim()[0], x1 * 1.04)


USED = ("aggregated", "late")  # late = aggregated after an eviction / timeout record (e.g. a withheld upload delivered on return)


def slot_split(d):
    """Slot-hours over the run: useful / wasted compute, held by a departed client, finished but waiting (sync barrier), idle (async)."""
    s, f = d["slots"], d["slots"].get("by_fate") or {}
    comp = lambda keep: sum(v[0] for k, v in f.items() if (k in USED) == keep) / 3600
    occ = s["busy_s"] + s["held_departed_s"]
    free = max(0.0, (s["capacity_s"] or occ) - occ) / 3600
    aw = free * (s.get("await_commit_frac") or 0.0)  # free slot, but every eligible idle client's update awaits its commit
    return {"useful_compute": comp(True), "wasted_compute": comp(False), "held_departed": s["held_departed_s"] / 3600,
            "waiting": s["wait_s"] / 3600, "idle": free - aw, "await_commit": aw}


def lost_per_departure_s(d):
    """Mean slot time between a client leaving mid-task and its slot's release (eviction, timeout or delivery)."""
    n = (d.get("departures") or {}).get("left_mid_task") or 0
    return d["slots"]["held_departed_s"] / n if n else None


STALL = (("useful_compute", "useful compute"), ("wasted_compute", "compute discarded"), ("held_departed", "held by departed client"),
         ("waiting", "finished, waiting (barrier)"), ("idle", "idle (no task)"), ("await_commit", "free, eligible clients await their commit"))
LOSS = (("wasted_compute", "Discarded compute", dict(hatch="//////", **HATCH)), ("held_departed", "Held by departed client", dict(color=INK, **SOLID)),
        ("waiting", "Waiting at round barrier", dict(hatch="......", **HATCH)), ("idle", "Idle slot", dict(color=LIGHT, **SOLID)),
        ("await_commit", "Clients await own commit", dict(hatch="-----", **HATCH)))
SLOT_NAMES = ("Felix", "FedBuff", "FedBuff*", "REFL", "FedDance", "FedAvg", "Oort")


def slots(D, base=False):
    """Share of slot time lost, by cause, per stack; 2 × 2 cells. base: §3 version, baselines only (no FLuX row)."""
    names0 = SLOT_NAMES[1:] if base else SLOT_NAMES
    loss = [(k, {"held_departed": "Held after client left (to timeout)", "idle": "Idle slot (no check-in yet)"}.get(k, lab) if base else lab, st_)
            for k, lab, st_ in LOSS]
    fig, axs = plt.subplots(2, 2, figsize=(COL_W, 2.7 if not base else 2.6), sharex=True, sharey=True)
    for ax, (ds, c) in zip(axs.flat, CELLS4):
        rows = [(n, slot_split(D.stack(ds, c, n)[0])) for n in names0 if D.stack(ds, c, n)]
        names = [n for n, _ in rows]
        parts = [[100 * sp[k] / sum(sp.values()) for _, sp in rows] for k, _, _ in loss]
        tot = hstack(ax, names, parts, [st_ for _, _, st_ in loss], [lab for _, lab, _ in loss])
        bar_ends(ax, names, tot, [f"{t:.0f}%" for t in tot], 2)
        ax.set_xlim(0, 100)
        if base:  # a strip above the first row holds the panel tag
            ax.set_ylim(len(names) - 0.5, -1.5)
        mark(ax, len(names0) - len(names), D.prelim)
    fig.supxlabel("Slot Time Lost (%)", fontsize=FONT["label"])
    h, l = axs[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside upper center", ncol=2, handlelength=1.6, columnspacing=1.0)
    for ax, (ds, c) in zip(axs.flat, CELLS4):
        place(ax, tag(ds, c), [(98, -1.05, "right", "center")] if base else [(98, y, "right", "center") for y in (0, 1, 2)], **TXT)
    return save(fig, "slots_base" if base else "slots")


def slots_base(D):
    return slots(D, base=True)


def stall(D):
    """Slot time each stack loses per client departure (log): FLuX evicts on the report, check-in stacks hold the slot to timeout."""
    rows = []
    for ds, c in [(ds, c) for ds in pr.DS for c in D.cells(["s50u", "smu"])]:
        for n in SLOT_NAMES:
            legs = D.stack(ds, c, n)
            if not legs:
                continue
            d, sp = legs[0], slot_split(legs[0])
            tot = sum(sp.values()) or 1.0
            lost = lost_per_departure_s(d)
            rows.append([DSN[ds], CELL[c], n, f"{d['end']['h']:.0f}", *(f"{sp[k]:.0f}" for k, _ in STALL),
                         f"{100 * (sp['wasted_compute'] + sp['held_departed'] + sp['idle']) / tot:.0f}%",  # await_commit, barrier: not lost
                         f"{100 * (d['departures']['share'] or 0):.1f}%", "–" if lost is None else f"{lost:.0f}",
                         sum(d["fates"]["counts"].get(k, 0) for k in ("timeout", "evicted", "late")),
                         "eviction" if d["config"].get("knowledge") == "notify" else "timeout"])
    table("stall", ["dataset", "trace", "stack", "run h", *(t + " (slot-h)" for _, t in STALL), "lost share",
                    "tasks whose client left", "slot time lost per departure (s)", "tasks released early", "released by"], rows,
          "slot-hours = Σ over tasks; async capacity = c(t) × run time, sync = its picks' task time; a run ends at the target or 120 h")
    fig, ax = plt.subplots(figsize=(COL_W, 1.75))
    style4 = {CELLS4[0]: ("o", INK), CELLS4[1]: ("s", INK), CELLS4[2]: ("o", "white"), CELLS4[3]: ("s", "white")}
    lost, miss = {}, 0
    for k, (ds, c) in enumerate(CELLS4):
        mk, fc = style4[(ds, c)]
        for y, n in enumerate(SLOT_NAMES):
            legs = D.stack(ds, c, n)
            v = lost_per_departure_s(legs[0]) if legs else None
            if v is None:
                miss += 1
                continue
            lost[(ds, c, n)] = v
            ax.plot(max(v, 1.0), y + (k - 1.5) * 0.16, mk, ms=3.6, mfc=fc, mec=INK, mew=0.8, ls="none", zorder=Z["marker"],
                    label=tag(ds, c) if y == 0 else None)
    rowlabels(ax, SLOT_NAMES)
    ax.set_xscale("log")
    ax.set(xlim=(0.7, 4000), xlabel="Slot Time Lost per Client Departure")
    ax.set_xticks([1, 10, 60, 600, 1800], ["≤1 s", "10 s", "1 min", "10 min", "30 min"])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    mark(ax, miss, D.prelim)
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="outside upper center", ncol=2, handlelength=1.0, columnspacing=1.0)
    r = [lost[(ds, c, "FedBuff")] / max(lost[(ds, c, "Felix")], 1.0) for ds, c in CELLS4 if (ds, c, "Felix") in lost and (ds, c, "FedBuff") in lost]
    if r:
        place(ax, f"{label('Felix')}: {bf(min(r), 'x')}–{bf(max(r), 'x')} less\nthan FedBuff", [(1.2, y, "left", "center") for y in (2.0, 3.0, 4.0, 5.0)],
              linespacing=1.1, **TXT)
    return save(fig, "stall")


ROUND = (("comp", "Slowest pick: compute", dict(color=LIGHT, **SOLID)), ("comm", "Slowest pick: transfer", dict(color=INK, **SOLID)),
         ("wait", "Other barrier wait", dict(hatch="//////", **HATCH)), ("gap", "Waiting for check-ins", dict(hatch="......", **HATCH)))
ROUND_NAMES = ("Oort", "Oort*", "REFL", "REFL*", "FedDance", "FedDance*", "FedAvg", "FedAvg*")


def round_parts(rr):
    comp, comm = st.median(r[2] for r in rr), st.median(r[3] for r in rr)
    gap = st.median(b[0] - a[1] for a, b in zip(rr, rr[1:])) if len(rr) > 1 else 0.0
    return {"comp": comp, "comm": comm, "wait": max(0.0, st.median(r[1] - r[0] for r in rr) - comp - comm), "gap": max(0.0, gap)}


def rounds(D):
    """Median sync round by component, deployable vs oracle (*), MobiPerf: the oracle only removes the check-in wait."""
    fig, axs = plt.subplots(1, 2, figsize=(COL_W, 1.9), sharey=True)
    for ax, ds in zip(axs, pr.DS):
        c = D.cells(["smu"])[0]
        rows = {}
        for n in ROUND_NAMES:
            legs = D.stack(ds, c, n)
            rr = (legs[0].get("rounds") or {}).get("rows") if legs else None
            if rr:
                rows[n] = round_parts(rr)
        names = list(rows)
        tot = hstack(ax, names, [[rows[n][k] for n in names] for k, _, _ in ROUND], [st_ for _, _, st_ in ROUND], [l for _, l, _ in ROUND])
        ax.set_xlim(0, max(tot) * 1.45)
        ratio = {n: sum(rows[n].values()) / sum(rows[n + "*"].values()) for n in names if n + "*" in rows}
        bar_ends(ax, names, tot, [bf(ratio[n], "x") + " oracle's" if n in ratio else "" for n in names], max(tot) * 0.03)
        ax.set_xlabel(f"Median Round Time (s)\n{DSN[ds]}, {CELL[c]}")
        mark(ax, len(ROUND_NAMES) - len(names), D.prelim)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside upper center", ncol=2, handlelength=1.6, columnspacing=1.0)
    return save(fig, "rounds")


def staleness(D):
    """Mean staleness per update, split: from concurrency vs from uploads held across a departure; cifar ablation cells."""
    cells = D.cells(["s50u", "smu"])
    names = ("Felix", "FedBuff", "FedBuff*")
    fig, axs = plt.subplots(1, len(cells), figsize=(COL_W, 1.15), sharey=True, squeeze=False)
    for ax, c in zip(axs[0], cells):
        got = [n for n in names if D.stack("cifar10", c, n)]
        rest, held = [], []
        for n in got:
            s = D.stack("cifar10", c, n)[0]["staleness"]
            n_all = sum(s["all"].values()) or 1
            held.append(s["sum_withheld"] / n_all)
            rest.append(max(0.0, s["sum_all"] / n_all - held[-1]))
        tot = hstack(ax, got, [rest, held], [dict(color=LIGHT, **SOLID), dict(color=INK, **SOLID)],
                     ["From concurrency", "Upload withheld while client offline"])
        ax.set_xlim(0, max(tot) * 1.5)
        bar_ends(ax, got, tot, [f"{100 * h_ / t:.0f}% withheld" for h_, t in zip(held, tot)], max(tot) * 0.03)
        ax.set_xlabel(f"Mean Staleness (versions)\nCIFAR-10, {CELL[c]}")
        mark(ax, len(names) - len(got), D.prelim)
    h, l = axs[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside upper center", ncol=2, handlelength=1.6, columnspacing=1.0)
    return save(fig, "staleness")


def picks(D, ds="cifar10"):
    """Async selection, FLuX vs FedBuff per cell of one dataset: correlation of picks with data size, and samples per update (ratio on top)."""
    cells = [(d_, c) for d_, c in CELLS4 if d_ == ds and D.stack(d_, c, "Felix") and D.stack(d_, c, "FedBuff")]
    fig, axs = plt.subplots(1, 2, figsize=(COL_W, 1.45))
    xs, w = range(len(cells)), 0.36
    for ax, key, ylab in ((axs[0], "corr_size", "Picks–Data Size\nCorrelation"), (axs[1], "samples_per_update", "Samples per Update")):
        vals = {n: [D.stack(d_, c, n)[0]["picks"][key] or 0.0 for d_, c in cells] for n in ("Felix", "FedBuff")}
        for k, n in enumerate(("Felix", "FedBuff")):
            ax.bar([x + (k - 0.5) * w for x in xs], vals[n], width=w, color=STACK[n][0], label=label(n), zorder=Z["bar"],
                   edgecolor=STACK[n][0], lw=0.5)
        ax.set_ylabel(ylab)
        if key == "corr_size":
            ax.axhline(0, color=INK, lw=LINE["axis"], zorder=Z["line"])
            ax.set_ylim(-0.8, 0.8)
        else:
            top = max(max(v) for v in vals.values())
            ax.set_ylim(0, top * 1.3)
            for x, a, b in zip(xs, vals["Felix"], vals["FedBuff"]):
                ax.text(x, max(a, b) + top * 0.04, bf(a / b, "x"), ha="center", va="bottom", **TXT)
        ax.set_xticks(list(xs), [CELL[c] for _, c in cells])
        ax.tick_params(axis="x", length=0)
        ax.grid(axis="x", visible=False)
    fig.supxlabel(DSN[ds], fontsize=FONT["label"])
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside upper center", ncol=2)
    return save(fig, "picks" if ds == "cifar10" else "picks_speech")


def picks_speech(D):
    return picks(D, "google_speech")


def concurrency(D, base=False):
    """FLuX's concurrency follows the eligible pool on MobiPerf; FedBuff's fixed c often exceeds it. base: §3 version, no FLuX line."""
    panels = [(ds, D.cells(["smu"])[0]) for ds in pr.DS]
    fig, axs = plt.subplots(1, len(panels), figsize=(COL_W, 1.55))
    over = {}
    for ax, (ds, c) in zip(axs, panels):
        legs = D.stack(ds, c, "Felix")
        if not legs:
            mark(ax, 1, D.prelim)
            continue
        cc = legs[0]["c"]
        h = [t / 3600 for t in cc["t"]]
        el = [e if e is not None else float("nan") for e in cc["eligible"]]
        ax.plot(h, smooth_c(el, 15), color=INK, lw=LINE["muted"], label="Eligible clients", zorder=Z["line"])
        if not base:
            ax.plot(h, smooth_c(cc["c"], 15), color=STACK["Felix"][0], lw=LINE["hero"], label=f"{label('Felix')} concurrency", zorder=Z["hero"])
        fb = D.stack(ds, c, "FedBuff")
        if fb and fb[0]["config"]["c"]:
            cfb = float(fb[0]["config"]["c"])
            ax.axhline(cfb, color=STACK["FedBuff"][0], lw=LINE["width"], label="FedBuff concurrency", zorder=Z["line"])
            ok = [e for e in el if e == e]
            over[ds] = 100 * sum(e < cfb for e in ok) / len(ok)
            ax.set_ylim(0.6 * min(ok if base else [v for v in cc["c"] if v]), max(max(ok), cfb) * 1.4)
        ax.set(xlim=(0, 120), xticks=[0, 24, 48, 72, 96, 120], xlabel=f"Time (hours)\n{DSN[ds]}, {CELL[c]}")
    axs[0].set_ylabel("Clients")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside upper center", ncol=3, handlelength=1.4, columnspacing=0.8)
    for ax, (ds, c) in zip(axs, panels):
        if ds in over:
            y1 = ax.get_ylim()[1]
            place(ax, f"FedBuff > eligible\n{bf(over[ds], 'pct')} of the time", [(2, y1 * f, "left", "top") for f in (0.98, 0.94)]
                  + [(118, y1 * 0.98, "right", "top")], linespacing=1.1, **TXT)
    return save(fig, "concurrency_base" if base else "concurrency")


def concurrency_base(D):
    return concurrency(D, base=True)


def oracle_gain(D):
    """§3.2: what an oracle at selection buys each stack: oracle minus deployable peak accuracy, per trace."""
    names = ("FedBuff", "Oort", "REFL", "FedDance", "FedAvg")
    shapes = dict(zip(pr.ALL, ("o", "^", "s", "D")))
    fig, axs = plt.subplots(1, 2, figsize=(COL_W, 1.65), sharey=True)
    for ax, ds in zip(axs, pr.DS):
        miss = 0
        for y, n in enumerate(names):
            for k, c in enumerate(D.cells(pr.ALL)):
                a, b = D.stack(ds, c, n), D.stack(ds, c, n + "*")
                if not (a and b):
                    miss += 1
                    continue
                g = 100 * (max(roll(b[0]["acc"]["acc"])) - max(roll(a[0]["acc"]["acc"])))
                ax.plot(g, y + (k - 1.5) * 0.15, shapes[c], ms=3.4, mfc="white" if c != "smu" else INK, mec=INK, mew=0.8, ls="none",
                        zorder=Z["marker"], label=CELL[c] if y == 0 and ds == next(iter(pr.DS)) else None)
        ax.axvline(0, color=INK, lw=LINE["axis"], zorder=Z["fill"])
        rowlabels(ax, names)
        ax.set_xlabel(f"Oracle Gain in Peak\nAccuracy (%), {DSN[ds]}")
        mark(ax, miss, D.prelim)
    for ax in axs:
        xs = [x for ln in ax.lines for x in ln.get_xdata()]
        ax.set_xlim(min(min(xs), -2) - 1.5, max(max(xs), 2) + 1.5)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside upper center", ncol=4, handletextpad=0.2, columnspacing=0.8)
    return save(fig, "oracle_gain")


def fill_slots(D):
    """§3.3 bound: FedBuff with every slot refilled the instant a client leaves or becomes eligible (arm b1) vs FedBuff, cifar syn_50."""
    ds, c = "cifar10", "s50u"
    fb, b1 = D.stack(ds, c, "FedBuff"), D.get(ds, c, "b1", "felix")
    fig, axs = plt.subplots(1, 2, figsize=(COL_W, 1.6), sharey=True)
    if not (fb and b1):
        mark(axs[0], int(not fb) + int(not b1), D.prelim)
        return save(fig, "fill_slots")
    fb, b1 = fb[0], b1[0]
    lab_b1 = "FedBuff, every slot refilled"
    for ax, x in zip(axs, ("h", "gb")):
        a = b1["acc"]
        ax.plot([t / 3600 for t in a["t"]] if x == "h" else a["gb"], smooth_c([100 * v for v in a["acc"]]), color=INK, lw=LINE["width"],
                marker="P", markevery=0.14, ms=3.2, mfc="white", mew=0.8, label=lab_b1, zorder=Z["line"])
        plot_stack(ax, fb, "FedBuff", x)
        ax.axhline(80, color=INK, lw=LINE["ref"], ls=(0, (3, 2)), zorder=Z["fill"])
    axs[0].set(xlim=(0, 96), xticks=[0, 24, 48, 72, 96], ylim=(60, 89.5), yticks=[60, 65, 70, 75, 80], xlabel="Time (hours)",
               ylabel="Test Accuracy (%)")
    axs[1].set_xscale("log")
    axs[1].set(xlim=(100, 8000), xlabel="Data Moved (GB, log scale)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h[::-1], l[::-1], loc="outside upper center", ncol=2)
    sb = lambda d: (lambda s_: s_["sum_all"] / (sum(s_["all"].values()) or 1))(d["staleness"])
    place(axs[0], f"Same 80%, with {bf(b1['updates']['rate_h'] / fb['updates']['rate_h'], 'x')} the\n"
          f"updates per hour, {bf(b1['hit']['h'] / fb['hit']['h'], 'x')} slower", [(1, 89.3, "left", "top")], linespacing=1.1, **TXT)
    place(axs[1], f"Staleness {sb(fb):.1f} → {sb(b1):.1f},\n{bf(b1['hit']['gb'] / fb['hit']['gb'], 'x')} the GB",
          [(110, 89.3, "left", "top")], linespacing=1.1, **TXT)
    return save(fig, "fill_slots")


VAR = {"fn": ("P", "-"), "fx_cfix": ("^", "--"), "fx_ca": ("D", ":")}  # FLuX variants: black, marker + dash per arm


def choice_replay(D):
    """§3.4: selection's choice is gone before the slots run out: ρ(picks, size) vs c / eligible, trace replay + measured runs."""
    import json
    cells = (("smu", "mobiperf_3st"), ("s50u", "syn_50_3st"))
    fig, axs = plt.subplots(1, 2, figsize=(COL_W, 1.6), sharey=True)
    kw = dict(ls="none", mew=0.9, zorder=Z["marker"])
    for ax, (c, tr) in zip(axs, cells):
        rp = json.load(open(os.path.join(HERE, "..", "paper_felix", "data", f"choice_replay_{tr}.json")))
        for pol, col, mk, lab in (("size", INK, "o", "Replay: size-weighted picks"), ("uniform", STACK["FedBuff"][0], "s", "Replay: uniform picks")):
            rs = sorted({r["ratio"] for r in rp["rows"]})
            ys = [st.mean(r["corr"] for r in rp["rows"] if r["ratio"] == x and r["policy"] == pol) for x in rs]
            ax.plot(rs, ys, color=col, lw=LINE["width"], label=lab, zorder=Z["line"])
        pool = rp["mean_eligible"]
        for a in ("fx", "fn", "fx_cfix", "fx_ca"):  # measured runs: size × loss selection at four c
            legs = D.stack("cifar10", c, "Felix") if a == "fx" else D.get("cifar10", c, a, "felix")
            if legs:
                x = st.mean(v for v in legs[0]["c"]["c"] if v) / pool
                ax.plot(x, legs[0]["picks"]["corr_size"], "o", ms=4.2, mfc="white", mec=INK,
                        label="Runs: size × loss picks" if a == "fx" else None, **kw)
        fb = D.stack("cifar10", c, "FedBuff")
        if fb:
            ax.plot(float(fb[0]["config"]["c"]) / pool, fb[0]["picks"]["corr_size"], "s", ms=3.6, mfc="white", mec=STACK["FedBuff"][0],
                    label="Runs: FedBuff (uniform)", **kw)
        ax.axhline(0, color=INK, lw=LINE["axis"], zorder=Z["fill"])
        ax.axvline(1, ymax=0.68, color=INK, lw=LINE["ref"], ls=(0, (3, 2)), zorder=Z["fill"])  # c = pool; stops below the callout
        ax.set(xlim=(0, 1.55), xticks=[0, 0.5, 1, 1.5], ylim=(-0.75, 1.0), xlabel=f"Concurrency / Eligible Pool\nCIFAR-10, {CELL[c]}")
        ax._rp = rp
    axs[0].set_ylabel("Picks–Data Size\nCorrelation")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside upper center", ncol=2, columnspacing=0.8, handlelength=1.4)
    for ax in axs:
        rows = ax._rp["rows"]
        at = lambda x, pol: st.mean(r["corr"] for r in rows if r["ratio"] == x and r["policy"] == pol)
        g0, g1 = (round(at(x, "size") - at(x, "uniform"), 2) + 0.0 for x in (0.2, 1.0))
        place(ax, f"Gap over uniform:\n{g0:+.2f} at 0.2, {g1:+.2f} at 1".replace("-", "−").replace("+0.00", "0.00"),
              [(1.52, 0.97, "right", "top"), (0.05, 0.97, "left", "top"), (0.05, -0.7, "left", "bottom")], linespacing=1.1, **TXT)
    return save(fig, "choice_replay")


def pace_levels(D):
    """§5.3: no fixed c is best at every accuracy level: time and GB to each level per concurrency arm, cifar MobiPerf + syn_50."""
    cells = ("smu", "s50u")
    arms = (("Felix", None), ("fn", "Fixed c = 60"), ("fx_cfix", "Fixed c = 30"), ("fx_ca", "c ∝ eligible"), ("FedBuff", None))
    fig, axs = plt.subplots(2, 2, figsize=(COL_W, 2.75), sharex="row")
    best = {}
    for i, c in enumerate(cells):
        tgt = 100 * float(D.stack("cifar10", c, "Felix")[0]["config"]["report_target"])
        marks = [50, 60, 65, 70, 75, 80]
        for a, lab in arms[::-1]:
            legs = D.stack("cifar10", c, a) if a in ("Felix", "FedBuff") else D.get("cifar10", c, a, "felix")
            if not legs:
                continue
            d = legs[0]
            pk = 100 * max(roll(d["acc"]["acc"]))
            xs = [x for x in np.arange(50, pk + 1e-6, 0.25)] + [pk]
            hit = [first_at(d, x / 100) for x in xs]
            me = [k for k, x in enumerate(xs) if any(abs(x - m) < 0.13 for m in marks)] + [len(xs) - 1]
            for j in (0, 1):
                ys = [hh[j] for hh in hit]
                if a in ("Felix", "FedBuff"):
                    hero = a == "Felix"
                    st_ = dict(color=STACK[a][0], lw=LINE["hero"] if hero else LINE["width"], marker=MARK[a], mfc=STACK[a][0] if hero else "white",
                               zorder=Z["hero" if hero else "line"])
                else:
                    st_ = dict(color=INK, lw=LINE["width"], ls=VAR[a][1], marker=VAR[a][0], mfc="white", zorder=Z["line"])
                axs[i][j].plot(xs, ys, markevery=me, ms=3.0, mew=0.8, label=label(lab or a), **st_)
            for m in marks:
                v = first_at(d, m / 100)
                if v and a != "FedBuff":
                    best.setdefault((c, m), []).append((v[0], lab or label(a)))
        for j in (0, 1):
            ax = axs[i][j]
            ax.set_yscale("log")
            ax.set_xlim(49, 81)
            ax.set_xticks([50, 60, 70, 80])
            if j == 0:
                ax.set_yticks([1, 10, 100], ["1", "10", "100"])
                ax.set_ylabel(f"Time to Level (hours)\nCIFAR-10, {CELL[c]}")
            else:
                ax.set_yticks([100, 1000], ["100", "1k"])
                ax.set_ylabel("Data to Level (GB)")
            ax.yaxis.set_minor_locator(plt.NullLocator())
            y0, y1 = ax.get_ylim()
            k = 12 if j == 0 else 1.6  # a strip above the time curves for the callout; the target line stops below it
            ax.set_ylim(y0, y1 * k)
            ax.axvline(tgt, ymax=np.log(y1 / y0) / np.log(k * y1 / y0), color=INK, lw=LINE["ref"], ls=(0, (3, 2)), zorder=Z["fill"])
    for ax in axs[-1]:
        ax.set_xlabel("Accuracy Level (%)")
    h, l = axs[0][0].get_legend_handles_labels()
    order = [l.index(x) for x in (label("Felix"), "Fixed c = 60", "Fixed c = 30", "c ∝ eligible", "FedBuff") if x in l]
    fig.legend([h[k] for k in order], [l[k] for k in order], loc="outside upper center", ncol=3, handlelength=1.8, columnspacing=0.8)
    for i, c in enumerate(cells):
        tgt = round(100 * float(D.stack("cifar10", c, "Felix")[0]["config"]["report_target"]))
        first = lambda m: ", ".join(sorted((n for t, n in best[(c, m)] if t <= 1.02 * min(best[(c, m)])[0]), key=lambda n: n != label("Felix"))
                                    ).replace("Fixed c", "c")
        y1 = axs[i][0].get_ylim()[1]
        place(axs[i][0], f"Fastest by level:\n60%: {first(60)}\n{tgt}%: {first(tgt)}", [(49.5, y1 * 0.95, "left", "top")],
              linespacing=1.1, **TXT)
    return save(fig, "pace_levels")


def pace_bytes(D):
    """§5.3: FLuX vs FLuX with fixed c = 60 (`fn`), accuracy vs GB: the ramp saves bytes on MobiPerf, a no-op on syn_50."""
    cells = D.cells(pr.hp.FB2_ABL)
    fig, axs = plt.subplots(1, len(cells), figsize=(COL_W, 1.6))
    for ax, c in zip(axs, cells):
        fx, fn = D.stack("cifar10", c, "Felix"), D.get("cifar10", c, "fn", "felix")
        if not (fx and fn):
            mark(ax, int(not fx) + int(not fn), D.prelim)
            continue
        fx, fn = fx[0], fn[0]
        tgt = 100 * float(fx["config"]["report_target"])
        a = fn["acc"]
        ax.plot(a["gb"], smooth_c([100 * v for v in a["acc"]]), color=INK, lw=LINE["width"], marker="P", markevery=0.14, ms=3.2,
                mfc="white", mew=0.8, label=f"{label('Felix')}, fixed c = 60", zorder=Z["line"])
        plot_stack(ax, fx, "Felix", "gb")
        ax.axhline(tgt, color=INK, lw=LINE["ref"], ls=(0, (3, 2)), zorder=Z["fill"])
        ax.set_xscale("log")
        ax.set(xlim=(100, 8000), ylim=(tgt - 15, tgt + 4.5), xlabel=f"Data Moved (GB, log scale)\nCIFAR-10, {CELL[c]}")
        if fx["hit"] and fn["hit"]:
            rg, rt = fn["hit"]["gb"] / fx["hit"]["gb"], fn["hit"]["h"] / fx["hit"]["h"]
            same = abs(rg - 1) < 0.05 and abs(rt - 1) < 0.05  # pool ≫ c_max: the ramp never engages
            ax._callout = ("Same GB and time:\npool stays above c" if same else
                           f"{bf(rg, 'x')} fewer GB,\n{bf(rt, 'x')} faster", tgt)
    axs[0].set_ylabel("Test Accuracy (%)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h[::-1], l[::-1], loc="outside upper center", ncol=2)
    for ax in axs:
        if hasattr(ax, "_callout"):
            t, tgt = ax._callout
            place(ax, t, [(110, tgt + 4.3, "left", "top"), (110, tgt - 1, "left", "top")], linespacing=1.1, **TXT)
    return save(fig, "pace_bytes")


def wasted(D):
    rows = []
    for ds in pr.DS:
        for c in D.cells(pr.hp.FB2_ABL):
            for n in ("Felix", *DEPLOY, *ORACLE):
                legs = D.stack(ds, c, n)
                if not legs:
                    rows.append([DSN[ds], CELL[c], n, "missing", "", "", "", ""])
                    continue
                g = legs[0]["fates"]["gb"]
                tot = sum(g.values()) or 1.0
                used = sum(g.get(k, 0) for k in USED)
                rows.append([DSN[ds], CELL[c], n, f"{used:.1f}", f"{g.get('timeout', 0):.1f}", f"{g.get('evicted', 0):.1f}",
                             f"{g.get('discarded', 0):.1f}", f"{100 * (1 - used / tot):.1f}%"])
    return table("wasted", ["dataset", "trace", "stack", "GB aggregated (incl. late)", "timed out", "evicted", "discarded", "paid, not used"], rows)


def knowledge(D):
    rows = []
    for ds in pr.DS:
        for c in D.cells(["smu", "s50u"]):
            for n in ("Felix", *DEPLOY, *ORACLE):
                legs = D.stack(ds, c, n)
                if not legs:
                    rows.append([DSN[ds], CELL[c], n, "missing", ""])
                    continue
                dp = legs[0]["departures"]
                lag = dp["lag_s"]["p50"]
                rows.append([DSN[ds], CELL[c], n, f"{100 * (dp['share'] or 0):.2f}%", "–" if lag is None else f"{lag:.0f} s"])
    return table("knowledge", ["dataset", "trace", "stack", "dispatches to clients leaving mid-task", "detection lag p50"], rows,
                 "forecast AUC rows: paper_felix/data/fx_n147_*.json (FX-N147)")


def overhead(D):
    rows = []
    for ds in pr.DS:
        for c in D.cells(pr.ALL):
            for n in ("Felix", "FedBuff", "REFL", "FedDance"):
                legs = D.stack(ds, c, n)
                if not legs:
                    continue
                o, ch = legs[0]["overhead"], max(legs[0]["overhead"]["client_h"], 1e-9)
                reports = o["trace_flips"] if n == "Felix" else o["checkins"]
                rows.append([DSN[ds], CELL[c], n, f"{reports / ch:.2f}", f"{o['wire_msgs'] / ch:.1f}",
                             f"{1e3 * o.get('commit_s_mean', 0):.1f} ms"])
    return table("overhead", ["dataset", "trace", "stack", "reports / client-h", "model msgs / client-h", "server commit"], rows,
                 "reports = Felix's eligibility notifies (one per flip) or check-ins")


# ---------------------------------------------------------------- ablations
def _arm_bars(D, name, arms, extra=()):
    """Peak accuracy (dots), then time and GB to target (bars) per arm on the cifar ablation cells; bar-end ratio vs FLuX, misses hatched."""
    cells = [(ds, c) for ds in pr.ABL_DS for c in D.cells(pr.hp.FB2_ABL)]
    series0 = lambda ds, c: [("FedBuff", D.stack(ds, c, "FedBuff")), *[(ARMS[a], D.get(ds, c, a, "felix")) for a in arms], *extra,
                             ("Felix", D.stack(ds, c, "Felix"))]
    n_rows = len(series0(*cells[0]))
    fig, axs = plt.subplots(len(cells), 3, figsize=(COL_W, 0.6 + 0.21 * n_rows * len(cells)), sharey=True, sharex="col", squeeze=False)
    for i, (ds, c) in enumerate(cells):
        series = [(n, legs) for n, legs in series0(ds, c)]
        fx = series[-1][1]
        names = [n for n, _ in series]
        ax = axs[i][0]  # accuracy first (success-metric hierarchy): peak 5-eval mean, comparable where runs go the full 120 h
        tgt = 100 * float(fx[0]["config"]["report_target"]) if fx else None
        peak = {n: 100 * max(max(roll(d["acc"]["acc"])) for d in legs) for n, legs in series if legs}
        for y, n in enumerate(names):
            if n in peak:
                col = STACK[n][0] if n in ("Felix", "FedBuff") else INK
                ax.plot(peak[n], y, MARK.get(n, "o"), ms=3.6, mfc=col if n == "Felix" else "white", mec=col, mew=0.9, ls="none",
                        zorder=Z["marker"])
        if tgt:
            ax.axvline(tgt, color=INK, lw=LINE["ref"], ls=(0, (3, 2)), zorder=Z["fill"])
        rowlabels(ax, [label(n) for n in names])
        ax.grid(axis="y", visible=False)
        full = fx and fx[0]["end"]["h"] >= 119  # runs stopped at the target tie on accuracy: no deltas
        ax._acc = (names, peak, full, tgt)
        if i == len(cells) - 1:
            ax.set_xlabel("Peak\nAccuracy (%)")
        for j, key in enumerate(("h", "gb"), 1):
            ax = axs[i][j]
            vals, txt, styles = [], [], []
            for n, legs in series:
                v = at_hit(legs, key) if legs else None
                ref = at_hit(fx, key)
                if v is None:  # missed the target (or missing): bar to the run's end, hatched
                    vals.append(mean(d["end"][key] for d in legs) if legs else 0.0)
                    txt.append("Missing" if not legs else "Missed")
                    styles.append("miss")
                else:
                    vals.append(v)
                    txt.append("" if n == "Felix" or ref is None else bf(v / ref, "x"))
                    styles.append(n)
            for y, (v, sty) in enumerate(zip(vals, styles)):
                col = STACK[sty][0] if sty in ("Felix", "FedBuff") else LIGHT
                ax.barh(y, v, height=0.68, zorder=Z["bar"], **(dict(hatch="//////", **HATCH) if sty == "miss" else dict(color=col, **SOLID)))
            rowlabels(ax, [label(n) for n in names])
            top = max(vals) if vals else 1.0
            ax.set_xlim(0, top * 1.15)
            bar_ends(ax, names, vals, txt, top * 0.03)
            ax.set_xticks([0, 60, 120] if key == "h" else [0, 2500, 5000], None if key == "h" else ["0", "2.5k", "5k"])
            if i == len(cells) - 1:
                ax.set_xlabel("Time to\nTarget (hours)" if key == "h" else "Data to\nTarget (GB)")
            mark(ax, sum(not legs for _, legs in series), D.prelim)
        axs[i][0].set_ylabel(tag(ds, c))
    pk = [v for row in axs for v in row[0]._acc[1].values()] + [row[0]._acc[3] for row in axs if row[0]._acc[3]]
    lo, hi = min(pk), max(pk)
    for row in axs:  # delta labels after the shared x range is final
        ax = row[0]
        names, peak, full, tgt = ax._acc
        ax.set_xlim(lo - 0.6, hi + 0.25 * (hi - lo) + 1.2)
        if not full:
            place(ax, f"All reach\n{tgt:.0f}%", [(lo - 0.4, len(names) - 1.5, "left", "center"), (lo - 0.4, 0.5, "left", "center")],
                  linespacing=1.0, **TXT)
            continue
        for y, n in enumerate(names):
            if n in peak:
                t = f"{peak[n]:.1f}%" if n == "Felix" else f"{peak[n] - peak['Felix']:+.1f}%".replace("-", "−")
                place(ax, t, [(peak[n] + 0.35, y, "left", "center"), (peak[n] - 0.35, y, "right", "center")],
                      weight="bold" if n == "Felix" else "normal", **TXT)
    return save(fig, name)


def ladder(D):
    return _arm_bars(D, "ladder", ("b1", "uo_o", "fn"))


def swap_availability(D):
    return _arm_bars(D, "swap_availability", ("fx_ck", "fx_or"))


def swap_selection(D):
    return _arm_bars(D, "swap_selection", ("fx_rnd", "fx_oort"))


def swap_aggregation(D):
    return _arm_bars(D, "swap_aggregation", ("fx_rold", "fx_r1"))


def swap_concurrency(D):
    return _arm_bars(D, "swap_concurrency", ("fn", "fx_cfix", "fx_ca"))


def rejected(D):
    rows = []
    for ds in pr.ABL_DS:
        for c in D.cells(pr.hp.FB2_ABL):
            fx = D.stack(ds, c, "Felix")
            fh, fg = at_hit(fx, "h"), at_hit(fx, "gb")
            for a in ("fx_ucb", "fx_e2", "fx_ck", "fx_rold", "fx_ca"):
                legs = D.get(ds, c, a, "felix")
                h, g = at_hit(legs, "h"), at_hit(legs, "gb")
                rows.append([DSN[ds], CELL[c], ARMS[a], "missing" if not legs else "✗" if h is None else
                             f"{100 * (h / fh - 1):+.0f}%" if fh else "–", "" if not legs or g is None or not fg else f"{100 * (g / fg - 1):+.0f}%"])
    return table("rejected", ["dataset", "trace", "idea", "Δ time to target vs Felix", "Δ GB"], rows)


FIGS = {f.__name__: f for f in (teaser, headline, curves, static, efficiency, slots, slots_base, stall, rounds, staleness, picks, picks_speech,
                                 concurrency, concurrency_base, wasted, knowledge, overhead, ladder, swap_availability, swap_selection,
                                 swap_aggregation, swap_concurrency, rejected, oracle_gain, fill_slots, choice_replay, pace_bytes, pace_levels)}
assert set(FIGS) == set(pr.FIGURES), set(FIGS) ^ set(pr.FIGURES)


def main(argv=None):
    global OUT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root")
    ap.add_argument("--seeds", nargs="+", type=int, default=[1])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--only", nargs="*", choices=list(FIGS))
    ap.add_argument("--out", help="output dir (default paper_felix/figs/expts)")
    a = ap.parse_args(argv)
    OUT = a.out or OUT
    D = Data(os.path.abspath(a.root or pr.default_root(a.smoke)), a.seeds, a.smoke)
    bad = 0
    for name in a.only or FIGS:
        try:
            print(f"{name}: {FIGS[name](D)}", flush=True)
        except Exception:  # one broken figure must not stop the rest
            bad += 1
            print(f"{name}: FAILED\n{traceback.format_exc()}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
