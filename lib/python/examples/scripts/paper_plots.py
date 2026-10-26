#!/usr/bin/env python3
"""FX-N153: the paper's evaluation figures and tables (felix_paper_outline Table 2), one function per row, from leg digests only.

  paper_plots.py [--root R] [--seeds 1] [--smoke] [--only teaser headline ...] [--out DIR]

Legs come from the manifest (paper_runs.py); a panel with missing legs is drawn and labelled MISSING (count), never dropped; any
figure not from final data (smoke, one seed) is labelled PRELIM. Figures -> paper_felix/figs/expts/ (PDF + PNG), tables -> .md.
"""
import argparse
import collections
import os
import statistics as st
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "paper_felix")]
import paper_runs as pr  # noqa: E402
from figlib import COL_W, FULL_W, LINE, ROW_H, STACK, plt, status  # noqa: E402

OUT = os.path.join(HERE, "..", "paper_felix", "figs", "expts")
STACKS = {("fx", "felix"): "Felix", ("ctl_srv", "fedbuff"): "FedBuff", ("fbstar_srv", "fedbuff"): "FedBuff*",
          ("sync_srv", "refl"): "REFL", ("sync_srv", "feddance"): "FedDance", ("sync_srv", "fedavg"): "FedAvg",
          ("sync_srv", "oort"): "Oort", ("sync_srv", "oort_star"): "Oort*", ("rstar_srv", "refl"): "REFL*",
          ("fdstar_srv", "feddance"): "FedDance*", ("favstar_srv", "fedavg"): "FedAvg*"}
DEPLOY, ORACLE = ("FedBuff", "Oort", "REFL", "FedDance", "FedAvg"), ("FedBuff*", "Oort*", "REFL*", "FedDance*", "FedAvg*")
SYNC = ("Oort", "REFL", "FedDance", "FedAvg", "Oort*", "REFL*", "FedDance*", "FedAvg*")
ARMS = {"b1": "+ Notify", "uo_o": "+ Select", "fn": "+ Agg (c 60)", "fx_cfix": "c ½ fixed", "fx_ca": "c ∝ eligible",
        "fx_ck": "check-in", "fx_or": "oracle", "fx_rold": "1/√(1+τ)", "fx_r1": "1/(1+τ)", "fx_rnd": "uniform", "fx_oort": "Oort's",
        "fx_ucb": "+ UCB", "fx_e2": "2 epochs"}
CELL = {"s0": "syn_0 static", "s0u": "syn_0", "s20u": "syn_20", "s50u": "syn_50", "smu": "MobiPerf"}
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
        self.prelim = smoke or len(seeds) < 3

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
    tgt = next((d["config"]["target"] for n in names for d in D.stack(ds, cell, n)), None)
    if tgt:
        ax.axhline(100 * float(tgt), color="#6b6a64", lw=LINE["ref"], ls=":")
    ax.set_xlabel("hours" if x == "t" else "GB")
    ax.set_ylabel("accuracy %")
    ax.set_title(f"{DSN[ds]} {CELL[cell]}")
    mark(ax, miss, D.prelim)


# ---------------------------------------------------------------- End to end
def teaser(D):
    fig, axs = plt.subplots(1, 2, figsize=(COL_W, ROW_H))
    for ax, x in zip(axs, ("t", "gb")):
        curve_panel(ax, D, "cifar10", D.cells(["smu"])[0], ("Felix", *DEPLOY), x)
    axs[1].legend(loc="lower right", ncol=2)
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
    cells = D.cells(pr.ALL)
    fig, axs = plt.subplots(len(pr.DS), len(cells), figsize=(FULL_W, ROW_H * 2), squeeze=False)
    for i, ds in enumerate(pr.DS):
        for j, c in enumerate(cells):
            curve_panel(axs[i][j], D, ds, c, ("Felix", *DEPLOY, *ORACLE))
    axs[0][-1].legend(fontsize=5, ncol=2)
    return save(fig, "curves")


# ---------------------------------------------------------------- why Felix wins
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


def _bars(ax, labels, parts, colors, names):
    bottom = [0.0] * len(labels)
    for part, col, nm in zip(parts, colors, names):
        ax.bar(range(len(labels)), part, bottom=bottom, color=col, label=nm, width=0.7)
        bottom = [b + p for b, p in zip(bottom, part)]
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")


def slots(D):
    panels = [("cifar10", "s50u"), ("cifar10", "smu"), ("google_speech", "smu")]
    panels = [(ds, c) for ds, c in panels if not D.smoke or c == "smu"]
    fig, axs = plt.subplots(1, len(panels), figsize=(FULL_W, ROW_H * 1.3), squeeze=False)
    names = ("Felix", "FedBuff", "FedBuff*", "REFL", "FedDance", "FedAvg", "Oort")
    for ax, (ds, c) in zip(axs[0], panels):
        lab, busy, held, idle, miss = [], [], [], [], 0
        for n in names:
            legs = D.stack(ds, c, n)
            if not legs:
                miss += 1
                continue
            s = legs[0]["slots"]
            tot = max(s["capacity_s"] or 0.0, s["busy_s"] + s["held_departed_s"]) or 1.0  # ramp: in flight can briefly exceed c
            lab.append(n)
            busy.append((s["compute_s"]) / tot)
            held.append(s["held_departed_s"] / tot)
            idle.append(((s["idle_s"] or 0.0) + s["wait_s"]) / tot)
        _bars(ax, lab, [busy, held, idle], ["#2a78d6", "#e34948", "#b1b0a9"], ["computing", "held by departed", "idle / barrier wait"])
        ax.set_ylabel("share of slot time")
        ax.set_title(f"{DSN[ds]} {CELL[c]}")
        mark(ax, miss, D.prelim)
    axs[0][0].legend(fontsize=5)
    return save(fig, "slots")


def rounds(D):
    fig, axs = plt.subplots(1, 2, figsize=(FULL_W, ROW_H * 1.3), squeeze=False)
    for ax, ds in zip(axs[0], pr.DS):
        c = D.cells(["smu"])[0]
        lab, comp, comm, wait, miss = [], [], [], [], 0
        for n in SYNC:
            legs = D.stack(ds, c, n)
            rr = (legs[0].get("rounds") or {}).get("rows") if legs else None
            if not rr:
                miss += 1
                continue
            lab.append(n)
            comp.append(st.median(r[2] for r in rr))
            comm.append(st.median(r[3] for r in rr))
            wait.append(max(0.0, st.median(r[1] - r[0] for r in rr) - comp[-1] - comm[-1]))
        _bars(ax, lab, [comp, comm, wait], ["#2a78d6", "#eda100", "#b1b0a9"], ["slowest: compute", "slowest: transfer", "other wait"])
        ax.set_ylabel("median round (s)")
        ax.set_title(f"{DSN[ds]} {CELL[c]}")
        mark(ax, miss, D.prelim)
    axs[0][0].legend(fontsize=5)
    return save(fig, "rounds")


def staleness(D):
    cells = D.cells(["s50u", "smu"])
    fig, axs = plt.subplots(1, len(cells), figsize=(COL_W, ROW_H * 1.2), squeeze=False)
    for ax, c in zip(axs[0], cells):
        lab, rest, held, miss = [], [], [], 0
        for n in ("FedBuff", "FedBuff*", "Felix"):
            legs = D.stack("cifar10", c, n)
            if not legs:
                miss += 1
                continue
            s = legs[0]["staleness"]
            n_all = sum(s["all"].values()) or 1
            lab.append(n)
            held.append(s["sum_withheld"] / n_all)
            rest.append(max(0.0, s["sum_all"] / n_all - held[-1]))
        _bars(ax, lab, [rest, held], ["#2a78d6", "#e34948"], ["in flight (concurrency)", "held across a departure"])
        ax.set_ylabel("mean staleness (versions)")
        ax.set_title(f"CIFAR-10 {CELL[c]}")
        mark(ax, miss, D.prelim)
    axs[0][0].legend(fontsize=5)
    return save(fig, "staleness")


def picks(D):
    cells = [(ds, c) for ds in pr.DS for c in D.cells(pr.hp.FB2_ABL)]
    metrics = (("corr_size", "ρ(picks, |D|)"), ("samples_per_update", "samples / update"), ("coverage", "coverage"),
               ("top20_share", "top-20% share"))
    fig, axs = plt.subplots(len(cells), len(metrics), figsize=(FULL_W, ROW_H * len(cells)), squeeze=False)
    names = ("Felix", "FedBuff", "Oort", "REFL", "FedDance", "FedAvg")
    for i, (ds, c) in enumerate(cells):
        for j, (k, title) in enumerate(metrics):
            ax, vals, lab, miss = axs[i][j], [], [], 0
            for n in names:
                legs = D.stack(ds, c, n)
                if not legs:
                    miss += 1
                    continue
                lab.append(n)
                vals.append(legs[0]["picks"][k] or 0.0)
            ax.bar(range(len(lab)), vals, color=[style(n)["color"] for n in lab])
            ax.set_xticks(range(len(lab)), lab, rotation=45, ha="right")
            ax.set_title(f"{title} · {DSN[ds]} {CELL[c]}", fontsize=6)
            mark(ax, miss, D.prelim)
    return save(fig, "picks")


def concurrency(D):
    panels = [("cifar10", "smu"), ("cifar10", "s50u"), ("google_speech", "smu")]
    panels = [(ds, c) for ds, c in panels if not D.smoke or c == "smu"]
    fig, axs = plt.subplots(1, len(panels), figsize=(FULL_W, ROW_H * 1.4), squeeze=False)
    fig.subplots_adjust(wspace=0.75)
    for ax, (ds, c) in zip(axs[0], panels):
        legs = D.stack(ds, c, "Felix")
        if legs:
            cc = legs[0]["c"]
            h = [t / 3600 for t in cc["t"]]
            ax.plot(h, cc["c"], label="Felix c", **style("Felix"))
            ax.plot(h, cc["eligible"], label="eligible", color="#6b6a64", lw=LINE["muted"])
            fb = D.stack(ds, c, "FedBuff")
            if fb and fb[0]["config"]["c"]:
                ax.axhline(float(fb[0]["config"]["c"]), label="FedBuff c", **style("FedBuff"))
            ax2 = ax.twinx()
            ax2.plot(h, [p if p is not None else float("nan") for p in cc["phi"]], color="#eda100", lw=LINE["muted"], ls="--", label="φ")
            ax2.set_ylim(0, 1.05)
            ax2.set_ylabel("loss phase φ")
        ax.set_xlabel("hours")
        ax.set_ylabel("clients")
        ax.set_title(f"{DSN[ds]} {CELL[c]}")
        mark(ax, 0 if legs else 1, D.prelim)
    axs[0][0].legend(fontsize=5)
    return save(fig, "concurrency")


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
                rows.append([DSN[ds], CELL[c], n, f"{g.get('aggregated', 0):.1f}", f"{g.get('timeout', 0):.1f}", f"{g.get('evicted', 0):.1f}",
                             f"{g.get('late', 0) + g.get('discarded', 0):.1f}", f"{100 * (1 - g.get('aggregated', 0) / tot):.1f}%"])
    return table("wasted", ["dataset", "trace", "stack", "GB aggregated", "timed out", "evicted", "late / discarded", "paid, not used"], rows)


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
    """time and GB to target per arm on the ablation cells: FedBuff and `fx` plus the run set's arms."""
    cells = [(ds, c) for ds in pr.DS for c in D.cells(pr.hp.FB2_ABL)]
    fig, axs = plt.subplots(2, len(cells), figsize=(FULL_W, ROW_H * 2.2), squeeze=False)
    for j, (ds, c) in enumerate(cells):
        series = [("FedBuff", D.stack(ds, c, "FedBuff")), *[(ARMS[a], D.get(ds, c, a, "felix")) for a in arms],
                  *extra, ("Felix", D.stack(ds, c, "Felix"))]
        miss = sum(not legs for _, legs in series)
        for i, key in enumerate(("h", "gb")):
            ax = axs[i][j]
            lab = [n for n, legs in series if legs]
            vals = [at_hit(legs, key) for _, legs in series if legs]
            ends = [mean(d["end"][key] for d in legs) for _, legs in series if legs]
            ax.bar(range(len(lab)), [v if v is not None else e for v, e in zip(vals, ends)],
                   color=["#2a78d6" if n == "Felix" else "#b1b0a9" for n in lab],
                   hatch=None, edgecolor=["#e34948" if v is None else "white" for v in vals])
            ax.set_xticks(range(len(lab)), lab, rotation=45, ha="right")
            ax.set_ylabel("hours to target" if key == "h" else "GB to target")
            if i == 0:
                ax.set_title(f"{DSN[ds]} {CELL[c]}")
            mark(ax, miss, D.prelim)
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
    for ds in pr.DS:
        for c in D.cells(pr.hp.FB2_ABL):
            fx = D.stack(ds, c, "Felix")
            fh, fg = at_hit(fx, "h"), at_hit(fx, "gb")
            for a in ("fx_ucb", "fx_e2", "fx_ck", "fx_rold", "fx_ca"):
                legs = D.get(ds, c, a, "felix")
                h, g = at_hit(legs, "h"), at_hit(legs, "gb")
                rows.append([DSN[ds], CELL[c], ARMS[a], "missing" if not legs else "✗" if h is None else
                             f"{100 * (h / fh - 1):+.0f}%" if fh else "–", "" if not legs or g is None or not fg else f"{100 * (g / fg - 1):+.0f}%"])
    return table("rejected", ["dataset", "trace", "idea", "Δ time to target vs Felix", "Δ GB"], rows)


FIGS = {f.__name__: f for f in (teaser, headline, curves, static, efficiency, slots, rounds, staleness, picks, concurrency, wasted,
                                 knowledge, overhead, ladder, swap_availability, swap_selection, swap_aggregation, swap_concurrency,
                                 rejected)}
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
