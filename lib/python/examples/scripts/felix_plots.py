#!/usr/bin/env python3
"""FELIX Scoreboard plots, rewritten in place per batch: accuracy vs time, traffic vs time, accuracy vs traffic.

  felix_plots.py POOL... [--prefix F42] [--new newp] [--old ctl] [--no-old] [--also felix:new:"Felix new, no pacing" ...]

One PNG per dataset (`<out>/felix_<dataset>.png`): columns = the batch's cells, lines = Felix new / old, --also arms, every baseline.
"""
import argparse
import bisect
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from accuracy_table import curve, dataset_of, rows_from_pools  # noqa: E402
from felix_claims import traffic  # noqa: E402

CELLS = (("s0", "syn_0 · static"), ("s0u", "syn_0 · stream"), ("s50u", "syn_50_3st · stream"))  # absent cells are dropped
BASE = (("fedbuff", "FedBuff", "#1baf7a", "-"), ("refl", "REFL", "#eda100", "-"), ("feddance", "FedDance", "#e87ba4", "-"),
        ("oort", "Oort", "#008300", "-"), ("oort_star", "Oort*", "#6250d6", "--"))
ALSO = {"refl:rstar": ("REFL*", "#eda100", "--"), "refl:rfc": ("REFL-fc", "#eda100", ":"),
        "felix:new": ("Felix new, no pacing", "#2a78d6", "--"), "felix:l5": ("Felix new, no pacing", "#2a78d6", "--"),
        "felix:fck": ("Felix by check-in", "#2a78d6", ":"), "feddance:fdstar": ("FedDance*", "#e87ba4", "--"),
        "fedbuff:fbstar": ("FedBuff*", "#1baf7a", "--")}  # F42 arms
SMOOTH = {"cifar10": 1, "google_speech": 3}  # speech stream evals swing ±10-20 pp


def leg(d):
    c, _ = curve(d)
    tr = traffic(d)
    ts, cum, s = [], [], 0.0
    for t, b in tr:
        s += b
        ts.append(t / 3600)
        cum.append(s / 1e9)
    gb = lambda h: cum[bisect.bisect_right(ts, h) - 1] if ts and h >= ts[0] else 0.0
    return [(t / 3600, 100 * a, gb(t / 3600)) for t, _, a in c], list(zip(ts[::50], cum[::50]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pools", nargs="+")
    ap.add_argument("--prefix", default="F42")
    ap.add_argument("--new", default="newp")
    ap.add_argument("--old", default="ctl")
    ap.add_argument("--no-old", action="store_true")
    ap.add_argument("--also", nargs="*", default=list(ALSO), help="extra baseline:arm series (known: %s)" % ", ".join(ALSO))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_metadata"))
    a = ap.parse_args(argv)
    series = [("felix", a.new, "Felix new", "#2a78d6", "-", 2.4)]
    if not a.no_old:
        series.append(("felix", a.old, "Felix old", "#eb6834", "-", 2.0))
    series += [(b, "ctl", lab, col, ls, 1.4) for b, lab, col, ls in BASE]
    series += [(*k.split(":"), *ALSO[k], 1.6) for k in a.also if k in ALSO]
    want = {}  # (dataset, cell, baseline, arm) -> sim dir
    for phase, b, _, _, sim in rows_from_pools(a.pools):
        parts = phase.split(f"{a.prefix}_", 1)[-1].split("_")
        if sim and len(parts) == 2:
            want[(dataset_of(sim), parts[0], b, parts[1])] = sim
    with ProcessPoolExecutor(32) as ex:
        data = dict(zip(want, ex.map(leg, want.values())))
    for ds in sorted({k[0] for k in data}):
        cells = [c for c in CELLS if any(k[0] == ds and k[1] == c[0] for k in data)]
        fig, ax = plt.subplots(3, len(cells), figsize=(5.4 * len(cells), 13), sharey="row", squeeze=False)
        for j, (cell, title) in enumerate(cells):
            for b, arm, lab, col, ls, lw in series:
                if (ds, cell, b, arm) not in data:
                    continue
                pts, tot = data[(ds, cell, b, arm)]
                k = SMOOTH.get(ds, 1)
                acc = [sum(p[1] for p in pts[max(0, i - k + 1):i + 1]) / len(pts[max(0, i - k + 1):i + 1]) for i in range(len(pts))]
                ax[0, j].plot([p[0] for p in pts], acc, ls, color=col, lw=lw, label=lab, marker="o", ms=3)
                ax[1, j].plot([t for t, _ in tot], [g for _, g in tot], ls, color=col, lw=lw, label=lab)
                ax[2, j].plot([p[2] for p in pts], acc, ls, color=col, lw=lw, label=lab, marker="o", ms=3)
            ax[0, j].set_title(title, fontsize=12)
            for i, (xl, yl) in enumerate((("hours (sim clock)", "test accuracy %"), ("hours (sim clock)", "model traffic, GB"),
                                           ("model traffic, GB", "test accuracy %"))):
                ax[i, j].set_xlabel(xl, fontsize=9)
                ax[i, 0].set_ylabel(yl, fontsize=10)
                ax[i, j].grid(alpha=0.25, lw=0.6)
                ax[i, j].spines[["top", "right"]].set_visible(False)
        h, l = next(((h, l) for h, l in (x.get_legend_handles_labels() for x in ax[0]) if len(h) == max(
            len(x.get_legend_handles_labels()[0]) for x in ax[0])))
        fig.legend(h, l, loc="upper center", ncol=len(l), fontsize=10, frameon=False, bbox_to_anchor=(0.5, 0.965))
        sm = ", accuracy: 3-eval rolling mean" if SMOOTH.get(ds, 1) > 1 else ""
        fig.suptitle(f"{a.prefix} · {ds} · sims, seed 1{sm}", fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        out = os.path.join(a.out, f"felix_{ds}.png")
        fig.savefig(out, dpi=100)
        print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
