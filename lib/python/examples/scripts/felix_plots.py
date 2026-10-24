#!/usr/bin/env python3
"""FELIX Scoreboard plots, rewritten in place per batch: Felix arms vs time, baselines vs time (Felix old as reference), accuracy vs traffic.

  felix_plots.py POOL... [--prefix F42] [--new newp] [--old ctl] [--no-old] [--also fedbuff:fbstar ...]

One PNG per dataset (`<out>/felix_<dataset>.png`): columns = the batch's cells; y zoomed to the post-warm-up band; lines labelled at their end.
"""
import argparse
import bisect
import os
import re
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
ALSO = {"refl:rstar": ("REFL*", "#eda100", "--"), "felix:fck": ("Felix by check-in", "#2a78d6", ":"),
        "feddance:fdstar": ("FedDance*", "#e87ba4", "--"), "fedbuff:fbstar": ("FedBuff*", "#1baf7a", "--"),
        "felix:l3b_srv:raw": ("Felix new, unserved", "#2a78d6", ":"), "felix:l3b_e3_srv": ("Felix new, 3 epochs", "#2a78d6", "--"),
        "fedavg:ctl": ("FedAvg", "#9c6b30", ":")}  # F42 / F43 arms
SMOOTH = {"cifar10": 3, "google_speech": 5}  # evals swing ±5 pp cifar, ±10-20 pp speech
WARM_H = 2.0  # y range starts from the lowest smoothed accuracy after this


def leg(d):
    if d.endswith(":raw"):  # FX-N128: a serveEma leg's unserved model
        from bench_report import leg as bench_leg
        return [(t / 3600, 100 * a, b / 1e9) for t, a, b in bench_leg(d[:-4], b'"agg_eval_raw"')[0]], []
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
    ap.add_argument("--base-arm", nargs="*", default=[], help="baseline:arm, e.g. fedbuff:ctl_srv:raw (F42f)")
    ap.add_argument("--basis", choices=("avg", "cur"), help="FX-N133: every series from its `_srv` leg's averaged or current model")
    a = ap.parse_args(argv)
    a.base_arm = dict(x.split(":", 1) for x in a.base_arm)
    series = [("felix", a.new, "Felix new", "#2a78d6", "-", 2.4)]
    if not a.no_old:
        series.append(("felix", a.old, "Felix old", "#eb6834", "-", 2.0))
    series += [(b, a.base_arm.get(b, "ctl"), lab, col, ls, 1.4) for b, lab, col, ls in BASE]
    series += [(*k.split(":", 1), *ALSO[k], 1.6) for k in a.also if k in ALSO]
    if a.basis:
        on = lambda arm: re.sub(r"_srv(:raw)?$", "", arm) + ("_srv" if a.basis == "avg" else "_srv:raw")
        a.old, seen = on(a.old), set()
        series = [s_ for s_ in ((b, on(arm), *r) for b, arm, *r in series) if s_[:2] not in seen and not seen.add(s_[:2])]
    want = {}  # (dataset, cell, baseline, arm) -> sim dir
    for phase, b, _, _, sim in rows_from_pools(a.pools):
        parts = phase.split(f"{a.prefix}_", 1)[-1].split("_", 1)
        if sim and len(parts) == 2:
            want[(dataset_of(sim), parts[0], b, parts[1])] = sim
            if parts[1].endswith("_srv"):
                want[(dataset_of(sim), parts[0], b, parts[1] + ":raw")] = sim + ":raw"
    with ProcessPoolExecutor(32) as ex:
        data = dict(zip(want, ex.map(leg, want.values())))
    smooth = lambda ys, k: [sum(ys[max(0, i - k + 1):i + 1]) / len(ys[max(0, i - k + 1):i + 1]) for i in range(len(ys))]
    for ds in sorted({k[0] for k in data}):
        cells = [c for c in CELLS if any(k[0] == ds and k[1] == c[0] for k in data)]
        k = SMOOTH.get(ds, 1)
        curves = {key: ([p[0] for p in pts], smooth([p[1] for p in pts], k), [p[2] for p in pts])
                  for key, (pts, _) in data.items() if key[0] == ds and pts}
        top = max((acc[-1] for _, acc, _ in curves.values()), default=0)  # stalled legs (< half the best final) don't set the range
        late = [a_ for (t, acc, _) in curves.values() if acc[-1] >= top / 2 for t_, a_ in zip(t, acc) if t_ >= WARM_H]
        ylo, yhi = (max(0, min(late) - 2), max(late) + 1.5) if late else (None, None)
        rows = (("Felix: new vs old", lambda b, arm: b == "felix"),
                ("Baselines (Felix old for reference)", lambda b, arm: b != "felix" or arm == a.old),
                ("Accuracy vs traffic (all)", lambda b, arm: True))
        fig, ax = plt.subplots(3, len(cells), figsize=(6.2 * len(cells), 13.5), sharey=True, squeeze=False)
        for j, (cell, title) in enumerate(cells):
            for i, (rtitle, keep) in enumerate(rows):
                ends = []
                for b, arm, lab, col, ls, lw in series:
                    key = (ds, cell, b, arm)
                    if key not in curves or not keep(b, arm):
                        continue
                    t, acc, gb = curves[key]
                    x = gb if i == 2 else t
                    ref = i == 1 and b == "felix"
                    ax[i, j].plot(x, acc, ls, color=col, lw=2.6 if ref else lw, alpha=0.9, label=lab, zorder=3 if ref else 2)
                    if i < 2:
                        ends.append([acc[-1], lab, col])
                ends.sort()
                for n_ in range(1, len(ends)):  # nudge end labels apart (min gap 4% of the y range)
                    ends[n_][0] = max(ends[n_][0], ends[n_ - 1][0] + 0.04 * (yhi - ylo))
                for y, lab, col in ends:
                    ax[i, j].annotate(lab, (1.01, y), xycoords=("axes fraction", "data"), fontsize=8, color="#52514e", va="center",
                                      annotation_clip=False)
                    ax[i, j].plot([1.0], [y], marker="s", ms=5, color=col, transform=ax[i, j].get_yaxis_transform(), clip_on=False)
                ax[i, j].set_title(f"{title} · {rtitle}" if i else title, fontsize=11 if i else 12)
                ax[i, j].set_xlabel("model traffic, GB" if i == 2 else "hours (sim clock)", fontsize=9)
                ax[i, j].grid(alpha=0.25, lw=0.6)
                ax[i, j].spines[["top", "right"]].set_visible(False)
                if ylo is not None:
                    ax[i, j].set_ylim(ylo, yhi)
            ax[0, j].set_title(f"{title} · Felix: new vs old", fontsize=11)
        for i in range(3):
            ax[i, 0].set_ylabel("test accuracy %", fontsize=10)
        h, l = {}, {}
        for x in ax[2]:
            for h_, l_ in zip(*x.get_legend_handles_labels()):
                h.setdefault(l_, h_)
        fig.legend(list(h.values()), list(h), loc="lower center", ncol=min(len(h), 6), fontsize=9, frameon=False)
        basis = {"avg": "averaged models", "cur": "current models"}.get(a.basis, "mixed models")
        fig.suptitle(f"{a.prefix} · {ds} · sims, seed 1 · {basis} · accuracy: {k}-eval rolling mean · y from the lowest value after {WARM_H:g} h (stalled legs clipped)",
                     fontsize=13)
        fig.tight_layout(rect=(0, 0.05, 0.93, 0.97))
        out = os.path.join(a.out, f"felix_{ds}.png")
        fig.savefig(out, dpi=100)
        print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
