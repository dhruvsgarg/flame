#!/usr/bin/env python3
"""Time and bytes to the final target (operator 10-24): every stack is compared at the moment it reaches the target, not at a horizon.

  target_report.py POOL... --target cifar10=0.75 google_speech=0.70 [--out DIR]

Hit = first eval whose 5-eval mean >= target. Per cell: time / GB to target, both normalized to the best stack (1.00); a stack
that never hits reports its last 5-eval mean, end time and GB. Reference = the fastest deployed (check-in) baseline to the target;
`*` oracle arms are shown but never the reference. At Felix's hit: each stack's 5-eval mean and GB (the vertical gap).
Figures: per cell accuracy vs time and vs GB (Felix, the reference, the rest muted); per dataset bars of normalized time / GB.
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_report import cells, leg  # noqa: E402

ORACLE = ("rstar", "fdstar", "oort_star", "fbstar")
NAMES = {"felix": "Felix", "fedbuff": "FedBuff", "refl": "REFL", "feddance": "FedDance", "fedavg": "FedAvg", "oort": "Oort",
         "oort_star": "Oort*"}
FELIX, REF, MUTED = "#2a78d6", "#eb6834", "#9a9893"  # palette.md slots 1 / 2; the rest recede


def name(b, arm):
    n = NAMES.get(b, b)
    if arm.startswith("rstar"):
        return "REFL*"
    if arm.startswith("fdstar"):
        return "FedDance*"
    return n if b != "felix" or arm == "fn_chr" else f"Felix ({arm})"


def mean5(pts):
    """[(t, 5-eval mean, bytes)] from [(t, acc, bytes)]."""
    return [(t, sum(p[1] for p in pts[max(0, i - 4):i + 1]) / len(pts[max(0, i - 4):i + 1]), b) for i, (t, _, b) in enumerate(pts)]


def at(pts, t):
    ys = [p for p in pts if p[0] <= t]
    return ys[-1] if ys else (t, 0.0, 0)


def cell_rows(legs, target):
    rows = []
    for (b, arm), (pts, end, nbytes) in legs.items():
        m = mean5(pts)
        hit = next((p for p in m if p[1] >= target), None)
        rows.append(dict(stack=name(b, arm), baseline=b, arm=arm, oracle=arm.startswith(ORACLE) or b == "oort_star",
                         hit=hit is not None, t_h=(hit or m[-1])[0] / 3600, gb=(hit or m[-1])[2] / 1e9,
                         acc=(hit or m[-1])[1], end_acc=m[-1][1], end_h=end / 3600, end_gb=nbytes / 1e9, curve=m))
    hits = [r for r in rows if r["hit"]]
    tmin, gmin = (min(r["t_h"] for r in hits), min(r["gb"] for r in hits)) if hits else (None, None)
    fx = next((r for r in rows if r["baseline"] == "felix" and r["arm"] == "fn_chr"), None)
    dep = [r for r in hits if r["baseline"] != "felix" and not r["oracle"]]
    ref = min(dep, key=lambda r: r["t_h"]) if dep else None
    for r in rows:
        r["t_norm"] = r["t_h"] / tmin if r["hit"] else None
        r["gb_norm"] = r["gb"] / gmin if r["hit"] else None
        if fx and fx["hit"]:
            p = at(r["curve"], fx["t_h"] * 3600)
            r["acc_at_felix"], r["gb_at_felix"] = p[1], p[2] / 1e9
    return rows, fx, ref


def table(ds, cell, target, rows, fx, ref):
    out = [f"\n## {ds} {cell}: target {100 * target:.1f}% (5-eval mean)"]
    if fx and fx["hit"] and ref:
        out.append(f"Felix vs the fastest deployed baseline ({ref['stack']}): **{ref['t_h'] / fx['t_h']:.2f}× sooner, "
                   f"{ref['gb'] / fx['gb']:.2f}× fewer GB** ({fx['t_h']:.1f} vs {ref['t_h']:.1f} h, {fx['gb']:.0f} vs {ref['gb']:.0f} GB)")
    elif fx and fx["hit"]:
        out.append("No deployed baseline reaches the target.")
    out += ["", "| stack | reached | h to target | GB to target | time ÷ best | GB ÷ best | at Felix's hit: acc % · GB | end: acc % · h · GB |",
            "|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (not r["hit"], r["t_h"] if r["hit"] else -r["end_acc"])):
        f2 = lambda x: "–" if x is None else f"{x:.2f}"
        gap = f"{100 * r['acc_at_felix']:.1f} · {r['gb_at_felix']:.0f}" if "acc_at_felix" in r else "–"
        out.append(f"| {r['stack']} | {'✓' if r['hit'] else '✗'} | {r['t_h']:.1f} | {r['gb']:.0f} | {f2(r['t_norm'])} | "
                   f"{f2(r['gb_norm'])} | {gap} | {100 * r['end_acc']:.1f} · {r['end_h']:.0f} · {r['end_gb']:.0f} |"
                   if r["hit"] else
                   f"| {r['stack']} | ✗ | – | – | – | – | {gap} | {100 * r['end_acc']:.1f} · {r['end_h']:.0f} · {r['end_gb']:.0f} |")
    return out


def curves(ds, cell, target, rows, fx, ref, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, (xi, xl, scale) in zip(axes, ((0, "hours", 1 / 3600), (2, "GB transferred", 1e-9))):
        for r in sorted(rows, key=lambda r: r is fx or r is ref):
            hi = r is fx or r is ref
            c = FELIX if r is fx else REF if r is ref else MUTED
            ax.plot([p[xi] * scale for p in r["curve"]], [100 * p[1] for p in r["curve"]], color=c, lw=2 if hi else 1,
                    alpha=1 if hi else 0.6, label=r["stack"] if hi else None)
            if r["hit"] and hi:
                x = (r["t_h"] if xi == 0 else r["gb"])
                ax.plot([x], [100 * target], "o", ms=8, color=c, mec="white", mew=2, zorder=5)
        ax.axhline(100 * target, color="#52514e", lw=1, ls="--")
        ax.text(ax.get_xlim()[0], 100 * target + 0.5, f" target {100 * target:.1f}%", va="bottom", ha="left", color="#52514e", fontsize=9)
        if fx and ref and fx["hit"]:
            a, b = (fx["t_h"], ref["t_h"]) if xi == 0 else (fx["gb"], ref["gb"])
            ax.annotate("", xy=(a, 100 * target - 2), xytext=(b, 100 * target - 2), arrowprops=dict(arrowstyle="<->", color="#0b0b0b", lw=1))
            ax.text((a + b) / 2, 100 * target - 3, f"{b / a:.2f}× {'sooner' if xi == 0 else 'fewer GB'}", ha="center", va="top", fontsize=10)
        ax.set_xlabel(xl)
        ax.set_ylabel("test accuracy % (5-eval mean)")
        ax.grid(alpha=0.2)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].legend(frameon=False, loc="lower right", title="grey: other baselines", title_fontsize=8)
    fig.suptitle(f"{ds} {cell}")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def bars(ds, per_cell, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cs = sorted(per_cell)
    fig, axes = plt.subplots(2, len(cs), figsize=(4.2 * len(cs), 6.5), constrained_layout=True, squeeze=False)
    for j, cell in enumerate(cs):
        rows = sorted(per_cell[cell][0], key=lambda r: (not r["hit"], r["t_norm"] or -r["end_acc"]))
        for i, key in enumerate(("t_norm", "gb_norm")):
            ax = axes[i][j]
            ys = [r[key] if r["hit"] else 0 for r in rows]
            cols = [FELIX if r["baseline"] == "felix" else MUTED for r in rows]
            ax.bar(range(len(rows)), ys, color=cols, width=0.7)
            for k, r in enumerate(rows):
                ax.text(k, (ys[k] or 0) + 0.05, f"{ys[k]:.2f}" if r["hit"] else f"✗ {100 * r['end_acc']:.0f}%", ha="center", va="bottom",
                        fontsize=7, rotation=0 if r["hit"] else 90)
            ax.set_ylim(0, max([1.0] + ys) * 1.3)
            ax.set_xticks(range(len(rows)), [r["stack"] for r in rows], rotation=60, ha="right", fontsize=8)
            ax.set_ylabel(("time" if key == "t_norm" else "GB") + " to target ÷ best")
            ax.set_title(cell if i == 0 else "")
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
    fig.suptitle(f"{ds}: normalized to the best stack (1.00); ✗ = not reached (end accuracy)")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pools", nargs="+")
    ap.add_argument("--target", nargs="+", required=True, help="dataset=accuracy, e.g. cifar10=0.75")
    ap.add_argument("--out", default=None, help="dir for figures + JSON (default: first pool)")
    a = ap.parse_args(argv)
    tg = {k: float(v) for k, v in (s.split("=") for s in a.target)}
    out = a.out or a.pools[0]
    os.makedirs(out, exist_ok=True)
    cs = cells(a.pools)
    keys = [(ds, tr, b, arm) for (ds, tr, arm, b) in cs]
    with Pool(min(32, len(keys) or 1)) as p:
        data = dict(zip(keys, p.map(leg, [cs[(ds, tr, arm, b)][0] for ds, tr, b, arm in keys])))
    groups = {}
    for (ds, tr, b, arm), v in data.items():
        if ds in tg:
            groups.setdefault(ds, {}).setdefault(tr, {})[(b, arm)] = v
    js = []
    for ds, byc in sorted(groups.items()):
        per = {}
        for cell, legs in sorted(byc.items()):
            rows, fx, ref = cell_rows(legs, tg[ds])
            per[cell] = (rows, fx, ref)
            print("\n".join(table(ds, cell, tg[ds], rows, fx, ref)))
            curves(ds, cell, tg[ds], rows, fx, ref, os.path.join(out, f"target_{ds}_{cell}.png"))
            js += [dict({k: v for k, v in r.items() if k != "curve"}, dataset=ds, cell=cell, target=tg[ds],
                        reference=bool(ref is r)) for r in rows]
        bars(ds, per, os.path.join(out, f"target_bars_{ds}.png"))
    json.dump(js, open(os.path.join(out, "target_report.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
