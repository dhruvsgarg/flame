"""§3 Motivation figures (Figs 2-6). Each function = one figure; real where the data exists, else MOCK with the planned source."""
import json
import os

import numpy as np

import extract as X
from figlib import COL_W, FONT, INK, LINE, PRED, TRACE, Z, place, plt, save, smooth

SEC = "s3_motivation"
DATA = os.path.join(X.HERE, "data")
TRACES = (("mobiperf_3st", "MobiPerf", *TRACE["MobiPerf"]), ("syn_50_3st", "syn_50", *TRACE["syn_50"]))
PRED_KEYS = (("now", "Eligible now (naive)"), ("prophet", "REFL Prophet"), ("feddance", "FedDance forecast"), ("seasonal", "Time-of-day vote"))


def _fx(trace):
    return json.load(open(os.path.join(DATA, f"fx_n147_{trace}.json")))


def _log_minutes(ax, ticks):
    ax.set_xscale("log")
    ax.set_xticks([m for m, _ in ticks], [t for _, t in ticks])
    ax.xaxis.set_minor_locator(plt.NullLocator())


def fig2_eligibility():
    """§3.1 (FX-N147, traces only): the eligible pool is small and churns; eligibility memory fades within hours."""
    fx = {t: _fx(t) for t, *_ in TRACES}
    fig, (a, b) = plt.subplots(1, 2, figsize=(COL_W, 1.6))
    for t, name, c, ls in TRACES[::-1]:
        s = fx[t]["stats"]
        d = np.arange(len(s["share_t"])) * s["share_dt_s"] / 86400
        if name == "MobiPerf":  # syn_50 sits flat at 50% by construction: shown only as the memoryless control in (b)
            a.plot(d, smooth(list(100 * np.array(s["share_t"])), 4), color=c, ls=ls, lw=LINE["width"], zorder=Z["line"])
        lags = sorted(fx[t]["autocorr"], key=float)
        b.plot([float(k) / 60 for k in lags], [fx[t]["autocorr"][k] for k in lags], color=c, ls=ls, marker="o", ms=3, mfc="white",
               lw=LINE["width"], zorder=Z["line"], label=name)
    a.set(xlim=(0, 6), ylim=(0, 40), xticks=range(7), xlabel="Time (days), MobiPerf", ylabel="Eligible Clients (%)")
    b.axhline(0, color=INK, lw=LINE["axis"], zorder=Z["fill"])
    _log_minutes(b, [(5, "5 min"), (60, "1 h"), (360, "6 h"), (1440, "1 day")])
    b.set(xlim=(4, 1800), ylim=(-0.1, 1.05), xlabel="Lag", ylabel="Eligibility Autocorrelation")
    h, l = b.get_legend_handles_labels()
    fig.legend(h[::-1], l[::-1], loc="outside upper center", ncol=2)
    s, ac = fx["mobiperf_3st"]["stats"], fx["mobiperf_3st"]["autocorr"]
    m = 100 * float(np.mean(s["share_t"]))
    place(a, rf"$\mathbf{{{m:.0f}\%}}$ eligible on average;" + "\n" + rf"$\mathbf{{{100 * s['left_1h']:.0f}\%}}$ of them gone 1 h later",
          [(0.1, 39.5, "left", "top"), (0.1, 38, "left", "top")], linespacing=1.1, fontsize=FONT["annot"])
    place(b, rf"$\mathbf{{{ac['3600']:.2f}}}$ at 1 h," + "\n" + rf"$\mathbf{{{ac['43200']:.2f}}}$ at 12 h",
          [(x, 0.1, "left", "bottom") for x in (18, 15, 22)], linespacing=1.1, fontsize=FONT["annot"])
    return save(fig, SEC, "fig2_eligibility")


def fig2_forecast():
    """§3.1 (FX-N147, MobiPerf): deployed forecasters vs the naive guess "eligible now stays eligible", scored against the trace."""
    m = _fx("mobiperf_3st")["metrics"]
    hs = sorted({int(k.split("_")[1]) for k in m})
    fig, axs = plt.subplots(1, 2, figsize=(COL_W, 1.6), sharey=True)
    for ax, q, xl in ((axs[0], "A", "Horizon h\nWho is eligible in [t+h, t+2h]"), (axs[1], "B", "Horizon h\nWhich eligible client stays for h")):
        for key, lab in PRED_KEYS:
            color, ls, mk = PRED[lab]
            ax.plot([h / 60 for h in hs], [m[f"{q}_{h}"][key]["auc"] for h in hs], color=color, ls=ls, marker=mk, ms=3.5,
                    mfc="white" if key != "now" else color, lw=LINE["hero"] if key == "now" else LINE["width"], label=lab,
                    zorder=Z["hero"] if key == "now" else Z["line"])
        ax.axhline(0.5, color=INK, lw=LINE["ref"], ls=(0, (3, 2)), zorder=Z["fill"])
        _log_minutes(ax, [(5, "5 min"), (15, "15 min"), (60, "1 h")])
        ax.set(xlim=(4, 75), ylim=(0.4, 1.02), xlabel=xl)
    axs[0].set_ylabel("AUC (1 = Perfect)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside upper center", ncol=2, columnspacing=1.0)
    a = {k: [m[f"A_{h}"][k]["auc"] for h in hs] for k, _ in PRED_KEYS}
    bb = {k: [m[f"B_{h}"][k]["auc"] for h in hs] for k, _ in PRED_KEYS}
    place(axs[1], "Coin flip", [(70, 0.508, "right", "bottom"), (5, 0.508, "left", "bottom")], fontsize=FONT["annot"])
    gap = [x - y for x, y in zip(a["now"], a["prophet"])]
    place(axs[0], f"Prophet: {bf_(-min(gap))} to {bf_(-max(gap))} vs naive", [(4.4, y, "left", "bottom") for y in (0.505, 0.51, 0.52)],
          fontsize=FONT["annot"])
    best = max(max(bb["prophet"]), max(bb["feddance"]))
    place(axs[1], f"Best forecaster: {best:.2f}", [(4.4, 0.95, "left", "top"), (4.4, 0.85, "left", "top")], fontsize=FONT["annot"])
    return save(fig, SEC, "fig2_forecast")


def bf_(v):
    return rf"$\mathbf{{{v:+.2f}}}$"


FIGS = (fig2_eligibility, fig2_forecast)  # Figs 3-6 now come from the FB2 evaluation figures (README)
