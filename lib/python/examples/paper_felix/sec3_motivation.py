"""§3 Motivation figures (Figs 2-6). Each function = one figure; real where the data exists, else MOCK with the planned source."""
import json
import os

import numpy as np

import extract as X
from figlib import FONT, FULL_W, INK, LINE, PRED, TRACE, Z, place, plt, save, smooth

SEC = "s3_motivation"
DATA = os.path.join(X.HERE, "data")
TRACES = (("mobiperf_3st", "MobiPerf", *TRACE["MobiPerf"]), ("syn_50_3st", "Syn-50%", *TRACE["Syn-50%"]))
PRED_KEYS = (("now", "Eligible now (naive)"), ("prophet", "REFL Prophet"), ("feddance", "FedDance forecast"), ("seasonal", "Time-of-day vote"))


def _fx(trace):
    return json.load(open(os.path.join(DATA, f"fx_n147_{trace}.json")))


def _log_minutes(ax, ticks):
    ax.set_xscale("log")
    ax.set_xticks([m for m, _ in ticks], [t for _, t in ticks])
    ax.xaxis.set_minor_locator(plt.NullLocator())


TASK_S = {"CIFAR-10": (26, 451, 677), "Speech": (276, 1280, 1920)}  # FLuX epoch task: median, slowest; task timeout (task_timeout.py)


def fig2_trace():
    """§3.1 (FX-N147, traces only), one figure*: the pool's size is steady but its members churn (a); eligible windows are short
    next to a task (b); eligibility memory fades within hours (c); forecasters do no better than "eligible now" (d, e)."""
    fx = {t: _fx(t) for t, *_ in TRACES}
    fig, axs = plt.subplots(1, 5, figsize=(FULL_W, 1.65), width_ratios=(1, 1.2, 0.95, 1, 1))
    a, b, c, d, e = axs
    kw = dict(fontsize=FONT["annot"])
    s = fx["mobiperf_3st"]["stats"]
    days = np.arange(len(s["share_t"])) * s["share_dt_s"] / 86400
    turn = 100 * float(np.mean(s["turnover_t"][1:]))
    a.plot(days, smooth(list(100 * np.array(s["share_t"])), 4), color=INK, lw=LINE["width"], zorder=Z["line"],
           label=f"Eligible: {100 * float(np.mean(s['share_t'])):.0f}% of clients")
    tdays = np.arange(len(s["turnover_t"])) * s["turnover_dt_s"] / 86400
    a.plot(tdays[1:], smooth(list(100 * np.array(s["turnover_t"][1:])), 3), color=INK, lw=LINE["muted"], ls=(0, (2, 1.2)), zorder=Z["line"],
           label=f"New in the last hour: {turn:.0f}%\nof the eligible pool")
    a.set(xlim=(0, 6), ylim=(0, 100), xticks=range(7), yticks=[0, 20, 40, 60], xlabel="(a) Time (days), MobiPerf", ylabel="Share (%)")
    a.legend(loc="upper left", fontsize=FONT["annot"], handlelength=1.4, borderaxespad=0.1, labelspacing=0.2)
    for t, name, col, ls in TRACES[::-1]:
        w = np.sort(np.array(fx[t]["stats"]["sessions_s"]) / 60)
        b.plot(w, 100 * np.arange(1, len(w) + 1) / len(w), color=col, ls=ls, lw=LINE["width"], zorder=Z["line"], label=name)
        lags = sorted(fx[t]["autocorr"], key=float)
        c.plot([float(k) / 60 for k in lags], [fx[t]["autocorr"][k] for k in lags], color=col, ls=ls, marker="o", ms=3, mfc="white",
               lw=LINE["width"], zorder=Z["line"])
    _log_minutes(b, [(1, "1 min"), (10, "10 min"), (60, "1 h"), (600, "10 h")])
    b.set(xlim=(0.3, 2000), ylim=(0, 150), yticks=[0, 25, 50, 75, 100], xlabel="(b) Eligible Window Length", ylabel="Windows (CDF, %)")
    for k, (ds, (med, slow, to)) in enumerate(TASK_S.items()):  # task-time ranges in a strip above the CDF
        y = 136 - 20 * k
        b.plot([med / 60, slow / 60], [y, y], color=INK, lw=LINE["width"], solid_capstyle="butt", zorder=Z["line"])
        b.plot([med / 60], [y], "o", ms=3, color=INK, zorder=Z["marker"])
        b.plot([slow / 60], [y], "|", ms=5, mew=1.0, color=INK, zorder=Z["marker"])
        b.plot([to / 60], [y], "x", ms=3.5, mew=0.9, color=INK, zorder=Z["marker"])
        b.text(to / 60 * 1.5, y, ds, ha="left", va="center", **kw)
    c.axhline(0, color=INK, lw=LINE["axis"], zorder=Z["fill"])
    _log_minutes(c, [(5, "5 min"), (60, "1 h"), (360, "6 h"), (1440, "1 day")])
    c.set(xlim=(4, 1800), ylim=(-0.1, 1.05), xlabel="(c) Lag", ylabel="Autocorrelation")
    m = fx["mobiperf_3st"]["metrics"]
    hs = sorted({int(k.split("_")[1]) for k in m})
    for ax, q, xl in ((d, "A", "(d) Horizon h: who is\neligible in [t+h, t+2h]"), (e, "B", "(e) Horizon h: eligible\nclient stays for h")):
        for key, lab in PRED_KEYS:
            color, ls, mk = PRED[lab]
            ax.plot([h / 60 for h in hs], [m[f"{q}_{h}"][key]["auc"] for h in hs], color=color, ls=ls, marker=mk, ms=3.5,
                    mfc="white" if key != "now" else color, lw=LINE["hero"] if key == "now" else LINE["width"], label=lab,
                    zorder=Z["hero"] if key == "now" else Z["line"])
        ax.axhline(0.5, color=INK, lw=LINE["ref"], ls=(0, (3, 2)), zorder=Z["fill"])
        _log_minutes(ax, [(5, "5 min"), (15, "15 min"), (60, "1 h")])
        ax.set(xlim=(4, 75), ylim=(0.3, 1.02), xlabel=xl)
    d.set_ylabel("Forecast AUC")
    e.sharey(d)
    e.tick_params(labelleft=False)
    h1, l1 = b.get_legend_handles_labels()
    h2, l2 = d.get_legend_handles_labels()
    fig.legend(h1[::-1] + h2, l1[::-1] + l2, loc="outside upper center", ncol=6, columnspacing=1.2)
    ac = fx["mobiperf_3st"]["autocorr"]
    w = np.array(fx["mobiperf_3st"]["stats"]["sessions_s"])
    sp = 100 * float((w < TASK_S["Speech"][1]).mean())
    xs = TASK_S["Speech"][1] / 60  # drop line from the slowest speech task to the CDF: the share of windows shorter than it
    b.plot([xs, xs], [sp, 116], color=INK, lw=LINE["ref"], ls=":", zorder=Z["fill"])
    b.plot([xs], [sp], "o", ms=3, color=INK, zorder=Z["marker"])
    place(b, rf"$\mathbf{{{sp:.0f}\%}}$", [(xs * 1.4, sp - 2, "left", "top"), (xs * 1.4, sp - 6, "left", "top")], **kw)
    place(c, rf"$\mathbf{{{ac['3600']:.2f}}}$ at 1 h," + "\n" + rf"$\mathbf{{{ac['43200']:.2f}}}$ at 12 h",
          [(x, 0.1, "left", "bottom") for x in (18, 15, 22)], linespacing=1.1, **kw)
    A = {k: [m[f"A_{h}"][k]["auc"] for h in hs] for k, _ in PRED_KEYS}
    B = {k: [m[f"B_{h}"][k]["auc"] for h in hs] for k, _ in PRED_KEYS}
    gap = [x - y for x, y in zip(A["now"], A["prophet"])]
    place(d, f"Prophet: {bf_(-min(gap))} to\n{bf_(-max(gap))} vs naive", [(4.4, y, "left", "bottom") for y in (0.31, 0.32)],
          linespacing=1.1, **kw)
    place(e, "Coin flip", [(70, 0.49, "right", "top"), (5, 0.49, "left", "top")], **kw)
    best = max(max(B["prophet"]), max(B["feddance"]))
    place(e, f"Best forecaster: {best:.2f}", [(4.4, 0.98, "left", "top"), (4.4, 0.88, "left", "top")], **kw)
    return save(fig, SEC, "fig2_trace")


def bf_(v):
    return rf"$\mathbf{{{v:+.2f}}}$"


FIGS = (fig2_trace,)  # Figs 3-6 now come from the FB2 evaluation figures (README)
