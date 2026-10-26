"""§3 Motivation figures (Figs 2-6). Each function = one figure; real where the data exists, else MOCK with the planned source."""
import json
import os

import numpy as np

import extract as X
from figlib import COL_W, FONT, FULL_W, INK, LINE, MUTED, PRED, ROW_H, STACK, TRACE, Z, plt, save, smooth, status

SEC = "s3_motivation"
DATA = os.path.join(X.HERE, "data")
TRACES = (("mobiperf_3st", "MobiPerf", *TRACE["MobiPerf"]), ("syn_50_3st", "syn_50", *TRACE["syn_50"]))
PRED_KEYS = (("now", "naive: eligible now"), ("prophet", "REFL Prophet"), ("feddance", "FedDance $V_m$"), ("seasonal", "time-of-day vote"))


def _fx(trace):
    return json.load(open(os.path.join(DATA, f"fx_n147_{trace}.json")))


def fig2_eligibility():
    """§3.1: eligibility is scarce, shifting and hard to forecast (FX-N147, trace-only; real)."""
    fig, ax = plt.subplots(2, 3, figsize=(FULL_W, 2 * ROW_H + 0.5))
    fx = {t: _fx(t) for t, *_ in TRACES}
    a = ax[0, 0]  # (a) eligible share over the trace
    for t, name, c, ls in TRACES:
        s = fx[t]["stats"]
        h = np.arange(len(s["share_t"])) * s["share_dt_s"] / 3600
        a.plot(h / 24, 100 * np.array(s["share_t"]), color=c, ls=ls, lw=LINE["muted"], label=name)
    a.axvspan(1, 1.5, color=MUTED, alpha=0.12, lw=0)
    a.text(1.25, 2, "12 h leg", fontsize=FONT["annot"], ha="center", color=MUTED)
    a.set(xlabel="trace day", ylabel="eligible (% of clients)", title="(a) eligible pool over time", ylim=(0, 80))
    a.legend(loc="upper right")
    b = ax[0, 1]  # (b) eligible-session length CDF
    for t, name, c, ls in TRACES:
        x = np.sort(np.array(fx[t]["stats"]["sessions_s"]) / 60)
        b.plot(x, np.arange(1, len(x) + 1) / len(x), color=c, ls=ls, label=name)
    b.set(xscale="log", xlabel="eligible session (min)", ylabel="CDF", title="(b) how long a client stays eligible")
    c_ = ax[0, 2]  # (c) autocorrelation by lag
    for t, name, c, ls in TRACES:
        lags = sorted(fx[t]["autocorr"], key=float)
        c_.plot([float(k) / 3600 for k in lags], [fx[t]["autocorr"][k] for k in lags], color=c, ls=ls, marker="o", ms=3, label=name)
    c_.axhline(0, color=MUTED, lw=LINE["axis"])
    c_.set(xscale="log", xlabel="lag (h)", ylabel="corr(eligible now, at lag)", title="(c) eligibility memory", ylim=(-0.1, 1))
    m = fx["mobiperf_3st"]["metrics"]
    hs = sorted({int(k.split("_")[1]) for k in m})
    for col, (q, title, ylab) in enumerate((("A", "(d) who is eligible in [t+h, t+2h]", "AUC (all clients)"),
                                             ("B", "(e) who stays eligible for h", "AUC (eligible now)"))):
        d = ax[1, col]
        for key, lab in PRED_KEYS:
            color, ls, mk = PRED[lab]
            d.plot([h / 60 for h in hs], [m[f"{q}_{h}"][key]["auc"] for h in hs], color=color, ls=ls, marker=mk, ms=3.5, label=lab)
        d.axhline(0.5, color=MUTED, lw=LINE["axis"], ls=":")
        d.set(xscale="log", xlabel="horizon h (min)", ylabel=ylab, title=title, ylim=(0.45, 1.0), xticks=[5, 15, 60])
        d.set_xticklabels(["5", "15", "60"])
    f = ax[1, 2]  # (f) Brier score among eligible clients (lower = better); "eligible now" = the base rate
    for key, lab in PRED_KEYS:
        color, ls, mk = PRED[lab]
        f.plot([h / 60 for h in hs], [m[f"B_{h}"][key]["brier"] for h in hs], color=color, ls=ls, marker=mk, ms=3.5)
    f.set(xscale="log", xlabel="horizon h (min)", ylabel="Brier score (eligible now)", title="(f) calibration of (e), lower better",
          xticks=[5, 15, 60], ylim=(0, 0.6))
    f.set_xticklabels(["5", "15", "60"])
    fig.legend(*ax[1, 0].get_legend_handles_labels(), loc="center", ncol=4, bbox_to_anchor=(0.5, 0.485))
    fig.tight_layout(h_pad=2.2)
    return save(fig, SEC, "fig2_eligibility")


def _sync_rounds(d):
    """Commit times of a sync leg (one agg_round per round)."""
    import glob
    import re
    f = glob.glob(os.path.join(d, "telemetry", "aggregator_*.jsonl"))[0]
    rx = re.compile(rb'"vclock_now": ([-0-9.e]+)')
    ts = []
    with open(f, "rb") as fh:
        for line in fh:
            if b'"agg_round"' in line and b'"task_to_perform": "eval"' not in line and (mm := rx.search(line)):
                ts.append(float(mm.group(1)))
    return np.diff(sorted(set(ts)))


def fig3_sync_oracle():
    """§3.2: oracle knowledge of who is eligible barely helps a sync stack (real F49 cifar; paper cell = speech)."""
    runs = X.f49("smu", ["REFL", "REFL*", "FedDance", "FedDance*", "Oort", "Oort*"])
    R = X.load(runs.values())
    fig, ax = plt.subplots(1, 3, figsize=(FULL_W, ROW_H + 0.3))
    for col, pair in enumerate((("REFL", "REFL*"), ("FedDance", "FedDance*"))):
        a = ax[col]
        for lab in pair:
            if lab not in runs:
                continue
            cv = R[runs[lab]]["curve"]
            color, ls = STACK[lab]
            a.plot([t / 3600 for t, *_ in cv], smooth([100 * x for _, x, _ in cv], 5), color=color, ls=ls, label=lab)
        a.set(xlabel="hours", ylabel="test accuracy (%)", title=f"({'ab'[col]}) {pair[0]}: check-in vs oracle")
        a.legend(loc="lower right")
    c = ax[2]
    for lab in ("REFL", "FedDance", "Oort"):
        if lab in runs:
            g = _sync_rounds(runs[lab])
            x = np.sort(g)
            c.plot(x, np.arange(1, len(x) + 1) / len(x), color=STACK[lab][0], label=f"{lab} (median {np.median(g):.0f} s)")
    c.set(xscale="log", xlabel="round length (s)", ylabel="CDF", title="(c) rounds wait for the slowest pick")
    c.legend(loc="lower right")
    fig.tight_layout()
    status(fig, "PRELIM", "F49 cifar MobiPerf 12 h, 1 seed; paper uses speech (F43/FB1)")
    return save(fig, SEC, "fig3_sync_oracle")


def fig4_slots():
    """§3.3: async FL idles slots (real F49 FedBuff), and filling them does not buy accuracy (MOCK: F44 b1 bound not in F49)."""
    fig, ax = plt.subplots(1, 2, figsize=(FULL_W * 0.75, ROW_H + 0.3), gridspec_kw={"width_ratios": [1.6, 1]})
    a = ax[0]
    d = X.f49("s50u", ["FedBuff"]).get("FedBuff")
    r = X.load([d])[d]
    taken, busy = np.array(r["inflight"]), np.array([(60 * m, v) for m, v in r["busy"] if m < 720])
    a.plot(taken[:, 0] / 3600, smooth(list(taken[:, 1]), 10), color=MUTED, lw=LINE["muted"], label="slots taken")
    a.plot(busy[:, 0] / 3600, smooth(list(busy[:, 1]), 10), color=STACK["FedBuff"][0], lw=LINE["width"], label="clients computing")
    a.fill_between(busy[:, 0] / 3600, smooth(list(busy[:, 1]), 10), np.interp(busy[:, 0], taken[:, 0], smooth(list(taken[:, 1]), 10)),
                   color=MUTED, alpha=0.15, lw=0)
    a.text(6, 49, f"held by departed / waiting ≈ {taken[:, 1].mean() - busy[:, 1].mean():.0f}", fontsize=FONT["annot"], color=INK, ha="center")
    a.axhline(60, color=MUTED, ls=":", lw=LINE["ref"])
    a.text(0.2, 62, "c = 60", fontsize=FONT["annot"], color=MUTED)
    a.set(xlabel="hours", ylabel="slots", title="(a) FedBuff, syn_50: slots taken vs computing", ylim=(0, 70))
    a.legend(loc="lower right")
    b = ax[1]  # MOCK from F44 numbers: FedBuff vs the every-slot-refilled bound
    names, upd, acc = ["FedBuff", "all slots\nrefilled"], [116, 213], [0.0, -1.2]
    b.bar(names, upd, color=[STACK["FedBuff"][0], MUTED], width=0.55, zorder=Z["bar"])
    for i, (u, d) in enumerate(zip(upd, acc)):
        b.text(i, u + 5, f"{u}k upd\n{d:+.1f} pp final", ha="center", fontsize=FONT["annot"])
    b.set(ylabel="updates in 12 h (k)", title="(b) 1.8× the work, lower final", ylim=(0, 270))
    fig.tight_layout()
    status(ax[0], "PRELIM", "F49 cifar, 1 seed")
    status(ax[1], "MOCK", "F44 numbers; re-run bound arm b1")
    return save(fig, SEC, "fig4_slots")


def fig5_staleness_source():
    """§3.4: departures, not slow devices, cause most staleness (real F49 FedBuff; held = ready → commit lag > 60 s)."""
    fig, ax = plt.subplots(figsize=(COL_W, ROW_H + 0.2))
    rows = []
    for cell in ("s50u", "smu"):
        d = X.f49(cell, ["FedBuff"]).get("FedBuff")
        if d:
            s = X.load([d])[d]["stale"]
            n, tau = s["held"][0] + s["work"][0], s["held"][1] + s["work"][1]
            rows.append((X.CELLS[cell].split(" ·")[0], 100 * s["held"][0] / n, 100 * s["held"][1] / tau, s["held"][1] / max(s["held"][0], 1)))
    y = np.arange(len(rows))
    ax.barh(y + 0.18, [r[1] for r in rows], 0.34, color=MUTED, label="share of updates")
    ax.barh(y - 0.18, [r[2] for r in rows], 0.34, color=STACK["FedBuff"][0], label="share of total staleness")
    for i, r in enumerate(rows):
        ax.text(r[1] + 1, i + 0.18, f"{r[1]:.1f}%", va="center", fontsize=FONT["annot"], color=INK)
        ax.text(r[2] + 1, i - 0.18, f"{r[2]:.0f}%  (mean τ {r[3]:.0f})", va="center", fontsize=FONT["annot"])
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set(xlabel="% (updates held across a departure)", xlim=(0, 100), title="Held uploads: few updates, most staleness")
    ax.legend(loc="upper right")
    ax.set_ylim(-0.6, len(rows) - 0.4 + 0.5)
    fig.tight_layout()
    status(fig, "PRELIM", "F49 cifar FedBuff 12 h, 1 seed")
    return save(fig, SEC, "fig5_staleness_source")


def fig6_choice():
    """§3.5: availability removes choice before capacity. (a) real: eligible pool vs c on MobiPerf; (b) MOCK: ρ(picks, size) vs c / pool."""
    fig, ax = plt.subplots(1, 2, figsize=(FULL_W * 0.75, ROW_H + 0.3))
    a = ax[0]
    for t, name, c, ls in TRACES:
        s = _fx(t)["stats"]
        dt = s["share_dt_s"]
        i0, i1 = int(86400 / dt), int(1.5 * 86400 / dt)
        h = np.arange(i1 - i0) * dt / 3600
        a.plot(h, 300 * np.array(s["share_t"][i0:i1]), color=c, ls=ls, label=f"eligible, {name}")
    a.axhline(60, color=STACK["FedBuff"][0], lw=LINE["muted"])
    a.text(11.8, 15, "FedBuff c = 60\n(20% of n)", fontsize=FONT["annot"], color=INK, ha="right")
    a.set(xlabel="hours into the leg", ylabel="clients (n 300)", title="(a) the pool barely exceeds c (real traces)")
    a.legend(loc="upper right", bbox_to_anchor=(1, 1.02))
    a.set_ylim(0, 230)
    b = ax[1]  # MOCK from F47 / F48a: utility selection vs FedBuff as c nears the pool
    x = np.array([0.2, 0.4, 0.6, 0.8, 0.9, 1.0])
    b.plot(x, [0.9, 0.85, 0.6, 0.1, -0.3, -0.45], color=STACK["Oort"][0], marker="o", ms=3, label="loss-based selection")
    b.plot(x, [-0.46] * len(x), color=STACK["FedBuff"][0], marker="s", ms=3, label="FedBuff (uniform)")
    b.axhline(0, color=MUTED, lw=LINE["axis"])
    b.set(xlabel="c / eligible clients", ylabel="ρ(picks, shard size)", title="(b) choice collapses as c → pool", ylim=(-0.7, 1.05))
    b.legend(loc="lower left")
    fig.tight_layout()
    status(ax[1], "MOCK", "sweep c on F48a cells")
    return save(fig, SEC, "fig6_choice")


FIGS = (fig2_eligibility, fig3_sync_oracle, fig4_slots, fig5_staleness_source, fig6_choice)
