"""§5 Design: one observation figure per mechanism (P13). Real where F49 has it, else MOCK with the planned source."""
import numpy as np

import extract as X
from figlib import COL_W, FONT, FULL_W, INK, LINE, MUTED, ROW_H, STACK, Z, plt, save, smooth, status

SEC = "s5_design"
FELIX, FEDBUFF = STACK["Felix"][0], STACK["FedBuff"][0]


def fig5_1_notify():
    """Felix learning eligibility by check-in instead of notify (K1, F42b): AUC loss. MOCK until K1 re-runs on FL1 cells."""
    fig, ax = plt.subplots(figsize=(COL_W * 0.8, ROW_H + 0.2))
    ds, d = ["cifar", "speech"], [-4.4, -10.0]
    ax.bar(ds, d, color=MUTED, width=0.5)
    for i, v in enumerate(d):
        ax.text(i, v - 0.6, f"{v:+.1f}", ha="center", va="top", fontsize=FONT["tick"])
    ax.axhline(0, color=INK, lw=LINE["axis"])
    ax.set(ylabel="AUC Δ (pp) vs notify", title="Felix by check-in (syn_50)", ylim=(-13, 1))
    fig.tight_layout()
    status(fig, "MOCK", "F42b K1 numbers; re-run `fck` on FL1 cells")
    return save(fig, SEC, "fig5_1_notify")


def _sel(r):
    from scipy.stats import spearmanr
    ids = [c for c in r["sizes"] if c in r["picks"]] + [c for c in r["sizes"] if c not in r["picks"]]
    p = [r["picks"].get(c, 0) for c in ids]
    s = [r["sizes"][c] for c in ids]
    return spearmanr(p, s).correlation, sum(pi * si for pi, si in zip(p, s)) / max(sum(p), 1)


def fig5_2_select():
    """(a, b) real F49: ρ(picks, shard size) and samples per update, Felix vs FedBuff; (c) MOCK speed-term ablation (F45-F46)."""
    fig, ax = plt.subplots(1, 3, figsize=(FULL_W, ROW_H + 0.3))
    cells = list(X.CELLS)
    for j, lab in enumerate(("FedBuff", "Felix")):
        vals = []
        for cell in cells:
            d = X.f49(cell, [lab]).get(lab)
            vals.append(_sel(X.load([d])[d]) if d else (np.nan, np.nan))
        x = np.arange(len(cells)) + (j - 0.5) * 0.36
        for k in (0, 1):
            ax[k].bar(x, [v[k] for v in vals], 0.34, color=STACK[lab][0], label=lab, zorder=Z["bar"])
    for k, (yl, t) in enumerate((("ρ(picks, shard size)", "(a) whom each stack trains"), ("samples per update", "(b) data per update"))):
        ax[k].set_xticks(range(len(cells)), [X.CELLS[c].split(" ·")[0] for c in cells])
        ax[k].set(ylabel=yl, title=t)
    ax[0].axhline(0, color=INK, lw=LINE["axis"])
    ax[0].legend(loc="upper right")
    c = ax[2]
    arms, h1, fin = ["utility", "utility + speed"], [44, 47.5], [56.0, 54.2]
    x = np.arange(2)
    c.bar(x - 0.18, h1, 0.34, color=MUTED, label="at 1 h")
    c.bar(x + 0.18, fin, 0.34, color=FELIX, label="final")
    c.set_xticks(x, arms)
    c.set(ylabel="accuracy (%)", title="(c) the speed term: early gain, final loss", ylim=(30, 62))
    c.legend(loc="upper right")
    fig.tight_layout()
    status(ax[0], "PRELIM", "F49 cifar, 1 seed")
    status(ax[1], "PRELIM", "F49 cifar, 1 seed")
    status(c, "MOCK", "F45-F46 numbers")
    return save(fig, SEC, "fig5_2_select")


def fig5_3_pace():
    """Felix-Pace on MobiPerf (real F49 `fn_chr`): c follows ½ the eligible pool as φ falls; GB vs accuracy with and without it."""
    d = X.f49("smu", ["Felix"])["Felix"]
    d_fixed = X.run_dir(X.F49, "F49_smu_fn", "felix")
    R = X.load([x for x in (d, d_fixed) if x])
    kc = np.array([r for r in R[d]["kc"] if None not in r[:4]], float)
    fig, ax = plt.subplots(1, 3, figsize=(FULL_W, ROW_H + 0.3))
    h = kc[:, 0] / 3600
    a = ax[0]
    a.plot(h, smooth(list(kc[:, 2]), 20), color=MUTED, lw=LINE["muted"], label="eligible clients")
    a.plot(h, smooth(list(np.minimum(60, 0.5 * kc[:, 2])), 20), color=MUTED, ls="--", lw=LINE["ref"], label="min(c_max, ½ eligible)")
    a.plot(h, kc[:, 1], color=FELIX, lw=LINE["width"], label="c (Felix-Pace)")
    a.set(xlabel="hours", ylabel="clients / slots", title="(a) concurrency follows the pool", ylim=(0, 100))
    a.legend(loc="upper right", ncol=2, fontsize=FONT["annot"])
    b = ax[1]
    b.plot(h, kc[:, 3], color=FELIX, lw=LINE["width"])
    b.set(xlabel="hours", ylabel="φ (loss / running peak)", title="(b) training phase φ", ylim=(0, 1.05))
    c = ax[2]
    for dd, lab, ls in ((d_fixed, "Felix, fixed c 60", "--"), (d, "Felix", "-")):
        if dd:
            cv = R[dd]["curve"]
            c.plot([g for *_, g in cv], smooth([100 * x for _, x, _ in cv], 5), color=FELIX, ls=ls, label=lab)
    c.set(xlabel="GB moved", ylabel="test accuracy (%)", title="(c) same accuracy, fewer bytes")
    c.legend(loc="lower right")
    fig.tight_layout()
    status(fig, "PRELIM", "F49 cifar MobiPerf 12 h, 1 seed")
    return save(fig, SEC, "fig5_3_pace")


def fig5_4_agg():
    """(a) analytic: staleness weight per rule (Felix-Agg = rate `new`, s 0.4 / a 0.25 / b 0.1); (b) MOCK time to 50% (F46)."""
    fig, ax = plt.subplots(1, 2, figsize=(FULL_W * 0.7, ROW_H + 0.3))
    tau = np.linspace(0, 40, 200)
    a = ax[0]
    a.plot(tau, 1 / np.sqrt(1 + tau), color=FEDBUFF, label="FedBuff 1/√(1+τ)")
    a.plot(tau, 1 / (1 + tau), color=MUTED, ls="--", label="1/(1+τ)")
    for u, ls in ((10, ":"), (1000, "-")):
        a.plot(tau, 0.4 * (1 + tau) ** -0.25 + 0.6 * (1 - (1 + u) ** -0.1), color=FELIX, ls=ls, label=f"Felix-Agg, loss utility {u}")
    a.set(xlabel="staleness τ (versions)", ylabel="update weight", title="(a) a large step, mild discount", ylim=(0, 1.05))
    a.legend(loc="upper right")
    b = ax[1]
    rules, lo, hi = ["Felix-Agg", "1/√(1+τ)", "1/(1+τ)"], [60, 120, 210], [80, 130, 280]
    b.barh(rules, [(x + y) / 2 for x, y in zip(lo, hi)], xerr=[[(y - x) / 2 for x, y in zip(lo, hi)]] * 2, color=[FELIX, FEDBUFF, MUTED],
           height=0.5, error_kw=dict(lw=LINE["ref"], capsize=2))
    b.invert_yaxis()
    b.set(xlabel="minutes to 50% (same selection, K 10)", title="(b) time to 50%")
    fig.tight_layout()
    status(b, "MOCK", "F46 ranges")
    return save(fig, SEC, "fig5_4_agg")


FIGS = (fig5_1_notify, fig5_2_select, fig5_3_pace, fig5_4_agg)
