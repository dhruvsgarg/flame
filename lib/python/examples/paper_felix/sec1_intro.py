"""§1 Introduction: Fig 1 teaser (accuracy vs time and vs GB, hardest realistic cell: cifar MobiPerf `smu`)."""
import numpy as np

import extract as X
from figlib import FONT, FULL_W, INK, LINE, MUTED, ROW_H, STACK, Z, plt, save, smooth, status

SEC = "s1_intro"
DEPLOYED = ("FedBuff", "REFL", "FedDance", "Oort", "FedAvg")


def _hit(cv, target, k=5):
    acc = smooth([a for _, a, _ in cv], k)
    i = next((i for i, a in enumerate(acc) if a >= target), None)
    return (cv[i][0], cv[i][2]) if i is not None else None


def fig1_teaser():
    """Target (until FX-N146 fixes it) = the best deployed baseline's peak 5-eval mean; reference = that baseline."""
    runs = X.f49("smu", ("Felix",) + DEPLOYED)
    R = X.load(runs.values())
    cv = {lab: R[d]["curve"] for lab, d in runs.items()}
    peak = {lab: max(smooth([a for _, a, _ in c], 5)) for lab, c in cv.items() if lab != "Felix"}
    ref = max(peak, key=peak.get)
    target = peak[ref]
    fig, ax = plt.subplots(1, 2, figsize=(FULL_W * 0.7, ROW_H + 0.35), sharey=True)
    for lab in ("Felix",) + DEPLOYED:
        if lab not in cv:
            continue
        color, ls = STACK[lab]
        lw, z = (LINE["hero"], Z["hero"]) if lab == "Felix" else (LINE["muted"], Z["line"])
        acc = smooth([100 * a for _, a, _ in cv[lab]], 5)
        ax[0].plot([t / 3600 for t, *_ in cv[lab]], acc, color=color, ls=ls, lw=lw, zorder=z, label=lab)
        ax[1].plot([g for *_, g in cv[lab]], acc, color=color, ls=ls, lw=lw, zorder=z)
    for a in ax:
        a.axhline(100 * target, color=MUTED, ls=":", lw=LINE["ref"])
    hf, hr = _hit(cv["Felix"], target), _hit(cv[ref], target)
    if hf and hr:
        ax[0].annotate(f"{hr[0] / hf[0]:.1f}× sooner than {ref}", (hf[0] / 3600, 100 * target), xytext=(hf[0] / 3600 + 1.5, 100 * target - 38),
                       fontsize=FONT["tick"], color=INK, arrowprops=dict(arrowstyle="->", color=INK, lw=LINE["axis"]))
        ax[1].annotate(f"{100 * (1 - hf[1] / hr[1]):.0f}% fewer GB", (hf[1], 100 * target), xytext=(hf[1] * 1.6, 100 * target - 14),
                       fontsize=FONT["tick"], color=INK, arrowprops=dict(arrowstyle="->", color=INK, lw=LINE["axis"]))
    ax[0].set(xlabel="hours", ylabel="test accuracy (%)")
    ax[1].set(xlabel="GB moved (down + up)", xscale="log")
    ax[0].text(0.2, 100 * target + 1, "target (placeholder)", fontsize=FONT["annot"], color=MUTED)
    fig.legend(*ax[0].get_legend_handles_labels(), loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.13))
    fig.tight_layout()
    status(fig, "PRELIM", f"F49 cifar MobiPerf 12 h, 1 seed; target = {ref}'s peak until FX-N146")
    return save(fig, SEC, "fig1_teaser")


FIGS = (fig1_teaser,)
