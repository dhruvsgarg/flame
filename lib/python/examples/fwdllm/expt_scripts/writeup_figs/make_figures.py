#!/usr/bin/env python3
"""Render every figure in `fl_fwd_ft_writeup.md` into `../../figs/`.

  ./make_figures.py            # all
  ./make_figures.py 4 5        # just those

Historical numbers come from P4's ledger via `ledger.py`; the 2026-08-20 arms
come from `extract.py`'s JSON cache. Run `extract.py` first.
"""
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

import ledger
from figstyle import (AQUA, BLUE, CRITICAL, GOOD, GRID, INK, INK2, INK3, ORANGE,
                      SURFACE, DATASET_COLOR, caption, despine, note, use_style)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
FIGS = os.path.normpath(os.path.join(HERE, "..", "..", "figs"))
DATASETS = ("agnews", "yahoo", "yelp-p")

PHI_STOP = 2.7            # the shipped Phi backstop
PHI_CLIFF_LO = 3.63       # highest Phi at which an arm still held its peak
PHI_CLIFF_HI = 4.23       # lowest Phi at which one did not


def arm(name):
    with open(os.path.join(DATA, f"{name}.json")) as fh:
        return json.load(fh)


def save(fig, stem):
    os.makedirs(FIGS, exist_ok=True)
    path = os.path.join(FIGS, f"{stem}.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote figs/{stem}.png")


# ----------------------------------------------------------------- figure 1
def fig1():
    """Why the norm can only grow, and what it costs when it grows too far."""
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(11, 4.1),
                                   gridspec_kw=dict(width_ratios=[1, 1.35]))

    # (a) geometry: successive perpendicular steps, drawn to scale.
    axa.set_title("A step perpendicular to $\\theta$ can only lengthen it", loc="left")
    theta = np.array([2.6, 0.0])
    axa.annotate("", xy=theta, xytext=(0, 0),
                 arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2.4))
    axa.text(1.3, -0.22, r"$\theta_t$", color=BLUE, fontsize=11, ha="center")
    step = np.array([0.0, 1.15])
    axa.annotate("", xy=theta + step, xytext=theta,
                 arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=2.4))
    axa.text(2.78, 0.55, r"$\Delta\theta$", color=ORANGE, fontsize=11)
    axa.annotate("", xy=theta + step, xytext=(0, 0),
                 arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.6,
                                 linestyle=(0, (4, 3))))
    axa.text(1.15, 0.78, r"$\theta_{t+1}$", color=INK2, fontsize=11, rotation=23)
    axa.plot([2.44, 2.44, 2.6], [0.0, 0.16, 0.16], color=INK3, lw=0.9)
    axa.text(0.02, 0.90,
             r"$\|\theta_{t+1}\|^2 = \|\theta_t\|^2 + \|\Delta\theta\|^2$",
             transform=axa.transAxes, fontsize=12, color=INK)
    axa.text(0.02, 0.135,
             "no cross-term: measured at $1.000\\pm0.005$ in every\n"
             "25-commit block of every arm. So the weights grow,\nevery commit, forever.",
             transform=axa.transAxes, fontsize=8, color=INK3, va="top")
    axa.set_xlim(-0.2, 3.6); axa.set_ylim(-1.05, 1.95)
    axa.set_aspect("equal"); axa.grid(False)
    axa.set_xticks([]); axa.set_yticks([])
    despine(axa, keep=())

    # (b) what that costs: accuracy lost by the end, against total inflation.
    rows = [r for r in ledger.load() if r["peak"] >= 0.80]
    held = [r for r in rows if r["peak"] - r["final"] <= 0.05]
    lost = [r for r in rows if r["peak"] - r["final"] > 0.05]
    axb.axvspan(PHI_CLIFF_HI, 14, color=CRITICAL, alpha=0.06, zorder=0)
    axb.axvline(PHI_STOP, color=INK3, lw=1.1, ls=(0, (5, 3)), zorder=1)
    axb.scatter([r["phi_obs"] for r in held], [r["peak"] - r["final"] for r in held],
                s=52, color=GOOD, edgecolor="white", linewidth=1.0, zorder=3,
                label=f"held its peak  (n={len(held)})")
    axb.scatter([r["phi_obs"] for r in lost], [r["peak"] - r["final"] for r in lost],
                s=72, color=CRITICAL, marker="X", edgecolor="white", linewidth=1.0,
                zorder=3, label=f"lost it  (n={len(lost)})")
    axb.set_xscale("log")
    axb.set_xticks([1, 1.5, 2, 3, 4, 6, 10])
    axb.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    axb.set_xlim(1.0, 14)
    axb.set_ylim(-0.03, 0.68)
    axb.set_xlabel(r"$\Phi$ — total inflation reached  ($\|\theta_T\|/\|\theta_0\|$, log scale)")
    axb.set_ylabel("accuracy lost by the end\n(peak − final)")
    axb.set_title("Past a threshold, the run gives back everything it learned", loc="left")
    axb.text(PHI_STOP * 0.95, 0.02, "shipped stop\n$\\Phi=2.7$", ha="right",
             va="bottom", fontsize=8, color=INK2)
    axb.text(4.6, 0.40, f"no arm holds\nabove $\\Phi={PHI_CLIFF_HI}$", fontsize=8,
             color=CRITICAL)
    axb.text(1.05, 0.24,
             f"every arm below\n$\\Phi={PHI_CLIFF_LO}$ holds its peak:\nworst loss 0.014",
             fontsize=8, color=GOOD)
    axb.legend(loc="upper left", bbox_to_anchor=(0.015, 0.99))
    despine(axb)
    caption(fig, "22 arms that learned (peak ≥ 0.80), from P4's ledger. Arms that never "
                 "learned are excluded: Φ governs losing what you learned, not failing to "
                 "learn. The cliff sits between Φ=3.63 and Φ=4.23 — the shipped stop at 2.7 "
                 "is conservative against it.")
    save(fig, "fig1_noise_compounds")


# ----------------------------------------------------------------- figure 2
def fig2():
    """Phi predicted from the step sizes alone vs Phi measured on the weights."""
    rows = ledger.load()
    plain = [r for r in rows if r["beta"] == 0]
    mom = [r for r in rows if r["beta"] > 0]
    err = [100 * (r["phi_obs"] - r["phi_pred"]) / r["phi_pred"] for r in plain]

    fig, (ax, axe) = plt.subplots(1, 2, figsize=(11, 4.2),
                                  gridspec_kw=dict(width_ratios=[1.15, 1]))
    lim = (0.98, 13)
    ax.plot(lim, lim, color=INK3, lw=1.1, ls=(0, (5, 3)), zorder=1)
    ax.scatter([r["phi_pred"] for r in plain], [r["phi_obs"] for r in plain],
               s=46, color=BLUE, edgecolor="white", linewidth=0.9, zorder=3,
               label=f"independent steps  (n={len(plain)})")
    ax.scatter([r["phi_pred"] for r in mom], [r["phi_obs"] for r in mom],
               s=74, color=ORANGE, marker="D", edgecolor="white", linewidth=0.9,
               zorder=3, label=f"server momentum $\\beta>0$  (n={len(mom)})")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(*lim); ax.set_ylim(*lim)
    for a in (ax.xaxis, ax.yaxis):
        a.set_major_formatter(plt.ScalarFormatter())
    ax.set_xticks([1, 1.5, 2, 3, 5, 10]); ax.set_yticks([1, 1.5, 2, 3, 5, 10])
    ax.set_xlabel(r"$\Phi$ predicted from the step sizes:  $e^{B}$")
    ax.set_ylabel(r"$\Phi$ measured on the weights:  $\|\theta_T\|/\|\theta_0\|$")
    ax.set_title("The budget law is exact over an 11× range", loc="left")
    note(ax, "the law's one documented exception:\ncorrelated steps inflate faster",
         xy=(mom[1]["phi_pred"], mom[1]["phi_obs"]), xytext=(1.02, 1.9), color=ORANGE)
    ax.legend(loc="lower right")
    despine(ax)

    axe.axvline(0, color=INK3, lw=1.0)
    hi = float(np.ceil(max(np.abs(err))))
    axe.hist(err, bins=np.linspace(-hi, hi, 33), color=BLUE, alpha=0.85)
    axe.set_xlabel("error in predicted $\\Phi$  (%)")
    axe.set_ylabel("arms")
    worst = max(plain, key=lambda r: abs(r["phi_obs"] - r["phi_pred"]) / r["phi_pred"])
    axe.set_title(f"Median miss {np.median(np.abs(err)):.2f}%; worst "
                  f"{max(np.abs(err)):.1f}%", loc="left")
    axe.text(0.97, 0.86, f"the worst is `{worst['run']}`, the\nhighest-$\\rho$ raw-SGD arm on\n"
             f"record ($\\rho$={worst['rho1']:.2f}) — the law's\naccuracy is $\\rho$-dependent",
             transform=axe.transAxes, ha="right", va="top", fontsize=7.5, color=INK3)
    despine(axe)
    caption(fig, "All 39 arms in P4's ledger, spanning ρ 0.0002–0.22, N 10–200, p 118k–450k, "
                 "and 66–3,353 commits. B is computed from the step sizes alone — no gradients, "
                 "no accuracy, no per-model constant. On the controlled arms the miss is under "
                 "0.3%; it grows to a few percent only on the extreme raw-SGD trajectories, "
                 "which is a documented property of the law rather than a surprise.")
    save(fig, "fig2_budget_law")


# ----------------------------------------------------------------- figure 3
def fig3():
    """Peak accuracy against progress banked."""
    rows = ledger.load()
    def dataset_of(r):
        return "yahoo" if "yahoo" in r["desc"].lower() else "agnews"

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    for ds in ("agnews", "yahoo"):
        same = [r for r in rows if dataset_of(r) == ds and r["rf"] == 16]
        other = [r for r in rows if dataset_of(r) == ds and r["rf"] != 16]
        ax.scatter([r["lam"] for r in same], [r["peak"] for r in same], s=46,
                   color=DATASET_COLOR[ds], edgecolor="white", linewidth=0.9,
                   zorder=3, label=f"{ds}  (n={len(same)})")
        if other:
            ax.scatter([r["lam"] for r in other], [r["peak"] for r in other], s=64,
                       facecolor="none", edgecolor=DATASET_COLOR[ds], linewidth=1.6,
                       zorder=3, label=f"{ds}, a different $p$  (n={len(other)})")

    for ds, dx, dy in (("agnews", 0.10, -0.085), ("yahoo", 0.12, -0.02),
                       ("yelp-p", 0.10, -0.10)):
        a = next(r for r in ledger.recent()
                 if r["dataset"] == ds and r["role"] == "controller")
        ax.scatter([a["lam"]], [a["peak"]], s=170, marker="*",
                   color=DATASET_COLOR[ds], edgecolor="white", linewidth=1.0, zorder=5)
        ax.annotate(f"FluxTune\n{ds}", xy=(a["lam"], a["peak"]),
                    xytext=(a["lam"] + dx, a["peak"] + dy), fontsize=8, color=INK2,
                    arrowprops=dict(arrowstyle="-", color=DATASET_COLOR[ds], lw=0.8,
                                    shrinkA=0, shrinkB=5))
    ax.scatter([], [], s=170, marker="*", color=INK3, edgecolor="white",
               label="FluxTune (2026-08-20)")

    ax.axhline(0.876, color=INK3, lw=1.0, ls=(0, (5, 3)))
    ax.text(2.46, 0.884, "best agnews arm on record 0.876", ha="right", fontsize=8,
            color=INK2)
    ax.set_xlabel(r"$\Lambda$ — progress banked  ($\sum \rho_t \cos_t$)")
    ax.set_ylabel("peak held-out accuracy")
    ax.set_title("Accuracy rises with progress banked — one curve per task, same shape",
                 loc="left")
    ax.set_xlim(-0.08, 2.52); ax.set_ylim(0.22, 0.95)
    ax.legend(loc="lower right", ncol=2, columnspacing=1.0)
    despine(ax)
    n16 = sum(1 for r in rows if r["rf"] == 16)
    caption(fig, f"Filled: {n16} arms at p=450k spanning both combination rules, both step "
                 "rules, both gates and α 0.1–1. Open: three arms at a different p, which "
                 "sit off their curve because Λ is a relative coordinate and does not "
                 "transfer across p. The two low yahoo points are that task's own curve, "
                 "not exceptions to agnews': the shape transfers, the absolute level is a "
                 "property of the task. FluxTune on yahoo reads 0.657 at Λ=0.994 — the "
                 "value pre-registered for 'the curve transfers across task'.")
    save(fig, "fig3_accuracy_vs_progress")


# ----------------------------------------------------------------- figure 4
def fig4():
    """FluxTune vs FluxTune-v2, per dataset, on the simulated clock."""
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.2))
    for ax, ds in zip(axes, DATASETS):
        ctl, base = arm(f"{ds}_controller"), arm(f"{ds}_control")
        cx, cy = np.array(ctl["acc"]).T
        bx, by = np.array(base["acc"]).T
        cx, bx = cx / 1000, bx / 1000
        ax.plot(bx, by, color=ORANGE, lw=1.7)
        ax.plot(cx, cy, color=BLUE, lw=1.7)

        ref = ledger.REFERENCE[ds]
        ax.axhline(ref, color=INK3, lw=1.1, ls=(0, (5, 3)))
        # If the controller cleared the reference, its end-label owns the space
        # above the line -- put the caption below it instead.
        above = cy.max() <= ref
        ax.text(0.02, ref, "backprop reference", fontsize=7.5, color=INK2,
                va="bottom" if above else "top",
                transform=ax.get_yaxis_transform())

        lo, hi = min(by.min(), cy.min()), max(ref, cy.max())
        pad = (hi - lo) * 0.34
        ax.set_ylim(lo - pad * 0.55, hi + pad * 0.30)

        # Direct end-labels rather than a legend box inside the data.
        ax.scatter([cx[-1]], [cy[-1]], s=58, color=BLUE, zorder=6,
                   edgecolor="white", linewidth=1.0)
        ax.text(cx[-1] + 1.2, cy[-1], f"{cy.max():.3f}", color=BLUE, fontsize=9,
                weight="bold", va="center")
        ax.text(bx[-1] + 1.2, by[-1], f"{by.max():.3f}", color=ORANGE, fontsize=9,
                weight="bold", va="center")

        # The compute span, drawn low and off the curves, with guides up to them.
        base_pk = by.max()
        bx_pk = bx[np.argmax(by)]
        if (cy >= base_pk).any():
            cross = cx[np.argmax(cy >= base_pk)]
            y0, y1 = ax.get_ylim()
            yb = y0 + (y1 - y0) * 0.055
            for xg in (cross, bx_pk):
                ax.plot([xg, xg], [yb, base_pk], color=INK3, lw=0.8,
                        ls=(0, (2, 3)), zorder=2)
            ax.annotate("", xy=(cross, yb), xytext=(bx_pk, yb),
                        arrowprops=dict(arrowstyle="<|-", color=INK, lw=1.3))
            ax.text((cross + bx_pk) / 2, yb + (y1 - y0) * 0.022,
                    f"{bx_pk / cross:.1f}× less compute", ha="center", va="bottom",
                    fontsize=8.5, color=INK)

        # A note in the empty lower-right, with no leader crossing the data. The
        # bullet carries the identity, so the text does not have to.
        txt = ("FluxTune stopped itself on its own budget, 42% of compute unused"
               if ds == "yelp-p" else
               "FluxTune killed by a monitoring bug, still climbing")
        ax.text(0.0, 1.015, txt, transform=ax.transAxes, va="bottom",
                fontsize=7.5, color=INK3)
        ax.set_xlim(-1.5, max(bx.max(), cx.max()) * 1.14)
        ax.set_title(ds, loc="left", pad=18)
        ax.set_xlabel("simulated wall clock  (1000 s)")
        despine(ax)
    axes[0].set_ylabel("held-out accuracy")
    for c, lbl in ((BLUE, "FluxTune — no learning knob supplied"),
                   (ORANGE, "FluxTune-v2 — static step, hand-searched on agnews")):
        axes[0].plot([], [], color=c, lw=2.2, label=lbl)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower left",
               bbox_to_anchor=(0.0, 1.0), ncol=2, frameon=False, fontsize=8.5)
    caption(fig, "Same stack on both arms; only the step rule differs. The arrow spans from "
                 "where FluxTune first reaches FluxTune-v2's best-ever accuracy to where v2 "
                 "finally reaches it. v2 never reaches FluxTune's accuracy, on any dataset, "
                 "given its entire budget.")
    save(fig, "fig4_fluxtune_vs_v2")


# ----------------------------------------------------------------- figure 5
def fig5():
    """The probe measures flat headroom; the mean combiner reports it shrinking."""
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(11.4, 4.2))
    for ds in DATASETS:
        pr = arm(f"{ds}_controller")["probes"]
        n = [p["n"] for p in pr]
        measured = [p["b_rem"] for p in pr]
        used = [p["b_max"] - p["B"] for p in pr]
        c = DATASET_COLOR[ds]
        axl.plot(n, measured, color=c, lw=2.0, marker="o", ms=5)
        axl.plot(n, used, color=c, lw=1.8, ls=(0, (4, 3)), marker="s", ms=4,
                 markerfacecolor="white")
        axr.plot(n, [p["rho_star"] for p in pr], color=c, lw=1.8, ls=(0, (4, 3)),
                 marker="s", ms=4, markerfacecolor="white")
        axr.plot(n, [np.sqrt(2 * b / 300) for b in measured], color=c, lw=2.0,
                 marker="o", ms=5)

    axl.set_xlabel("probe firing  (every 150 commits)")
    axl.set_ylabel("remaining headroom,  $\\ln \\Phi_{knee}$")
    axl.set_title("The probe says the road is not shrinking.\nThe average says it is.", loc="left")
    axl.set_ylim(0, 0.78)
    axl.text(5.6, 0.40, "measured by the probe\n(solid, filled circles)",
             fontsize=8, color=INK2)
    axl.text(1.35, 0.055, "used by FluxTune\n(dashed, open squares)",
             fontsize=8, color=INK2)
    axl.annotate("", xy=(8, 0.246), xytext=(8, 0.084),
                 arrowprops=dict(arrowstyle="<|-|>", color=CRITICAL, lw=1.4))
    axl.text(7.85, 0.163, "2.9×", ha="right", fontsize=9, color=CRITICAL, weight="bold")
    despine(axl)

    axr.set_xlabel("probe firing  (every 150 commits)")
    axr.set_ylabel(r"step size $\rho^*$")
    axr.set_title("So the step is annealed 1.7× below\nwhat the measurement supports", loc="left")
    axr.set_ylim(0, 0.075)
    axr.text(5.2, 0.058, "$\\rho^*$ the latest\nmeasurement supports", fontsize=8, color=INK2)
    axr.text(5.2, 0.016, "$\\rho^*$ actually used", fontsize=8, color=INK2)
    despine(axr)

    for ds in DATASETS:
        axl.plot([], [], color=DATASET_COLOR[ds], lw=2.2, label=ds)
    axl.legend(loc="upper right", ncol=3, columnspacing=1.0)
    caption(fig, "All 19 probe firings across the three FluxTune arms. Solid: what the probe "
                 "measured. Dashed: what the mean-of-all-senses combiner reported to the "
                 "step rule. Since ρ* = √(2·headroom/T_res), a 2.9× understatement in "
                 "headroom is a 1.7× understatement in step size.")
    save(fig, "fig5_the_combiner_throttles")


# ----------------------------------------------------------------- figure 6
def fig6():
    """The knee sits below the grid the probe searches."""
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.9), sharey=True)
    for ax, ds in zip(axes, DATASETS):
        pr = arm(f"{ds}_controller")["probes"]
        p = pr[-1]
        labels = {"agnews": 4, "yahoo": 10, "yelp-p": 2}
        chance = 1.0 / labels[ds]
        base = p["base_acc"]
        phis = [q[0] for q in p["curve"]]
        norm = [(q[1] - chance) / (base - chance) for q in p["curve"]]
        c = DATASET_COLOR[ds]

        ax.axhspan(-0.28, 0.0, color=INK3, alpha=0.07, zorder=0)
        ax.axvspan(min(phis), max(phis), color=c, alpha=0.06, zorder=0)
        ax.axhline(0.5, color=CRITICAL, lw=1.2, ls=(0, (5, 3)), zorder=2)
        ax.plot([1.0] + phis, [1.0] + norm, color=c, lw=1.8, marker="o", ms=6,
                markerfacecolor="white", zorder=3)
        ax.scatter([1.0], [1.0], s=52, color=INK3, zorder=4)
        ax.scatter([p["phi_knee"]], [0.5], s=90, marker="v", color=CRITICAL,
                   edgecolor="white", linewidth=0.9, zorder=5)
        ax.set_xlim(0.95, 4.25)
        ax.set_xlabel(r"$\Phi$ — inflation injected")
        ax.set_title(f"{ds}   ·   firing {p['n']}, commit {p['commit']}", loc="left")
        ax.text(min(phis) + 0.06, -0.20, "the grid it searches", fontsize=7.5, color=c)
        despine(ax)
    axes[0].set_ylabel("accuracy, chance-normalised\n(1 = untouched, 0 = chance)")
    axes[0].set_ylim(-0.28, 1.12)
    axes[0].text(1.02, 1.03, "the model,\nuntouched", fontsize=7.5, color=INK2)
    axes[0].text(2.05, 0.55, "knee level", fontsize=7.5, color=CRITICAL)
    axes[2].annotate("reported knee: an extrapolation\nfrom the anchor and one point",
                     xy=(1.28, 0.5), xytext=(1.5, 0.80), fontsize=7.5, color=CRITICAL,
                     arrowprops=dict(arrowstyle="-", color=CRITICAL, lw=0.8,
                                     shrinkA=0, shrinkB=6))
    caption(fig, "The last firing of each FluxTune arm. The probe searches Φ ∈ {1.5 … 4}, a "
                 "range sized from offline measurements. Live, the very first point it tests "
                 "is already below the knee level, so the crossing lies to the LEFT of the "
                 "whole grid and the reported knee is an extrapolation between the synthetic "
                 "anchor at Φ=1 and that one point — the four points at 2.5–4.0 do no work at "
                 "all. The ruler starts past the mark it is meant to read.")
    save(fig, "fig6_ruler_starts_past_the_mark")




# ----------------------------------------------------------------- figure 7
TAIL_FRAC = 0.20          # window for the tail-slope fit -- the most recent
                          # fifth of the run, the most conservative choice


def fig7():
    """Does more budget still buy accuracy, at the point each run ended?"""
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.3))
    for ax, ds in zip(axes, DATASETS):
        rows = np.array(arm(f"{ds}_controller")["acc_budget"], dtype=float)
        B, acc = rows[:, 1], rows[:, 3]
        ref, c = ledger.REFERENCE[ds], DATASET_COLOR[ds]
        ax.plot(B, acc, color=c, lw=1.0, alpha=0.35)

        edges = np.linspace(0, B.max(), 11)
        mids, means = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            seg = acc[(B > lo) & (B <= hi)]
            if seg.size:
                mids.append((lo + hi) / 2); means.append(seg.mean())
        ax.plot(mids, means, color=c, lw=2.4, marker="o", ms=5)

        m = B > B.max() * (1 - TAIL_FRAC)
        slope, icept = np.polyfit(B[m], acc[m], 1)
        gap = ref - acc.max()
        ax.axhline(ref, color=INK3, lw=1.1, ls=(0, (5, 3)))
        ax.text(0.02, ref, "backprop reference", fontsize=7.5, color=INK2,
                va="bottom" if acc.max() <= ref else "top",
                transform=ax.get_yaxis_transform())

        if gap <= 0:
            lines = [f"tail slope  dAcc/dB = {slope:.2f}", "already past the reference"]
            col, xmax = GOOD, B.max() * 1.10
        else:
            need = gap / slope
            phi = np.exp(B.max() + need)
            reach = phi < PHI_CLIFF_LO
            col = GOOD if reach else CRITICAL
            lines = [f"tail slope  dAcc/dB = {slope:.2f}",
                     f"needs $\\Delta B$ = {need:.2f}  \u2192  $\\Phi$ = {phi:.1f}",
                     ("inside the cliff at $\\Phi$=3.63" if reach
                      else "far beyond any safe $\\Phi$")]
            xmax = B.max() * 1.10
            if reach:
                xs = np.array([B.max(), B.max() + need])
                ax.plot(xs, icept + slope * xs, color=col, lw=1.6, ls=(0, (2, 2)))
                ax.scatter([B.max() + need], [ref], s=64, marker="*", color=col,
                           edgecolor="white", linewidth=0.9, zorder=6)
                xmax = B.max() + need * 1.25
        ax.text(0.985, 0.035, "\n".join(lines), transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8, color=col)
        ax.set_title(ds, loc="left")
        ax.set_xlabel("$B$ — budget spent")
        ax.set_xlim(-0.03, xmax)
        despine(ax)
    axes[0].set_ylabel("held-out accuracy")
    caption(fig, "Held-out accuracy against budget spent, for the three FluxTune arms. Faint: "
                 "every eval. Bold: the mean in ten equal-width bands of B. The slope is fitted "
                 "by least squares over the last 20% of each run — the most recent, and the most "
                 "conservative, window. agnews is already past its reference. yahoo needs only "
                 "ΔB≈0.33, which lands at Φ≈2.8: past the shipped 2.7 rail but far inside the "
                 "Φ=3.63 cliff. yelp-p has flattened, so no amount of extra budget closes its "
                 "0.060 gap — that one is not a stopping problem at all.")
    save(fig, "fig7_does_more_budget_help")


FIGURES = {1: fig1, 2: fig2, 3: fig3, 4: fig4, 5: fig5, 6: fig6, 7: fig7}

if __name__ == "__main__":
    use_style()
    want = [int(a) for a in sys.argv[1:]] or sorted(FIGURES)
    for k in want:
        FIGURES[k]()
