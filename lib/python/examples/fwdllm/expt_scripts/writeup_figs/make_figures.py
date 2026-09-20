#!/usr/bin/env python3
"""Render every figure in `fl_fwd_ft_writeup.md` into `../../figs/`.

  ./make_figures.py            # all
  ./make_figures.py 4 5        # just those

Historical numbers come from P4's ledger via `ledger.py`; the 2026-08-20 runs
come from `extract.py`'s JSON cache. Run `extract.py` first.
"""
import json
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import ledger
from figstyle import (AQUA, BLUE, CRITICAL, GOOD, GRID, INK, INK2, INK3, ORANGE,
                      SURFACE, WARNING, DATASET_COLOR, caption, despine, note,
                      use_style)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
FIGS = os.path.normpath(os.path.join(HERE, "..", "..", "figs"))
DATASETS = ("agnews", "yahoo", "yelp-p")

PHI_STOP = 2.7            # the shipped Phi backstop
PHI_CLIFF_LO = 3.63       # highest Phi at which an run still held its peak
PHI_CLIFF_HI = 4.23       # lowest Phi at which one did not


def phi_to_deg(phi):
    """Phi = sec(drift angle), because every step is perpendicular to theta."""
    phi = np.asarray(phi, dtype=float)
    return np.degrees(np.arccos(np.clip(1.0 / np.maximum(phi, 1e-9), -1.0, 1.0)))


def deg_to_phi(deg):
    deg = np.asarray(deg, dtype=float)
    return 1.0 / np.maximum(np.cos(np.radians(np.clip(deg, 0.0, 89.0))), 1e-9)


def run_data(name):
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
             "25-commit block of every run. So the weights grow,\nevery commit, forever.",
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
    axb.text(4.6, 0.40, f"no run holds\nabove $\\Phi={PHI_CLIFF_HI}$", fontsize=8,
             color=CRITICAL)
    axb.text(1.05, 0.24,
             f"every run below\n$\\Phi={PHI_CLIFF_LO}$ holds its peak:\nworst loss 0.014",
             fontsize=8, color=GOOD)
    axb.legend(loc="upper left", bbox_to_anchor=(0.015, 0.99))
    despine(axb)
    caption(fig, "22 runs that learned (peak ≥ 0.80), from P4's ledger. Runs that never "
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
    axe.set_ylabel("runs")
    worst = max(plain, key=lambda r: abs(r["phi_obs"] - r["phi_pred"]) / r["phi_pred"])
    axe.set_title(f"Median miss {np.median(np.abs(err)):.2f}%; worst "
                  f"{max(np.abs(err)):.1f}%", loc="left")
    axe.text(0.97, 0.86, f"the worst is `{worst['run']}`, the\nhighest-$\\rho$ raw-SGD run on\n"
             f"record ($\\rho$={worst['rho1']:.2f}) — the law's\naccuracy is $\\rho$-dependent",
             transform=axe.transAxes, ha="right", va="top", fontsize=7.5, color=INK3)
    despine(axe)
    caption(fig, "All 39 runs in P4's ledger, spanning ρ 0.0002–0.22, N 10–200, p 118k–450k, "
                 "and 66–3,353 commits. B is computed from the step sizes alone — no gradients, "
                 "no accuracy, no per-model constant. On the controlled runs the miss is under "
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
    ax.text(2.46, 0.884, "best agnews run on record 0.876", ha="right", fontsize=8,
            color=INK2)
    ax.set_xlabel(r"$\Lambda$ — progress banked  ($\sum \rho_t \cos_t$)")
    ax.set_ylabel("peak held-out accuracy")
    ax.set_title("Accuracy rises with progress banked — one curve per task, same shape",
                 loc="left")
    ax.set_xlim(-0.08, 2.52); ax.set_ylim(0.22, 0.95)
    ax.legend(loc="lower right", ncol=2, columnspacing=1.0)
    despine(ax)
    n16 = sum(1 for r in rows if r["rf"] == 16)
    caption(fig, f"Filled: {n16} runs at p=450k spanning both combination rules, both step "
                 "rules, both gates and α 0.1–1. Open: three runs at a different p, which "
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
        ctl, base = run_data(f"{ds}_anchor"), run_data(f"{ds}_control")
        cx, cy = np.array(ctl["acc"]).T
        bx, by = np.array(base["acc"]).T
        cx, bx = cx / 1000, bx / 1000
        ax.plot(bx, by, color=ORANGE, lw=1.7)
        ax.plot(cx, cy, color=BLUE, lw=1.7)

        tgt, ref = ledger.FL_TARGET[ds], ledger.REFERENCE[ds]
        ax.axhline(tgt, color=GOOD, lw=1.4, ls=(0, (5, 3)))
        ax.text(0.02, tgt, "FL target", fontsize=7.5, color=GOOD, weight="bold",
                va="bottom" if cy.max() <= tgt else "top",
                transform=ax.get_yaxis_transform())
        ax.axhline(ref, color=INK3, lw=0.9, ls=(0, (2, 3)))
        ax.text(0.55, ref, "centralized ceiling", fontsize=7, color=INK3,
                ha="center", va="bottom" if ref < tgt else "top",
                transform=ax.get_yaxis_transform())

        lo, hi = min(by.min(), cy.min()), max(tgt, ref, cy.max())
        pad = (hi - lo) * 0.34
        ax.set_ylim(lo - pad * 0.55, hi + pad * 0.30)

        # Direct labels rather than a legend box inside the data. These runs run
        # past their peak, so the marker goes on the peak, not the last point.
        ip = int(np.argmax(cy))
        ax.scatter([cx[ip]], [cy[ip]], s=80, marker="*", color=BLUE, zorder=6,
                   edgecolor="white", linewidth=1.0)
        ax.text(cx[ip], cy[ip] + (hi - lo) * 0.055, f"{cy.max():.3f}", color=BLUE,
                fontsize=9, weight="bold", ha="center")
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
        ax.text(0.0, 1.015, "ran to its compute ceiling; no stop fired",
                transform=ax.transAxes, va="bottom", fontsize=7.5, color=INK3)
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
    caption(fig, "Same stack on both runs; only the step rule differs. The arrow spans from "
                 "where FluxTune first reaches FluxTune-v2's best-ever accuracy to where v2 "
                 "finally reaches it. v2 never reaches FluxTune's accuracy on any dataset, given "
                 "its entire budget, and clears no FL target. Star = FluxTune's peak; these runs "
                 "then run past it, because nothing stops them. The centralized ceiling is a "
                 "plumbing check, not a bar -- it is a 10-client exact-gradient run.")
    save(fig, "fig4_fluxtune_vs_v2")


# ----------------------------------------------------------------- figure 5
def fig5():
    """The probe measures flat headroom; the mean combiner reports it shrinking."""
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(11.4, 4.2))
    for ds in DATASETS:
        pr = run_data(f"{ds}_controller")["probes"]
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
    caption(fig, "All 19 probe firings across the three FluxTune runs. Solid: what the probe "
                 "measured. Dashed: what the mean-of-all-senses combiner reported to the "
                 "step rule. Since ρ* = √(2·headroom/T_res), a 2.9× understatement in "
                 "headroom is a 1.7× understatement in step size.")
    save(fig, "fig5_the_combiner_throttles")


# ----------------------------------------------------------------- figure 6
def fig6():
    """The knee sits below the grid the probe searches."""
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.9), sharey=True)
    for ax, ds in zip(axes, DATASETS):
        pr = run_data(f"{ds}_controller")["probes"]
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
    caption(fig, "The last firing of each FluxTune run. The probe searches Φ ∈ {1.5 … 4}, a "
                 "range sized from offline measurements. Live, the very first point it tests "
                 "is already below the knee level, so the crossing lies to the LEFT of the "
                 "whole grid and the reported knee is an extrapolation between the synthetic "
                 "anchor at Φ=1 and that one point — the four points at 2.5–4.0 do no work at "
                 "all. The ruler starts past the mark it is meant to read.")
    save(fig, "fig6_ruler_starts_past_the_mark")




# ----------------------------------------------------------------- figure 7
TAIL_FRAC = 0.20          # window the 08-20 tail-slope fit used -- kept so the
                          # refuted prediction is redrawn exactly as it was made


def _smooth(v, w=11):
    return np.array([v[max(0, i - w + 1):i + 1].mean() for i in range(len(v))])


def fig7():
    """The tail-slope prediction, drawn against the run that tested it.

    The 08-20 runs stopped mid-rise; a least-squares slope over their last 20%
    of B said yahoo had 0.23 of accuracy per unit budget left. The 08-21 anchor
    runs spent ~0.4 more B on each dataset and settled the question.
    """
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.3))
    for ax, ds in zip(axes, DATASETS):
        old = np.array(run_data(f"{ds}_controller")["acc_budget"], dtype=float)
        new_ = np.array(run_data(f"{ds}_anchor")["acc_budget"], dtype=float)
        tgt, ref, c = ledger.FL_TARGET[ds], ledger.REFERENCE[ds], DATASET_COLOR[ds]

        Bo, ao = old[:, 1], old[:, 3]
        Bn, an = new_[:, 1], new_[:, 3]
        sn = _smooth(an)

        ax.plot(Bn, an, color=c, lw=0.8, alpha=0.22)
        ax.plot(Bn, sn, color=c, lw=2.2, label="08-21, anchor")
        ax.plot(Bo, _smooth(ao), color=INK3, lw=1.5, ls=(0, (4, 2)),
                label="08-20, where the slope was fitted")

        # the prediction, redrawn from the 08-20 run's own tail
        m = Bo > Bo.max() * (1 - TAIL_FRAC)
        slope, icept = np.polyfit(Bo[m], ao[m], 1)
        xs = np.linspace(Bo.max(), Bn.max(), 40)
        ax.plot(xs, icept + slope * xs, color=CRITICAL, lw=1.5, ls=(0, (2, 2)))

        ip = int(np.argmax(an))            # raw peak, per P4.4's scoring rule
        ax.scatter([Bn[ip]], [an[ip]], s=70, marker="*", color=c,
                   edgecolor="white", linewidth=0.9, zorder=6)
        ax.axhline(tgt, color=GOOD, lw=1.4, ls=(0, (5, 3)))
        ax.text(0.02, tgt, "FL target", fontsize=7.5, color=GOOD, weight="bold",
                va="bottom" if an.max() <= tgt else "top",
                transform=ax.get_yaxis_transform())
        ax.axhline(ref, color=INK3, lw=0.9, ls=(0, (2, 3)))
        ax.text(0.98, ref, "centralized ceiling", fontsize=7, color=INK3, ha="right",
                va="bottom" if ref < tgt else "top",
                transform=ax.get_yaxis_transform())

        io = int(np.argmax(ao))            # both runs scored on their raw peak
        pred = slope * (Bn[ip] - Bo[io])
        got = an[ip] - ao[io]
        ax.text(0.985, 0.035,
                f"predicted  {pred:+.3f}\nmeasured   {got:+.3f}",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8.5, color=CRITICAL, family="DejaVu Sans Mono")
        ax.set_title(ds, loc="left")
        ax.set_xlabel("$B$ — budget spent")
        despine(ax)
    axes[0].set_ylabel("held-out accuracy")
    axes[0].legend(loc="lower left", fontsize=7.5, bbox_to_anchor=(0.03, 0.02))
    caption(fig, "The dashed grey curve is where each 08-20 run stopped; the red dotted line is "
                 "its own tail slope extended forward -- the prediction. The solid curve is the "
                 "08-21 run that actually spent that budget (star = its peak; both runs scored on "
                 "their raw peak, per P4.4). On yahoo the slope promised +0.092 and delivered "
                 "+0.005, a 17x over-prediction; agnews missed by 3.4x and yelp-p came out with "
                 "the wrong sign. All three datasets are saturated, and a slope fitted inside the "
                 "rise says nothing about the plateau.")
    save(fig, "fig7_does_more_budget_help")


# ----------------------------------------------------------------- figure 8
def fig8():
    """Q1: the gate's ruler stretches with the thing it measures.

    Per-commit `var`/||theta|| series were not retained for any FwdLLM run
    (the pre-08-16 portfolio is off disk), so the shapes here are the measured
    endpoints -- 6x norm growth, 36x floor drift -- carried by the laws that
    produced them. Re-run to plot the series directly.
    """
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(11, 4.0))
    t = np.linspace(0, 1000, 400)
    norm = 1 + 5 * t / 1000                      # 1 -> 6x over the run
    floor = 0.3 / 36 * norm ** 2                 # var ~ ||g||^2, and 6^2 = 36

    axa.set_title("The threshold is fixed. The ruler is not.", loc="left")
    axa.plot(t, floor, color=ORANGE, label="achievable $var$ floor")
    axa.axhline(0.3, color=CRITICAL, lw=1.6, ls=(0, (4, 3)))
    axa.set_yscale("log")
    axa.text(0.04, 0.905, "var_threshold = 0.3  — fixed, for the whole run",
             transform=axa.transAxes, color=CRITICAL, fontsize=8.5, weight="bold")
    note(axa, "$\\|\\theta\\|$ grows 6$\\times$\n$\\Rightarrow$ floor drifts 36$\\times$",
         (700, floor[280]), (250, 0.11), color=ORANGE)
    axa.text(0.97, 0.53, "the floor rises to MEET the threshold\n"
                         "$\\Rightarrow$ $N$ must grow 36$\\times$ to keep the gate firing",
             transform=axa.transAxes, ha="right", fontsize=8, color=INK2)
    axa.set_xlabel("commit"); axa.set_ylabel("per-coordinate variance  (log)")
    despine(axa)

    axb.set_title("What that forces: an anneal nobody asked for", loc="left")
    axb.plot(t, norm ** 2, color=BLUE, label="pool size  $N \\propto \\|\\theta\\|^2$")
    axb.set_ylabel("$N$ / $N_0$", color=BLUE)
    axb.tick_params(axis="y", colors=BLUE)
    ax2 = axb.twinx()
    ax2.plot(t, 1 / norm, color=AQUA)
    ax2.set_ylabel("$\\rho$ / $\\rho_0$", color=AQUA)
    ax2.tick_params(axis="y", colors=AQUA)
    ax2.grid(False)
    ax2.set_ylim(0, 1.08)
    axb.text(0.30, 0.90, "$\\rho \\propto 1/\\|\\theta\\|$, and $\\|\\theta\\| \\propto \\sqrt{t}$\n"
                         "$\\Rightarrow$ Robbins-Monro BY ACCIDENT",
             transform=axb.transAxes, va="top", fontsize=8.5, color=INK)
    axb.text(0.30, 0.22, "not a fix: $\\Sigma\\rho^2$ still diverges\n"
                         "$\\Rightarrow$ collapse deferred to commit 1,200-1,700,\n"
                         "     not prevented",
             transform=axb.transAxes, va="top", fontsize=8, color=CRITICAL)
    axb.set_xlabel("commit")
    despine(axb, keep=("left", "bottom"))
    despine(ax2, keep=("right",))

    caption(fig, "The FwdLLM commit gate holds var <= 0.3, but var = 2b^2||g||^2/N carries units of "
                 "||theta||^2. Measured over one run: ||theta|| grows 6x, so the achievable variance "
                 "floor drifts 36x while the threshold stays put (left). Holding a fixed threshold "
                 "against a stretching ruler forces N to grow as ||theta||^2 and rho to fall as "
                 "1/||theta|| (right) -- a decaying step schedule the designers never wrote, and the "
                 "only reason the system does not diverge on the timescales it was run for. "
                 "ILLUSTRATIVE: the 6x and 36x endpoints are measured; the per-commit series were not "
                 "retained and this needs a re-run to plot directly.")
    save(fig, "fig8_the_stretching_ruler")


# ----------------------------------------------------------------- figure 9
def fig9():
    """Q1: one threshold, three different meanings across heterogeneity."""
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    alphas = ["0.1", "1", "100"]
    floors = [1.33, 0.64, 0.28]
    cols = [CRITICAL, GOOD, WARNING]
    bars = ax.bar(alphas, floors, color=cols, width=0.56, zorder=3)
    ax.axhline(0.3, color=INK, lw=1.8, ls=(0, (4, 3)), zorder=4)
    ax.text(-0.44, 2.02, "var_threshold = 0.3 — one number, shipped.\n"
                         "It is the $\\alpha$=1, $K$=20 corner of an ($\\alpha$, $K$) surface.",
            color=INK, fontsize=8.5, ha="left", va="top", weight="bold")
    verdicts = ["4.4$\\times$ ABOVE 0.3:\ngate can barely\never fire",
                "2.1$\\times$ above at this $K$\n-- it only lands ON 0.3\nat $K$=20",
                "BELOW 0.3: gate\nfires on the first\nreading"]
    for b, f, v, c in zip(bars, floors, verdicts, cols):
        ax.text(b.get_x() + b.get_width() / 2, f + 0.06, f"{f:.2f}", ha="center",
                fontsize=9.5, weight="bold", color=c)
        ax.text(b.get_x() + b.get_width() / 2, f + 0.20, v, ha="center",
                fontsize=7.6, color=INK2, va="bottom")
    ax.set_ylim(0, 2.28)
    ax.set_xlabel("Dirichlet $\\alpha$  (client heterogeneity)")
    ax.set_ylabel("achievable $var$ floor")
    ax.set_title("The same threshold means three different things", loc="left")
    despine(ax)
    caption(fig, "Achievable variance floor across a 1000x sweep in Dirichlet alpha, all other "
                 "settings held. The shipped threshold 0.3 lands exactly on the alpha=1, K=20 floor "
                 "-- one point of a two-dimensional (alpha, K) surface, on one model, on one dataset. "
                 "Move heterogeneity and the same number is either unreachable or trivially "
                 "satisfied. Retuning it does not help: it fits a different corner. This is what "
                 "'nothing transferred' means concretely. Floors measured over the 4 alpha-sweep runs at "
                 "matched K; the shipped 0.3 comes from a different corner of the same surface "
                 "(alpha=1, K=20), which is the point: one number cannot describe a two-dimensional "
                 "surface. K=10 survives at alpha=100 while K=20 does not stabilise at alpha=0.1.")
    save(fig, "fig9_one_threshold_three_meanings")


# ---------------------------------------------------------------- figure 10
def fig10():
    """Q2: the cliff is a budget reading, so going faster arrives sooner."""
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(11.4, 4.3),
                                   gridspec_kw=dict(width_ratios=[1.25, 1]))

    # (a) Phi trajectories. FluxTune is logged per commit; the v1-style runs are
    # geometric interpolations pinned to their two logged endpoints; FwdLLM is
    # the documented sqrt(t) extrapolation.
    axa.set_title("$\\Phi$ over a run: who reaches the cliff, and when", loc="left")
    axa.axhspan(PHI_CLIFF_LO, PHI_CLIFF_HI, color=CRITICAL, alpha=0.13, lw=0)
    axa.axhline(PHI_CLIFF_HI, color=CRITICAL, lw=1.0, ls=(0, (3, 3)))
    axa.text(1830, PHI_CLIFF_HI + 0.45, "above 4.23: every run lost its peak",
             color=CRITICAL, fontsize=8, ha="right", weight="bold")
    axa.text(1830, 2.85, "below 3.63: every run held it",
             color=GOOD, fontsize=8, ha="right", weight="bold")

    v1 = [("200358", 195, 5.74), ("013806", 328, 9.47), ("200325", 177, 4.23)]
    for run, T, phi in v1:
        t = np.linspace(1, T, 200)
        axa.plot(t, phi ** (t / T), color=ORANGE, lw=1.9,
                 alpha=0.95 if run == "200325" else 0.55)
    axa.text(355, 7.6, "FluxTune-v1\n(async + JVP, raw SGD)\nstraight through the band",
             color=ORANGE, fontsize=8.5, weight="bold")

    t = np.linspace(1, 1800, 400)
    fw = np.sqrt(1 + t / 98.0)                # rho ~ 1/||theta||, ||theta|| ~ sqrt(t)
    axa.plot(t, fw, color=BLUE, lw=2.0)
    hit = t[np.searchsorted(fw, PHI_CLIFF_LO)]
    axa.scatter([hit], [PHI_CLIFF_LO], s=70, marker="X", color=BLUE, zorder=6)
    note(axa, "FwdLLM crosses here:\ncommit 1,200-1,700 --\nnobody ever ran it this far",
         (hit, PHI_CLIFF_LO), (1245, 1.95), color=BLUE)
    axa.text(60, 3.05, "FwdLLM\n(accidental anneal)", color=BLUE, fontsize=8.5,
             weight="bold")

    for ds in DATASETS:
        b = np.array(run_data(f"{ds}_controller")["budget"], dtype=float)
        axa.plot(b[:, 0], np.exp(b[:, 2]), color=GOOD, lw=1.8, alpha=0.85)
    axa.text(430, 1.16, "FluxTune (this work)\nlevels off below the band",
             color=GOOD, fontsize=8.5, weight="bold")

    axa.set_xlim(0, 1850); axa.set_ylim(0.9, 10.2)
    axa.set_yscale("log")
    axa.set_yticks([1, 2, 3, 4, 6, 10])
    axa.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axa.set_xlabel("commit"); axa.set_ylabel("$\\Phi = \\|\\theta_t\\|/\\|\\theta_0\\|$  (log)")
    despine(axa)

    # (b) what each run walked away with, against where it ended up.
    axb.set_title("What the run kept, against where it stopped", loc="left")
    axb.axvspan(PHI_CLIFF_LO, PHI_CLIFF_HI, color=CRITICAL, alpha=0.13, lw=0)
    rows = sorted([r for r in ledger.load() if r["peak"] >= 0.80],
                  key=lambda r: r["phi_obs"])
    for i, r in enumerate(rows):
        drop = r["peak"] - r["final"]
        col = CRITICAL if drop > 0.05 else GOOD
        axb.plot([r["phi_obs"]] * 2, [r["peak"], r["final"]], color=col, lw=1.4,
                 zorder=3)
        axb.scatter([r["phi_obs"]], [r["peak"]], s=26, facecolor="white",
                    edgecolor=col, linewidth=1.3, zorder=4)
        axb.scatter([r["phi_obs"]], [r["final"]], s=30, color=col, zorder=5)
        if drop > 0.4:
            up = r["phi_obs"] < 10
            axb.annotate(r["run"], (r["phi_obs"], r["final"]),
                         xytext=(0, 7 if up else -7), textcoords="offset points",
                         fontsize=7, color=CRITICAL, ha="center",
                         va="bottom" if up else "top")
    axb.set_xscale("log")
    axb.set_xticks([1, 2, 3, 4, 6, 12])
    axb.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axb.set_xlabel("$\\Phi$ at the end of the run  (log)")
    axb.set_ylabel("accuracy")
    axb.text(0.03, 0.17, "hollow = peak reached\nfilled  = what it ended with",
             transform=axb.transAxes, fontsize=7.6, color=INK2)
    axb.set_ylim(0.20, 0.93)
    despine(axb)

    caption(fig, "The damage threshold is a Phi -- a budget reading, not a clock reading. "
                 "FluxTune-v1 added async aggregation and JVP-magnitude probe selection, worth 1.7x "
                 "progress per commit (rho*sqrt(N) = 1.68 vs 1.01); with a raw SGD step and nothing "
                 "counting the spend, that speed also carries it through the cliff band inside the "
                 "run window. FwdLLM's dimensionally-wrong gate anneals rho as 1/||theta||, so its "
                 "own crossing sits out at commit 1,200-1,700 -- past where it was ever run. It does "
                 "not avoid the failure; it defers it. Right: every run that reached 0.80, peak to "
                 "final against where it stopped. LEFT PANEL PARTLY ILLUSTRATIVE: FluxTune is logged "
                 "per commit; the v1 runs are geometric interpolations pinned to their two logged "
                 "endpoints; FwdLLM is the documented sqrt(t) extrapolation. The pre-08-16 run logs "
                 "are off disk -- re-run to plot the true series.")
    save(fig, "fig10_fast_is_not_safe")


# ---------------------------------------------------------------- figure 11
def fig11():
    """Q3: who sets what -- hand-set knobs versus sensed quantities."""
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(11.4, 4.2))
    HAND, SENSE, PROBE, OPER = CRITICAL, GOOD, BLUE, INK2

    def box(ax, xy, w, h, label, color, sub=None, fill=0.10):
        x, y = xy
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color, alpha=fill,
                                   edgecolor=color, linewidth=1.5, zorder=2))
        ax.text(x + w / 2, y + h * (0.60 if sub else 0.5), label, ha="center",
                va="center", fontsize=8.6, weight="bold", color=INK, zorder=3)
        if sub:
            ax.text(x + w / 2, y + h * 0.26, sub, ha="center", va="center",
                    fontsize=7.2, color=INK2, zorder=3)

    def arrow(ax, a, b, color=INK3, style="-|>"):
        ax.annotate("", xy=b, xytext=a,
                    arrowprops=dict(arrowstyle=style, color=color, lw=1.3))

    for ax in (axa, axb):
        ax.set_xlim(-0.75, 10); ax.set_ylim(0.25, 9.8)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        despine(ax, keep=())

    axa.set_title("FwdLLM: every number is hand-set", loc="left", color=CRITICAL)
    box(axa, (0.4, 7.6), 4.2, 1.5, "var_threshold = 0.3", HAND,
        "per (model, dataset, $\\alpha$, K)")
    box(axa, (5.4, 7.6), 4.2, 1.5, "learning rate $\\eta$", HAND, "hand-searched")
    box(axa, (0.4, 5.0), 9.2, 1.5, "commit gate:  pool until  $var \\leq 0.3$", HAND,
        "$var \\propto \\|g\\|^2$ -- carries units, so the setpoint drifts 36$\\times$")
    box(axa, (0.4, 2.4), 9.2, 1.5, "step:  $\\Delta\\theta = \\eta \\cdot G$", HAND,
        "$\\rho$ is an OUTCOME, never chosen")
    arrow(axa, (2.5, 7.5), (2.5, 6.6)); arrow(axa, (7.5, 7.5), (7.5, 6.6))
    arrow(axa, (5.0, 4.9), (5.0, 4.0))
    axa.annotate("", xy=(0.4, 5.6), xytext=(0.4, 3.2),
                 arrowprops=dict(arrowstyle="-|>", color=CRITICAL, lw=1.5,
                                 connectionstyle="arc3,rad=0.45"))
    axa.text(-0.45, 4.35, "runaway: $\\|\\theta\\|\\uparrow \\Rightarrow \\|g\\|\\uparrow "
                          "\\Rightarrow$ step $\\uparrow$", fontsize=7.8,
             color=CRITICAL, rotation=90, ha="center", va="center", weight="bold")
    axa.text(5.0, 1.2, "nothing measures what the run is spending",
             ha="center", fontsize=8.4, color=CRITICAL, weight="bold")

    axb.set_title("FluxTune: three deployment facts, the rest is sensed", loc="left",
                  color=GOOD)
    box(axb, (0.3, 8.2), 9.4, 1.2, "OPERATOR:  model + PEFT scheme  ·  rank $\\Rightarrow p$  "
                                   "·  compute budget", OPER, "deployment facts, not tuning knobs",
        fill=0.07)
    box(axb, (0.3, 5.9), 4.3, 1.5, "$B_{max}$ -- WRONG WALL", WARNING,
        "injection probe, ~6 fwd / 150\nreads a FROZEN model: 1.8x low", fill=0.13)
    axb.text(2.62, 5.66, "the only sensed number -- and the open defect (fig 13)",
             ha="center", va="top", fontsize=7.2, color=CRITICAL, weight="bold",
             zorder=4)
    box(axb, (5.4, 5.9), 4.3, 1.5, "$\\rho^* = \\sqrt{2(B_{max}-B)/T_{res}}$", SENSE,
        "law C -- a RATE, not a deadline")
    box(axb, (5.4, 3.5), 4.3, 1.3, "gate:  $N = p(\\rho^*/s)^2/P$", SENSE,
        "dimensionless")
    box(axb, (0.3, 3.5), 4.3, 1.3, "$B \\;+\\!\\!=\\; \\frac{1}{2}\\ln(1+\\rho^2)$", SENSE,
        "exact, free, from $\\rho$ alone")
    box(axb, (0.3, 1.2), 9.4, 1.3, "commit  ·  $\\theta \\leftarrow \\theta - \\rho\\|\\theta\\| "
                                   "G/\\|G\\|$", SENSE, "trust-ratio: $\\rho$ is enacted, to 8.7e-5")
    arrow(axb, (4.6, 6.65), (5.4, 6.65))
    arrow(axb, (7.55, 5.8), (7.55, 4.9))
    arrow(axb, (5.4, 4.15), (4.6, 4.15))
    arrow(axb, (2.45, 3.4), (2.45, 2.6))
    axb.annotate("", xy=(0.3, 6.1), xytext=(0.3, 2.4),
                 arrowprops=dict(arrowstyle="-|>", color=GOOD, lw=1.5,
                                 connectionstyle="arc3,rad=-0.28"))
    axb.text(-0.45, 4.3, "closed on a MEASURED budget", fontsize=7.8, color=GOOD,
             rotation=90, ha="center", va="center", weight="bold")
    axb.text(5.0, 0.45, "no learning rate · no threshold · no cohort width · no run length",
             ha="center", fontsize=8.2, color=GOOD, weight="bold")
    axb.text(5.0, 7.90, "everything below the operator row is arithmetic or measured -- "
                        "except the box in amber", ha="center", fontsize=7.4,
             color=INK2, style="italic")

    caption(fig, "Left: in FwdLLM every quantity in the loop is a constant somebody chose, and the two "
                 "that matter most -- the variance threshold and the learning rate -- carry units of "
                 "the thing they are supposed to regulate, so they mean something different on every "
                 "model, dataset and heterogeneity level. Right: FluxTune takes three deployment "
                 "facts and senses the rest. B is exact arithmetic on the step sizes it already "
                 "chose; B_max is measured on the running model with forward passes only; rho* and "
                 "the pool size follow. The constants that remain (s, T_res=300, the Phi rail) are "
                 "derived, provably unable to buy accuracy, or a safety rail under test -- and all "
                 "were validated at a single p. The amber box is the exception and it is the "
                 "open defect: B_max is read off a FROZEN model's tolerance to injected noise, "
                 "which figure 13 shows is a different quantity from the wall a re-fitting run "
                 "meets -- so the loop's arithmetic is sound and its target is not.")
    save(fig, "fig11_who_sets_what")



# ---------------------------------------------------------------- figure 12
def fig12():
    """Every run peaks at the same Phi, then gives it back."""
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(12.6, 4.5),
                                   gridspec_kw={"width_ratios": [1.55, 1]})

    peaks = {}
    for ds in DATASETS:
        rows = np.array(run_data(f"{ds}_anchor")["acc_budget"], dtype=float)
        B, acc = rows[:, 1], rows[:, 3]
        phi, sm = np.exp(B), _smooth(acc)
        c = DATASET_COLOR[ds]
        ip = int(np.argmax(sm))
        peaks[ds] = (phi[ip], sm[ip])
        # normalised so three datasets at different accuracy share one axis
        axa.plot(phi, sm - sm[ip], color=c, lw=2.0)
        axa.scatter([phi[ip]], [0], s=80, marker="*", color=c,
                    edgecolor="white", linewidth=0.9, zorder=6)
        axa.text(1.35 + 1.55 * DATASETS.index(ds), 0.058,
                 f"{ds}  $\\Phi$={np.round(phi[ip] + 1e-9, 2):.2f}", color=c, fontsize=8.5,
                 ha="left", va="center", weight="bold")

    lo, hi = min(p for p, _ in peaks.values()), max(p for p, _ in peaks.values())
    axa.axvspan(lo, hi, color=GOOD, alpha=0.10, zorder=0)
    axa.text((lo + hi) / 2, 0.030, "peaks\n2.82-3.00", ha="center", va="center",
             fontsize=8, color=GOOD, weight="bold")
    axa.axvline(PHI_STOP, color=WARNING, lw=1.4, ls=(0, (4, 2)))
    axa.text(PHI_STOP - 0.06, -0.062, "shipped rail 2.7\n(never actually ran)",
             ha="right", fontsize=7.8, color=WARNING)
    axa.axvspan(PHI_CLIFF_LO, PHI_CLIFF_HI, color=CRITICAL, alpha=0.09, zorder=0)
    axa.text((PHI_CLIFF_LO + PHI_CLIFF_HI) / 2, 0.030, "portfolio\ncliff",
             ha="center", va="center", fontsize=8, color=CRITICAL)
    axa.axhline(0, color=INK3, lw=0.8)
    axa.set_xlim(1.0, 6.2)
    axa.set_ylim(-0.16, 0.072)
    axa.set_xlabel("$\\Phi$ — how much the model has inflated")
    sec = axa.secondary_xaxis("top", functions=(phi_to_deg, deg_to_phi))
    sec.set_xticks([0, 37, 60, 70, 74, 76, 80])
    sec.set_xlabel("the same axis, read as a drift angle off $\\theta_0$   "
                   "($1/\\Phi$ = retention)", fontsize=8.5)
    sec.tick_params(labelsize=7.6)
    axa.set_ylabel("held-out accuracy, relative to this run's peak")
    axa.set_title("Three tasks, one peak location", loc="left")
    despine(axa)

    # right: what each candidate stopping rule would have delivered
    rules = ["$\\Phi$=2.7", "$\\Phi$=3.0", "saturation", "$\\Phi$=3.63", "ran to\nthe ceiling"]
    cost = {"agnews":  [-0.0045, -0.0057, -0.0050, -0.0077, -0.0294],
            "yahoo":   [-0.0049, -0.0003, -0.0078, -0.0149, -0.1365],
            "yelp-p":  [-0.0050, -0.0020, -0.0053, -0.0141, -0.1067]}
    x = np.arange(len(rules)); w = 0.26
    for k, ds in enumerate(DATASETS):
        axb.bar(x + (k - 1) * w, cost[ds], w, color=DATASET_COLOR[ds], label=ds)
    axb.axhline(0, color=INK3, lw=0.9)
    axb.set_xticks(x); axb.set_xticklabels(rules, fontsize=7.8)
    axb.set_ylabel("accuracy given up vs the peak")
    axb.set_title("What each stopping rule costs", loc="left")
    axb.legend(loc="lower left", fontsize=7.8)
    despine(axb)

    caption(fig, "Left: the three 08-21 runs, each shifted so its own peak sits at zero, against "
                 "Phi. Peaks land at 2.82 / 3.00 / 2.91 on 4-, 10- and 2-class tasks -- a band of "
                 "0.18 -- and every run then decays, gently to about 3.3 and faster after. Right: "
                 "the cost of stopping at each candidate rule. The shipped 2.7 rail gives up 0.005 "
                 "everywhere; 3.0 and the sized saturation detector are the two best rules and "
                 "agree with each other; running to the vclock ceiling, which is what these runs "
                 "actually did because both shipped stops are inert, gives up 0.03 to 0.14. "
                 "The top axis reads the same trajectory as an angle: Phi = sec(drift), so the "
                 "peaks sit at 69-70 degrees off the pretrained point and the cliff at 74-76.")
    save(fig, "fig12_where_every_run_peaks")



# ---------------------------------------------------------------- figure 13
def fig13():
    """Injected inflation is not earned inflation: the probe measures another wall."""
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.1), sharey=True)
    labels = {"agnews": 4, "yahoo": 10, "yelp-p": 2}

    for ax, ds in zip(axes, DATASETS):
        d = run_data(f"{ds}_anchor")
        rows = np.array(d["acc_budget"], dtype=float)
        phi, sm = np.exp(rows[:, 1]), _smooth(rows[:, 3])
        chance, c = 1.0 / labels[ds], DATASET_COLOR[ds]
        peak = sm.max()

        def norm(a):
            return (np.asarray(a, dtype=float) - chance) / (peak - chance)

        ax.axhspan(-0.14, 0.05, color=INK3, alpha=0.09, lw=0, zorder=0)
        if ds == "agnews":
            ax.text(1.02, -0.10, "chance", fontsize=7.4, color=INK2, ha="left")

        # earned: the run's own accuracy at the commit where Phi reached that value
        ax.plot(phi, norm(sm), color=c, lw=2.3, zorder=4)
        ip = int(np.argmax(sm))
        ax.scatter([phi[ip]], [norm(sm[ip])], marker="*", s=130, color=c,
                   edgecolor="white", linewidth=0.9, zorder=6)
        ax.text(phi[ip], 1.10, f"peaks at $\\Phi$={np.round(phi[ip] + 1e-9, 2):.2f}",
                ha="center",
                fontsize=8, color=c, weight="bold")

        # injected: every probe firing from commit 600 on
        fires = [p for p in d["probes"] if p["commit"] >= 600]
        grid = [q[0] for q in fires[0]["curve"]]
        M = np.array([[norm(dict(p["curve"])[g]) for g in grid] for p in fires])
        ax.fill_between(grid, M.min(0), M.max(0), color=CRITICAL, alpha=0.13, lw=0,
                        zorder=2)
        ax.plot(grid, M.mean(0), color=CRITICAL, lw=1.8, ls=(0, (4, 2)), marker="o",
                ms=5.5, markerfacecolor="white", zorder=5)

        # the one comparison the whole section turns on
        i2 = int(np.argmin(np.abs(phi - 2.0)))
        lo, hi = float(M.mean(0)[grid.index(2.0)]), float(norm(sm[i2]))
        ax.annotate("", xy=(2.0, hi), xytext=(2.0, lo),
                    arrowprops=dict(arrowstyle="<->", color=INK, lw=1.3))
        ax.plot([2.0], [lo], marker="_", ms=9, color=CRITICAL)

        ax.set_xlim(0.95, 4.25)
        ax.set_xlabel("$\\Phi$ — inflation, however it was reached")
        ax.set_title(f"{ds}   ·   {len(fires)} probe firings", loc="left")
        despine(ax)

    axes[0].set_ylim(-0.14, 1.19)
    axes[0].set_ylabel("held-out accuracy\n(chance-corrected, 1 = this run's peak)")
    axes[0].plot([], [], color=INK, lw=2.3, label="EARNED — the run at this $\\Phi$")
    axes[0].plot([], [], color=CRITICAL, lw=1.8, ls=(0, (4, 2)), marker="o", ms=5.5,
                 markerfacecolor="white", label="INJECTED — the probe at this $\\Phi$")
    axes[0].legend(loc="center left", fontsize=7.8, frameon=False,
                   bbox_to_anchor=(0.02, 0.42))
    axes[1].text(2.08, 0.52, "same $\\Phi$,\ntwo answers", fontsize=8.2, color=INK,
                 weight="bold")

    caption(fig, "Both curves are chance-corrected and normalised to each run's own peak, so "
                 "the three tasks share one axis. Solid: the run's accuracy at the commit "
                 "where its own Phi reached that value. Dashed: the injection probe's reading "
                 "at the same Phi, averaged over every firing from commit 600 on; band = "
                 "min-max over firings. At Phi = 2 the probe reports CHANCE on all three "
                 "datasets while the run is within 3% of its peak, and every injected point "
                 "at Phi >= 2 reads chance on all 40 firings. The only difference between the "
                 "two protocols is whether the classifier was allowed to re-fit between "
                 "increments -- so most of the wall is re-fittable, and B_max, the quantity "
                 "law C anneals against, has never measured the wall the budget is about.")
    save(fig, "fig13_injected_is_not_earned")


FIGURES = {1: fig1, 2: fig2, 3: fig3, 4: fig4, 5: fig5, 6: fig6, 7: fig7,
           8: fig8, 9: fig9, 10: fig10, 11: fig11, 12: fig12, 13: fig13}

if __name__ == "__main__":
    use_style()
    want = [int(a) for a in sys.argv[1:]] or sorted(FIGURES)
    for k in want:
        FIGURES[k]()
