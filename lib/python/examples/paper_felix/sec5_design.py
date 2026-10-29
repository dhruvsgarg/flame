"""§5 Design: the analytic Felix-Agg weight curve; the per-mechanism data figures come from scripts/paper_plots.py (README)."""
import numpy as np

from figlib import COL_W, FONT, INK, LINE, STACK, Z, label, place, plt, save

SEC = "s5_design"
FELIX, FEDBUFF = STACK["Felix"][0], STACK["FedBuff"][0]


def fig5_4_agg():
    """§5.4 (analytic, no runs): update weight vs staleness τ per rule; Felix-Agg = rate `new` (s 0.4, a 0.25, b 0.1)."""
    fig, ax = plt.subplots(figsize=(COL_W, 1.6))
    tau = np.linspace(0, 40, 200)
    ax.plot(tau, 1 / np.sqrt(1 + tau), color=FEDBUFF, lw=LINE["width"], label="FedBuff: 1/√(1+τ)")
    ax.plot(tau, 1 / (1 + tau), color=INK, ls="--", lw=LINE["width"], label="1/(1+τ)")
    for u, ls in ((1000, "-"), (10, ":")):
        ax.plot(tau, 0.4 * (1 + tau) ** -0.25 + 0.6 * (1 - (1 + u) ** -0.1), color=FELIX, ls=ls, lw=LINE["hero"],
                label=f"{label('Felix')}-Agg, loss utility {u}", zorder=Z["hero"])
    ax.set(xlim=(0, 40), ylim=(0, 1.05), xlabel="Staleness τ (versions)", ylabel="Update Weight")
    fig.legend(*ax.get_legend_handles_labels(), loc="outside upper center", ncol=2, columnspacing=0.8)
    w = lambda t, u=1000: 0.4 * (1 + t) ** -0.25 + 0.6 * (1 - (1 + u) ** -0.1)
    place(ax, rf"At τ = 10: $\mathbf{{{w(10) / (1 / np.sqrt(11)):.1f}\times}}$ FedBuff's weight",
          [(39, 0.98, "right", "top"), (39, 0.9, "right", "top")], fontsize=FONT["annot"])
    return save(fig, SEC, "fig5_4_agg")


FIGS = (fig5_4_agg,)  # 5.1-5.3 now come from the FB2 evaluation figures (swap_availability, picks / swap_selection, pace_bytes)
