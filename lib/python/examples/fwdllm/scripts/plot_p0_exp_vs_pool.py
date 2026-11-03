"""P0 A/B: FluxTune `exp` vs `pool` rho schedule (DistilBERT, AG News, N=90). Accuracy and Phi vs commits."""
import glob
import math
import os
import re

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXPTS = os.path.join(HERE, "..", "experiments")
RUNS = {  # label -> (run prefix, color, rho(t))
    "exp": ("run_20261018_044202", "#2a78d6", lambda t: 0.0671 * math.exp(-(t - 1) / 600)),
    "pool": ("run_20261018_064840", "#eb6834", lambda t: 0.0671),
}
EVAL_EVERY = 4  # --eval-every-n-commits
SMOOTH = 10  # evals per rolling mean


def accs(prefix):
    log = glob.glob(os.path.join(EXPTS, prefix + "*", "*aggregator.log"))[0]
    return np.array([float(re.search(r"'acc': ([0-9.]+)", l).group(1))
                     for l in open(log) if "results after eval are" in l])


fig, (ax, axp) = plt.subplots(2, 1, figsize=(8, 7), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
for name, (prefix, color, rho) in RUNS.items():
    a = accs(prefix)
    c = EVAL_EVERY * np.arange(1, len(a) + 1)
    sm = np.convolve(a, np.ones(SMOOTH) / SMOOTH, mode="valid")
    ax.plot(c, a, color=color, lw=0.8, alpha=0.25)
    ax.plot(c[SMOOTH - 1:], sm, color=color, lw=2, label=f"{name} (peak {a.max():.3f})")
    ax.annotate(name, (c[-1], sm[-1]), xytext=(6, 0), textcoords="offset points", color="#0b0b0b", va="center")
    phi = np.exp(np.cumsum([0.5 * math.log1p(rho(t) ** 2) for t in range(1, c[-1] + 1)]))
    axp.plot(np.arange(1, c[-1] + 1), phi, color=color, lw=2, label=name)
    axp.annotate(f"{name}: {phi[-1]:.2f}", (c[-1], phi[-1]), xytext=(6, 0), textcoords="offset points",
                 color="#0b0b0b", va="center")

ax.axvline(490, color="#52514e", lw=1, ls=":")
ax.text(497, 0.795, "pool halts:\nΦ = 3 rail (c490)", color="#52514e", fontsize=9)
for y in (0.86, 0.87):
    ax.axhline(y, color="#52514e", lw=0.8, ls="--", alpha=0.6)
    ax.text(1180, y, f"{y:.2f}", color="#52514e", fontsize=8, va="center", ha="right")
ax.set_ylim(0.78, 0.885)  # zoomed: both arms pass 0.78 by ~c150
ax.set_ylabel("Test accuracy")
ax.set_title(f"P0: FluxTune ρ schedule, exp vs pool (DistilBERT · AG News · N=90, 1 seed)\n"
             f"zoomed to acc ≥ 0.78; faint = per eval; bold = {SMOOTH}-eval rolling mean", fontsize=11)
ax.legend(loc="lower right", frameon=False)

axp.axhline(3.0, color="#52514e", lw=1, ls=":")
axp.set_ylabel("Φ = Π√(1+ρ²)")
axp.set_xlabel("Server commits")
axp.set_xlim(0, 1180)
for a_ in (ax, axp):
    a_.grid(axis="y", color="#e4e3df", lw=0.6)
    a_.spines[["top", "right"]].set_visible(False)

out = os.path.join(HERE, "..", "p0_exp_vs_pool_acc_phi.png")
fig.tight_layout()
fig.savefig(out, dpi=150)
print(os.path.abspath(out))
