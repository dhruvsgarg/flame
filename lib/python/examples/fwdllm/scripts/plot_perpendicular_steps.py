"""§5.2 figure: server updates are perpendicular to θ, so a step is a turn of arctan ρ; set ρ, not η.

Reads `server_update` (trainable_weight_norm, trainable_delta_norm) from the ag60 runs (DistilBERT · AG News, 1 seed).
cos(θ_t, Δ_t) comes from ‖θ_{t+1}‖² = ‖θ_t‖² + 2⟨θ_t, Δ_t⟩ + ‖Δ_t‖².
"""
import glob
import json
import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "expt_scripts"))
from plotlib.baselines import FLUXTUNE_COLOR, style_for  # noqa: E402

EXPTS = os.path.join(HERE, "..", "experiments")
P = 450_340  # DistilBERT trainable params
FWD = style_for("fwdllm").color
# label -> (run glob, color, linestyle)
ARMS = {
    "FwdLLM (SGD)": ("run_20261022_072501_ag60_A0_*", FWD, "-"),
    "FwdLLM + fixed buffer (SGD)": ("run_20261022_074205_ag60_A3_*", FWD, "--"),
    "FwdLLM + normalized step": ("run_20261022_075429_ag60_A2_*", "#E69E66", "-"),
    "FluxTune": ("run_20261022_072504_ag60_A6_*", FLUXTUNE_COLOR, "-"),
    "FluxTune + momentum 0.5": ("run_20261022_094100_ag60_Smom_*", "#6baed6", "--"),
}


def load(pattern):
    run = glob.glob(os.path.join(EXPTS, pattern))[0]
    f = glob.glob(os.path.join(run, "telemetry", "aggregator*.jsonl"))[0]
    ev = [json.loads(l) for l in open(f) if '"server_update"' in l]
    ev = [e for e in ev if e.get("event") == "server_update" and not e.get("step_skipped")]
    w = np.array([e["trainable_weight_norm"] for e in ev])
    d = np.array([e["trainable_delta_norm"] for e in ev])
    # The logged delta sits one event apart on the SGD and trust-ratio paths; keep the pairing that closes the identity.
    best = None
    for dd in (d[:-1], d[1:]):
        cos = (w[1:] ** 2 - w[:-1] ** 2 - dd ** 2) / (2 * w[:-1] * dd)
        if best is None or np.std(cos) < np.std(best[0]):
            best = (cos, dd)
    cos, dd = best
    rho = dd / w[:-1]
    return dict(cos=cos, rho=rho, obs=w[1:] / w[0], pred=np.cumprod(np.sqrt(1 + rho ** 2)))


data = {k: load(v[0]) for k, v in ARMS.items()}
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#888", "axes.labelcolor": "#333", "xtick.color": "#555", "ytick.color": "#555"})
fig, axs = plt.subplots(2, 2, figsize=(9, 6.4))
ax = axs.ravel()

# (a) geometry
a = ax[0]
a.set_aspect("equal")
a.axis("off")
rho = 0.45  # exaggerated for legibility
t0, step = np.array([1.0, 0.0]), np.array([0.0, rho])
for start, vec, col, lab in [((0, 0), t0, "#333", r"$\theta_t$"), ((1, 0), step, FLUXTUNE_COLOR, r"$\Delta_t$"),
                             ((0, 0), t0 + step, "#333", r"$\theta_{t+1}$")]:
    a.add_patch(FancyArrowPatch(start, tuple(np.add(start, vec)), arrowstyle="-|>", mutation_scale=12, lw=1.6, color=col))
a.text(0.5, -0.09, r"$\theta_t$", ha="center", va="top")
a.text(1.04, rho / 2, r"$\Delta_t \perp \theta_t$", color=FLUXTUNE_COLOR, va="center")
a.text(0.42, 0.27, r"$\theta_{t+1}$", ha="right")
a.add_patch(Arc((0, 0), 0.6, 0.6, theta1=0, theta2=math.degrees(math.atan(rho)), color="#555", lw=1))
a.text(0.34, 0.06, r"$\arctan\rho$", fontsize=7)
a.plot([0.94, 0.94, 1.0], [0, 0.06, 0.06], color="#888", lw=0.8)
a.text(0.0, -0.32, r"$\rho = \|\Delta_t\| / \|\theta_t\|$" "\n" r"$\|\theta_{t+1}\| = \|\theta_t\|\sqrt{1+\rho^2}$"
       "\n" r"turn per update $= \arctan\rho \approx \rho$ rad", fontsize=7.5, va="top")
a.set_xlim(-0.05, 1.45)
a.set_ylim(-0.62, 0.55)
a.set_title("(a) A perpendicular step is a turn", loc="left", fontsize=8.5)

# (b) measured cos(θ, Δ) per update
b = ax[1]
band = 2 / math.sqrt(P)
b.axvspan(-band * 1e3, band * 1e3, color="#ddd", lw=0, zorder=0)
b.text(band * 1e3 + 1, -0.45, r"random direction $\pm 2/\sqrt{p}$", fontsize=6.5, color="#666")
rng = np.random.default_rng(0)
for i, (k, (_, col, ls)) in enumerate(reversed(ARMS.items())):
    c = data[k]["cos"] * 1e3
    b.scatter(c, i + rng.uniform(-0.18, 0.18, len(c)), s=4, color=col, alpha=0.55, lw=0)
    b.text(np.median(c), i + 0.32, f"median {np.median(c):+.1f}", fontsize=6, ha="center", color="#333")
b.set_yticks(range(len(ARMS)))
b.set_yticklabels(list(reversed(ARMS)), fontsize=7)
b.set_xlabel(r"cos($\theta_t$, $\Delta_t$)  ($\times 10^{-3}$)")
b.set_xlim(-10, 75)
b.set_ylim(-0.6, len(ARMS) - 0.4)
b.set_title("(b) Updates are perpendicular to θ, unless momentum is on", loc="left", fontsize=8.5)

# (c) predicted vs observed norm growth
c = ax[2]
for k, (_, col, ls) in ARMS.items():
    if "fixed buffer" not in k:  # grows 14x; error listed below
        c.plot(data[k]["pred"], data[k]["obs"], color=col, ls=ls, lw=2, label=k)
lim = [1, 2.25]
c.plot(lim, lim, color="#999", lw=0.8, ls=":", zorder=0)
c.set_xlim(lim)
c.set_ylim(lim)
c.set_xlabel(r"predicted $\prod_t \sqrt{1+\rho_t^2}$")
c.set_ylabel(r"observed $\|\theta_t\| / \|\theta_0\|$")
c.legend(fontsize=6.5, frameon=False, loc="upper left")
c.text(1.55, 1.02, "predicted / observed at run end:\n" + "\n".join(
    f"{k.split(' (')[0][:24]}: {100 * (data[k]['pred'][-1] / data[k]['obs'][-1] - 1):+.1f}%" for k in ARMS), fontsize=6, va="bottom")
c.set_title("(c) So weight growth follows from ρ alone", loc="left", fontsize=8.5)

# (d) turn per update
d = ax[3]
for k in ["FwdLLM (SGD)", "FwdLLM + fixed buffer (SGD)", "FluxTune"]:
    _, col, ls = ARMS[k]
    ang = np.arctan(data[k]["rho"])
    d.plot(np.arange(1, len(ang) + 1), ang, color=col, ls=ls, lw=1.2 if "SGD" in k else 2, alpha=0.8 if "SGD" in k else 1, label=k)
d.set_xlabel("server update")
d.set_ylabel("turn per update (rad)")
d.set_ylim(0, None)
d.legend(fontsize=6.5, frameon=False, loc="upper left")
d.set_title("(d) SGD's turn per update is an outcome; FluxTune sets it", loc="left", fontsize=8.5)

fig.text(0.995, 0.005, "DistilBERT · AG News, ag60 preview runs (1 seed, stopped at 0.60)", ha="right", fontsize=6, color="#888")
fig.tight_layout()
out = os.path.join(HERE, "..", "figs", "fig_perpendicular_steps")
fig.savefig(out + ".png", dpi=200)
fig.savefig(out + ".pdf")
print(out + ".png")
for k in ARMS:
    print(f"{k:30s} n={len(data[k]['cos'])} cos med {np.median(data[k]['cos']):+.5f} "
          f"p5/p95 {np.percentile(data[k]['cos'], 5):+.4f}/{np.percentile(data[k]['cos'], 95):+.4f} "
          f"growth obs {data[k]['obs'][-1]:.3f} pred {data[k]['pred'][-1]:.3f}")
