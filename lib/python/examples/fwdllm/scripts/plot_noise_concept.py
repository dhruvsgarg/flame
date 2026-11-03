"""§3.1 concept options: a forward-gradient estimate is almost all noise; averaging helps, selection does not.

Simulation, not run data: exact gradient g = e1, Gaussian perturbations v, estimate ĝ = (g·v) v, p = DistilBERT's.
Writes figs/concept_noise_opt{A,B,C,D}.png.
"""
import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "expt_scripts"))
from plotlib.baselines import FLUXTUNE_COLOR, style_for  # noqa: E402

# ---- CONFIG ------------------------------------------------------------------
P_DIM = 450_340      # trainable params
P_PERT = 10          # perturbations per client update
N_POOL = 50          # client updates per server update
K_MAX = 1000         # longest running average in option B
TRIALS = 4
SEED = 0
SELECT_COLOR = style_for("fwdllm").color
SINGLE_COLOR = "#888888"
MEAN_COLOR = FLUXTUNE_COLOR
OUT = os.path.join(HERE, "..", "figs", "concept_noise_opt{}")
# ------------------------------------------------------------------------------

rng = np.random.default_rng(SEED)
plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False})


def draw(k):
    """k estimates (g·v) v with g = e1, as float32 rows."""
    v = rng.standard_normal((k, P_DIM), dtype=np.float32)
    return v[:, :1] * v


def parts(vec):
    """(component along g, perpendicular length, cosine)."""
    a = float(vec[0])
    b = float(np.sqrt(np.dot(vec, vec) - a * a))
    return a, b, a / math.hypot(a, b)


def rules():
    """Same 500 perturbations, combined four ways."""
    est = draw(P_PERT * N_POOL).reshape(N_POOL, P_PERT, P_DIM)
    jvp = np.abs(est[:, :, 0])  # |g·v|² ordering = |JVP| ordering
    best = est[np.arange(N_POOL), jvp.argmax(1)]
    return {
        "1 perturbation": parts(est[0, 0]),
        f"best of {P_PERT} by |JVP|": parts(best[0]),
        f"mean of {P_PERT}": parts(est[0].mean(0)),
        f"server update: {N_POOL} × best": parts(best.mean(0)),
        f"server update: {N_POOL} × mean": parts(est.mean((0, 1))),
    }


def save(fig, tag):
    fig.tight_layout()
    fig.savefig(OUT.format(tag) + ".png", dpi=200)
    fig.savefig(OUT.format(tag) + ".pdf")
    plt.close(fig)
    print(OUT.format(tag) + ".png")


R = [rules() for _ in range(TRIALS)]
med = {k: tuple(np.median([r[k][i] for r in R]) for i in range(3)) for k in R[0]}
COLORS = [SINGLE_COLOR, SELECT_COLOR, MEAN_COLOR, SELECT_COLOR, MEAN_COLOR]

# ---- A: high dimension makes every direction perpendicular --------------------
fig, ax = plt.subplots(figsize=(5, 3.3))
for d, col in [(3, "#cccccc"), (100, "#999999"), (10_000, "#555555"), (P_DIM, MEAN_COLOR)]:
    x = rng.standard_normal(200_000)
    cos = x / np.sqrt(x ** 2 + rng.chisquare(d - 1, x.size))
    ax.hist(cos, bins=np.linspace(-1, 1, 801), density=True, histtype="step", lw=1.6, color=col,
            label=f"p = {d:,}" + ("  (DistilBERT adapters)" if d == P_DIM else ""))
ax.set_yscale("log")
ax.set_xlim(-1, 1)
ax.set_xlabel("cos(random perturbation, true gradient)")
ax.set_ylabel("density (log)")
ax.legend(frameon=False, fontsize=7.5)
ax.set_title(f"A random direction is perpendicular in high dimension: |cos| ≈ 1/√p = {1 / math.sqrt(P_DIM):.4f}",
             loc="left", fontsize=8.5)
save(fig, "A")

# ---- B: alignment vs number averaged; selection does not move along the line ---
fig, ax = plt.subplots(figsize=(5, 3.5))
ks = np.unique(np.logspace(0, math.log10(K_MAX), 40).astype(int))
curves = []
for _ in range(TRIALS):
    s, out, k0 = np.zeros(P_DIM, np.float32), [], 0
    for k in ks:
        s += draw(k - k0).sum(0)
        k0 = k
        out.append(parts(s)[2])
    curves.append(out)
ax.plot(ks, np.median(curves, 0), color=MEAN_COLOR, lw=2, label="average of k perturbations (simulated)")
ax.plot(ks, np.sqrt(ks / P_DIM), color=MEAN_COLOR, lw=1, ls=":", label=r"theory $\sqrt{k/p}$")
for k, lab, x in [(f"best of {P_PERT} by |JVP|", "best of 10", P_PERT), (f"server update: {N_POOL} × best", f"{N_POOL} × best of 10", P_PERT * N_POOL)]:
    ax.scatter([x], [med[k][2]], color=SELECT_COLOR, s=40, zorder=3)
    ax.annotate(lab, (x, med[k][2]), xytext=(6, -10), textcoords="offset points", color=SELECT_COLOR, fontsize=7.5)
ax.axvline(P_PERT * N_POOL, color="#bbb", lw=0.8, ls="--")
ax.text(P_PERT * N_POOL * 1.05, 0.0012, f"FluxTune server update\nP·N = {P_PERT * N_POOL}", fontsize=7, color="#666")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("perturbations computed")
ax.set_ylabel("cos(estimate, true gradient)")
ax.legend(frameon=False, fontsize=7.5, loc="upper left")
ax.set_title("Averaging climbs as √k; selection uses the compute but stays low", loc="left", fontsize=8.5)
save(fig, "B")

# ---- C: signal vs noise length per rule ---------------------------------------
fig, ax = plt.subplots(figsize=(5.4, 3.3))
names = list(med)
y = np.arange(len(names))[::-1]
ax.barh(y + 0.18, [med[k][0] for k in names], height=0.34, color=COLORS, label="along true gradient (signal)")
ax.barh(y - 0.18, [med[k][1] for k in names], height=0.34, color=COLORS, alpha=0.35, label="perpendicular (noise)")
for yi, k in zip(y, names):
    ax.text(med[k][1] * 1.15, yi - 0.18, f"cos {med[k][2]:.4f}", va="center", fontsize=7)
ax.set_yticks(y)
ax.set_yticklabels(names, fontsize=7.5)
ax.set_xscale("log")
ax.set_xlabel("length (units of ‖g‖)")
ax.legend(frameon=False, fontsize=7, loc="lower right")
ax.set_title("Selection grows signal and noise together; averaging shrinks noise", loc="left", fontsize=8.5)
save(fig, "C")

# ---- D: angle from the true gradient ------------------------------------------
fig, (ax, az) = plt.subplots(1, 2, figsize=(7, 3.3), gridspec_kw=dict(width_ratios=[1, 1.4]))
ax.set_aspect("equal")
ax.axis("off")
ax.annotate("", (1, 0), (0, 0), arrowprops=dict(arrowstyle="-|>", color="#222", lw=2))
ax.text(1.02, -0.06, "true gradient (backprop)", fontsize=7.5, va="top")
th = np.linspace(0, math.pi / 2, 100)
ax.plot(np.cos(th), np.sin(th), color="#ddd", lw=0.8)
for k, col in zip(names, COLORS):
    ang = math.acos(med[k][2])
    ax.annotate("", (math.cos(ang), math.sin(ang)), (0, 0), arrowprops=dict(arrowstyle="-|>", color=col, lw=1.2))
ax.text(0.03, 1.05, "every estimate ≈ 90°", fontsize=7.5)
ax.set_xlim(-0.1, 1.6)
ax.set_ylim(-0.2, 1.2)
ax.set_title("Seen at full scale", loc="left", fontsize=8.5)
for yi, (k, col) in enumerate(zip(names[::-1], COLORS[::-1])):
    deg = math.degrees(math.acos(med[k][2]))
    az.barh(yi, 90 - deg, color=col)
    az.text(90 - deg + 0.05, yi, f"{deg:.2f}°", va="center", fontsize=7)
az.set_yticks(range(len(names)))
az.set_yticklabels(names[::-1], fontsize=7.5)
az.set_xlabel("degrees closer to the gradient than a perpendicular (90° − angle)")
az.set_title("Zoomed: averaging tilts toward the gradient", loc="left", fontsize=8.5)
save(fig, "D")

for k in names:
    print(f"{k:28s} signal {med[k][0]:.3f} noise {med[k][1]:9.1f} cos {med[k][2]:.5f}")
