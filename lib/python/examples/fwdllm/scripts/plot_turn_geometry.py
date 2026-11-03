"""§5.2 concept figure: the same turn needs a step proportional to ‖θ‖ (Δ = ρ‖θ‖), since updates are ⟂ θ.

LAYOUT "overlay": both θ on one axes; equal turns share one ray, the old step turns the longer θ less.
LAYOUT "side_by_side": short θ left, longer θ right.
Edit the CONFIG block and re-run; writes figs/fig_turn_geometry.{png,pdf}.
"""
import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "expt_scripts"))
from plotlib.baselines import FLUXTUNE_COLOR, style_for  # noqa: E402

# ---- CONFIG ------------------------------------------------------------------
LAYOUT = "overlay"          # "overlay" or "side_by_side"
DIRECTION_DEG = 25          # direction of θ in both panels
NORM_SMALL = 1.5            # ‖θ‖ early in training (left)
NORM_LARGE = 3.0            # ‖θ‖ after the weights grew (right)
RHO = 0.35                  # relative step = tan(turn); exaggerated for legibility (recipe: 0.07)
SHOW_SAME_STEP = True       # right panel: also apply the left panel's step (a fixed absolute step, SGD-like)
XLIM, YLIM = (-0.2, 3.6), (-0.2, 2.5)
FIGSIZE = (9, 4)            # side_by_side
FIGSIZE_OVERLAY = (5.6, 4.4)
FILL_TRIANGLES = True       # overlay: shade the two similar triangles
SMALL_THETA_COLOR = "#aaaaaa"
SMALL_STEP_COLOR = "#6baed6"  # lighter blue for the short θ's step
THETA_COLOR = "#333333"
STEP_COLOR = FLUXTUNE_COLOR                 # Δ = ρ‖θ‖
SAME_STEP_COLOR = style_for("fwdllm").color  # fixed absolute step
ARROW_LW, ARROW_HEAD = 1.8, 12
LABEL_FS, NOTE_FS, TITLE_FS = 9, 8, 9.5
ARC_RADIUS_FRAC = 0.35      # turn arc radius as a fraction of ‖θ‖
ARC_LABEL_PAD = 0.08        # arc label distance past the arc's upper end
THETA_LABEL_OFFSET = 0.18   # θ label distance below the vector
STEP_LABEL_OFFSET = 0.08    # step label distance past the step, along θ
TITLES = (r"(a) $\|\theta\|$ = {small:g}: step $\rho\|\theta\|$ turns θ by $\arctan\rho$",
          r"(b) $\|\theta\|$ = {large:g}: the same turn needs a {ratio:g}× longer step")
TITLE_OVERLAY = r"Same turn $\arctan\rho$, step $\Delta = \rho\|\theta\|$ grows with $\|\theta\|$"
OUT = os.path.join(HERE, "..", "figs", "fig_turn_geometry")
# ------------------------------------------------------------------------------

u = np.array([math.cos(math.radians(DIRECTION_DEG)), math.sin(math.radians(DIRECTION_DEG))])  # θ direction
n = np.array([-u[1], u[0]])                                                                      # ⟂ direction
TURN = math.atan(RHO)


def arrow(ax, start, end, color, lw=ARROW_LW, ls="-"):
    ax.add_patch(FancyArrowPatch(tuple(start), tuple(end), arrowstyle="-|>", mutation_scale=ARROW_HEAD,
                                 lw=lw, color=color, linestyle=ls, shrinkA=0, shrinkB=0))


def right_angle(ax, corner, size=0.09, color="#888"):
    pts = np.array([corner - size * u, corner - size * u + size * n, corner + size * n])
    ax.plot(pts[:, 0], pts[:, 1], color=color, lw=0.8)


def turn_arc(ax, radius, angle, color, label, at="end"):
    ax.add_patch(Arc((0, 0), 2 * radius, 2 * radius, theta1=DIRECTION_DEG,
                     theta2=DIRECTION_DEG + math.degrees(angle), color=color, lw=1.1))
    phi = math.radians(DIRECTION_DEG) + (angle if at == "end" else angle / 2)
    ax.text((radius + ARC_LABEL_PAD) * math.cos(phi), (radius + ARC_LABEL_PAD) * math.sin(phi), label, color=color,
            fontsize=NOTE_FS, ha="right" if at == "end" else "left", va="bottom" if at == "end" else "center")


def setup(ax, title):
    ax.set_aspect("equal")
    ax.axhline(0, color="#ccc", lw=0.8, zorder=0)
    ax.axvline(0, color="#ccc", lw=0.8, zorder=0)
    ax.grid(color="#eee", lw=0.6, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_xlabel("weight coordinate 1", fontsize=NOTE_FS)
    ax.tick_params(labelsize=NOTE_FS - 1)
    ax.set_title(title, fontsize=TITLE_FS, loc="left")


def draw_turn(ax, norm, step_len, color, label, label_side="right", label_xy=None):
    """θ (norm along u), a ⟂ step of step_len, the new θ, and its turn arc."""
    tip = norm * u
    step = step_len * n
    angle = math.atan(step_len / norm)
    arrow(ax, tip, tip + step, color)
    arrow(ax, (0, 0), tip + step, color, lw=1.0, ls="--")
    side = 1 if label_side == "right" else -1
    ax.text(*(tip + step / 2 + side * STEP_LABEL_OFFSET * u if label_xy is None else label_xy), label.format(step=step_len, turn=angle),
            color=color, fontsize=NOTE_FS, ha="left" if side > 0 else "right", va="center")
    return angle


def draw_theta(ax, norm, sub, color=THETA_COLOR, lw=ARROW_LW, label_at=0.5, head=True):
    tip = norm * u
    if head:
        arrow(ax, (0, 0), tip, color, lw=lw)
    else:  # band + tip dot, so a longer θ drawn on top stays readable
        ax.plot([0, tip[0]], [0, tip[1]], color=color, lw=lw, solid_capstyle="butt", zorder=1)
        ax.plot(*tip, "o", color=color, ms=5, zorder=3)
    right_angle(ax, tip)
    ax.text(*(label_at * tip - THETA_LABEL_OFFSET * n), rf"$\theta_{{{sub}}}$", color=color, fontsize=LABEL_FS,
            ha="center", va="center")


def overlay():
    fig, ax = plt.subplots(figsize=FIGSIZE_OVERLAY)
    setup(ax, TITLE_OVERLAY)
    ax.set_ylabel("weight coordinate 2", fontsize=NOTE_FS)
    if FILL_TRIANGLES:
        for norm, col in ((NORM_LARGE, STEP_COLOR), (NORM_SMALL, SMALL_STEP_COLOR)):
            tri = np.array([(0, 0), norm * u, norm * u + RHO * norm * n])
            ax.fill(tri[:, 0], tri[:, 1], color=col, alpha=0.08, lw=0)
    draw_theta(ax, NORM_SMALL, "t", color=SMALL_THETA_COLOR, lw=ARROW_LW + 3, head=False)  # wide band under the long θ
    draw_theta(ax, NORM_LARGE, "t'", label_at=0.8)
    a = draw_turn(ax, NORM_LARGE, RHO * NORM_LARGE, STEP_COLOR, r"$\Delta' = \rho\|\theta_{{t'}}\|$ = {step:.2f}")
    draw_turn(ax, NORM_SMALL, RHO * NORM_SMALL, SMALL_STEP_COLOR, r"$\Delta = \rho\|\theta_t\|$ = {step:.2f}",
              label_xy=NORM_SMALL * u - 0.2 * n)
    turn_arc(ax, ARC_RADIUS_FRAC * NORM_SMALL, a, STEP_COLOR, f"same turn\n{a:.2f} rad", at="end")
    if SHOW_SAME_STEP:
        a = draw_turn(ax, NORM_LARGE, RHO * NORM_SMALL, SAME_STEP_COLOR, r"$\Delta$ reused on $\theta_{{t'}}$" "\nturns only {turn:.2f} rad")
        turn_arc(ax, 1.25 * ARC_RADIUS_FRAC * NORM_LARGE, a, SAME_STEP_COLOR, "")
    ax.text(0.98, 0.03, rf"$\Delta' / \Delta = \|\theta_{{t'}}\| / \|\theta_t\|$ = {NORM_LARGE / NORM_SMALL:g}×", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=NOTE_FS, color="#333")
    return fig


def side_by_side():
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=FIGSIZE, sharey=True)
    setup(ax_l, TITLES[0].format(small=NORM_SMALL))
    setup(ax_r, TITLES[1].format(large=NORM_LARGE, ratio=NORM_LARGE / NORM_SMALL))
    ax_l.set_ylabel("weight coordinate 2", fontsize=NOTE_FS)

    draw_theta(ax_l, NORM_SMALL, "t")
    a = draw_turn(ax_l, NORM_SMALL, RHO * NORM_SMALL, STEP_COLOR, r"$\Delta = \rho\|\theta\|$ = {step:.2f}")
    turn_arc(ax_l, ARC_RADIUS_FRAC * NORM_SMALL, a, STEP_COLOR, f"{a:.2f} rad")

    draw_theta(ax_r, NORM_LARGE, "t'")
    a = draw_turn(ax_r, NORM_LARGE, RHO * NORM_LARGE, STEP_COLOR, r"$\Delta = \rho\|\theta\|$ = {step:.2f}")
    turn_arc(ax_r, ARC_RADIUS_FRAC * NORM_LARGE, a, STEP_COLOR, f"{a:.2f} rad")
    if SHOW_SAME_STEP:
        a = draw_turn(ax_r, NORM_LARGE, RHO * NORM_SMALL, SAME_STEP_COLOR, "same step {step:.2f}\nturns only {turn:.2f} rad")
        turn_arc(ax_r, 0.6 * ARC_RADIUS_FRAC * NORM_LARGE, a, SAME_STEP_COLOR, "")
    return fig


fig = overlay() if LAYOUT == "overlay" else side_by_side()
fig.tight_layout()
fig.savefig(OUT + ".png", dpi=200)
fig.savefig(OUT + ".pdf")
print(OUT + ".png")
