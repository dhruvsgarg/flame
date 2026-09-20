"""3.1's `B_max` sensor: inflation injected rather than waited for.

The METHOD is not new (model §5.5b) -- `scripts/probe_inflation_damage.py` has
run it standalone since B-1. What lives here is the arithmetic both the offline
probe and the LIVE aggregator need, so the knee that sizes a run's budget is
defined in exactly one place.

Add isotropic Gaussian noise to the trainable slice scaled so `||theta_tr||`
grows by `Phi`, read held-out accuracy back at each `Phi`, and take the knee.
Forward passes only, no gradients -- the same operator set the method already
restricts itself to.

**Read the knee on CHANCE-NORMALIZED accuracy, never raw.** Post-collapse every
dataset floors at its own chance level `1/K` (B-1: 0.24 on agnews' 4 classes,
0.11 on yahoo's 10, 0.51 on yelp-p's 2), so a raw threshold means three
different things on three datasets and would reproduce none of B-1's numbers.

**Biased conservative, deliberately.** It noises a model that cannot re-fit,
while real training re-fits continuously, so it reads the knee ~0.6-1.2 low
(§7.1) -- it under-spends budget, never over-spends.
"""
import math

# Model §5.5b's ~6 evals. Spans B-1's whole measured range (yahoo/yelp-p ~2.0-2.3,
# agnews ~3.0-3.5) with a point either side of both.
PHI_GRID = (1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
KNEE_LEVEL = 0.5                # normalized accuracy defining the knee (B-1)
MIN_BASE_MARGIN = 0.05          # base must clear chance by this to be readable


def normalized(acc, base_acc, num_labels):
    """(acc - chance) / (base - chance). 1.0 at the unperturbed model, 0.0 at
    chance, and comparable across datasets with different `K`."""
    chance = 1.0 / num_labels
    den = base_acc - chance
    return (acc - chance) / den if den > 0 else float("nan")


def knee(phis, accs, base_acc, num_labels, level=KNEE_LEVEL):
    """Largest `Phi` whose normalized accuracy still clears `level`.

    Linearly interpolated between the two bracketing grid points, with `Phi`=1.0
    (normalized 1.0 by construction) as the implicit left anchor so the bracket
    always exists. Returns None when the base model is too close to chance for
    the curve to mean anything -- better to keep the current B_max than to size
    a budget off a degenerate head.
    """
    if base_acc - 1.0 / num_labels < MIN_BASE_MARGIN:
        return None
    pts = [(1.0, 1.0)] + [
        (float(p), normalized(a, base_acc, num_labels)) for p, a in zip(phis, accs)
    ]
    pts.sort(key=lambda t: t[0])
    prev_phi, prev_n = pts[0]
    for phi, n in pts[1:]:
        if n < level:
            if prev_n == n:
                return prev_phi
            # linear in Phi between the bracketing points
            return prev_phi + (prev_phi - phi) * (level - prev_n) / (prev_n - n)
        prev_phi, prev_n = phi, n
    return pts[-1][0]          # never crossed: the grid is the lower bound


def b_max_from_knee(phi_knee):
    return math.log(phi_knee) if phi_knee and phi_knee > 1.0 else None


def noise_scale(phi):
    """`||eps||` as a multiple of `||theta_tr||` for orthogonal noise to grow the
    norm by exactly `phi`: `||theta + eps||^2 = ||theta||^2 + ||eps||^2`, so
    `||eps|| = ||theta||*sqrt(phi^2 - 1)`. Isotropic noise in p~4.5e5 dimensions
    is orthogonal to any fixed vector to `1/sqrt(p)`, so this is exact enough
    that the realised norm ratio can be logged as a check rather than assumed.
    """
    return math.sqrt(max(0.0, phi * phi - 1.0))
