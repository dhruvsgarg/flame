"""Exercise row P3's retention probe: cos(theta_t, theta_0) against 1/Phi.

Checks (a) theta_0 is stashed at the FIRST commit and refused afterwards -- a
    theta stashed at commit 5 is not theta_0 and a probe that pretended
    otherwise would report a wrong number silently,
(b) the identity it exists to test actually holds under exactly perpendicular
    steps: cos(theta_t,theta_0)*Phi = 1, which is the prediction a run must
    reproduce to 1.00 +/- 0.02,
(c) a step with a RADIAL component breaks it in the stated direction, so the
    probe can tell the two apart -- otherwise it would confirm the prediction
    whatever the geometry, and
(d) it is off by default and costs nothing when off.
"""
import math
import sys

import torch

sys.path.insert(0, "/home/dgarg39/flame/lib/python")
from examples.fwdllm.aggregator.FedSgdAggregator import FedSGDAggregator as A  # noqa: E402


class _M(torch.nn.Module):
    """A trainable slice and a frozen one, so the probe's requires_grad filter
    is exercised rather than assumed."""

    def __init__(self, p=4096, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.tr = torch.nn.Parameter(torch.randn(p, generator=g))
        self.frozen = torch.nn.Parameter(torch.randn(p, generator=g),
                                         requires_grad=False)


class _T:
    def __init__(self, m):
        self.model = m


def make(every=1, p=4096):
    a = object.__new__(A)
    a._retention_every = every
    a._theta_0 = None
    a._theta_0_norm = 0.0
    a._commit_count = 0
    a._B = 0.0
    a.trainer = _T(_M(p))
    return a


def step(a, rho, radial=0.0, gen=None):
    """One commit: a step of relative length rho, perpendicular to theta unless
    `radial` says otherwise. Banks B exactly as the commit path does."""
    with torch.no_grad():
        th = a.trainer.model.tr
        d = torch.randn(th.shape, generator=gen)
        d -= th * (float((d * th).sum()) / float(th.pow(2).sum()))   # project out
        d /= d.norm()
        if radial:
            d = (1 - radial) * d + radial * th / th.norm()
            d /= d.norm()
        th.add_(d * (rho * float(th.norm())))
    a._commit_count += 1
    a._B += 0.5 * math.log1p(rho ** 2)


# (a) theta_0 is stashed at commit 0 and refused after
a = make()
a._stash_theta_0()
assert a._theta_0 is not None and a._theta_0_norm > 0
n0 = a._theta_0_norm
a._stash_theta_0()                       # idempotent
assert a._theta_0_norm == n0
b = make()
b._commit_count = 5                      # too late to be theta_0
b._stash_theta_0()
assert b._theta_0 is None and b._retention_every == 0
print(f"  theta_0 stash         : taken at commit 0 (||theta_0||={n0:.3f}), "
      f"refused at commit 5 and the probe turns itself off")

# (b) THE PREDICTION: perpendicular steps give cos*Phi = 1
g = torch.Generator().manual_seed(3)
for _ in range(400):
    step(a, 0.05, gen=g)
th = a.trainer.model.tr.detach()
cos = float((th * a._theta_0[0]).sum()) / (float(th.norm()) * n0)
phi = math.exp(a._B)
assert abs(cos * phi - 1.0) < 0.02, cos * phi
print(f"  perpendicular steps   : cos={cos:.4f} Phi={phi:.4f} "
      f"cos*Phi={cos * phi:.4f} (P3' gate: 1.00 +/- 0.02), "
      f"drift={math.degrees(math.acos(cos)):.1f} deg")

# and the emitter runs on the real path without faulting
a._log_retention()

# (c) a radial component breaks it, and in the stated direction -- a step that
# overlaps theta_0 leaves MORE retention than Phi predicts, so cos*Phi > 1
c = make()
c._stash_theta_0()
g = torch.Generator().manual_seed(4)
for _ in range(400):
    step(c, 0.05, radial=0.5, gen=g)
th = c.trainer.model.tr.detach()
cos_r = float((th * c._theta_0[0]).sum()) / (float(th.norm()) * c._theta_0_norm)
assert cos_r * math.exp(c._B) > 1.02, cos_r * math.exp(c._B)
print(f"  radial component      : cos*Phi={cos_r * math.exp(c._B):.4f} > 1 -- "
      f"the probe DISCRIMINATES, it does not just confirm")

# (d) off by default costs nothing
d = make(every=0)
d._stash_theta_0()
d._log_retention()
assert d._theta_0 is None
print("  every=0               : nothing stashed, nothing emitted")

print("\nALL RETENTION-PROBE CHECKS PASS")
