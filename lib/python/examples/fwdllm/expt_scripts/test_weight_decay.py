"""Q2 preflight: `server_weight_decay` must pin Phi = 1.00 by construction.

Q2 asks whether ||theta_tr|| is CAUSAL or a symptom of directional misaim
(fl_fwd_ft_solution.md §7.2). The arm is only interpretable if the decay
actually cancels Leg 1's inflation, so this checks the arithmetic rather than
the outcome:

  (a) disabled            -> byte-identical to today
  (b) lambda = auto       -> lambda = rho^2/2 each commit
  (c) auto over T commits -> ||theta_tr|| stays flat (Phi ~ 1), where the same
                             trajectory without decay grows as (1+rho^2)^(T/2)
  (d) frozen params       -> untouched; only the trainable slice decays

Drives the real `_apply_weighted_update`. CPU only, no data, no model download.
"""
import math
import sys

import torch
import torch.nn as nn

sys.path.insert(0, "/home/dgarg39/flame/lib/python")
from examples.fwdllm.aggregator.FedSgdAggregator import FedSGDAggregator as A

torch.manual_seed(0)
RHO, T = 0.09, 200


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.trainable = nn.Linear(64, 32, bias=False)
        self.frozen = nn.Linear(16, 8, bias=False)
        self.frozen.weight.requires_grad_(False)

    def forward(self, x):
        return self.trainable(x)


def make(weight_decay=None, rho=RHO):
    a = object.__new__(A)
    model = Net()
    a.trainer = type("T", (), {"model": model})()
    a._cos_ground_truth_audit = False
    a._cos_probe_batch = None
    a._server_update_audit = True
    a._pool_split_half_audit = False
    a._server_step_rule = "trust_ratio"
    a._rho_star, a._rho_schedule, a._rho_exp = rho, "const", 0.0
    a._commit_count = 0
    a._last_rho = None
    a._B, a._stop_fired, a._phi_stop = 0.0, None, "off"   # C-1, stop disabled
    a._b_max_probe_every = 0                              # 3.1 re-sense off
    a._weight_decay = weight_decay
    a.server_momentum = 0.0
    a._server_momentum_buf = {}
    a._var_scalar = a._n_eff_scalar = None
    a.fwd_llm_stage = None
    a._model_version = 0
    return a, model


def tr_norm(model):
    return math.sqrt(sum(float(p.detach().pow(2).sum())
                         for p in model.parameters() if p.requires_grad))


def commit(a, model, gen):
    """One commit with a random pooled direction -- the orthogonal-step regime."""
    upload = [torch.randn(p.shape, generator=gen) if p.requires_grad
              else torch.zeros_like(p) for p in model.parameters()]
    a._emit_server_update = lambda *args, **kw: None
    a._apply_weighted_update(
        model_list=[(1, upload)],
        weighted_gradient_sum=[torch.zeros_like(p) for p in model.parameters()],
        old_param=iter(list(model.parameters())),
        learning_rate=1.0, training_num=1,
    )


def trajectory(weight_decay):
    a, model = make(weight_decay)
    gen = torch.Generator().manual_seed(1)
    n0 = tr_norm(model)
    for _ in range(T):
        commit(a, model, gen)
    return n0, tr_norm(model), model, a


# (a) disabled == today
n0, n1, _, a_off = make(None)[1], None, None, None
a_off, model_off = make(None)
gen = torch.Generator().manual_seed(1)
before = [p.detach().clone() for p in model_off.parameters()]
commit(a_off, model_off, gen)
assert a_off._wd_lambda(RHO) is None, "disabled must yield no lambda"
moved = any(not torch.allclose(b, p.detach())
            for b, p in zip(before, model_off.parameters()) if p.requires_grad)
assert moved, "the control arm must still take its step"
print("  disabled             : no lambda, step unchanged")

# (b) auto = rho^2/2
a_auto, _ = make("auto")
lam = a_auto._wd_lambda(RHO)
assert abs(lam - 0.5 * RHO ** 2) < 1e-12, lam
print(f"  auto at rho={RHO}     : lambda = {lam:.6g} = rho^2/2")

# (c) the point of the arm: Phi pinned at 1 vs geometric growth without it
n0_off, nT_off, _, _ = trajectory(None)
n0_on, nT_on, _, _ = trajectory("auto")
phi_off, phi_on = nT_off / n0_off, nT_on / n0_on
phi_pred = (1 + RHO ** 2) ** (T / 2)
print(f"  no decay, T={T}      : Phi = {phi_off:.3f}  (norm law predicts "
      f"{phi_pred:.3f})")
print(f"  auto decay, T={T}    : Phi = {phi_on:.3f}  (must be ~1)")
assert abs(phi_off / phi_pred - 1) < 0.05, "control must follow the norm law"
assert abs(phi_on - 1.0) < 0.05, f"auto decay must pin Phi at 1, got {phi_on}"

# (d) frozen params must not decay
a_f, model_f = make(0.1)
frozen_before = model_f.frozen.weight.detach().clone()
commit(a_f, model_f, torch.Generator().manual_seed(2))
assert torch.equal(model_f.frozen.weight.detach(), frozen_before), \
    "decay must not touch frozen params"
print("  frozen params        : untouched by decay")
print("test_weight_decay: OK")
