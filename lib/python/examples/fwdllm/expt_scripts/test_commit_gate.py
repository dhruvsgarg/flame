"""Exercise S-C's `_gate_satisfied` / `_n_required` on a stub aggregator.

Checks (a) commit_gate=var is byte-identical to the old `var <= thr` test,
(b) N_req = p*(rho/s)^2/G_rule, so it falls as rho anneals and rises with p,
(c) the gate commits exactly when the measured pool reaches N_req, and
(d) with no rho yet (raw_sgd, first commit) it refuses, leaving the cap to fire.
"""
import sys, types, torch
sys.path.insert(0, "/home/dgarg39/flame/lib/python")
from examples.fwdllm.aggregator.FedSgdAggregator import FedSGDAggregator as A

P_TRAIN, S = 1040932, 0.4


def make(gate="n_target", rule="trust_ratio", rho=0.01, g_rule=2.988, n=100):
    a = object.__new__(A)
    a._commit_gate = gate; a._gate_safety_s = S
    a._server_step_rule = rule; a._rho_star = rho
    a._rho_schedule = "const"; a._rho_exp = 0.55; a._commit_count = 0
    a._p_trainable = P_TRAIN; a._g_rule = g_rule; a._last_rho = None
    a._n_eff_scalar = float(n)
    a.grad_for_var_check_list = [torch.zeros(1)] * n
    a.var = torch.tensor(0.5); a.var_threshold = 0.3
    return a


# (a) var mode untouched
a = make(gate="var")
assert a._gate_satisfied() is False, "var=0.5 > thr=0.3 must not commit"
a.var = torch.tensor(0.2)
assert a._gate_satisfied() is True, "var=0.2 <= thr=0.3 must commit"
print("  var gate            : unchanged (0.5 -> POOL, 0.2 -> COMMIT)")

# (b) N_req closed form, and its two dependencies
for rho in (0.02, 0.01, 0.005):
    want = P_TRAIN * (rho / S) ** 2 / 2.988
    got = make(rho=rho)._n_required()
    assert abs(got - want) < 1e-6, (rho, got, want)
    print(f"  N_req  rho*={rho:<6} select : {got:9.1f}")
print(f"  N_req  rho*=0.01   mean P=10 : {make(rho=0.01, g_rule=10.0)._n_required():9.1f}"
      f"   (G_rule 10 vs 2.988 -> 3.35x smaller pool)")

# (c) commit exactly at the boundary
req = make(rho=0.005)._n_required()
below, above = int(req) - 1, int(req) + 2
assert make(rho=0.005, n=below)._gate_satisfied() is False
assert make(rho=0.005, n=above)._gate_satisfied() is True
print(f"  boundary at N={req:.1f} : n={below} POOL, n={above} COMMIT")

# (d) raw_sgd before any commit has no rho -> refuse, let the cap fire
a = make(rule="raw_sgd")
assert a._n_required() is None and a._gate_satisfied() is False
a._last_rho = 0.005
assert abs(a._n_required() - P_TRAIN * (0.005 / S) ** 2 / 2.988) < 1e-6
print("  raw_sgd cold start  : refuses until a rho exists, then sizes from it")
