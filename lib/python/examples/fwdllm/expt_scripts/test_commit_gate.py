"""Exercise S-C's `_gate_satisfied` / `_n_required` on a stub aggregator.

Checks (a) commit_gate=var is byte-identical to the old `var <= thr` test,
(b) N_req = p*(rho/s)^2/G_rule, so it falls as rho anneals and rises with p,
(c) the gate commits exactly when the measured pool reaches N_req,
(d) with no rho yet (raw_sgd, first commit) it refuses, leaving the cap to fire,
and (e) gate_rho_ref=setpoint holds N_req flat under the anneal where `annealed`
    decays it to zero -- the §22.3a composition bug and its fix.
"""
import sys, types, torch
sys.path.insert(0, "/home/dgarg39/flame/lib/python")
from examples.fwdllm.aggregator.FedSgdAggregator import FedSGDAggregator as A

# PRODUCTION p (handoff §7): create_model counts 1,040,932 but the trainer drops
# pre_classifier before any probe is drawn, so N_req is sized off 450,340.
P_TRAIN, S = 450340, 0.4


def make(gate="n_target", rule="trust_ratio", rho=0.01, g_rule=2.988, n=100,
         schedule="const", rho_ref="annealed", commit=0):
    a = object.__new__(A)
    a._commit_gate = gate; a._gate_safety_s = S; a._gate_rho_ref = rho_ref
    a._server_step_rule = rule; a._rho_star = rho
    a._rho_schedule = schedule; a._rho_exp = 0.55; a._commit_count = commit
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

# (e) the §22.3a composition bug: under `rm`, `annealed` decays N_req to nothing
#     (measured 28.1 -> 0.0 by commit ~20, gate floors at I=1 for 1,271 commits)
#     while `setpoint` holds it. Same rho* and schedule; only the reference differs.
print("  gate_rho_ref under rm anneal (rho*=0.02, exp=0.55, mean P=10):")
print(f"    {'commit':>7} {'rho_t':>10} {'N_req annealed':>15} {'N_req setpoint':>15}")
for c in (0, 20, 200, 1200):
    ann = make(rho=0.02, g_rule=10.0, schedule="rm", rho_ref="annealed", commit=c)
    setp = make(rho=0.02, g_rule=10.0, schedule="rm", rho_ref="setpoint", commit=c)
    n_a, n_s = ann._n_required(), setp._n_required()
    assert abs(n_s - P_TRAIN * (0.02 / S) ** 2 / 10.0) < 1e-6, "setpoint must not move"
    print(f"    {c:>7} {ann._rho_star_now():>10.6f} {n_a:>15.2f} {n_s:>15.2f}")
ann_0 = make(rho=0.02, g_rule=10.0, schedule="rm", commit=0)._n_required()
ann_1200 = make(rho=0.02, g_rule=10.0, schedule="rm", commit=1200)._n_required()
assert ann_1200 < ann_0 / 100, "annealed must collapse -- that is the bug"
assert ann_1200 < 1.0, "and to below one upload, hence the I=1 floor"
print(f"    annealed collapses {ann_0:.1f} -> {ann_1200:.3f} ({ann_0 / ann_1200:.0f}x); "
      f"setpoint flat at {P_TRAIN * (0.02 / S) ** 2 / 10.0:.1f}")

# (f) `setpoint` changes nothing when there is no anneal, and nothing under raw_sgd
for sched in ("const",):
    a = make(rho=0.01, schedule=sched, rho_ref="setpoint", commit=50)
    b = make(rho=0.01, schedule=sched, rho_ref="annealed", commit=50)
    assert abs(a._n_required() - b._n_required()) < 1e-9
a = make(rule="raw_sgd", rho_ref="setpoint"); a._last_rho = 0.005
assert abs(a._n_required() - P_TRAIN * (0.005 / S) ** 2 / 2.988) < 1e-6
print("  gate_rho_ref no-ops : identical under const, and under raw_sgd (no setpoint exists)")
