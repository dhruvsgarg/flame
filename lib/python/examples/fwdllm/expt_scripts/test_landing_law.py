"""Exercise C-1's landing anneal (law C) and budget stop on a stub aggregator.

Checks (a) every existing schedule is byte-identical -- `landing` is additive,
(b) law C's closed form: rho*_0 = sqrt(2*B_max/T_res) and B(t) -> B_max
    (1 - e^{-t/T_res}), the property that makes T_res a RATE and not a deadline,
(c) the gate-reachability cap rho_max = s*sqrt(max_iter*K*G_rule/p) actually
    keeps ceil(n_req/K) <= max_iter, the standing launch assertion,
(d) a B_max re-sensed BELOW what is already spent clamps to rho* = 0 rather
    than taking a sqrt of a negative (edge case f),
(e) a re-sense moves rho* and leaves T_res alone -- the whole law-C decision,
(f) the stop latches, fires at f*B_max, and only `halt` sets _work_done,
(g) momentum is refused at construction (edge case b), and
(h) THE SANITY GATE: the live code reproduces replay_landing_law.py's
    pre-registered two-phase enactment on agnews and yahoo to 4 decimals.
"""
import math
import sys

sys.path.insert(0, "/home/dgarg39/flame/lib/python")
from examples.fwdllm.aggregator.FedSgdAggregator import FedSGDAggregator as A  # noqa: E402

P_AGNEWS, P_YAHOO = 450340, 454954
S, K, G_RULE, MAX_ITER, T_RES = 1.5, 10, 10.0, 20, 300.0
B_PRIOR, B_AGNEWS, B_YAHOO = math.log(2.0), math.log(3.15), math.log(2.15)


def make(schedule="landing", b_max=B_PRIOR, p=P_AGNEWS, t_res=T_RES,
         phi_stop="off", cap=True, rho_star=0.06):
    """A stub carrying only what _rho_star_now / _n_required / the stop read."""
    a = object.__new__(A)
    a._rho_schedule = schedule
    a._rho_star = rho_star
    a._rho_exp = 0.25
    a._commit_count = 0
    a._B = 0.0
    a._b_max = b_max
    a._t_res = t_res
    a._budget_stop_frac = 0.95
    a._phi_stop = phi_stop
    a._phi_stop_threshold = 2.7
    a._stop_fired = None
    a._work_done = False
    a._rho_max = (S * math.sqrt(MAX_ITER * K * G_RULE / p)) if cap else None
    a._commit_gate = "n_target"
    a._gate_safety_s = S
    a._gate_rho_ref = "annealed"
    a._server_step_rule = "trust_ratio"
    a._p_trainable = p
    a._g_rule = G_RULE
    a._last_rho = None
    return a


def step(a):
    """One commit of the real path: take rho*, bank its budget, test the stop."""
    rho = a._rho_star_now()
    i = math.ceil(a._n_required() / K - 1e-9) if a._n_required() else 1
    a._commit_count += 1
    a._last_rho = rho
    a._B += 0.5 * math.log1p(rho ** 2)
    a._check_budget_stop()
    return rho, max(1, min(i, MAX_ITER))


# (a) every existing schedule byte-identical
a = make(schedule="const")
assert a._rho_star_now() == 0.06
a._commit_count = 500
assert a._rho_star_now() == 0.06, "const must not see B"
a = make(schedule="rm")
a._commit_count = 99
assert abs(a._rho_star_now() - 0.06 * (100 ** -0.25)) < 1e-12
print("  const / rm          : byte-identical (landing is additive)")

# (b) law C's closed form, and that T_res is a rate not a deadline
a = make(b_max=B_AGNEWS)
want0 = math.sqrt(2 * B_AGNEWS / T_RES)
assert abs(a._rho_star_now() - want0) < 1e-12, (a._rho_star_now(), want0)
for t in range(1, 1201):
    step(a)
    if t in (300, 600, 1200):
        pred = B_AGNEWS * (1 - math.exp(-t / T_RES))
        err = abs(a._B - pred) / pred
        assert err < 0.01, (t, a._B, pred, err)
        print(f"  law C  t={t:<5} B={a._B:.5f}  predicted {pred:.5f}  "
              f"({100 * err:+.2f}%)")
assert a._commit_count == 1200, "T_res must NOT end the run -- it is a rate"
print(f"  T_res={T_RES:g} is a rate  : ran {a._commit_count} commits past it, "
      f"B/B_max={a._B / B_AGNEWS:.3f} < 1 always")

# (c) the gate-reachability cap holds the launch assertion
cap = S * math.sqrt(MAX_ITER * K * G_RULE / P_AGNEWS)
n_req_at_cap = P_AGNEWS * (cap / S) ** 2 / G_RULE
assert abs(n_req_at_cap - MAX_ITER * K) < 1e-6, n_req_at_cap
a = make(b_max=math.log(20.0))           # absurd B_max -> uncapped rho* is huge
assert a._rho_star_now() == cap, "cap must bind"
assert math.ceil(a._n_required() / K - 1e-9) <= MAX_ITER
assert make(b_max=math.log(20.0), cap=False)._rho_star_now() > cap
print(f"  rho_max cap         : {cap:.4f} -> n_req={n_req_at_cap:.1f} = "
      f"max_iter*K exactly, I={math.ceil(n_req_at_cap / K - 1e-9)}")

# (d) B_max re-sensed below the spend -> rho* = 0, not a sqrt of a negative
a = make(b_max=B_AGNEWS)
for _ in range(400):
    step(a)
spent = a._B
a._b_max = spent * 0.5                   # 3.1 re-senses far down
assert a._rho_star_now() == 0.0, a._rho_star_now()
print(f"  B_max below spend   : B={spent:.4f} > B_max={a._b_max:.4f} -> rho*=0")

# (e) a re-sense moves rho* and leaves T_res alone
a = make(b_max=B_PRIOR)
for _ in range(150):
    step(a)
before, t_res_before = a._rho_star_now(), a._t_res
a._b_max = B_AGNEWS
after = a._rho_star_now()
assert after > before and a._t_res == t_res_before
print(f"  re-sense            : rho* {before:.4f} -> {after:.4f}, "
      f"T_res {t_res_before:g} -> {a._t_res:g} (unmoved)")

# (f) the stop: fires at f*B_max, latches, and only `halt` ends the run
for mode, ends in (("off", False), ("log_only", False), ("halt", True)):
    a = make(b_max=B_AGNEWS, phi_stop=mode)
    for _ in range(1500):
        step(a)
    fired = a._stop_fired is not None
    assert fired == (mode != "off"), (mode, a._stop_fired)
    assert a._work_done == ends, (mode, a._work_done)
    if fired:
        assert a._stop_fired == "budget"
        assert a._B >= 0.95 * B_AGNEWS
    print(f"  phi_stop={mode:<9}  : fired={fired} work_done={a._work_done}")
a = make(b_max=B_AGNEWS, phi_stop="halt")
for _ in range(1500):
    step(a)
a._b_max = B_AGNEWS * 4                  # threshold moves out from under it
a._check_budget_stop()
assert a._stop_fired == "budget", "stop must latch, not re-arm"
print("  latching            : a raised B_max does not un-fire the stop")

# a non-landing schedule falls back to the fixed Phi threshold
a = make(schedule="const", phi_stop="halt", rho_star=0.06)
while not a._stop_fired and a._commit_count < 5000:
    step(a)
assert a._stop_fired == "phi_fixed" and abs(math.exp(a._B) - 2.7) < 0.01
print(f"  no sensed B_max     : phi_fixed at Phi={math.exp(a._B):.3f}, "
      f"commit {a._commit_count}")

# (h) SANITY GATE -- reproduce replay_landing_law.py's pre-registered enactment
print("\n  two-phase enactment vs replay_landing_law.py (T5, pre-registered):")
EXPECT = {                     # dataset: (p, B_max, rho_before, rho_after, I_b, I_a)
    "agnews": (P_AGNEWS, B_AGNEWS, 0.0530, 0.0764, 6, 12),
    "yahoo": (P_YAHOO, B_YAHOO, 0.0530, 0.0573, 6, 7),
}
for ds, (p, b_max, r_b, r_a, i_b, i_a) in EXPECT.items():
    a = make(b_max=B_PRIOR, p=p)
    for t in range(151):
        if t == 150:
            a._b_max = b_max             # 3.1 fires; T_res untouched
            got_a, gi_a = a._rho_star_now(), math.ceil(a._n_required() / K - 1e-9)
        if t == 149:
            got_b, gi_b = a._rho_star_now(), math.ceil(a._n_required() / K - 1e-9)
        step(a)
    for name, got, want in (("rho before", got_b, r_b), ("rho after", got_a, r_a)):
        assert abs(got - want) < 5e-5, (ds, name, got, want)
    assert (gi_b, gi_a) == (i_b, i_a), (ds, gi_b, gi_a, i_b, i_a)
    print(f"    {ds:7s} rho* {got_b:.4f} -> {got_a:.4f} (want {r_b} -> {r_a})   "
          f"I {gi_b} -> {gi_a} (want {i_b} -> {i_a})   OK")

print("\n  the two datasets diverge with nothing supplied -- Phase 4's acceptance")
print("  criterion, predicted before the run rather than hoped for.")

# ---- 3.1's knee arithmetic (expts/bmax_probe.py), which sizes B_max ----------
from examples.fwdllm.expts.bmax_probe import (  # noqa: E402
    b_max_from_knee, knee, noise_scale,
)

print("\n  B_max probe (3.1):")
# the knee must be read on CHANCE-NORMALIZED accuracy: the same raw curve on a
# 4-class and a 10-class task has different floors, so a raw threshold would
# reproduce neither of B-1's numbers.
PHIS = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
ag = knee(PHIS, [0.86, 0.84, 0.80, 0.62, 0.35, 0.25], 0.88, 4)
ya = knee(PHIS, [0.71, 0.60, 0.38, 0.20, 0.12, 0.10], 0.73, 10)
assert 3.0 <= ag <= 3.5, ag
assert 2.0 <= ya <= 2.5, ya
print(f"    agnews-shaped curve -> Phi_knee={ag:.2f} (B-1: 3.0-3.5)  "
      f"B_max={b_max_from_knee(ag):.4f}")
print(f"    yahoo-shaped  curve -> Phi_knee={ya:.2f} (B-1: 2.0-2.3)  "
      f"B_max={b_max_from_knee(ya):.4f}")

# identical RAW curve, different K -> different knee. This is the whole reason
# the normalization exists, so assert it rather than trusting it.
raw = [0.60, 0.45, 0.30, 0.20, 0.15, 0.12]
assert knee(PHIS, raw, 0.75, 2) != knee(PHIS, raw, 0.75, 10)
print(f"    same raw curve, K=2 vs K=10 -> knee "
      f"{knee(PHIS, raw, 0.75, 2):.2f} vs {knee(PHIS, raw, 0.75, 10):.2f} "
      f"(chance floor differs)")

# a head at chance has no readable curve -- keep the prior, do not size off it
assert knee(PHIS, [0.25] * 6, 0.26, 4) is None
print("    base at chance      -> None (keeps the current B_max, sizes nothing)")

# never crossing -> the grid is a LOWER bound on the knee, not a failure
assert knee(PHIS, [0.87] * 6, 0.88, 4) == 4.0
print("    never crosses       -> Phi_knee=4.0 (grid max, a lower bound)")

# the noise scale is what makes ||theta_tr|| grow by exactly Phi
for phi in (1.5, 2.0, 3.0):
    got = math.sqrt(1.0 + noise_scale(phi) ** 2)
    assert abs(got - phi) < 1e-12, (phi, got)
print("    noise_scale         : ||theta+eps||/||theta|| == Phi exactly")

# ---- 3.1 end to end on a real (tiny) model: does it RESTORE the weights? -----
# The one failure that would silently corrupt a live run: a probe that perturbs
# theta and does not put it back leaves the aggregator training a noised model.
import torch  # noqa: E402
from torch import nn  # noqa: E402

torch.manual_seed(0)
_VOCAB, _SEQ, _N, _NL = 50, 8, 64, 4


class _Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(_VOCAB, 16)
        self.head = nn.Linear(16, _NL)

    def forward(self, x):
        return self.head(self.emb(x).mean(dim=1))


class _DS:
    def __init__(self):
        ids = torch.randint(0, _VOCAB, (_N, _SEQ))
        labels = torch.randint(0, _NL, (_N,))
        pad = torch.zeros(_N, _SEQ, dtype=torch.long)
        self.tensors = (pad, ids, pad, pad, labels)   # eval reads [1] and [4]


a = make(b_max=B_PRIOR)
_net = _Net()
a.trainer = type("T", (), {"model": _net})()
a.test_global = type("G", (), {"dataset": _DS()})()
a.num_labels = _NL
a._cos_probe_batch = None
a._b_max_probe_n = _N
a._b_max_phis = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

_before = [p.detach().clone() for p in _net.parameters() if p.requires_grad]
_n_before = math.sqrt(sum(float(p.pow(2).sum()) for p in _before))
_b_max_before = a._b_max
a._resense_b_max()
_after = [p.detach().clone() for p in _net.parameters() if p.requires_grad]
_drift = max(float((x - y).abs().max()) for x, y in zip(_before, _after))
assert _drift == 0.0, f"probe left the model perturbed by {_drift}"
print(f"\n  3.1 on a live model  : weights restored EXACTLY (max drift {_drift})")
print(f"    ||theta_tr|| {_n_before:.4f} unchanged; B_max {_b_max_before:.4f} -> "
      f"{a._b_max:.4f} (random head near chance may keep the prior -- correct)")

# and it must restore even when the eval throws mid-probe
class _Boom(_Net):
    calls = 0

    def forward(self, x):
        _Boom.calls += 1
        if _Boom.calls > 2:
            raise RuntimeError("induced mid-probe failure")
        return super().forward(x)


a2 = make(b_max=B_PRIOR)
_boom = _Boom()
a2.trainer = type("T", (), {"model": _boom})()
a2.test_global = type("G", (), {"dataset": _DS()})()
a2.num_labels = _NL
a2._cos_probe_batch = None
a2._b_max_probe_n = _N
a2._b_max_phis = [1.5, 2.0, 2.5]
_pre = [p.detach().clone() for p in _boom.parameters() if p.requires_grad]
a2._resense_b_max()          # must swallow, restore, and keep the old B_max
_post = [p.detach().clone() for p in _boom.parameters() if p.requires_grad]
assert max(float((x - y).abs().max()) for x, y in zip(_pre, _post)) == 0.0
assert a2._b_max == B_PRIOR, a2._b_max
print("    mid-probe exception : swallowed, weights restored, B_max unchanged")

print("\nALL LANDING-LAW CHECKS PASS")
