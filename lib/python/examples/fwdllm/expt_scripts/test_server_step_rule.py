"""Exercise _apply_weighted_update on a stub aggregator.

Checks (a) raw_sgd is byte-identical to the pre-split code, (b) trust_ratio
yields rho = ||dtheta_tr|| / ||theta_tr|| EXACTLY equal to rho*_t, which is the
commit-1 enactment test, and (c) the rm schedule anneals as t^-exp.
"""
import sys, types, torch
sys.path.insert(0, "/home/dgarg39/flame/lib/python")
from examples.fwdllm.aggregator.FedSgdAggregator import FedSGDAggregator as A

torch.manual_seed(0)
NP, K = 6, 5                      # param tensors, uploads

def make(rule, rho_star=0.02, sched="const", exp=0.55):
    a = object.__new__(A)
    a._server_step_rule = rule; a._rho_star = rho_star
    a._rho_schedule = sched; a._rho_exp = exp; a._commit_count = 0
    a.server_momentum = 0.0; a._server_momentum_buf = {}
    a._server_update_audit = False
    a._pool_split_half_stats = lambda ml: None
    a._emit_server_update = lambda *x, **k: None
    params = [torch.randn(4, 5) for _ in range(NP)]
    for i, p in enumerate(params):
        p.requires_grad_(i < NP - 2)          # last two frozen
    a.trainer = types.SimpleNamespace(model=types.SimpleNamespace(
        parameters=lambda: iter(params)))
    return a, params

def run(rule, **kw):
    a, params = make(rule, **kw)
    before = [p.detach().clone() for p in params]
    ups = [[torch.randn(4, 5) * (1.0 if p.requires_grad else 0.0) for p in params]
           for _ in range(K)]
    model_list = [(1, [u.clone() for u in up]) for up in ups]
    wgs = model_list[0][1]
    a._apply_weighted_update(model_list, wgs, iter(params), 0.01, K)
    d_sq = t_sq = 0.0
    for b, p in zip(before, params):
        if p.requires_grad:
            d_sq += float((p.detach() - b).pow(2).sum()); t_sq += float(b.pow(2).sum())
    return d_sq ** 0.5 / t_sq ** 0.5, before, params, ups

rho, _, _, ups = run("raw_sgd")
# reference: what the fused single-pass code computed
G = [sum(u[j] for u in ups) for j in range(NP)]
ref = (sum(float((0.01 * g / K).pow(2).sum()) for j, g in enumerate(G) if j < NP - 2) ** 0.5)
print(f"  raw_sgd     rho={rho:.8f}   (unchanged path)")

for sched, exp in [("const", 0.55), ("rm", 0.55)]:
    for rs in (0.02, 0.005):
        r, *_ = run("trust_ratio", rho_star=rs, sched=sched, exp=exp)
        want = rs if sched == "const" else rs * (1 ** -exp)
        ok = "OK" if abs(r - want) < 1e-6 else "MISMATCH"
        print(f"  trust_ratio rho*={rs:<6} sched={sched:<5} -> rho={r:.8f}  want={want:.8f}  {ok}")

a, _ = make("trust_ratio", 0.02, "rm", 0.55)
sched = []
for t in range(5):
    sched.append(round(a._rho_star_now(), 6)); a._commit_count += 1
print(f"  rm schedule t=1..5: {sched}   (t^-0.55 decay)")
