"""Predicted commit-1 effect of probe_combine=mean: ||u|| falls by sqrt(E*P).

E = E[v_par^2] for the shipped coin-flip-top-2 rule = 2.988 at P=10, so the
sanity check in handoff sec 15.2 is rho -> rho / 5.47 on the first commit.
"""
import math, torch

torch.manual_seed(0)
p, P, T = 20000, 10, 4000
g = torch.randn(p); g /= g.norm()          # unit gradient
sel_sq = mean_sq = 0.0
E_acc = 0.0
for _ in range(T):
    V = torch.randn(P, p)
    d = V @ g                               # d_i = <g, v_i>
    order = torch.argsort(d.abs())
    pick = order[-1 - int(torch.randint(0, 2, (1,)))]   # coin-flip top-2
    E_acc += float(d[pick] ** 2 / (d ** 2).mean())
    sel_sq += float((d[pick] * V[pick]).pow(2).sum())
    mean_sq += float((d.unsqueeze(1) * V).mean(0).pow(2).sum())
E = E_acc / T
print(f"  E[v_par^2] coin-flip top-2, P={P} : {E:.3f}   (handoff: 2.988)")
print(f"  measured  ||u_select|| / ||u_mean|| : {math.sqrt(sel_sq/mean_sq):.3f}")
print(f"  predicted sqrt(E*P)                : {math.sqrt(E*P):.3f}")
