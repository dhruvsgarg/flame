"""Validate _compute_n_eff against pools with a KNOWN effective n.

Case 1 (ideal): n iid uploads u_k = d_k * v_k, one shared gradient.  n_eff ~= n.
Case 2 (heterogeneous): uploads split across G groups with different gradient
        directions.  n_eff must fall well below n -- that is the whole claim.
Case 3 (duplicated): each upload repeated r times.  Effective n is n/r, not n.
"""
import sys, math
import torch

sys.path.insert(0, "/home/dgarg39/flame/lib/python")
from examples.fwdllm.trainer.forward_training.fwdgrad_utils import calculate_var


def n_eff(lst, var):
    m = lst[0].numel()
    mean_sq = sum(float(t.pow(2).sum()) for t in lst) / len(lst)
    return 2.0 * mean_sq / (m * var)


def build(n, m, groups=1, repeat=1, seed=0):
    g = torch.Generator().manual_seed(seed)
    dirs = [torch.randn(m, generator=g) for _ in range(groups)]
    out = []
    for k in range(n // repeat):
        gr = dirs[k % groups]
        v = torch.randn(m, generator=g)
        d = float(torch.dot(gr, v) / math.sqrt(m))
        u = d * v
        out.extend([u.clone() for _ in range(repeat)])
    return out


torch.manual_seed(0)
m = 4096
print(f"{'case':34s} {'n':>5s} {'n_eff':>9s} {'ratio':>7s}")
for label, kw in [
    ("ideal iid pool",                dict(n=200, groups=1,  repeat=1)),
    ("ideal, larger pool",            dict(n=400, groups=1,  repeat=1)),
    ("heterogeneous: 4 groups",       dict(n=200, groups=4,  repeat=1)),
    ("heterogeneous: 20 groups",      dict(n=200, groups=20, repeat=1)),
    ("heterogeneous: 100 groups",     dict(n=200, groups=100, repeat=1)),
    ("duplicated x2 (n_eff -> n/2)",  dict(n=200, groups=1,  repeat=2)),
    ("duplicated x4 (n_eff -> n/4)",  dict(n=200, groups=1,  repeat=4)),
]:
    lst = build(m=m, **kw)
    v = float(calculate_var(lst))
    ne = n_eff(lst, v)
    print(f"{label:34s} {kw['n']:5d} {ne:9.1f} {ne/kw['n']:7.2f}")
