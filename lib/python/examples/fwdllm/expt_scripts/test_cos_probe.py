"""Exercise B1's ground-truth cos(G,g) probe end-to-end on a stub aggregator.

The probe is worthless if `weighted_gradient_sum`'s indices do not line up with
`model.parameters()` order, and that misalignment is silent -- it would just
return a plausible-looking small cosine forever. So this drives the REAL
`_apply_weighted_update`, with a pool constructed so the answer is known:

  (a) pool == g          -> cos = +1
  (b) pool == -g         -> cos = -1
  (c) pool orthogonal    -> cos = 0
  (d) random pool        -> |cos| ~ 1/sqrt(p), the isotropy floor
  (e) audit off          -> no probe, no cost, and the update is unchanged

Runs on a 2-layer net with a TensorDataset shaped like agnews' (index 1 = input
ids, index 4 = labels), CPU only, no GPU/model/data downloads.
"""
import math
import sys

import torch
import torch.nn as nn

sys.path.insert(0, "/home/dgarg39/flame/lib/python")
from examples.fwdllm.aggregator.FedSgdAggregator import FedSGDAggregator as A

torch.manual_seed(0)
NUM_LABELS, VOCAB, SEQ, N = 4, 50, 8, 32


class Net(nn.Module):
    """Embedding + mean-pool + linear. Small, but a real autograd graph."""

    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, 16)
        self.head = nn.Linear(16, NUM_LABELS)

    def forward(self, x):
        return self.head(self.emb(x).mean(dim=1))


class _DS:
    def __init__(self):
        ids = torch.randint(0, VOCAB, (N, SEQ))
        labels = torch.randint(0, NUM_LABELS, (N,))
        pad = torch.zeros(N, SEQ, dtype=torch.long)
        # agnews layout: eval_model reads tensors[1] (ids) and tensors[4] (labels)
        self.tensors = (pad, ids, pad, pad, labels)


class _Global:
    def __init__(self):
        self.dataset = _DS()


def make(audit=True, seed=0):
    a = object.__new__(A)
    torch.manual_seed(seed)  # same weights every arm, so ||g|| is comparable
    model = Net()
    a.trainer = type("T", (), {"model": model})()
    a.test_global = _Global()
    a.num_labels = NUM_LABELS
    a._cos_ground_truth_audit = audit
    a._cos_probe_batch = None
    a._cos_probe_batch_size = N
    a._server_update_audit = True
    a._pool_split_half_audit = False
    a._server_step_rule = "raw_sgd"
    a._commit_count = 0
    a._last_rho = None
    a._weight_decay = None
    a.server_momentum = 0.0
    a._server_momentum_buf = {}
    a._var_scalar = None
    a._n_eff_scalar = None
    a.fwd_llm_stage = None
    a._model_version = 0
    return a, model


def run(pool_from_g, seed=0):
    """Build a one-upload pool of `f(g)` and return the logged (cos, ||G||, ||g||)."""
    a, model = make(seed=seed)
    grads, g_norm = a._cos_probe_gradient()
    upload = [pool_from_g(g, i).clone() if g is not None else torch.zeros_like(p)
              for i, (g, p) in enumerate(zip(grads, model.parameters()))]
    captured = {}
    a._emit_server_update = lambda *args, **kw: captured.update(kw)
    a._apply_weighted_update(
        model_list=[(1, upload)],
        weighted_gradient_sum=[torch.zeros_like(p) for p in model.parameters()],
        old_param=iter(list(model.parameters())),
        learning_rate=1.0,
        training_num=1,
    )
    return captured["cos_gt"], grads, g_norm


# (a)/(b) exact alignment: pool == +-g must give +-1 to float precision
cos_gt, grads, g_norm = run(lambda g, i: g)
assert abs(cos_gt[0] - 1.0) < 1e-5, cos_gt
print(f"  pool = +g            : cos = {cos_gt[0]:+.8f}  (must be +1)")
assert abs(cos_gt[2] - g_norm) < 1e-5, "||g|| must match the probe's own norm"

cos_gt, _, _ = run(lambda g, i: -g)
assert abs(cos_gt[0] + 1.0) < 1e-5, cos_gt
print(f"  pool = -g            : cos = {cos_gt[0]:+.8f}  (must be -1)")

# scale invariance: cos is a direction test, ||G|| carries the magnitude
cos_gt, _, gn = run(lambda g, i: 37.5 * g)
assert abs(cos_gt[0] - 1.0) < 1e-5
print(f"  pool = 37.5*g        : cos = {cos_gt[0]:+.8f}, ||G|| = {cos_gt[1]:.4f} "
      f"= 37.5*||g|| = {37.5 * gn:.4f}")
assert abs(cos_gt[1] - 37.5 * gn) < 1e-3

# (c) orthogonal by construction: flip the sign of half of each block
def _half_flip(g, i):
    out = g.clone().flatten()
    out[: out.numel() // 2] *= -1
    return out.view_as(g)


# not exactly orthogonal, but must be far off +-1 -- catches "cos is always 1"
cos_gt, _, _ = run(_half_flip)
assert abs(cos_gt[0]) < 0.9, cos_gt
print(f"  pool = half-flipped g: cos = {cos_gt[0]:+.8f}  (must not be +-1)")

# (d) a random pool sits at the isotropy floor 1/sqrt(p) -- the sanity check that
#     the probe is measuring direction and not an artefact of the accumulation
p = sum(x.numel() for x in grads if x is not None)
cs = []
for seed in range(20):
    # the MODEL is fixed (seed=0 inside make); only the pool is redrawn
    _g = torch.Generator().manual_seed(100 + seed)
    cs.append(run(lambda g, i: torch.randn(g.shape, generator=_g))[0][0])
rms = (sum(c * c for c in cs) / len(cs)) ** 0.5
print(f"  random pool, p={p:<5}   : rms cos = {rms:.4f} vs 1/sqrt(p) = "
      f"{1 / math.sqrt(p):.4f}  ({rms * math.sqrt(p):.2f}x)")
assert 0.3 < rms * math.sqrt(p) < 3.0, "random pool must land on the isotropy floor"

# (e) audit off: no probe at all, and the applied update is byte-identical
a_off, model_off = make(audit=False)
captured = {}
a_off._emit_server_update = lambda *args, **kw: captured.update(kw)
before = [p.detach().clone() for p in model_off.parameters()]
upload = [torch.ones_like(p) for p in model_off.parameters()]
a_off._apply_weighted_update(
    model_list=[(1, upload)],
    weighted_gradient_sum=[torch.zeros_like(p) for p in model_off.parameters()],
    old_param=iter(list(model_off.parameters())),
    learning_rate=0.5, training_num=1,
)
assert captured["cos_gt"] is None, "audit off must not emit a cos"
assert a_off._cos_probe_batch is None, "audit off must not even cache a batch"
for b, p_ in zip(before, model_off.parameters()):
    assert torch.allclose(p_.detach(), b - 0.5), "update must be unchanged"
print("  audit off            : no cos emitted, batch not cached, update unchanged")

# the probe must leave no grads behind on the live model
a_on, model_on = make()
a_on._cos_probe_gradient()
assert all(p.grad is None for p in model_on.parameters()), "probe must clear grads"
print("  probe hygiene        : model left with no .grad, eval/train mode restored")

# ---------------------------------------------------------------- B17 (§3.9)
# The reference batch must be REPRESENTATIVE. test_index_list is per-client test
# shards concatenated in client order and never shuffled, so under niid_label
# partitioning the head of the tensor is one client's Dirichlet-skewed shard.
# Slicing [:n] there gave a 75%-single-class reference whose gradient is
# ANTI-correlated (-0.46) with the true held-out one, voiding every cos ever
# logged. This asserts the probe no longer reads the head of the tensor.
class _ClusteredDS:
    """Labels sorted into contiguous per-class blocks -- the pathological case."""

    def __init__(self, n=400):
        ids = torch.randint(0, VOCAB, (n, SEQ))
        labels = torch.arange(n) * NUM_LABELS // n      # 0000...1111...2222...
        pad = torch.zeros(n, SEQ, dtype=torch.long)
        self.tensors = (pad, ids, pad, pad, labels)


def _dominant_share(labels):
    lab = labels.view(-1).tolist()
    return max(lab.count(c) for c in set(lab)) / len(lab)


a_sk, _ = make()
a_sk.test_global = type("G", (), {"dataset": _ClusteredDS()})()
a_sk._cos_probe_batch_size = 64
a_sk._cos_probe_gradient()
share = _dominant_share(a_sk._cos_probe_batch[1])
head_share = _dominant_share(_ClusteredDS().tensors[4][:64])
assert head_share == 1.0, "the fixture must actually be pathological"
assert share < 0.5, (
    f"probe reference is class-skewed at {share:.2f} -- it is reading the head "
    f"of an unshuffled, client-ordered test tensor (B17 regression)"
)
print(f"  B17 reference batch  : dominant-class share {share:.2f} "
      f"(unshuffled head would be {head_share:.2f})")

# and it must be the SAME batch every commit, or trends are not comparable
first = a_sk._cos_probe_batch[1].clone()
a_sk._cos_probe_gradient()
assert torch.equal(a_sk._cos_probe_batch[1], first), "reference batch must be fixed"
a_sk2, _ = make()
a_sk2.test_global = type("G", (), {"dataset": _ClusteredDS()})()
a_sk2._cos_probe_batch_size = 64
a_sk2._cos_probe_gradient()
assert torch.equal(a_sk2._cos_probe_batch[1], first), "must match across arms"
print("  B17 reference batch  : identical across commits and across arms")
