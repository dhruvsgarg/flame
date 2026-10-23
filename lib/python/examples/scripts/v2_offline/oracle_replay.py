"""Closed-loop oracle replay from a Felix leg's checkpoint: N versions under each selection / aggregation policy.

Every policy starts from the same checkpoint with the same client data (visible counts from the leg's telemetry at the replay clock),
trains with Felix's trainer (1 epoch, SGD 0.04), aggregates K = 10 updates per version (staleness 0: picks train on the current model).
Clock: a version costs Σ task_s(picks) / c slot-seconds (c = 15 tasks in flight). Accuracy on test half B; oracles look at half A.
Selection: random · old (|D|·q_post, size-driven) · new (|D|·q_pre / task s × balance) · oracle (top-K marginal value per second,
every candidate trained) · oracle_upd (per update). Aggregation on random picks: uniform · old rate · new rate · oracle weights.
Busy rule: last version's picks are not eligible (they are still in flight in the async system).
V4 candidates (FX-N128): agg_util (w ∝ |D| q_pre, nothing new sent) · agg_self (w ∝ |D| × own held-out gain, one more client pass) ·
agg_ls_oracle / agg_ls_fed (uniform direction, server step α from ALPHAS judged on test half A / on FED_CLIENTS available clients'
held-out samples = federated validation).
"""
import argparse, json, math, random, statistics as st, sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/dgarg39/flame/lib/python/examples/scripts/v2_offline")
import oracle_probe as op  # noqa: E402
from flame.optimizer.bn_buffers import is_bn_stat  # noqa: E402

K, CSLOT = 10, 15
ALPHAS, FED_CLIENTS = (0.125, 0.25, 0.5, 1.0, 1.5, 2.0), 20


def old_rate(u):
    return 0.4 * 1.0 + 0.6 * (1.5 - (1 + u) ** -0.1)


def held_idx(pr, c, train_idx):
    """The client's visible held-out samples: every 10th arrival of the visible prefix (train-visible / 0.9)."""
    n = int(round(len(train_idx) / 0.9))
    return [p for j, p in enumerate(c.order[:n]) if j % pr.k_hold == pr.k_hold - 1]


@torch.no_grad()
def train_loss(pr, m, params, idx):
    """Mean nll of `params` on train-set samples `idx` (held-out data lives in the train set)."""
    if not idx:
        return 0.0
    m.eval()
    s = torch.tensor(idx, device=pr.dev)
    return float(F.nll_loss(pr.fwd_p(m, params, pr.b.X[s]), pr.b.Y[s]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--hours", nargs="+", type=float, default=[1.5, 4, 8])
    ap.add_argument("--versions", type=int, default=40)
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--policies", nargs="+", default=["random", "old", "new_sel", "new", "oracle", "oracle_upd",
                                                       "agg_uniform", "agg_old", "agg_new", "agg_oracle"])
    ap.add_argument("--gpu", default="2")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pr = op.Probe(a.run, a.gpu)
    m = pr.b.model
    out = []
    for h in a.hours:
        ts0, path = min(pr.ckpts, key=lambda x: abs(x[0] - h * 3600))
        sd0 = torch.load(path, map_location=pr.dev, weights_only=False)["state_dict"]
        m.load_state_dict(sd0)
        L0, A0 = pr.loss_acc(m, pr.B)
        print(f"== {a.run.split('_dbg_')[1][:40]} t {ts0/3600:.1f} h  start acc {A0:.3f} loss {L0:.3f}", flush=True)
        for pol in a.policies:
            m.load_state_dict(sd0)
            rng = random.Random(7)
            t, picks, busy, curve, alphas = ts0, {c.tid: 0 for c in pr.b.cl}, set(), [(0, 0.0, A0, L0)], []
            for v in range(1, a.versions + 1):
                base = {k: x.clone() for k, x in m.state_dict().items()}
                cands = []
                for c in pr.b.cl:
                    if c.tid in busy:
                        continue
                    if c.trace:
                        import bisect
                        k_ = bisect.bisect_right(c.tt, t) - 1
                        if (c.trace[k_][1] if k_ >= 0 else "AVL_TRAIN") != "AVL_TRAIN":
                            continue
                    idx = pr.train_idx(c, t)
                    if idx:
                        cands.append((c, idx))
                need_all = pol in ("old", "oracle", "oracle_upd")
                info = {}
                if need_all:  # train every candidate (oracles; old needs its post-training loss)
                    LA, _ = pr.loss_acc(m, pr.A) if pol != "old" else (None, None)
                    for c, idx in cands:
                        c.idx, c.seen = idx, 0
                        m.load_state_dict(base)
                        delta, q_pre, q_post, _, _ = pr.b.local_train(m, c, len(idx))
                        val = None
                        if pol != "old":
                            p = {k: (base[k] + delta[k] / K if not is_bn_stat(k) else base[k]) for k in base}
                            val = LA - pr.loss_acc(m, pr.A, p)[0]
                        info[c.tid] = dict(delta=delta, q_post=q_post, val=val, s=c.task_s(len(idx)), n=len(idx))
                    m.load_state_dict(base)
                if pol in ("random",) or pol.startswith("agg_"):
                    chosen = rng.sample(cands, min(K, len(cands)))
                elif pol == "old":
                    chosen = sorted(cands, key=lambda x: -len(x[1]) * info[x[0].tid]["q_post"])[:K]
                elif pol in ("new", "new_sel"):
                    mean_p = max(st.mean(picks.values()), 1e-9)
                    def sc(x):
                        c, idx = x
                        q = pr.b.rms_loss(m, idx[:64])
                        return len(idx) * q / c.task_s(len(idx)) / (1 + picks[c.tid] / mean_p)
                    chosen = sorted(cands, key=lambda x: -sc(x))[:K]
                elif pol == "oracle":
                    chosen = sorted(cands, key=lambda x: -info[x[0].tid]["val"] / info[x[0].tid]["s"])[:K]
                else:  # oracle_upd
                    chosen = sorted(cands, key=lambda x: -info[x[0].tid]["val"])[:K]
                deltas, ws_old, ws_new, ws_util, ws_self, cost = [], [], [], [], [], 0.0
                for c, idx in chosen:
                    m.load_state_dict(base)
                    q_pre = pr.b.rms_loss(m, idx[:64]) if pol == "agg_util" else None
                    hold = held_idx(pr, c, idx) if pol == "agg_self" else []
                    pre_h = train_loss(pr, m, base, hold) if hold else None
                    if c.tid in info:
                        d = info[c.tid]["delta"]; qp = info[c.tid]["q_post"]
                    else:
                        c.idx, c.seen = idx, 0
                        m.load_state_dict(base)
                        d, _, qp, _, _ = pr.b.local_train(m, c, len(idx))
                    deltas.append(d)
                    ws_old.append(old_rate(len(idx) * qp))
                    ws_new.append(len(idx) ** 0.65)
                    if q_pre is not None:
                        ws_util.append(len(idx) * q_pre)
                    if hold:
                        post = {k: (base[k] + d[k] if not is_bn_stat(k) else base[k]) for k in base}
                        ws_self.append(len(idx) * max(1e-3, pre_h - train_loss(pr, m, post, hold)))
                    elif pol == "agg_self":
                        ws_self.append(len(idx) * 1e-3)
                    cost += c.task_s(len(idx))
                    picks[c.tid] += 1
                ws_new = [w / st.mean(ws_new) for w in ws_new]
                if pol == "agg_old":
                    w = ws_old
                elif pol == "agg_new":
                    w = ws_new
                elif pol == "agg_oracle":
                    w = torch.ones(len(deltas), device=pr.dev, requires_grad=True)
                    opt = torch.optim.Adam([w], lr=0.1)
                    keys = [k for k in base if not is_bn_stat(k)]
                    g = torch.Generator().manual_seed(v)
                    for _ in range(40):
                        s = pr.A[torch.randperm(len(pr.A), generator=g)[:512].to(pr.dev)]
                        p = dict(base)
                        for k in keys:
                            p[k] = base[k] + sum(w[i] * dl[k] for i, dl in enumerate(deltas)) / K
                        opt.zero_grad()
                        F.nll_loss(pr.fwd_p(m, p, pr.b.Xt[s]), pr.b.Yt[s]).backward()
                        opt.step()
                        with torch.no_grad():
                            w.clamp_(min=0)
                    w = w.detach().tolist()
                elif pol == "agg_uniform":
                    w = [1.0] * len(deltas)
                elif pol in ("agg_util", "agg_self"):
                    raw = ws_util if pol == "agg_util" else ws_self
                    w = [x / st.mean(raw) for x in raw]
                elif pol in ("agg_ls_oracle", "agg_ls_fed"):
                    if pol == "agg_ls_fed":
                        pool = [c for c, _ in rng.sample(cands, min(FED_CLIENTS, len(cands)))]
                        judge = [j for c in pool for j in held_idx(pr, c, pr.train_idx(c, t))]
                    keys = [k for k in base if not is_bn_stat(k)]
                    def cand(al):
                        return {k: (base[k] + al * sum(dl[k] for dl in deltas) / K if k in keys else base[k]) for k in base}
                    score = (lambda p: pr.loss_acc(m, pr.A, p)[0]) if pol == "agg_ls_oracle" else (lambda p: train_loss(pr, m, p, judge))
                    al = min(ALPHAS, key=lambda x: score(cand(x)))
                    alphas.append(al)
                    w = [al] * len(deltas)
                elif pol.startswith("agg_fixed"):  # agg_fixed2: fixed server step 2
                    w = [float(pol[9:].replace("p", "."))] * len(deltas)
                elif pol == "new":
                    w = ws_new  # new Felix = new selection + C5 weights
                else:
                    w = ws_old  # selection policies share old Felix's rate
                m.load_state_dict({k: (base[k] + sum(wi * dl[k] for wi, dl in zip(w, deltas)) / K if not is_bn_stat(k) else base[k])
                                   for k in base})
                busy = {c.tid for c, _ in chosen}
                t += cost / CSLOT
                if v % a.every == 0:
                    L, A = pr.loss_acc(m, pr.B)
                    curve.append((v, (t - ts0) / 60, A, L))
            fin = curve[-1]
            print(f"   {pol:11} v {fin[0]}  +{fin[1]:6.1f} min  acc {fin[2]:.3f}  loss {fin[3]:.3f}  | acc by version "
                  + " ".join(f"{x[2]*100:.1f}" for x in curve[1:]) + "  | min " + " ".join(f"{x[1]:.0f}" for x in curve[1:]), flush=True)
            out.append(dict(run=a.run, h=ts0 / 3600, pol=pol, curve=curve, alphas=alphas))
            json.dump(out, open(a.out, "w"))


if __name__ == "__main__":
    main()
