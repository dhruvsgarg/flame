#!/usr/bin/env python3
"""FX-N132 (d), R26 model step: how much a per-update value signal could buy by selection, before building one.

  value_headroom.py LEG.yaml [--versions 20,100,400] [--k 10] [--gpu 0] [--json-out R.json]

The leg's own client shards (fl_bench.resolve) drive a FedBuff-like trajectory (K uniform picks per version, 1 local epoch,
the leg's client lr / batch, server lr 1, mean of deltas). At each probe version every client trains one update from the
same model; its value = test-loss drop on half A of the test set when applied alone at weight 1/K. Then, on half B, the
buffer of K picked by each rule is applied (mean of deltas) and its loss drop reported: the buffer oracle (greedy on A),
top-K by value, random K, a diverse buffer (greedy min pairwise cosine of the deltas), a label-balanced buffer
(greedy min variance over classes of the summed classifier-bias deltas), top-K by |D|, by q, by |D| q. Also: ρ(value, signal) and value persistence ρ between probe versions.
q estimators (Q-S3): `q` pre-step first batch (Felix new), `qo` Oort EMA over the pass, `qv` rms loss on a held-out
`--holdout` share of the shard (never trained), `qt` rms loss on the whole train shard (reference).
Full shards (late-phase streams); no availability.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fl_bench  # noqa: E402  (sets sys.path for the example)


def spearman(x, y):
    rk = lambda v: {i: r for r, i in enumerate(sorted(range(len(v)), key=lambda i: v[i]))}
    rx, ry, n = rk(x), rk(y), len(x)
    return 1 - 6 * sum((rx[i] - ry[i]) ** 2 for i in range(n)) / (n * (n * n - 1)) if n > 2 else float("nan")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("yaml")
    ap.add_argument("--versions", default="20,100,400")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--holdout", type=float, default=0.0, help="held-out share of each shard (enables qv)")
    ap.add_argument("--json-out")
    a = ap.parse_args(argv)
    import fl_data
    exp, _, trainers, _, _ = fl_bench.resolve(a.yaml)
    hp0 = trainers[0]["hyperparameters"]
    from types import SimpleNamespace
    spec = fl_data.spec_for(SimpleNamespace(dataset_name=exp.trainer.dataset.name))
    dev = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")
    lr, bs = float(hp0.get("learningRate") or 0.04), int(hp0.get("batchSize") or 32)
    train, test = fl_data.in_memory(spec.train()), fl_data.in_memory(spec.test())
    tx = torch.stack([test[i][0] for i in range(len(test))]).to(dev)
    ty = torch.tensor([int(test[i][1]) for i in range(len(test))]).to(dev)
    half = len(ty) // 2
    A, B = (tx[:half], ty[:half]), (tx[half:], ty[half:])
    shards, held = [], []
    for t in trainers:
        idx = list(t["hyperparameters"]["trainer_indices_list"])
        k = min(len(idx) - 1, max(1, round(len(idx) * a.holdout))) if a.holdout > 0 else 0  # keep >= 1 train sample
        st = lambda ii: (torch.stack([train[i][0] for i in ii]).to(dev), torch.tensor([int(train[i][1]) for i in ii]).to(dev))
        shards.append(st(idx[k:] if k else idx))
        held.append(st(idx[:k]) if k > 0 else None)
    rng = random.Random(a.seed)
    torch.manual_seed(a.seed)
    model = spec.model().to(dev)
    flat = lambda m: torch.cat([p.detach().reshape(-1) for p in m.state_dict().values() if torch.is_floating_point(p)])

    def load(m, v):
        i = 0
        for p in m.state_dict().values():
            if torch.is_floating_point(p):
                n = p.numel()
                p.copy_(v[i:i + n].view_as(p))
                i += n

    def loss_on(v, data, p=1):
        """Mean loss^p of v on data, eval mode (p 2 = Oort's statistic)."""
        load(work, v)
        work.eval()
        with torch.no_grad():
            return sum(F.nll_loss(work(fl_data.model_input(data[0][j:j + 1000])), data[1][j:j + 1000],
                                  reduction="none").pow(p).sum().item()
                       for j in range(0, len(data[1]), 1000)) / len(data[1])

    def local(v, c):
        """One local epoch from v on client c: (delta, (pre-step rms loss of the first batch, Oort EMA rms over the pass))."""
        load(work, v)
        work.train()
        opt = torch.optim.SGD(work.parameters(), lr=lr)
        x, y = shards[c]
        perm = torch.randperm(len(y), device=dev)
        q = ema = None
        for j in range(0, len(y), bs):
            b = perm[j:j + bs]
            out = work(fl_data.model_input(x[b]))
            lv = F.nll_loss(out, y[b], reduction="none")
            sq = lv.detach().pow(2).mean().item()
            q = sq ** 0.5 if q is None else q
            ema = sq if ema is None else 0.8 * ema + 0.2 * sq  # the trainer's FedScale EMA (syncfl trainer.oort_loss)
            opt.zero_grad()
            lv.mean().backward()
            opt.step()
        return flat(work) - v, (q, ema ** 0.5)

    work = spec.model().to(dev)
    v = flat(model)
    probes, done, out = sorted(int(x) for x in a.versions.split(",")), 0, []
    n = len(shards)
    for target in probes:
        while done < target:  # trajectory: K uniform picks, mean of deltas
            v = v + torch.stack([local(v, c)[0] for c in rng.sample(range(n), a.k)]).mean(0)
            done += 1
        base_a, base_b = loss_on(v, A), loss_on(v, B)
        res = [local(v, c) for c in range(n)]
        val = [base_a - loss_on(v + d / a.k, A) for d, _ in res]
        size = [len(shards[c][1]) for c in range(n)]
        q, qo = [x[0] for _, x in res], [x[1] for _, x in res]
        qt = [loss_on(v, shards[c], 2) ** 0.5 for c in range(n)]
        sig = {"|D|": size, "q": q, "|D| q": [s * x for s, x in zip(size, q)], "|D| qo": [s * x for s, x in zip(size, qo)],
               "|D| qt": [s * x for s, x in zip(size, qt)]}
        if a.holdout > 0:  # a 1-sample shard holds nothing out: its train loss stands in
            qv = [loss_on(v, held[c] or shards[c], 2) ** 0.5 for c in range(n)]
            sig.update({"qv": qv, "|D| qv": [s * x for s, x in zip(size, qv)]})
        gain = lambda picks: base_b - loss_on(v + torch.stack([res[c][0] for c in picks]).mean(0), B)
        top = lambda s: sorted(range(n), key=lambda c: -s[c])[:a.k]
        rand = sum(gain(rng.sample(range(n), a.k)) for _ in range(8)) / 8

        def greedy(score):
            picks = []
            for _ in range(a.k):
                picks.append(max((c for c in range(n) if c not in picks), key=lambda c: score(picks + [c])))
            return picks
        D = torch.stack([d for d, _ in res])
        U = D / D.norm(dim=1, keepdim=True)
        oracle_set = greedy(lambda p: base_a - loss_on(v + D[p].mean(0), A))  # buffer oracle on A, graded on B
        diverse = greedy(lambda p: -float((U[p] @ U[p].T).sum()) if len(p) > 1 else float(D[p[0]].norm()))
        nb = list(model.state_dict().values())[-1].numel()  # classifier bias: its delta mirrors the client's label mix
        Bd = D[:, -nb:]
        balanced = greedy(lambda p: -float(Bd[p].sum(0).var()))
        row = dict(version=target, base_loss=base_b, oracle=gain(top(val)), random=rand, buffer_oracle=gain(oracle_set),
                   diverse=gain(diverse), balanced=gain(balanced),
                   **{f"top {k}": gain(top(s)) for k, s in sig.items()},
                   **{f"rho {k}": spearman(val, s) for k, s in sig.items()}, value=val)
        if out:
            row["rho prev value"] = spearman(out[-1]["value"], val)
        out.append(row)
        print(f"v{target}: loss {base_b:.3f} · buffer loss drop: buffer oracle {row['buffer_oracle']:+.4f} diverse "
              f"{row['diverse']:+.4f} label-balanced {row['balanced']:+.4f} top-value {row['oracle']:+.4f} random {rand:+.4f} "
              + " ".join(f"{k} {row['top ' + k]:+.4f}" for k in sig) + " · ρ(value, ·) "
              + " ".join(f"{k} {row['rho ' + k]:+.2f}" for k in sig)
              + (f" · ρ(value, prev value) {row['rho prev value']:+.2f}" if "rho prev value" in row else ""), flush=True)
    if a.json_out:
        json.dump(out, open(a.json_out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
