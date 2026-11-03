#!/usr/bin/env python3
"""Oracle probe on a Felix leg's checkpoints: how close are old / new Felix selection and aggregation to an oracle?

At each probe time t: load the leg's checkpoint nearest t, rebuild every client's train-visible data exactly as the trainer saw it
(stream order seeded by trainer id, 10% holdout, visible count from the leg's trainer telemetry), train each available client one
task (Felix: 1 epoch, SGD 0.04), and measure on test half A:
  selection oracle  v_i = L(θ) − L(θ + Δ_i / K)  (marginal value of the update in a K buffer), per slot-second v_i / task_s_i
  aggregation oracle w* = argmin_{w ≥ 0} L(θ + Σ w_i Δ_i / K) for a buffer; reported on test half B.
"""
import argparse, bisect, glob, hashlib, json, math, os, random, statistics as st, sys
from argparse import Namespace

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.func import functional_call

EX = "/home/dgarg39/flame/lib/python/examples"
sys.path.insert(0, os.path.join(EX, "scripts"))
import utility_bench as ub  # noqa: E402
from flame.availability.trace import load_trace  # noqa: E402
from flame.optimizer.bn_buffers import is_bn_stat  # noqa: E402

K, C = 10, 15


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def leg_info(run):
    """trainer id -> [(stream_clock_s, train-visible)], round -> vclock, picks per trainer before t (for balance)."""
    vis = {}
    for f in glob.glob(run + "/telemetry/trainer_*.jsonl"):
        for line in open(f, "rb"):
            if b'"trainer_round"' in line:
                e = json.loads(line)
                if e.get("task_to_perform") == "train" and e.get("stream_clock_s") is not None:
                    vis.setdefault(e["end_id"], []).append((float(e["stream_clock_s"]), int(e["dataset_size"])))
    for v in vis.values():
        v.sort()
    return vis


def vis_at(series, t, total_train):
    if not series:
        return None
    ts = [x for x, _ in series]
    k = bisect.bisect_right(ts, t)
    if k == 0:
        return series[0][1]
    if k == len(series):
        return series[-1][1]
    (t0, v0), (t1, v1) = series[k - 1], series[k]
    return int(round(v0 + (v1 - v0) * (t - t0) / max(t1 - t0, 1e-9)))


class Probe:
    def __init__(self, run, gpu):
        self.run = run
        self.ds = "google_speech" if "google_speech" in run else "cifar10"
        self.trace = "syn_50_3st" if "syn_50_3st" in run else "syn_0"
        os.environ["FLAME_TRACE_OFFSET_S"] = "86400"
        a = Namespace(horizon_h=1, trace_offset_h=24.0, dataset=self.ds, n=0, cache=str(ub.fl_data.cache_dir("flame_utility_bench")),
                      probe=64, rho=0.5)
        self.dev = f"cuda:{gpu}"
        self.b = ub.Bench(a, "syn_0", "random", 1, self.dev)  # model, data, fwd, local_train
        spec = ub.DATA[self.ds]
        reg = yaml.safe_load(open(os.path.join(ub.META, "trainer_registry.yaml")))["trainers"]
        spl = yaml.safe_load(open(os.path.join(ub.META, "dataset_splits", spec["split"])))["trainer_data_splits"]
        self.vis = leg_info(run)
        for c in self.b.cl:
            tid = reg[c.key]["task_id"]
            idx = list(spl[c.key])
            seed = int(hashlib.sha256(str(tid).encode()).hexdigest(), 16) % (2 ** 31)
            order = torch.randperm(len(idx), generator=torch.Generator().manual_seed(seed)).tolist()
            self.k_hold = 10
            c.order = [idx[j] for j in order]
            c.tid = tid
            c.trace = [] if self.trace == "syn_0" else [(float(t), s) for t, s in load_trace(self.trace, c.key).items()]
            c.tt = [t for t, _ in c.trace]
        nt = len(self.b.Yt)
        g = torch.Generator().manual_seed(0)
        perm = torch.randperm(nt, generator=g)
        cap = 4000 if self.ds == "cifar10" else 3000
        self.A, self.B = perm[:cap].to(self.dev), perm[cap:2 * cap].to(self.dev)
        self.ckpts = sorted((torch.load(p, map_location="cpu", weights_only=False)["sim_time_s"], p)
                            for p in glob.glob(run + "/checkpoints/round_*.pt"))

    def train_idx(self, c, t):
        n_vis = vis_at(self.vis.get(c.tid, []), t, None)
        if n_vis is None:
            return []
        # train-visible = visible prefix minus every 10th arrival; invert dataset_size -> prefix length
        pos = [p for j, p in enumerate(c.order) if j % self.k_hold != self.k_hold - 1]
        return pos[:max(1, n_vis)]

    @torch.no_grad()
    def loss_acc(self, model, sub, params=None):
        model.eval()
        tot, hit, n = 0.0, 0, 0
        for i in range(0, len(sub), 500):
            s = sub[i:i + 500]
            x = self.b.Xt[s]
            out = self.b.fwd(model, x) if params is None else self.fwd_p(model, params, x)
            tot += float(F.nll_loss(out, self.b.Yt[s], reduction="sum"))
            hit += int((out.argmax(1) == self.b.Yt[s]).sum())
            n += len(s)
        return tot / n, hit / n

    def fwd_p(self, model, params, x):
        if self.ds == "cifar10":
            return functional_call(model, params, (x,))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return functional_call(model, params, (ub.fl_data.model_input(x),)).float()

    def at(self, t_h):
        t = t_h * 3600
        ts, path = min(self.ckpts, key=lambda x: abs(x[0] - t))
        sd = torch.load(path, map_location=self.dev, weights_only=False)["state_dict"]
        m = self.b.model
        m.load_state_dict(sd)
        base = {k: v.clone() for k, v in m.state_dict().items()}
        L0, A0 = self.loss_acc(m, self.A)
        rows = []
        for c in self.b.cl:
            if c.trace:
                k = bisect.bisect_right(c.tt, ts + 86400 * 0) - 1  # load_trace already applies the offset
                state = c.trace[k][1] if k >= 0 else "AVL_TRAIN"
                if state != "AVL_TRAIN":
                    continue
            idx = self.train_idx(c, ts)
            if not idx:
                continue
            c.idx, c.seen = idx, 0
            vis = len(idx)
            m.load_state_dict(base)
            delta, q_pre, q_post, _, bn = self.b.local_train(m, c, vis)
            m.load_state_dict(base)
            q_pre64 = self.b.rms_loss(m, idx[:64])
            p = {k: (base[k] + delta[k] / K if not is_bn_stat(k) else base[k]) for k in base}
            L1, _ = self.loss_acc(m, self.A, p)
            task_s = c.task_s(vis)
            rows.append(dict(c=c, idx=idx, vis=vis, task_s=task_s, q_pre=q_pre, q_pre64=q_pre64, q_post=q_post, v=L0 - L1, delta=delta, bn=bn))
        return ts, L0, A0, base, rows

    def agg(self, base, buf, steps=60):
        """Buffer -> {scheme: (loss_B, acc_B, weights)} with uniform, old rate, new C5 and oracle weights (τ = 0)."""
        m = self.b.model
        keys = [k for k in base if not is_bn_stat(k)]
        bnm = {k: sum(r["bn"][k] for r in buf) / len(buf) for k in base if is_bn_stat(k) and k in buf[0]["bn"]}

        def params(w):
            p = dict(base)
            p.update(bnm)
            for k in keys:
                p[k] = base[k] + sum(w[i] * r["delta"][k] for i, r in enumerate(buf)) / len(buf)
            return p

        u = [r["vis"] * r["q_post"] for r in buf]
        old = [0.4 + 0.6 * (1.5 - (1 + x) ** -0.1) for x in u]
        s65 = [r["vis"] ** 0.65 for r in buf]
        new = [x / st.mean(s65) for x in s65]
        out = {}
        for name, w in (("uniform", [1.0] * len(buf)), ("old", old), ("new", new)):
            out[name] = (*self.loss_acc(m, self.B, params(torch.tensor(w, device=self.dev))), w)
        w = torch.ones(len(buf), device=self.dev, requires_grad=True)
        opt = torch.optim.Adam([w], lr=0.1)
        g = torch.Generator().manual_seed(1)
        for _ in range(steps):
            s = self.A[torch.randperm(len(self.A), generator=g)[:512].to(self.dev)]
            opt.zero_grad()
            F.nll_loss(self.fwd_p(m, params(w), self.b.Xt[s]), self.b.Yt[s]).backward()
            opt.step()
            with torch.no_grad():
                w.clamp_(min=0)
        wd = w.detach()
        out["oracle"] = (*self.loss_acc(m, self.B, params(wd)), wd.tolist())
        # best global scale on uniform weights (step size alone)
        best = None
        for sc in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0):
            l, a_ = self.loss_acc(m, self.A, params(torch.full((len(buf),), sc, device=self.dev)))
            if best is None or l < best[0]:
                best = (l, sc)
        out["scale*"] = (*self.loss_acc(m, self.B, params(torch.full((len(buf),), best[1], device=self.dev))), [best[1]])
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--hours", nargs="+", type=float, default=[1, 3, 6, 9])
    ap.add_argument("--gpu", default="2")
    ap.add_argument("--draws", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pr = Probe(a.run, a.gpu)
    res = []
    for h in a.hours:
        ts, L0, A0, base, rows = pr.at(h)
        v = np.array([r["v"] for r in rows])
        sec = np.array([r["task_s"] for r in rows])
        vps = v / sec
        scores = {
            "old: |D|·q_post": [r["vis"] * r["q_post"] for r in rows],
            "new: |D|·q_pre/s": [r["vis"] * r["q_pre64"] / r["task_s"] for r in rows],
            "|D|": [r["vis"] for r in rows],
            "1/s": [1 / r["task_s"] for r in rows],
            "q_pre": [r["q_pre64"] for r in rows],
            "|D|/s": [r["vis"] / r["task_s"] for r in rows],
        }
        top_or = np.sort(vps)[::-1][:C].sum()
        sel = {"random": dict(rho_ps=0.0, rho_v=0.0, captured=C * vps.mean() / top_or)}
        for name, s in scores.items():
            s = np.array(s)
            top = np.argsort(-s)[:C]
            sel[name] = dict(rho_ps=spearman(s, vps), rho_v=spearman(s, v), captured=vps[top].sum() / top_or)
        # aggregation: random buffers and score-picked buffers
        rng = random.Random(int(h * 10))
        aggs = {}
        bufs = [("random", [rows[i] for i in rng.sample(range(len(rows)), min(K, len(rows)))]) for _ in range(a.draws)]
        for name in ("old: |D|·q_post", "new: |D|·q_pre/s"):
            bufs.append((name, [rows[i] for i in np.argsort(-np.array(scores[name]))[:K]]))
        bufs.append(("oracle top-K", [rows[i] for i in np.argsort(-vps)[:K]]))
        ncls = int(pr.b.Yt.max()) + 1

        def ent(buf, w):
            h = torch.zeros(ncls)
            for wi, r_ in zip(w, buf):
                h += wi * torch.bincount(pr.b.Y[r_["idx"]].cpu(), minlength=ncls).float() / len(r_["idx"])
            h = h / h.sum()
            return float(-(h[h > 0] * h[h > 0].log()).sum() / math.log(ncls))

        for bname, buf in bufs:
            o = pr.agg(base, buf)
            d_ = {k: (x[0], x[1], [float(y) for y in x[2]]) for k, x in o.items()}
            d_["ent"] = (ent(buf, [1] * len(buf)), ent(buf, o["oracle"][2]), [])
            aggs.setdefault(bname, []).append(d_)
        r = dict(run=os.path.basename(a.run), h=ts / 3600, L0=L0, A0=A0, n=len(rows), neg=int((v < 0).sum()),
                 v_cv=float(v.std() / abs(v.mean())) if v.mean() else None, vps_cv=float(vps.std() / abs(vps.mean())),
                 sel=sel, agg=aggs,
                 clients=[dict(vis=r_["vis"], s=r_["task_s"], qpre=r_["q_pre64"], qpost=r_["q_post"], v=r_["v"]) for r_ in rows])
        res.append(r)
        print(f"t {ts/3600:.1f} h  acc {A0:.3f}  n {len(rows)}  neg {r['neg']}  v cv {r['v_cv']:.2f}  v/s cv {r['vps_cv']:.2f}", flush=True)
        for k_, x in sel.items():
            print(f"   sel {k_:18} rho(v/s) {x['rho_ps']:+.2f} rho(v) {x['rho_v']:+.2f} captured {x['captured']:.2f}", flush=True)
        for bname, lst in aggs.items():
            m_ = {s_: st.mean(d[s_][0] for d in lst) for s_ in lst[0]}
            ac = {s_: st.mean(d[s_][1] for d in lst) for s_ in lst[0]}
            ws = [d["oracle"][2] for d in lst]
            print(f"   agg {bname:18} " + " ".join(f"{s_} {m_[s_]:.4f}/{100*ac[s_]:.1f}" for s_ in m_ if s_ != "ent")
                  + f"  | label-entropy uniform {m_['ent']:.2f} oracle-w {ac['ent']:.2f} | w* sum {st.mean(sum(w) for w in ws):.1f} max-share {st.mean(max(w)/max(sum(w),1e-9) for w in ws):.2f}", flush=True)
        json.dump(res, open(a.out, "w"))


if __name__ == "__main__":
    main()
