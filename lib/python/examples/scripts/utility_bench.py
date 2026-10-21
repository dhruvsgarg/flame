#!/usr/bin/env python3
"""FX-N122: in-process async FL microbenchmark for Felix's train utility (ROBUST R26: model before runs).

  utility_bench.py --scenario syn_0|syn_50_3st|mobiperf_usage|mobiperf_drift --arms oort_post size nov ... [--seeds 1 2 3]

Real model, data and timing; only the selector's statistical term changes between arms:
- cifar CifarNet, the reference Dirichlet α 0.1 split's first n trainers, Felix's trainer (SGD 0.04, batch 32, one epoch over the
  visible shard), FedBuff server (lr 1, K updates per version, Felix's weight 0.4 (1+τ)^-0.25 + 0.6 (1.5 − (1+u)^-0.1)).
- task time = FedScale device model (registry compute ms / sample, bandwidth; 3 passes + 2 transfers; FX-D197).
- availability = the real trace (state at dispatch; Felix's eviction when a client leaves AVL_TRAIN mid-task), c tasks in flight.
- streams: `usage` = FX-N120 model A (data accrues while the device is off the charger, 6-day MobiPerf 3-state trace, real time),
  legs starting 24 h into the trace (FX-N123); `drift` = the same, each client's data arriving class by class (new data = new classes).
- selection = Felix's: normalized clipped utility + Oort's recency bonus, × (pref / d)^2 past the 10th-percentile duration, top pick per
  freed slot, Oort exploration of unseen clients. Beliefs update at contact (train return, or an eval task in *_eval arms).
Arms (statistical term u, "_d" = u / predicted task seconds): felix |D|·q after training (today) · oortpen felix + Oort's reference
speed penalty (10th percentile, no pacer) · pre |D|·q of the received model
(FX-N119) · size |D| · pow |D|^β·q · nov n_new·q_new + ρ n_seen·q_seen · nov_eval + eval refresh · *_oracle true current values · random.
Writes one JSON line per (arm, seed) to --out; `--report` prints the arm table from those lines.
"""
import argparse
import bisect
import copy
import hashlib
import heapq
import json
import math
import os
import random
import statistics
import sys
import time

import torch
import torch.nn.functional as F
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "async_cifar10"))
sys.path.insert(0, os.path.join(HERE, "..", "..", ".."))
import fl_data  # noqa: E402
import stream_schedule  # noqa: E402
from flame.availability.trace import load_trace  # noqa: E402

META = os.path.join(HERE, "..", "_metadata")
MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
# dataset -> model, Felix's batch (baseline_reference.yaml), model kbit (FX-D197), split, G3 n, static-leg horizon (F39_RUNTIME_S)
DATA = {"cifar10": dict(batch=32, kbit=2.15e3 * 8, split="cifar10_alpha0.1_n300.yaml", n=75, horizon=10800),
        "google_speech": dict(batch=16, kbit=29.0e3 * 8, split="google_speech_alpha0.1_n100.yaml", n=50, horizon=21600)}
SCEN = {  # trace, stream, horizon s (None = the dataset's static horizon)
    "syn_0": ("syn_0", None, None), "syn_50_3st": ("syn_50_3st", None, None),
    "mobiperf_usage": ("mobiperf_3st_50", "usage", 86400), "mobiperf_drift": ("mobiperf_3st_50", "drift", 86400),
    # MobiPerf-derived data growth (6-day trace, real time) under syn_50_3st availability: a real choice set (~37 trainable of 75)
    "s50_usage": ("syn_50_3st", "usage", 43200), "s50_drift": ("syn_50_3st", "drift", 43200),
}
# arm = statistical term [+ "_d": per predicted second of task time]; felix = today's (post-training loss, no speed term: Felix
# starts Oort's penalty at the 70th percentile and the pacer lifts it to infinity by ~1.9 h, PR40a)
ARMS = ("felix", "oortpen", "pre", "size_d", "pre_d", "pow_d", "nov", "nov_d", "nov_eval_d", "pre_oracle_d", "nov_oracle_d", "random")
BETA = 0.65  # PR40a D2: cifar progress per update ~ samples^0.65


def load_data(ds, train_idx, cache, dev):
    """(X, Y, X_test, Y_test) on dev. cifar: normalized float; speech: int16 PCM of the clients' clips + the test set, cached."""
    if ds == "cifar10":
        from torchvision.datasets import CIFAR10
        norm = lambda d: ((torch.from_numpy(d.data).float().div_(255).permute(0, 3, 1, 2) - torch.tensor(MEAN).view(1, 3, 1, 1))
                          / torch.tensor(STD).view(1, 3, 1, 1)).to(dev)
        tr = CIFAR10(str(fl_data.dataset_dir("cifar10")), train=True, download=False)
        te = CIFAR10(str(fl_data.dataset_dir("cifar10")), train=False, download=False)
        return norm(tr), torch.tensor(tr.targets).to(dev), norm(te), torch.tensor(te.targets).to(dev)
    path = os.path.join(cache, f"speech_{hashlib.sha256(str(train_idx).encode()).hexdigest()[:12]}.pt")
    if os.path.exists(path):
        d = torch.load(path)
    else:
        tr, te = fl_data.SpeechCommands("training"), fl_data.SpeechCommands("testing")
        X = torch.zeros(fl_data.SPEECH_TRAIN_SIZE, 1, fl_data.SPEECH_LEN, dtype=torch.int16)
        Y = torch.zeros(fl_data.SPEECH_TRAIN_SIZE, dtype=torch.long)
        for j in train_idx:
            X[j], Y[j] = tr.pcm16(j)
        tx = [te.pcm16(j) for j in range(len(te))]
        d = dict(X=X, Y=Y, Xt=torch.stack([x for x, _ in tx]), Yt=torch.tensor([y for _, y in tx]))
        os.makedirs(cache, exist_ok=True)
        torch.save(d, path)
    return d["X"].to(dev), d["Y"].to(dev), d["Xt"].to(dev), d["Yt"].to(dev)


class Client:
    def __init__(self, i, key, idx, labels, reg, trace, stream, rng, offset_s=0.0, kbit=DATA["cifar10"]["kbit"]):
        self.i, self.key = i, key
        order = list(range(len(idx)))
        rng.shuffle(order)
        if stream == "drift":  # class by class, classes in a random order
            cls = sorted(set(labels[idx[j]] for j in order))
            rng.shuffle(cls)
            rank = {c: r for r, c in enumerate(cls)}
            order.sort(key=lambda j: rank[labels[idx[j]]])
        self.idx = [idx[j] for j in order]
        self.ms, self.bw, self.kbit = reg["compute_ms_per_sample"], reg["bandwidth_kbps"], kbit
        self.trace = trace  # sorted [(t, state)]
        self.tt = [t for t, _ in trace]
        self.stream = stream
        if stream:  # FX-N120 model A (continuous accrual off the charger), starting offset_s into the trace (FX-N123)
            cfg = {"enabled": "True", "mode": "usage", "initial_frac": 0.0, "full_data_available_after_s": 1}
            self.sched = stream_schedule.from_config(cfg, reg["task_id"], scale=1.0, offset=offset_s)
        # beliefs
        self.contacted, self.last_pick_v, self.belief_ts = False, 0, -1.0
        self.b_vis = self.b_seen = self.b_q = self.b_qnew = None
        self.seen = 0  # samples trained on (prefix length at the last train)
        self.last_d = 0.0
        self.picks = 0

    def state(self, t):
        if not self.trace:
            return "AVL_TRAIN"
        k = bisect.bisect_right(self.tt, t) - 1
        return self.trace[k][1] if k >= 0 else "AVL_TRAIN"

    def leaves_train_at(self, t):
        """First time after t the client is not AVL_TRAIN (inf if never)."""
        k = bisect.bisect_right(self.tt, t)
        for tt, s in self.trace[k:]:
            if s != "AVL_TRAIN":
                return tt
        return math.inf

    def visible(self, t):
        if not self.stream:
            return len(self.idx)
        return self.sched.visible(len(self.idx), t)

    def task_s(self, samples, passes=3, transfers=2):
        return passes * samples * self.ms / 1000.0 + transfers * self.kbit / self.bw


class Bench:
    def __init__(self, a, scenario, arm, seed, dev):
        self.a, self.arm, self.seed, self.dev = a, arm, seed, dev
        trace_name, stream, self.T = SCEN[scenario]
        if a.horizon_h:
            self.T = a.horizon_h * 3600
        self.offset = (a.trace_offset_h if a.trace_offset_h is not None else (24.0 if stream else 0.0)) * 3600
        os.environ["FLAME_TRACE_OFFSET_S"] = str(self.offset)  # availability on the same shifted clock (load_trace)
        rng = random.Random(seed)
        torch.manual_seed(seed)
        self.ds = a.dataset
        spec = DATA[self.ds]
        self.batch, self.kbit = spec["batch"], spec["kbit"]
        if not self.T:
            self.T = spec["horizon"]
        n = a.n or spec["n"]
        reg = yaml.safe_load(open(os.path.join(META, "trainer_registry.yaml")))["trainers"]
        spl = yaml.safe_load(open(os.path.join(META, "dataset_splits", spec["split"])))["trainer_data_splits"]
        keys = sorted(spl, key=lambda k: int(k.split("_")[-1]))[:n]
        X, Y, self.Xt, self.Yt = load_data(self.ds, sorted({j for k in keys for j in spl[k]}), a.cache, dev)
        self.X, self.Y = X, Y
        labels = Y.tolist() if self.ds == "cifar10" else None
        self.cl = []
        for i, k in enumerate(keys):
            tr = [] if trace_name == "syn_0" else [(float(t), s) for t, s in load_trace(trace_name, k).items()]
            lab = labels if labels is not None else {j: int(Y[j]) for j in spl[k]}
            self.cl.append(Client(i, k, list(spl[k]), lab, reg[k], tr, stream, random.Random(f"{seed}:{k}"), self.offset, self.kbit))
        self.model = (fl_data.CifarNet() if self.ds == "cifar10" else fl_data.ResNet34_1D()).to(dev)
        self.audits, self.npicks = [], 0
        self.rng = rng

    # --- utilities ---------------------------------------------------------
    def fwd(self, model, x):
        """Speech: int16 PCM -> float, bf16 autocast (FX-D168); cifar: as is."""
        if self.ds == "cifar10":
            return model(x)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
            return model(fl_data.model_input(x)).float()

    @torch.no_grad()
    def rms_loss(self, model, idx):
        if not idx:
            return 0.0
        sel = idx[:self.a.probe]
        model.eval()
        per = F.nll_loss(self.fwd(model, self.X[sel]), self.Y[sel], reduction="none")
        return float(per.pow(2).mean().sqrt())

    def stat(self, c, t):
        """The arm's utility of client c at time t (None = unexplored)."""
        arm, rho = self.arm, self.a.rho
        base = arm[:-2] if arm.endswith("_d") else arm
        if base.endswith("_oracle"):
            vis = c.visible(t)
            q = self.rms_loss(self.model, c.idx[:vis])
            new = max(0, vis - c.seen)
            if base == "pre_oracle":
                u = vis * q
            else:
                qn = self.rms_loss(self.model, c.idx[c.seen:vis]) if new else 0.0
                u = new * qn + rho * c.seen * q
        else:
            if not c.contacted:
                return None
            vis = c.b_vis
            if base in ("felix", "oortpen", "pre"):
                u = vis * c.b_q
            elif base == "size":
                u = float(vis)
            elif base == "pow":
                u = vis ** BETA * c.b_q
            else:  # nov, nov_eval
                u = max(0, vis - c.seen) * (c.b_qnew or c.b_q) + rho * c.seen * c.b_q
        return u / c.task_s(vis) if arm.endswith("_d") else u

    def true_value(self, c, t):
        """Ground truth for the audit: the novelty value of c's update now (its unseen samples' loss + ρ x its seen ones), per second."""
        vis = c.visible(t)
        q = self.rms_loss(self.model, c.idx[:vis])
        qn = self.rms_loss(self.model, c.idx[c.seen:vis]) if vis > c.seen else 0.0
        return (max(0, vis - c.seen) * qn + self.a.rho * c.seen * q) / c.task_s(vis)

    def audit(self, t, cands, chosen):
        """Mis-selection: where the pick ranks among candidates by true value (1 = best), and its regret vs the best."""
        vals = [(self.true_value(c, t), c) for c in cands]
        best = max(v for v, _ in vals)
        mine = next(v for v, c in vals if c is chosen)
        self.audits.append(dict(pct=sum(v <= mine for v, _ in vals) / len(vals), regret=(best - mine) / best if best > 0 else 0.0))

    def pick(self, t, v, cands):
        c = self._pick(t, v, cands)
        self.npicks += 1
        if self.a.audit_every and self.npicks % self.a.audit_every == 0 and len(cands) > 1:
            self.audit(t, cands, c)
        return c

    def _pick(self, t, v, cands):
        if self.arm == "random":
            return self.rng.choice(cands)
        unexplored = [c for c in cands if not c.contacted and "_oracle" not in self.arm]
        eps = max(0.2, 0.9 * 0.95 ** v)
        if unexplored and (self.rng.random() < eps or len(unexplored) == len(cands)):
            return self.rng.choice(unexplored)
        scored = [(c, self.stat(c, t)) for c in cands]
        scored = [(c, u) for c, u in scored if u is not None]
        if not scored:
            return self.rng.choice(cands)
        us = sorted(u for _, u in scored)
        lo, hi = us[0], us[min(len(us) - 1, int(0.98 * len(us)))]
        pref = math.inf
        if self.arm == "oortpen":  # Oort's reference speed penalty: 10th-percentile preferred duration, (pref / d)^2 beyond it
            known = sorted(c.last_d for c in self.cl if c.contacted)
            pref = known[min(len(known) - 1, int(0.10 * len(known)))] if known else math.inf
        best, bs = None, -1e18
        for c, u in scored:  # Oort's normalized, clipped reward + recency bonus
            r = (min(u, hi) - lo) / (hi - lo) if hi > lo else 1.0
            r += math.sqrt(0.1 * math.log(max(v, 2)) / max(c.last_pick_v, 1))
            if c.contacted and c.last_d > pref:
                r *= (pref / c.last_d) ** 2
            if r > bs:
                best, bs = c, r
        return best

    # --- training ----------------------------------------------------------
    def local_train(self, base, c, vis):
        m = copy.deepcopy(base)
        opt = torch.optim.SGD(m.parameters(), lr=0.04)
        idx = c.idx[:vis]
        g = torch.Generator().manual_seed(hash((self.seed, c.i, c.picks)) & 0x7FFFFFFF)
        perm = torch.randperm(len(idx), generator=g).tolist()
        q_pre = self.rms_loss(m, [idx[j] for j in perm])
        new_idx = c.idx[c.seen:vis]
        q_new = self.rms_loss(m, new_idx) if new_idx else None
        m.train()
        for b in range(0, len(idx), self.batch):
            sel = [idx[j] for j in perm[b:b + self.batch]]
            opt.zero_grad()
            F.nll_loss(self.fwd(m, self.X[sel]), self.Y[sel]).backward()
            opt.step()
        m.eval()
        q_post = self.rms_loss(m, [idx[j] for j in perm])
        delta = {k: (p.detach() - base.state_dict()[k]) for k, p in m.state_dict().items()}
        return delta, q_pre, q_post, q_new

    @torch.no_grad()
    def test_acc(self):
        self.model.eval()
        hits = sum(int((self.fwd(self.model, self.Xt[i:i + 1000]).argmax(1) == self.Yt[i:i + 1000]).sum()) for i in range(0, len(self.Yt), 1000))
        return hits / len(self.Yt)

    @torch.no_grad()
    def client_accs(self, t):
        out = []
        for c in self.cl:
            idx = c.idx[:c.visible(t)]
            out.append(float(sum(int((self.fwd(self.model, self.X[idx[i:i + 1000]]).argmax(1) == self.Y[idx[i:i + 1000]]).sum())
                                 for i in range(0, len(idx), 1000)) / len(idx)))
        return out

    def run(self):
        a, T = self.a, self.T
        v, buf, busy, inflight = 0, [], set(), []  # inflight heap: (t_done, seq, client, base_v, base_model, vis)
        durations, snapshots = {}, {0: copy.deepcopy(self.model)}
        seq, evals, nbytes, evicted, eval_tasks = 0, [], 0.0, 0, 0
        evq = []  # pending eval returns (t, seq, client, base_model, vis)
        next_eval, t = 0.0, 0.0
        eval_every = T / 24

        def dispatch(now):
            nonlocal seq, nbytes
            while len(inflight) < a.c:
                cands = [c for c in self.cl if c.i not in busy and c.state(now) == "AVL_TRAIN"]
                if not cands:
                    return
                c = self.pick(now, v, cands)
                vis = c.visible(now)
                d = c.task_s(vis)
                leave = c.leaves_train_at(now)
                busy.add(c.i)
                c.picks += 1
                c.last_pick_v = max(v, 1)
                seq += 1
                nbytes += self.kbit / 8 * 1e3  # download
                heapq.heappush(inflight, (min(now + d, leave), seq, c, v, vis, now + d <= leave, d))

        def eval_refresh(now):
            nonlocal seq, nbytes, eval_tasks
            if "eval" not in self.arm:
                return
            three = any(c.trace for c in self.cl)
            idle = [c for c in self.cl if c.i not in busy and c.contacted and now - c.belief_ts >= a.eval_gate_s
                    and c.state(now) == ("AVL_EVAL" if three else "AVL_TRAIN")]
            idle.sort(key=lambda c: c.belief_ts)
            for c in idle[:a.eval_per_version]:
                seq += 1
                eval_tasks += 1
                nbytes += self.kbit / 8 * 1e3
                heapq.heappush(evq, (now + c.task_s(c.visible(now), passes=1, transfers=1), seq, c, copy.deepcopy(self.model)))
                busy.add(c.i)

        dispatch(0.0)
        while t < T:
            nxt = min([x[0] for x in inflight[:1]] + [x[0] for x in evq[:1]] + [next_eval, t + 60.0])
            t = nxt
            if t >= T:
                break
            if t >= next_eval:
                evals.append((t, self.test_acc()))
                next_eval += eval_every
            while evq and evq[0][0] <= t:
                _, _, c, m = heapq.heappop(evq)
                busy.discard(c.i)
                vis = c.visible(t)
                c.b_vis, c.b_q, c.belief_ts = vis, self.rms_loss(m, c.idx[:vis]), t
                c.b_qnew = self.rms_loss(m, c.idx[c.seen:vis]) if vis > c.seen else None
            while inflight and inflight[0][0] <= t:
                t_done, _, c, base_v, vis, ok, d = heapq.heappop(inflight)
                busy.discard(c.i)
                if not ok:
                    evicted += 1
                    continue
                delta, q_pre, q_post, q_new = self.local_train(snapshots[base_v], c, vis)
                nbytes += self.kbit / 8 * 1e3  # upload
                durations[c.i] = d
                c.contacted, c.belief_ts, c.b_vis, c.seen = True, t, vis, vis
                c.b_q = q_post if self.arm in ("felix", "oortpen") else q_pre
                c.last_d = d
                c.b_qnew = None
                u = vis * q_pre
                tau = v - base_v
                w = 0.4 * (1 + tau) ** -0.25 + 0.6 * (1.5 - (1 + u) ** -0.1)
                buf.append((w, delta))
                if len(buf) >= a.k:
                    sd = self.model.state_dict()
                    for kname in sd:
                        sd[kname] = sd[kname] + sum(w_ * dl[kname] for w_, dl in buf) / len(buf)
                    self.model.load_state_dict(sd)
                    buf, v = [], v + 1
                    snapshots[v] = copy.deepcopy(self.model)
                    live = {x[3] for x in inflight} | {v}
                    for old in [k for k in snapshots if k not in live]:
                        del snapshots[old]
                    eval_refresh(t)
            dispatch(t)
        accs = self.client_accs(T)
        srt = sorted(accs)
        picks = sorted((c.picks for c in self.cl), reverse=True)
        return dict(arm=self.arm, seed=self.seed, versions=v, evicted=evicted, eval_tasks=eval_tasks, gb=nbytes / 1e9, curve=evals,
                    client_mean=statistics.mean(accs), worst10=statistics.mean(srt[:max(1, len(srt) // 10)]),
                    top20=sum(picks[:max(1, len(picks) // 5)]) / max(1, sum(picks)), starved=sum(p <= 2 for p in picks),
                    pick_pct=statistics.median([x["pct"] for x in self.audits]) if self.audits else None,
                    regret=statistics.mean([x["regret"] for x in self.audits]) if self.audits else None)


def summarize(r, T):
    c = r["curve"]
    tail = [x for t, x in c if t >= 0.7 * T]
    tta = {}
    for th in (0.30, 0.35, 0.40):
        tta[th] = next((c[i][0] / 60 for i in range(1, len(c)) if (c[i][1] + c[i - 1][1]) / 2 >= th), None)
    return dict(last30=statistics.mean(tail) if tail else None, tta=tta)


def report(path):
    rows = [json.loads(x) for x in open(path)]
    by = {}
    for r in rows:
        by.setdefault((r.get("dataset", "cifar10")[:6] + " " + r["scenario"], r["arm"]), []).append(r)
    print("| scenario | arm | seeds | last-30% acc | min to 30 / 35 / 40% | versions | GB | worst-10% client | top-20% share | ≤ 2 picks | evicted | evals | pick rank (1 = best) | regret |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    med = lambda xs: statistics.median([x for x in xs if x is not None]) if any(x is not None for x in xs) else None
    f = lambda x, n=1: "–" if x is None else f"{x:.{n}f}"
    for (sc, arm), rs in sorted(by.items()):
        T = rs[0]["T"]
        ss = [summarize(r, T) for r in rs]
        last = [s["last30"] for s in ss]
        tt = " / ".join(f(med([s["tta"][th] for s in ss]), 0) for th in (0.30, 0.35, 0.40))
        print(f"| {sc} | {arm} | {len(rs)} | {f(100 * med(last))} ({f(100 * min(last))}-{f(100 * max(last))}) | {tt} "
              f"| {f(med([r['versions'] for r in rs]), 0)} | {f(med([r['gb'] for r in rs]), 1)} | {f(100 * med([r['worst10'] for r in rs]))} "
              f"| {f(med([r['top20'] for r in rs]), 2)} | {f(med([r['starved'] for r in rs]), 0)} | {f(med([r['evicted'] for r in rs]), 0)} "
              f"| {f(med([r['eval_tasks'] for r in rs]), 0)} | {f(med([r.get('pick_pct') for r in rs]), 2)} | {f(med([r.get('regret') for r in rs]), 2)} |")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", choices=sorted(SCEN))
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[1])
    ap.add_argument("--dataset", choices=sorted(DATA), default="cifar10")
    ap.add_argument("--n", type=int, default=0, help="population (default: the dataset's G3 n)")
    ap.add_argument("--cache", default=os.path.expanduser("~/.cache/flame_utility_bench"))
    ap.add_argument("--c", type=int, default=15)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--rho", type=float, default=0.5, help="nov: weight of already-trained samples")
    ap.add_argument("--probe", type=int, default=64, help="samples per loss probe (FX-N119)")
    ap.add_argument("--eval-gate-s", type=float, default=300.0)
    ap.add_argument("--audit-every", type=int, default=10, help="mis-selection audit on every Nth pick (0 = off)")
    ap.add_argument("--eval-per-version", type=int, default=10)
    ap.add_argument("--horizon-h", type=float, default=0.0, help="override the scenario's horizon")
    ap.add_argument("--trace-offset-h", type=float, default=None, help="leg start into the trace (default: 24 h with a stream, else 0)")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--out", default="")
    ap.add_argument("--report", default="", help="print the table from a --out file and exit")
    a = ap.parse_args(argv)
    if a.report:
        report(a.report)
        return 0
    dev = "cpu" if a.gpu == "cpu" or not torch.cuda.is_available() else f"cuda:{a.gpu}"
    for seed in a.seeds:
        for arm in a.arms:
            t0 = time.time()
            b = Bench(a, a.scenario, arm, seed, dev)
            r = b.run()
            r.update(scenario=a.scenario, dataset=a.dataset, T=b.T, wall_s=time.time() - t0, rho=a.rho)
            s = summarize(r, b.T)
            print(f"{a.scenario} {arm} s{seed}: last30 {100 * s['last30']:.1f}% v {r['versions']} worst10 {100 * r['worst10']:.1f}% "
                  f"top20 {r['top20']:.2f} evicted {r['evicted']} evals {r['eval_tasks']} pick rank {r['pick_pct']} regret {r['regret']} "
                  f"wall {r['wall_s']:.0f}s", flush=True)
            if a.out:
                with open(a.out, "a") as fh:
                    fh.write(json.dumps(r) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
