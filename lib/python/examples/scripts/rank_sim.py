#!/usr/bin/env python3
"""FX-N109 offline ranking simulator: how stale beliefs and utility logic shape picks over 12-48 h (no training).

Real per-client |D| (split), compute ms/sample + bandwidth (registry), usage-stream growth (stream_schedule, start +24 h),
availability (trace, same offset). Loss model calibrated on PR40a ctl legs (cifar 3 h / speech 6 h):
  per-sample loss on trained data q = L(t) * h_i(t);  L(t) = 2.0 (1 + t/1800)^-0.17  (1.91 -> 1.45 over 3 h measured)
  log h_i = a_i + b_i(t): a_i ~ N(0, 0.13), b_i OU sd 0.065 / 3 h  (measured p10/p90 0.84/1.22, half-to-half Spearman 0.85-0.91)
  observation noise 10% (measured consecutive-task |d hardness| 0.10-0.14);  loss on not-yet-trained samples = r * q (swept).
Each logic runs closed-loop (its own picks decide when its beliefs refresh). Logics:
  oort      sync rounds K=10 x1.1, 20-step tasks, |D|q + UCB, (pref/d)^2 past the pacer's preferred duration, cut-off sampling
  felix     async c=15 epoch tasks, |D|q + UCB, Oort penalty from the 70th pct fading out by 1.9 h (today)
  pre_d     felix with u = |D|q / predicted task s
  nov_d     u = (n_new q_new + rho n_seen q_seen) / task s; stale: n_new = 0 after a train
  nov_eval_d nov_d + eval refresh (<= 10 per version of AVL_EVAL idle clients, belief age >= 300 s)
  random    async c=15, uniform
"""
import argparse
import bisect
import math
import os
import random
import statistics as st
import sys
from multiprocessing import Pool

import numpy as np
import yaml

EX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(EX, "async_cifar10"))
sys.path.insert(0, os.path.join(EX, "..", "..", ".."))
import stream_schedule  # noqa: E402
from flame.availability.trace import load_trace  # noqa: E402

META = os.path.join(EX, "_metadata")
DATA = {"cifar10": dict(kbit=2.15e3 * 8, split="cifar10_alpha0.1_n300.yaml", n=75, batch=32, ob=30, pref_pct=10),
        "google_speech": dict(kbit=29.0e3 * 8, split="google_speech_alpha0.1_n100.yaml", n=50, batch=16, ob=15, pref_pct=30)}
LOGICS = ("oort", "felix", "pre_d", "nov_d", "nov_eval_d", "random")
BUCKETS = (12, 24, 36, 48)
RHO = 0.5


def L(t):
    return 2.0 * (1 + t / 1800.0) ** -0.17


class Cl:
    pass


def build(ds, trace, offset, seed):
    spec = DATA[ds]
    reg = yaml.safe_load(open(os.path.join(META, "trainer_registry.yaml")))["trainers"]
    spl = yaml.safe_load(open(os.path.join(META, "dataset_splits", spec["split"])))["trainer_data_splits"]
    keys = sorted(spl, key=lambda k: int(k.split("_")[-1]))[:spec["n"]]
    os.environ["FLAME_TRACE_OFFSET_S"] = str(offset)
    rng = np.random.default_rng(seed)
    cfg = {"enabled": "True", "mode": "usage", "initial_frac": 0.0, "full_data_available_after_s": 1}
    cls = []
    for i, k in enumerate(keys):
        c = Cl()
        c.i, c.N = i, len(spl[k])
        c.ms, c.bw = reg[k]["compute_ms_per_sample"], reg[k]["bandwidth_kbps"]
        tr = sorted((float(t), s) for t, s in load_trace(trace, k).items())
        c.tt, c.ss = [t for t, _ in tr], [s for _, s in tr]
        sch = stream_schedule.from_config(cfg, reg[k]["task_id"], scale=1.0, offset=offset)
        c.vis_grid = np.array([sch.visible(c.N, m * 60.0) for m in range(48 * 60 + 2)])  # per minute
        c.a = rng.normal(0, 0.13)
        c.b = rng.normal(0, 0.065)
        c.b_t = 0.0
        cls.append(c)
    return cls


def state(c, t):
    k = bisect.bisect_right(c.tt, t) - 1
    return c.ss[k] if k >= 0 else "AVL_TRAIN"


def leaves(c, t):
    k = bisect.bisect_right(c.tt, t)
    for j in range(k, len(c.tt)):
        if c.ss[j] != "AVL_TRAIN":
            return c.tt[j]
    return math.inf


def vis(c, t):
    return max(1, int(c.vis_grid[min(len(c.vis_grid) - 1, int(t // 60))]))


class Sim:
    def __init__(self, ds, logic, trace, r, seed, T):
        self.ds, self.logic, self.r, self.T = ds, logic, r, T
        self.spec = DATA[ds]
        self.cl = build(ds, trace, 86400.0, seed)
        self.rng = random.Random(seed * 1000 + LOGICS.index(logic))
        self.nrng = np.random.default_rng(seed)  # same hardness drift across logics
        for c in self.cl:
            c.seen, c.contacted, c.bt, c.bvis, c.bseen, c.bq, c.bqn, c.last_pick, c.picks, c.last_d = 0, False, -1, 0, 0, 0, None, 1, 0, 0
        self.audit = []
        self.cur_pref = self.oort_pref = math.inf
        self.ncand = []

    def h(self, c, t):  # OU drift of hardness, advanced lazily (deterministic per client stream)
        dt = t - c.b_t
        if dt > 0:
            phi = math.exp(-dt / 10800.0)
            c.b = c.b * phi + math.sqrt(1 - phi * phi) * 0.065 * self.nrng.normal()
            c.b_t = t
        return math.exp(c.a + c.b)

    def task_s(self, c, v):
        if self.logic == "oort":
            samples = 20 * min(self.spec["ob"], v)
        else:
            samples = v
        return 3 * samples * c.ms / 1000.0 + 2 * self.spec["kbit"] / c.bw

    def truth(self, c, t, hh):  # fresh components: (vis, n_new, q_seen, q_new)
        v = vis(c, t)
        qs = L(t) * hh
        return v, max(0, v - c.seen), qs, self.r * qs

    def score(self, kind, v, nnew, qs, qn, d):
        if kind in ("oort", "felix"):
            q = (qs * (v - nnew) + qn * nnew) / v
            return v * q
        if kind == "pre_d":
            q = (qs * (v - nnew) + qn * nnew) / v
            return v * q / d
        return (nnew * qn + RHO * (v - nnew) * qs) / d  # nov

    def belief(self, c):
        nnew = max(0, c.bvis - c.bseen)
        qn = c.bqn if c.bqn is not None else c.bq * 1.0
        return self.score(self._kind(), c.bvis, nnew, c.bq, qn, self.task_s(c, c.bvis))

    def _kind(self):
        return "nov" if self.logic.startswith("nov") else self.logic

    def fresh(self, c, t, hh):
        v, nnew, qs, qn = self.truth(c, t, hh)
        return self.score(self._kind(), v, nnew, qs, qn, self.task_s(c, v))

    def yard(self, c, t, hh):  # common yardsticks on fresh state, Felix-style epoch task time
        v, nnew, qs, qn = self.truth(c, t, hh)
        d = 3 * v * c.ms / 1000.0 + 2 * self.spec["kbit"] / c.bw
        return v / d, (v * (qs * (v - nnew) + qn * nnew) / v) / d, (nnew * qn + RHO * (v - nnew) * qs) / d, nnew / v

    def observe(self, c, t, hh, train):
        v, nnew, qs, qn = self.truth(c, t, hh)
        noise = math.exp(self.rng.gauss(0, 0.10))
        c.bvis, c.bt = v, t
        if train:
            c.bq = ((qs * (v - nnew) + qn * nnew) / v) * noise
            c.seen, c.bseen, c.bqn = v, v, None
            c.contacted = True
        else:  # eval: measures new and seen loss separately
            c.bq, c.bseen = qs * noise, c.seen
            c.bqn = qn * noise if nnew else None

    def pick(self, t, rnd, cands, k=1):
        if self.logic == "random":
            return self.rng.sample(cands, min(k, len(cands)))
        unexp = [c for c in cands if not c.contacted]
        eps = max(0.2, 0.9 * 0.95 ** rnd)
        out = []
        nexp = sum(self.rng.random() < eps for _ in range(k)) if unexp else 0
        if unexp and len(unexp) == len(cands):
            nexp = k
        out += self.rng.sample(unexp, min(nexp, len(unexp)))
        rest = [c for c in cands if c.contacted and c not in out]
        if not rest or len(out) >= k:
            return out[:k] if out else self.rng.sample(cands, 1)
        us = [self.belief(c) for c in rest]
        srt = sorted(us)
        lo, hi = srt[0], srt[min(len(srt) - 1, int(0.98 * len(srt)))]
        pref = math.inf
        known = sorted(c.last_d for c in self.cl if c.contacted)
        if known:
            self.oort_pref = known[min(len(known) - 1, int(self.pref_pct / 100 * len(known)))]
        if self.logic == "oort":
            pref = self.oort_pref
        elif self.logic == "felix" and t < 1.9 * 3600:
            pref = known[min(len(known) - 1, int(0.70 * len(known)))]
        self.cur_pref = pref
        sc = []
        for c, u in zip(rest, us):
            s = (min(u, hi) - lo) / (hi - lo) if hi > lo else 1.0
            s += math.sqrt(0.1 * math.log(max(rnd, 2)) / max(c.last_pick, 1))
            if c.last_d > pref:
                s *= (pref / c.last_d) ** 2
            sc.append((s, c))
        sc.sort(key=lambda x: -x[0])
        need = k - len(out)
        if self.logic == "oort":  # cut-off: sample from clients scoring >= cut x the need-th score, by score
            cut = sc[min(need, len(sc)) - 1][0] * self.spec.get("cut", 0.7 if self.ds == "cifar10" else 0.4)
            pool = [x for x in sc if x[0] >= cut]
            while pool and need:
                tot = sum(s for s, _ in pool)
                x = self.rng.random() * tot
                acc = 0
                for j, (s, c) in enumerate(pool):
                    acc += s
                    if acc >= x:
                        out.append(c)
                        pool.pop(j)
                        break
                need -= 1
        else:
            out += [c for _, c in sc[:need]]
        return out

    def pen(self, d):
        return min(1.0, (self.cur_pref / d) ** 2) if d > 0 and self.cur_pref < math.inf else 1.0

    def do_audit(self, t, cands, chosen):
        self.ncand.append((t, len(cands)))
        if len(cands) < 4:
            return
        hs = {c.i: self.h(c, t) for c in cands}
        idx = [j for j, c in enumerate(cands) if c.contacted]
        rec = dict(t=t)
        if len(idx) >= 4:
            # own full score (stat x penalty): belief uses last-seen d, fresh uses today's d
            b = [self.belief(cands[j]) * self.pen(cands[j].last_d) for j in idx]
            f = [self.fresh(cands[j], t, hs[cands[j].i]) * self.pen(self.task_s(cands[j], vis(cands[j], t))) for j in idx]
            rec["sp"] = spearman(b, f)
            k = max(1, len(idx) // 5)
            tb = set(sorted(range(len(b)), key=lambda j: -b[j])[:k])
            tf = set(sorted(range(len(f)), key=lambda j: -f[j])[:k])
            rec["top"] = len(tb & tf) / k
            rec["err"] = st.median(abs(bb / ff - 1) for bb, ff in zip(b, f) if ff > 0)
            rec["age"] = st.median((t - cands[j].bt) / 3600 for j in idx)
            jb, jf = max(range(len(b)), key=lambda j: b[j]), max(range(len(f)), key=lambda j: f[j])
            rec["sreg"] = (f[jf] - f[jb]) / f[jf] if f[jf] > 0 else 0.0
            rec["swrong"] = float(jb != jf)
            # structural disparity on fresh state: each logic's fresh stat vs size and vs samples/s
            fs = {}
            for j in idx:
                c = cands[j]
                v, nn, qs, qn = self.truth(c, t, hs[c.i])
                de = 3 * v * c.ms / 1000.0 + 2 * self.spec["kbit"] / c.bw
                fs.setdefault("size", []).append(v)
                fs.setdefault("sps", []).append(v / de)
                fs.setdefault("Dq", []).append(v * (qs * (v - nn) + qn * nn) / v)
                fs.setdefault("Dq_d", []).append(fs["Dq"][-1] / de)
                fs.setdefault("nov_d", []).append((nn * qn + RHO * (v - nn) * qs) / de)
                do = 3 * 20 * min(self.spec["ob"], v) * c.ms / 1000.0 + 2 * self.spec["kbit"] / c.bw
                fs.setdefault("oortfull", []).append(fs["Dq"][-1] * min(1.0, (self.oort_pref / do) ** 2))
            for name in ("Dq", "Dq_d", "nov_d", "oortfull"):
                rec["S_" + name + "~size"] = spearman(fs[name], fs["size"])
                rec["S_" + name + "~sps"] = spearman(fs[name], fs["sps"])
            rec["S_Dq_d~oortfull"] = spearman(fs["Dq_d"], fs["oortfull"])
        med_sps = st.median(vis(c, t) / (3 * vis(c, t) * c.ms / 1000.0 + 2 * self.spec["kbit"] / c.bw) for c in self.cl)
        med_D = st.median(vis(c, t) for c in self.cl)
        for c in chosen:
            v = vis(c, t)
            rec.setdefault("sps_rel", []).append(v / (3 * v * c.ms / 1000.0 + 2 * self.spec["kbit"] / c.bw) / med_sps)
            rec.setdefault("D_rel", []).append(v / med_D)
            rec.setdefault("newfrac", []).append(max(0, v - c.seen) / v)
        self.audit.append(rec)

    def run(self):
        T, c_inf = self.T, 15
        self.pref_pct = self.spec["pref_pct"]
        t, rnd, busy, heap, nupd = 0.0, 1, set(), [], 0
        import heapq
        seq = 0
        next_eval = 0.0
        pacer_hist = []
        if self.logic == "oort":
            K = 10
            while t < T:
                cands = [c for c in self.cl if state(c, t) == "AVL_TRAIN"]
                if not cands:
                    t += 60
                    continue
                ch = self.pick(t, rnd, cands, k=min(len(cands), round(K * 1.1)))
                if rnd % 2 == 0:
                    self.do_audit(t, cands, ch)
                fin = []
                for c in ch:
                    d = self.task_s(c, vis(c, t))
                    ok = t + d <= leaves(c, t)
                    c.picks += 1
                    c.last_pick = rnd
                    if ok:
                        fin.append((t + d, c, d))
                fin.sort(key=lambda x: x[0])
                done = fin[:K]
                t_end = done[-1][0] if done else t + 60
                util = 0.0
                for te, c, d in done:
                    hh = self.h(c, te)
                    self.observe(c, te, hh, True)
                    c.last_d = d
                    util += c.bvis * c.bq
                pacer_hist.append(util)
                if len(pacer_hist) >= 2 * 10 and rnd % 10 == 0:  # Oort pacer: utility stalls -> relax preferred duration
                    if sum(pacer_hist[-10:]) <= sum(pacer_hist[-20:-10]):
                        self.pref_pct = min(100, self.pref_pct + 20)
                t = t_end
                rnd += 1
            return self.summary()
        # async
        def dispatch(now):
            nonlocal seq
            while len(heap) < c_inf:
                cands = [c for c in self.cl if c.i not in busy and state(c, now) == "AVL_TRAIN"]
                if not cands:
                    return
                c = self.pick(now, rnd, cands)[0]
                seq += 1
                if seq % 5 == 0:
                    self.do_audit(now, cands, [c])
                else:
                    self.ncand.append((now, len(cands)))
                d = self.task_s(c, vis(c, now))
                busy.add(c.i)
                c.picks += 1
                c.last_pick = rnd
                heapq.heappush(heap, (min(now + d, leaves(c, now)), seq, c, now + d <= leaves(c, now), d))
        dispatch(0.0)
        while t < T:
            t = min(heap[0][0] if heap else t + 60, t + 60)
            while heap and heap[0][0] <= t:
                te, _, c, ok, d = heapq.heappop(heap)
                busy.discard(c.i)
                if ok:
                    self.observe(c, te, self.h(c, te), True)
                    c.last_d = d
                    nupd += 1
                    if nupd % 10 == 0:
                        rnd += 1
                        if self.logic == "nov_eval_d":
                            idle = [x for x in self.cl if x.i not in busy and x.contacted and te - x.bt >= 300
                                    and state(x, te) == "AVL_EVAL"]
                            idle.sort(key=lambda x: x.bt)
                            for x in idle[:10]:
                                self.observe(x, te, self.h(x, te), False)
            dispatch(t)
        return self.summary()

    def summary(self):
        out = {}
        for bi, hi in enumerate(BUCKETS):
            lo = BUCKETS[bi - 1] if bi else 0
            rs = [r for r in self.audit if lo * 3600 <= r["t"] < hi * 3600]
            def m(key, agg=st.median):
                vals = [r[key] for r in rs if key in r]
                return agg(vals) if vals else None
            def flat(key):
                vals = [x for r in rs for x in r.get(key, [])]
                return st.mean(vals) if vals else None
            out[hi] = dict(sp=m("sp"), top=m("top", st.mean), err=m("err"), age=m("age"), sreg=m("sreg", st.mean),
                           swrong=m("swrong", st.mean), sps_rel=flat("sps_rel"), D_rel=flat("D_rel"), newfrac=flat("newfrac"),
                           **{k: m(k) for k in {kk for r in rs for kk in r if kk.startswith("S_")}})
        for bi, hi in enumerate(BUCKETS):
            lo = BUCKETS[bi - 1] if bi else 0
            v = [n for tt, n in self.ncand if lo * 3600 <= tt < hi * 3600]
            out[hi]["ncand"] = st.mean(v) if v else 0
        picks = sorted((c.picks for c in self.cl), reverse=True)
        out["top20"] = sum(picks[:len(picks) // 5]) / max(1, sum(picks))
        out["never"] = sum(c.picks == 0 for c in self.cl)
        return out


def spearman(a, b):
    from scipy.stats import spearmanr
    v = spearmanr(a, b).correlation
    return 0.0 if v != v else float(v)


def job(args):
    ds, logic, trace, r, seed, T = args
    return args, Sim(ds, logic, trace, r, seed, T).run()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["cifar10", "google_speech"])
    ap.add_argument("--traces", nargs="+", default=["syn_50_3st", "mobiperf_3st_50"])
    ap.add_argument("--r", nargs="+", type=float, default=[1.0, 1.5, 3.0])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--hours", type=float, default=48)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    jobs = [(ds, lg, tr, r, s, a.hours * 3600) for ds in a.datasets for tr in a.traces for r in a.r for lg in LOGICS for s in a.seeds]
    with Pool(min(len(jobs), os.cpu_count() // 2)) as p:
        res = p.map(job, jobs)
    import json
    if a.out:
        with open(a.out, "w") as fh:
            for args, r in res:
                fh.write(json.dumps(dict(args=args, res={str(k): v for k, v in r.items()})) + "\n")


if __name__ == "__main__":
    main()
