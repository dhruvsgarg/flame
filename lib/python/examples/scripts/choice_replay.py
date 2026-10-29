#!/usr/bin/env python3
"""F5b: does selection keep its choice as concurrency nears the eligible pool? Trace replay, no training.

  choice_replay.py [--traces mobiperf_3st syn_50_3st] [--ratios ...] [--seeds 3] [--procs 64] [--out DIR]

Replays a run's window of the trace (cifar: n 300, +24 h offset, 120 h) with c slots kept full: a free slot is filled at once from the
idle eligible clients, sampled p ∝ |D| (FLuX's |D| q with q = 1: no training, so no loss) or uniformly (FedBuff); a task lasts the
device model's epoch time (`task_timeout.task_times_s`) and ends early when its client leaves (the slot is refilled at once).
c = ratio × the window's mean eligible pool. Metric = the digest's `picks.corr_size`: Pearson(data size, picks) over picked clients.
Writes `<out>/choice_replay_<trace>.json` (default paper_felix/data).
"""
import argparse
import heapq
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from flame.availability.trace import raw_events  # noqa: E402
from flame.launch import task_timeout  # noqa: E402

OUT = os.path.join(HERE, "..", "paper_felix", "data")
META = os.path.join(HERE, "..", "_metadata")
DS, N, T0, RUN_S = "cifar10", 300, 86400.0, 432000.0  # the FB2 cifar legs: n 300, trace offset +24 h, 120 h
RATIOS = (0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5)


def setup(trace):
    """Per client: eligible intervals in the window (s from window start), data size, epoch task time."""
    prof = yaml.safe_load(open(os.path.join(META, "datasets.yaml")))["datasets"][DS]
    keys = sorted(yaml.safe_load(open(os.path.join(META, "trainer_registry.yaml")))["trainers"])[:N]
    shard = task_timeout._shards(prof["reference_split"])
    dur = dict(zip(sorted(shard), task_timeout.task_times_s("felix", DS, prof)))
    wins = []
    for k in keys:
        ev = [(float(t), str(s).split(".")[-1]) for t, s in raw_events(trace, k)]
        w, cur = [], None
        for t, s in ev + [(T0 + RUN_S, "END")]:
            if s == "AVL_TRAIN" and cur is None:
                cur = t
            elif s != "AVL_TRAIN" and cur is not None:
                a, b = max(cur, T0), min(t, T0 + RUN_S)
                if b > a:
                    w.append((a - T0, b - T0))
                cur = None
        wins.append(w)
    return keys, wins, np.array([shard[k] for k in keys], float), np.array([dur[k] for k in keys])


def replay(args):
    trace, ratio, policy, seed, keys, wins, size, dur = args
    rng = np.random.default_rng(seed)
    n = len(keys)
    pool = sum(b - a for w in wins for a, b in w) / RUN_S
    c = max(1, round(ratio * pool))
    ev = [(a, 1, i) for i, w in enumerate(wins) for a, b in w] + [(b, 0, i) for i, w in enumerate(wins) for a, b in w]
    heapq.heapify(ev)  # (time, kind, client): 0 = leaves, 1 = joins, 2 = task ends (with a task id)
    elig, busy, task = np.zeros(n, bool), np.zeros(n, bool), np.zeros(n, int)
    picks, free, now = np.zeros(n, int), c, 0.0
    while ev:
        now, kind, i, *tid = heapq.heappop(ev)
        if kind == 1:
            elig[i] = True
        elif kind == 0:
            elig[i] = False
            if busy[i]:  # client left mid-task: evicted, slot refilled now
                busy[i], free, task[i] = False, free + 1, task[i] + 1
        elif busy[i] and tid[0] == task[i]:
            busy[i], free, task[i] = False, free + 1, task[i] + 1
        if ev and ev[0][0] == now:
            continue  # settle every event at this instant before filling
        cand = np.flatnonzero(elig & ~busy)
        k = min(free, len(cand))
        if k:
            p = size[cand] / size[cand].sum() if policy == "size" else None
            for j in rng.choice(cand, k, replace=False, p=p):
                busy[j], picks[j] = True, picks[j] + 1
                if now + dur[j] < RUN_S:
                    heapq.heappush(ev, (now + dur[j], 2, int(j), int(task[j])))
            free -= k
    m = picks > 0
    corr = float(np.corrcoef(size[m], picks[m])[0, 1]) if m.sum() > 2 else None
    return trace, ratio, policy, seed, c, pool, corr, int(picks.sum())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traces", nargs="*", default=["mobiperf_3st", "syn_50_3st"])
    ap.add_argument("--ratios", nargs="*", type=float, default=list(RATIOS))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--procs", type=int, default=64)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args(argv)
    jobs = []
    for tr in a.traces:
        keys, wins, size, dur = setup(tr)
        jobs += [(tr, r, pol, s, keys, wins, size, dur) for r in a.ratios for pol in ("size", "uniform") for s in range(a.seeds)]
    with Pool(a.procs) as p:
        res = p.map(replay, jobs, chunksize=1)
    for tr in a.traces:
        rows = [r for r in res if r[0] == tr]
        out = {"trace": tr, "dataset": DS, "n": N, "run_s": RUN_S, "offset_s": T0, "mean_eligible": rows[0][5], "metric": "pearson(size, picks)",
               "rows": [dict(ratio=r, policy=pol, seed=s, c=c, corr=corr, picks=k) for _, r, pol, s, c, _, corr, k in rows]}
        path = os.path.join(a.out, f"choice_replay_{tr}.json")
        json.dump(out, open(path, "w"), indent=1)
        print(f"== {tr}: mean eligible {out['mean_eligible']:.1f} -> {path}")
        for r in a.ratios:
            cs = {pol: np.mean([x[6] for x in rows if x[1] == r and x[2] == pol]) for pol in ("size", "uniform")}
            print(f"  c/eligible {r:4.2f}: size {cs['size']:+.2f}  uniform {cs['uniform']:+.2f}")


if __name__ == "__main__":
    main()
