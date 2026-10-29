#!/usr/bin/env python3
"""FX-N147: is device eligibility forecastable? Trace-only (no runs): the deployed forecasters vs "eligible now", per trace.

  eligibility_forecast.py [--traces mobiperf_3st syn_50_3st] [--n 300] [--days 1 5] [--every-min 30] [--out DIR] [--procs 96]

Eligible = AVL_TRAIN (AVL_EVAL counts as not eligible, as for every baseline). Query times t every `--every-min` over trace days
[d0, d1) (day 0 is history: Prophet needs a full day), each jittered (seeded) off the traces' 10-min event grid. Horizons h = 5 min,
15 min, 1 h. Predictors, each from history before t only:
- prophet: REFL's per-device Prophet (`refl_predictors.ProphetForecaster._model`: daily seasonality, flat growth, refit each trace
  day on that day's past), p = mean forecast over the window.
- feddance: V = 1 − exp(−5λ), λ = share of the last 50 rounds (μ 240 s) the client was eligible (its per-round check-in count).
- seasonal: the client's share eligible over the same window on earlier days (REFL-fc's majority-vote input).
- now: eligible at t (persistence; what a pushed notify gives the server).
Questions: A (every client) eligible for all of [t + h, t + 2h] (REFL's query shape); B (clients eligible at t, i.e. a selection)
stays eligible for all of [t, t + h] (a task of length h survives). In B "now" is 1 for everyone, so it scores as the base rate.
Metrics: AUC (rank) and Brier. Also eligibility autocorrelation by lag and the trace statistics §3.1 plots. Writes
`<out>/fx_n147_<trace>.json` (default: examples/paper_felix/data).
"""
import argparse
import bisect
import json
import math
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from flame.availability.refl_predictors import DAY_S, ProphetForecaster, _avl_train_share  # noqa: E402
from flame.availability.trace import raw_events  # noqa: E402

STEP = 300.0  # grid / Prophet bin, s
HORIZONS = (300.0, 900.0, 3600.0)  # a sync round / async task (~4-5 min) up to a long task
LAGS = (300, 900, 1800, 3600, 7200, 14400, 28800, 43200, 86400)
MU, FD_ROUNDS, FD_K = 240.0, 50, 5  # FedDance: round length, history window, prediction window (Alg. 1 defaults)
PREDS = ("prophet", "feddance", "seasonal", "now")
OUT = os.path.join(HERE, "..", "paper_felix", "data")


def _client(args):
    trace, key, times, states, d0, d1, every, end = args
    import logging
    logging.getLogger().setLevel(logging.WARNING)
    rng = np.random.default_rng(int(key.split("_")[-1]))
    times = [float(t) for t in times]
    states = [str(s).split(".")[-1] for s in states]
    on = lambda t: (states[bisect.bisect_right(times, t) - 1] if times and t >= times[0] else "AVL_TRAIN") == "AVL_TRAIN"  # noqa: E731
    share = lambda a, b: _avl_train_share(times, states, a, b) if times else 1.0  # noqa: E731
    pf = object.__new__(ProphetForecaster)  # the deployed model's code, without loading the registry per process
    pf._ev, pf.step_s, pf.growth, pf.fits, pf._fit = {key: (times, states)}, STEP, "flat", 0, {}

    def prophet(t, a, b):
        day = int(t // DAY_S)
        if pf._fit.get(key, (None,))[0] != day:
            pf._fit[key] = pf._model(key, day)
        _, g0, yhat = pf._fit[key]
        i, j = int((a - g0) // STEP), int((b - g0) // STEP)
        return float(yhat[max(i, 0):max(j, i + 1)].mean())

    def feddance(t):
        lam = sum(on(t - k * MU) for k in range(1, FD_ROUNDS + 1)) / FD_ROUNDS
        return 1.0 - math.exp(-lam * FD_K)

    def seasonal(t, a, b):
        sh = [share(a - k * DAY_S, b - k * DAY_S) for k in range(1, int(t // DAY_S) + 1)]
        return sum(sh) / len(sh)

    rows = {h: {"A": [], "B": []} for h in HORIZONS}  # (truth, prophet, feddance, seasonal, now)
    qs = np.arange(d0 * DAY_S, d1 * DAY_S, every)
    for t in qs + rng.uniform(0, every, len(qs)):
        if t + 2 * max(HORIZONS) > end:
            break
        now, fd = on(t), feddance(t)
        for h in HORIZONS:
            a, b = t + h, t + 2 * h
            rows[h]["A"].append((share(a, b) >= 1.0, prophet(t, a, b), fd, seasonal(t, a, b), float(now)))
            if now:
                rows[h]["B"].append((share(t, t + h) >= 1.0, prophet(t, t, t + h), fd, seasonal(t, t, t + h), 1.0))
    grid = np.arange(0.0, end, STEP)
    e = np.array([on(t) for t in grid], dtype=np.int8)
    sess, cur = [], None  # eligible session lengths, s
    for t, s in zip(times, states):
        if s == "AVL_TRAIN" and cur is None:
            cur = t
        elif s != "AVL_TRAIN" and cur is not None:
            sess.append(t - cur)
            cur = None
    return key, {h: {q: np.array(v, dtype=float).reshape(-1, 5) for q, v in r.items()} for h, r in rows.items()}, e, sess


def auc(y, p):
    from scipy.stats import rankdata
    y = np.asarray(y, bool)
    npos, nneg = y.sum(), (~y).sum()
    if not npos or not nneg:
        return None
    r = rankdata(p)
    return float((r[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def run(trace, n, d0, d1, every, procs):
    import yaml
    reg = yaml.safe_load(open(os.path.join(HERE, "..", "_metadata", "trainer_registry.yaml")))["trainers"]
    keys = sorted(reg)[:n]
    ev = {k: raw_events(trace, k) for k in keys}
    end = max([float(e[-1][0]) for e in ev.values() if e] or [d1 * DAY_S + 2 * max(HORIZONS)])  # the last state holds to here
    jobs = [(trace, k, [t for t, _ in ev[k]], [s for _, s in ev[k]], d0, d1, every, end) for k in keys]
    with Pool(procs) as pool:
        res = pool.map(_client, jobs, chunksize=1)
    out = {"trace": trace, "n": n, "days": [d0, d1], "every_s": every, "step_s": STEP, "mu_s": MU, "metrics": {}}
    for h in HORIZONS:
        for q in ("A", "B"):
            m = np.concatenate([r[1][h][q] for r in res if len(r[1][h][q])])
            y = m[:, 0].astype(bool)
            row = {"n": int(len(y)), "pos_rate": float(y.mean())}
            for i, name in enumerate(PREDS, 1):
                p = m[:, i] if not (q == "B" and name == "now") else np.full(len(y), y.mean())
                row[name] = {"auc": auc(y, p) if not (q == "B" and name == "now") else 0.5,
                             "brier": float(np.mean((p - y) ** 2))}
            out["metrics"][f"{q}_{int(h)}"] = row
    E = np.stack([r[2] for r in res]).astype(float)
    L = E.shape[1]  # clients x 5-min grid
    lo, hi = int(d0 * DAY_S / STEP), int(min(d1 * DAY_S / STEP, L))
    ac = {}
    for lag in LAGS:
        k = int(lag / STEP)
        x, z = E[:, lo:hi - k].ravel(), E[:, lo + k:hi].ravel()
        ac[lag] = float(np.corrcoef(x, z)[0, 1])
    out["autocorr"] = ac
    H = E[:, lo:hi:12]  # hourly
    leg0, leg1 = int(DAY_S / STEP), int(1.5 * DAY_S / STEP)  # FX-N123: legs start +24 h; 12 h legs
    out["stats"] = {
        "share_t": E.mean(0)[::3].tolist(), "share_dt_s": 3 * STEP,  # eligible share every 15 min, full trace
        "client_frac": E[:, lo:hi].mean(1).tolist(),
        "ever_12h": float((E[:, leg0:leg1].max(1) > 0).mean()),
        "sessions_s": [s for r in res for s in r[3]],
        "left_1h": float((H[:, :-1] * (1 - H[:, 1:])).sum() / max(H[:, :-1].sum(), 1)),  # eligible now, not in 1 h
        **turnover(trace, n),
    }
    return out


def turnover(trace, n):
    """F2a overlay, no forecasts: hourly share of the eligible pool that was not eligible 1 h earlier, full trace."""
    import yaml
    keys = sorted(yaml.safe_load(open(os.path.join(HERE, "..", "_metadata", "trainer_registry.yaml")))["trainers"])[:n]
    ev = {k: raw_events(trace, k) for k in keys}
    grid = np.arange(0.0, max(float(e[-1][0]) for e in ev.values() if e), 3600.0)  # the last state holds to the trace's end
    rows = []
    for k in keys:
        times, states = [float(t) for t, _ in ev[k]], [str(s).split(".")[-1] for _, s in ev[k]]
        rows.append([(states[bisect.bisect_right(times, t) - 1] if times and t >= times[0] else "AVL_TRAIN") == "AVL_TRAIN" for t in grid])
    E = np.array(rows, float)
    new = (E[:, 1:] * (1 - E[:, :-1])).sum(0) / np.maximum(E[:, 1:].sum(0), 1)
    return {"turnover_t": [0.0, *new.tolist()], "turnover_dt_s": 3600.0}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traces", nargs="*", default=["mobiperf_3st", "syn_50_3st"])
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--days", type=float, nargs=2, default=[1, 5])
    ap.add_argument("--every-min", type=float, default=30)
    ap.add_argument("--procs", type=int, default=96)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--turnover-only", action="store_true", help="add only stats.turnover_t to the existing JSONs (no forecasts)")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    if a.turnover_only:
        for tr in a.traces:
            path = os.path.join(a.out, f"fx_n147_{tr}.json")
            r = json.load(open(path))
            r["stats"].update(turnover(tr, a.n))
            json.dump(r, open(path, "w"))
            print(f"{tr}: turnover mean {100 * np.mean(r['stats']['turnover_t'][1:]):.0f}% -> {path}")
        return
    for tr in a.traces:
        r = run(tr, a.n, a.days[0], a.days[1], a.every_min * 60, a.procs)
        path = os.path.join(a.out, f"fx_n147_{tr}.json")
        json.dump(r, open(path, "w"))
        print(f"== {tr} (n {r['n']}, days {r['days']}) -> {path}")
        print("| question | h | pos | " + " | ".join(f"{p} AUC · Brier" for p in PREDS) + " |")
        for k, m in r["metrics"].items():
            q, h = k.split("_")
            cells = " | ".join(f"{m[p]['auc']:.2f} · {m[p]['brier']:.3f}" if m[p]["auc"] is not None else "–" for p in PREDS)
            print(f"| {q} | {int(h) // 60} min | {m['pos_rate']:.2f} | {cells} |")
        print("autocorr by lag:", {f"{k // 60}m": round(v, 2) for k, v in r["autocorr"].items()})
        print(f"ever eligible in a 12 h leg: {r['stats']['ever_12h']:.2f}; eligible now but not in 1 h {r['stats']['left_1h']:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
