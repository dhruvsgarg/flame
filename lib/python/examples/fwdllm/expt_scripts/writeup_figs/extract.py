#!/usr/bin/env python3
"""Pull the 2026-08-20 arms out of `experiments/` into a small JSON cache.

The aggregator telemetry is 1.7-2.8 GB per arm, so every figure reading it
directly would re-scan tens of gigabytes. One pass here, cached; the figure
script never touches `experiments/`.

  ./extract.py            # writes data/*.json, skips what already exists
  ./extract.py --force    # re-scan
"""
import argparse
import bisect
import glob
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.normpath(os.path.join(HERE, "..", "..", "experiments"))
OUT = os.path.join(HERE, "data")

ARMS = {
    "agnews_controller": "run_20260820_152215_fluxtune_agnews*",
    "agnews_control":    "run_20260820_021843_fluxtune_agnews*",
    "yahoo_controller":  "run_20260820_125003_fluxtune_yahoo*",
    "yahoo_control":     "run_20260820_151619_fluxtune_yahoo*",
    "yelp-p_controller": "run_20260820_125010_fluxtune_yelp-p*",
    "yelp-p_control":    "run_20260820_161751_fluxtune_yelp-p*",
}

# [BmaxProbe] commit=150 B_max 0.693147 -> 0.507491 (sensed=... n=1 B_rem=0.234791
#   Phi_knee=1.265 policy=mean) base_acc=0.744 B=0.2727 rho*=0.0395635 ... curve[1.5:0.277 ...]
_PROBE = re.compile(
    r"\[BmaxProbe\] commit=(\d+) B_max [\d.]+ -> ([\d.]+) "
    r"\(sensed=([\d.]+) n=(\d+) B_rem=([\d.]+) Phi_knee=([\d.]+) [^)]*\) "
    r"base_acc=([\d.]+) B=([\d.]+) rho\*=([\d.eE+-]+).*?curve\[([^\]]*)\]"
)


def acc_vs_vclock(agg_jsonl):
    """Held-out accuracy resampled onto the simulated clock.

    `agg_eval` carries no vclock, and `vclock_charge` carries no accuracy, so
    the two streams are joined on wall timestamp and interpolated.
    """
    vt, vv, ev = [], [], []
    with open(agg_jsonl, errors="replace") as fh:
        for line in fh:
            if '"vclock_charge"' in line:
                try:
                    d = json.loads(line)
                    vt.append(d["ts"]); vv.append(d["vclock_now"])
                except Exception:
                    pass
            elif '"agg_eval"' in line:
                try:
                    d = json.loads(line)
                    ev.append((d["ts"], d["test-accuracy"]))
                except Exception:
                    pass
    def to_vclock(ts):
        i = bisect.bisect_left(vt, ts)
        if i <= 0:
            return vv[0] if vv else 0.0
        if i >= len(vt):
            return vv[-1]
        t0, t1, v0, v1 = vt[i - 1], vt[i], vv[i - 1], vv[i]
        return v1 if t1 == t0 else v0 + (v1 - v0) * (ts - t0) / (t1 - t0)
    return [[round(to_vclock(t), 1), a] for t, a in ev]


def budget_trace(agg_jsonl):
    """(commit, rho, B/B_max, trainable_weight_norm) per commit."""
    rows = []
    with open(agg_jsonl, errors="replace") as fh:
        for line in fh:
            if '"server_update"' not in line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            rows.append([len(rows) + 1, d.get("rho"), d.get("budget_frac"),
                         d.get("trainable_weight_norm")])
    return rows


def probes(agg_log):
    out = []
    with open(agg_log, errors="replace") as fh:
        for line in fh:
            if "[BmaxProbe] commit=" not in line:
                continue
            m = _PROBE.search(line)
            if not m:
                continue
            (commit, b_max, sensed, n, b_rem, knee,
             base_acc, B, rho, curve) = m.groups()
            pts = []
            for tok in curve.split():
                phi, acc = tok.split(":")
                pts.append([float(phi), float(acc)])
            out.append(dict(commit=int(commit), b_max=float(b_max),
                            sensed=float(sensed), n=int(n), b_rem=float(b_rem),
                            phi_knee=float(knee), base_acc=float(base_acc),
                            B=float(B), rho_star=float(rho), curve=pts))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    for name, pattern in ARMS.items():
        dest = os.path.join(OUT, f"{name}.json")
        if os.path.exists(dest) and not a.force:
            print(f"  [skip] {name}")
            continue
        hits = glob.glob(os.path.join(RUNS, pattern))
        if not hits:
            print(f"  [MISS] {name} -- no run dir matching {pattern}")
            continue
        run = hits[0]
        agg_jsonl = glob.glob(os.path.join(run, "telemetry", "aggregator_*.jsonl"))[0]
        agg_log = glob.glob(os.path.join(run, "*aggregator.log"))[0]
        print(f"  [scan] {name} <- {os.path.basename(run)}")
        payload = dict(
            run=os.path.basename(run),
            acc=acc_vs_vclock(agg_jsonl),
            budget=budget_trace(agg_jsonl),
            probes=probes(agg_log),
        )
        with open(dest, "w") as fh:
            json.dump(payload, fh)
        print(f"         {len(payload['acc'])} evals, {len(payload['budget'])} commits, "
              f"{len(payload['probes'])} probe fires")


if __name__ == "__main__":
    main()
