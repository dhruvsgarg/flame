#!/usr/bin/env python3
"""FX-D197: did every task get the FedScale task time of its trainer's device? Per leg, per baseline.

  task_time_audit.py RUN_DIR... [--pools POOL...]

Train tasks: expected = (3 x samples x ms/sample / 1000 + 2 x model kbit / kbps) / delay factor (registry device, the trainer's
reported samples and model size), vs the charged budget (minus the I9 pass) and the charged duration max(gpu, budget).
Eval tasks: charged span (aggregator commit sct - dispatch send) vs 1 x visible shard x ms + one download (from the leg's train
samples). Prints mismatches (> 1 ms) and the mean / p50 / p90 train and eval time per leg.
"""
import argparse
import glob
import json
import os
import statistics as st
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from accuracy_table import rows_from_pools  # noqa: E402

REG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_metadata", "trainer_registry.yaml")


def devices():
    reg = yaml.safe_load(open(REG))["trainers"]
    return {t["task_id"]: (float(t["compute_ms_per_sample"]), float(t["bandwidth_kbps"])) for t in reg.values()}


def _factor(run_dir):
    try:
        hp = json.load(open(os.path.join(run_dir, "aggregator_config.json"))).get("hyperparameters", {})
    except OSError:
        hp = {}
    f = hp.get("trainingDelayFactor") or hp.get("training_delay_factor")
    return float(f) if f else 1.0


def q(xs):
    xs = sorted(xs)
    return f"mean {st.mean(xs):6.1f} p50 {xs[len(xs) // 2]:6.1f} p90 {xs[int(0.9 * (len(xs) - 1))]:6.1f}" if xs else "–"


def audit(run_dir, dev):
    fac, train, bad, ex, shard = _factor(run_dir), [], 0, [], {}
    for f in glob.glob(os.path.join(run_dir, "telemetry", "trainer_*.jsonl")):
        tid = os.path.basename(f)[len("trainer_"):-len(".jsonl")]
        for line in open(f):
            if '"trainer_round"' not in line:
                continue
            e = json.loads(line)
            if e.get("task_to_perform", "train") != "train" or e.get("model_samples") is None:
                continue
            ms, kbps = dev[tid]
            exp = (3 * e["model_samples"] * ms / 1000 + 2 * e["model_kbit"] / kbps) / fac
            got = e["training_budget_s"] - (e.get("utility_pass_s") or 0.0) - (e.get("val_pass_s") or 0.0)
            dur = e["sim_round_duration_s"] if e.get("sim_round_duration_s") is not None else e["training_budget_s"]
            if abs(got - exp) > 1e-3 or dur + 1e-6 < e["training_budget_s"]:
                bad += 1
                ex.append((tid[-4:], round(got, 3), round(exp, 3)))
            train.append(e["training_budget_s"])
            shard[tid] = (e.get("visible_samples") or e["model_samples"], e["model_kbit"])
    evals = []
    agg = glob.glob(os.path.join(run_dir, "telemetry", "aggregator_*.jsonl"))
    if agg:
        sent = {}
        for line in open(agg[0]):
            if '"dispatch"' in line and '"eval"' in line:
                e = json.loads(line)
                sent[(e["end_id"], e.get("round"))] = e.get("sim_send_ts")
            elif '"agg_round"' in line and '"eval"' in line:
                e = json.loads(line)
                for t in e.get("contributing_trainers") or []:
                    s0 = max((v for (k, r), v in sent.items() if k == t and v is not None and v <= (e.get("sim_completion_ts_recv") or 0)),
                             default=None)
                    if s0 is not None and e.get("sim_completion_ts_recv") is not None:
                        evals.append(e["sim_completion_ts_recv"] - s0)
    return dict(n=len(train), bad=bad, ex=ex[:3], train=train, evals=evals)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--pools", nargs="*", default=[])
    a = ap.parse_args(argv)
    legs = [(os.path.basename(r.rstrip("/")), r) for r in a.runs]
    legs += [(f"{ph} {b}", d) for ph, b, _t, real, sim in rows_from_pools(a.pools) for d in (real, sim) if d]
    dev, fails = devices(), 0
    for name, d in legs:
        if not os.path.isdir(d):
            continue
        r = audit(d, dev)
        fails += r["bad"] > 0
        mode = "real" if d.rstrip("/").endswith("_real") else "sim"
        print(f"{name[:60]:60s} {mode:4s} train n {r['n']:5d} mismatch {r['bad']:3d} {r['ex'] or ''} | train {q(r['train'])} | "
              f"eval n {len(r['evals'])} {q(r['evals'])}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
