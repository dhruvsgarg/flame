#!/usr/bin/env python3
"""FedDance sprint sanity checks on stored runs (FELIX_READINESS "FedDance sprint", S-steps).

  fd_sanity.py RUN_DIR [RUN_DIR ...] [--k 10] [--overcommit 1.3] [--expect 'lr=0.04 batch=32 epochs=1 optimizer=sgd']
  fd_sanity.py --pair REAL_DIR SIM_DIR [--rounds 20]

Per run (PASS/FAIL each): config applied ([TRAINER_HP]), every commit has exactly K updates, picks <= ceil(overcommit*K) and
never an UN_AVL client, FedDance utilities vary (not degenerate), no update_rejected / NaN loss, evals arrive and accuracy rises.
--pair: per (trainer, round) training losses equal in real and sim up to the first pick fork (FX-D144), same commit rounds.
"""
import argparse
import glob
import json
import math
import os
import re
import sys


LOSS_RTOL = 1e-4


def agg(run):
    f = glob.glob(os.path.join(run, "telemetry", "aggregator_*.jsonl"))
    return [json.loads(ln) for ln in open(f[0])] if f else []


def losses(run):
    out = {}
    for f in glob.glob(os.path.join(run, "telemetry", "trainer_*.jsonl")):
        for ln in open(f):
            r = json.loads(ln)
            if r.get("event") == "trainer_round" and r.get("final_loss") is not None:
                out[(f[-9:-6], r["round"])] = r["final_loss"]
    return out


def check_run(run, k, oc, expect):
    res, ev = [], agg(run)
    hp = next((ln for f in glob.glob(os.path.join(run, "*trainers.log")) for ln in open(f, errors="replace")
               if "[TRAINER_HP]" in ln), "")
    got = dict(re.findall(r"(\w+)=([^\s]+)", hp.split("[TRAINER_HP]")[-1])) if hp else {}
    bad = {kk: (v, got.get(kk)) for kk, v in expect.items() if got.get(kk) != v}
    res.append(("config applied", not bad and bool(got), bad or got))
    rounds = [e for e in ev if e.get("event") == "agg_round"]
    sizes = sorted({len(e.get("contributing_trainers") or []) for e in rounds})
    res.append((f"commits have K={k} updates", bool(rounds) and sizes == [k], f"{len(rounds)} rounds, sizes {sizes}"))
    sel = [e for e in ev if e.get("event") == "selection" and e.get("task") == "train"]
    cap = math.ceil(oc * k)
    over = [e["round"] for e in sel if len(e.get("chosen") or []) > cap]
    unav = [e["round"] for e in sel for c in e.get("chosen") or []
            if (e.get("per_trainer") or {}).get(c, {}).get("avl_state") == "UN_AVL"]
    res.append((f"picks <= {cap}, none UN_AVL", not over and not unav, f"over {over[:5]} unavail {unav[:5]}"))
    utils = [round(v["utility"], 6) for e in sel[len(sel) // 4:] for v in (e.get("per_trainer") or {}).values()
             if v.get("utility") is not None]
    res.append(("utilities vary", len(set(utils)) > 5, f"{len(set(utils))} distinct of {len(utils)}"))
    rej = sum(e.get("event") == "update_rejected" for e in ev)
    nan = sum(1 for v in losses(run).values() if v != v)
    res.append(("no rejects / NaN", rej == 0 and nan == 0, f"rejected {rej}, NaN losses {nan}"))
    acc = [e["test-accuracy"] for e in ev if e.get("event") == "agg_eval" and e.get("test-accuracy") is not None]
    res.append(("evals arrive, accuracy rises", len(acc) >= 2 and acc[-1] > acc[0], f"{len(acc)} evals, {acc[:1]} -> {acc[-1:]}"))
    return res


def check_pair(real, sim, n_rounds):
    lr, ls = losses(real), losses(sim)
    common = sorted(x for x in set(lr) & set(ls) if x[1] <= n_rounds)
    rel = {x: abs(lr[x] - ls[x]) / max(1e-12, abs(lr[x])) for x in common}
    diff = [x for x in common if rel[x] > LOSS_RTOL]  # GPU float order leaves ~1e-6; a different RNG stream gives ~1e-2
    rr = [e["round"] for e in agg(real) if e.get("event") == "agg_round"]
    sr = [e["round"] for e in agg(sim) if e.get("event") == "agg_round"]
    return [("losses identical (FX-D144)", bool(common) and not diff, f"{len(common)} (trainer, round) pairs, max rel {max(rel.values(), default=0):.1e}, over {LOSS_RTOL:g}: {diff[:3]}"),
            ("commit rounds match", rr[:n_rounds] == sr[:n_rounds], f"real {len(rr)} sim {len(sr)} rounds")]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--pair", nargs=2)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--overcommit", type=float, default=1.3)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--expect", default="lr=0.04 batch=32 epochs=1 optimizer=sgd")
    a = ap.parse_args(argv)
    expect = dict(kv.split("=", 1) for kv in a.expect.split())
    fails = 0
    jobs = [(r, check_run(r, a.k, a.overcommit, expect)) for r in a.runs]
    if a.pair:
        jobs.append((" vs ".join(os.path.basename(p.rstrip("/")) for p in a.pair), check_pair(*a.pair, a.rounds)))
    for name, res in jobs:
        print(f"== {os.path.basename(name.rstrip('/'))}")
        for what, ok, detail in res:
            fails += not ok
            print(f"  {'PASS' if ok else 'FAIL'}  {what}: {detail}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
