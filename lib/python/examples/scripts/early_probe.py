#!/usr/bin/env python3
"""PR27 early probes: one PASS/FAIL/WAIT line per FX-D12x claim, from telemetry of runs started after --since (epoch s).

  early_probe.py --since 1791600000 [--watch SECONDS]   # --watch re-probes until no WAIT remains (or the time is up)
"""
import argparse
import glob
import json
import math
import os
import time

EXP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "async_cifar10", "experiments")


def runs(since, *subs):
    out = [d for d in glob.glob(f"{EXP}/run_*") if os.path.getmtime(d) >= since and all(s in d for s in subs)]
    return sorted(out)


def events(run, kind, role="aggregator"):
    for f in glob.glob(f"{run}/telemetry/{role}*.jsonl"):
        for ln in open(f):
            try:
                r = json.loads(ln)
            except ValueError:
                continue
            if r.get("event") == kind:
                yield r


def probe_A(since):  # D122/D124: sim feddance speech syn_50 commits <= K per version and stamps every speed
    sims = runs(since, "gs_G0U_syn_50", "feddance", "_sim")
    rs = [r for r in events(sims[-1], "agg_round")] if sims else []
    if len(rs) < 5:
        return "WAIT", f"A sim rounds={len(rs)} < 5"
    over = [r["round"] for r in rs if len(r["contributing_trainers"]) > 5]
    zero = [r["round"] for r in rs if any(s == 0 for s in r["trainer_speed_s"])]
    return ("PASS" if not over and not zero else "FAIL"), f"A rounds>K={over} zero-speed rounds={zero}"


def probe_C(since):  # D123: real vs sim preferred duration and first picks, speech syn_20 oort
    pair = {m: runs(since, "gs_T3_syn_20s", "_oort_n", f"_{m}") for m in ("real", "sim")}
    if not all(pair.values()):
        return "WAIT", "C legs not started"
    sel = {m: [r for r in events(pair[m][-1], "selection") if r.get("round_preferred_duration_s")] for m in pair}
    n = min(len(sel["real"]), len(sel["sim"]))
    if n < 6:
        return "WAIT", f"C selections={n} < 6"
    gap = max(abs(a["round_preferred_duration_s"] - b["round_preferred_duration_s"]) / b["round_preferred_duration_s"]
              for a, b in zip(sel["real"][:n], sel["sim"][:n]))
    forks = sum(set(a["chosen"]) != set(b["chosen"]) for a, b in zip(sel["real"][:n], sel["sim"][:n]))
    return ("PASS" if gap < 0.01 else "FAIL"), f"C pref gap max {gap:.4f} (<0.01), picks forked {forks}/{n} (was fork at #3)"


def probe_D(since):  # D107: cifar oort finite through 10 rounds
    rs = runs(since, "_G0U_", "_oort", "_real")
    rs = [d for d in rs if "gs_" not in d]
    if not rs:
        return "WAIT", "D leg not started"
    ar = list(events(rs[-1], "agg_round"))
    bad = [r["round"] for r in ar if any(not math.isfinite(u) for u in r.get("stat_utility", []) if u is not None)]
    rej = sum(1 for _ in events(rs[-1], "update_rejected"))
    if len(ar) < 10 and not bad and not rej:
        return "WAIT", f"D rounds={len(ar)} < 10"
    return ("PASS" if not bad and not rej else "FAIL"), f"D rounds={len(ar)} nonfinite={bad} rejected={rej}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", type=float, required=True)
    ap.add_argument("--watch", type=int, default=0)
    a = ap.parse_args()
    end = time.time() + a.watch
    while True:
        res = [(p.__name__[-1], *p(a.since)) for p in (probe_A, probe_C, probe_D)]
        for k, v, msg in res:
            print(f"[{time.strftime('%H:%M:%S')}] probe {k} {v}: {msg}", flush=True)
        if not any(v == "WAIT" for _, v, _ in res) or time.time() > end:
            break
        time.sleep(120)


if __name__ == "__main__":
    main()
