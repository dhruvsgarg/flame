#!/usr/bin/env python3
"""F42 (R27): how availability knowledge turns into stalls, per sim leg (telemetry + the leg's own trace; no runs).

  predictor_audit.py POOL... [--runs RUN ...]

Per leg: train picks; survival = share of picks that stay AVL_TRAIN for the leg's median task time after dispatch (ground truth);
REFL: survival of the picks its predictor flagged as leaving soon (`priority_ids`) vs the rest; FedDance: Spearman of its believed
availability V_m with survival; abandons / h (slots held to the timeout) and wasted slot-hours (dead picks x timeout); knowledge
mode (R27). Felix (notify) is the reference: it evicts at the flip, so a dead pick costs it one flip, not a timeout.
"""
import argparse
import glob
import json
import os
import re
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "async_cifar10", "scripts"))
from accuracy_table import rows_from_pools  # noqa: E402
from parity import event_invariants as ev  # noqa: E402
from selection_audit import spearman  # noqa: E402


def _share_train(trace, a, b):
    from flame.availability.trace import state_at
    if not trace:
        return 1.0
    ts = [a] + [t for t in trace.irange(a, b, inclusive=(False, False))] + [b]
    on = sum((t1 - t0) for t0, t1 in zip(ts, ts[1:]) if str(state_at(trace, t0)).endswith("AVL_TRAIN"))
    return on / (b - a) if b > a else 1.0


def audit(run_dir):
    run = ev.load_run(run_dir)
    gt = ev._ground_truth(run)
    durs = [e.get("sim_round_duration_s") for evs in run["trainers"].values() for e in evs
            if e.get("event") == "trainer_round" and e.get("task_to_perform", "train") == "train" and e.get("sim_round_duration_s")]
    d = st.median(durs) if durs else 60.0
    hp = run["cfg"].get("hyperparameters", {}) or {}
    know = hp.get("avail_knowledge") or ("notify" if str(hp.get("proactive_inflight_evict")) == "True"
                                         else "oracle" if str(hp.get("avail_select_filter")) == "True" else "none")
    picks, prio, vm, surv = 0, [], [], []
    for e in ev._events(run, "selection", task="train"):
        t = e.get("vclock_now")
        new = ev._newly_chosen(e) or e.get("chosen") or []  # REFL's random mode leaves explore / exploit empty
        if t is None or not new:
            continue
        pri = set(e.get("priority_ids") or [])
        for c in new:
            s = _share_train((gt or {}).get(c), t, t + d) >= 1.0
            picks += 1
            surv.append(s)
            if "priority_ids" in e:
                prio.append((c in pri, s))
            v = ((e.get("per_trainer") or {}).get(c) or {}).get("feddance_V")
            if v is not None:
                vm.append((v, s))
    ab = [x for x in ev._events(run, "abandon_timeout") if x.get("reason") != "aware_boundary_eviction"]
    to = float(hp.get("send_timeout_wait_s") or 0)
    hours = max((e.get("vclock_now") or 0) for e in run["agg"] if e.get("event") == "agg_round") / 3600 or 1
    out = dict(know=know, picks=picks, survive=sum(surv) / max(1, len(surv)), abandons_h=len(ab) / hours,
               wasted_slot_h=len(ab) * to / 3600 / hours, task_s=d)
    if prio:
        p = [s for f, s in prio if f]
        q = [s for f, s in prio if not f]
        out.update(prio_share=len(p) / len(prio), survive_prio=sum(p) / max(1, len(p)), survive_rest=sum(q) / max(1, len(q)))
    if vm:
        out["rho_V_survive"] = spearman([v for v, _ in vm], [float(s) for _, s in vm])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pools", nargs="*")
    ap.add_argument("--runs", nargs="*", default=[])
    a = ap.parse_args(argv)
    legs = [(p, b, s) for p, b, _, _, s in rows_from_pools(a.pools) if s] + [(os.path.basename(r), "", r) for r in a.runs]
    print("| leg | baseline | knowledge | picks | survive | REFL flagged share · survive flagged / rest | FedDance ρ(V, survive) "
          "| abandons / h | wasted slot-h / h |\n|---|---|---|---|---|---|---|---|---|")
    for phase, b, d in legs:
        if not os.path.isdir(d):
            continue
        b = b or re.search(r"_dbg_(?:google_speech_)?(\w+?)_n\d+", d).group(1)
        r = audit(d)
        pr = f"{r['prio_share']:.2f} · {r['survive_prio']:.2f} / {r['survive_rest']:.2f}" if "prio_share" in r else "–"
        fd = f"{r['rho_V_survive']:+.2f}" if "rho_V_survive" in r else "–"
        print(f"| {phase} | {b} | {r['know']} | {r['picks']} | {r['survive']:.2f} | {pr} | {fd} | {r['abandons_h']:.1f} "
              f"| {r['wasted_slot_h']:.2f} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
