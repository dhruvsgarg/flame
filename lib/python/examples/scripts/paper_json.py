#!/usr/bin/env python3
"""Paper results as data: one JSON per dataset of every reportable metric per cell × stack (felix_paper_outline Table 2), from leg digests.

  paper_json.py [--root R] [--seeds 1] [--out DIR] [--datasets cifar10 google_speech] [--no-build-root]

Reads the same manifest and digests as paper_plots.py. Writes results_<dataset>.json (default paper_felix/figs/expts/): re-run after
legs land; a stack whose leg is not done is `"status": "pending"`, never dropped, and `meta.complete` says whether every leg is in.
Root: a scratch root of symlinks over RUN_ROOT (cifar's sync / oracle pools live in `cifar_e50`, lr decay every 50) with this tree's
code key, so the manifest accepts the pools; rebuilt on every run unless --no-build-root.
"""
import argparse
import datetime
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE]
import paper_plots as pp  # noqa: E402
import paper_runs as pr  # noqa: E402

OUT = os.path.join(HERE, "..", "paper_felix", "figs", "expts")
OVERLAY = {"cifar10": ("cifar_e50", ("sync_srv", "rstar_srv", "fdstar_srv", "favstar_srv"))}  # pools taken from the overlay dir
HOURS = (6, 12, 24, 48, 72, 96, 120)
LEVELS = {"cifar10": (50, 60, 70, 75, 80), "google_speech": (50, 60, 70, 80, 84, 88)}
NAMES = ("Felix", *pp.DEPLOY, *pp.ORACLE)
CELL_TRACE = {"s0u": "syn_0", "s20u": "syn_20_3st", "s50u": "syn_50_3st", "smu": "mobiperf_3st"}
ABL = {"ladder": ("b1", "uo_o", "fn"), "swap_availability": ("fx_ck", "fx_or"), "swap_selection": ("fx_rnd", "fx_oort"),
       "swap_aggregation": ("fx_rold", "fx_r1"), "swap_concurrency": ("fx_cfix", "fx_ca"), "rejected": ("fx_ucb", "fx_e2")}


def build_root(run_root, dst, key):
    """Symlink root of `run_root` (+ overlays) at dst, code keys forced to `key`."""
    for ds, sub in pr.DS.items():
        d = os.path.join(dst, sub)
        os.makedirs(d, exist_ok=True)
        for f in os.listdir(d):
            if os.path.islink(os.path.join(d, f)):
                os.unlink(os.path.join(d, f))
        for f in sorted(os.listdir(os.path.join(run_root, sub))):
            if f != "code_key.txt":
                os.symlink(os.path.join(run_root, sub, f), os.path.join(d, f))
        if ds in OVERLAY and os.path.isdir(os.path.join(run_root, OVERLAY[ds][0])):
            over = os.path.join(run_root, OVERLAY[ds][0])
            for f in sorted(os.listdir(over)):
                if any(f.endswith("_" + a) or f"_{a}_" in f for a in OVERLAY[ds][1]) and f.startswith("FB2E2E"):
                    os.unlink(os.path.join(d, f)) if os.path.lexists(os.path.join(d, f)) else None
                    os.symlink(os.path.join(over, f), os.path.join(d, f))
        open(os.path.join(d, "code_key.txt"), "w").write(key + "\n")


def avg(xs):
    xs = [x for x in xs if x is not None]
    return st.mean(xs) if xs else None


def rnd(x, n=2):
    return None if x is None else round(x, n)


def smooth(acc, w):
    """5-eval trailing mean of the accuracy series."""
    return [st.mean(acc[max(0, i - w + 1):i + 1]) for i in range(len(acc))]


def first_cross(d, level):
    """(h, gb, updates) at which the trailing mean first reaches level (fraction), else None."""
    a = d["acc"]
    for i, m in enumerate(smooth(a["acc"], d["config"].get("window") or 5)):
        if m >= level:
            return a["t"][i] / 3600, a["gb"][i], a["updates"][i]
    return None


def cell_target(d, ds, cell):
    """The cell's reporting target: the leg's config target lowered by trace_targets.json (older digests lack report_target)."""
    ov = json.load(open(os.path.join(HERE, "..", "_metadata", "trace_targets.json"))).get(ds, {}).get(CELL_TRACE[cell])
    return d["config"]["target"] if ov is None else min(d["config"]["target"], ov)


def acc_at(d, hours):
    """Trailing-mean accuracy at the last eval at or before `hours` (None past the run's end)."""
    a = d["acc"]
    if hours > d["end"]["h"] + 0.25:
        return None
    m = smooth(a["acc"], d["config"].get("window") or 5)
    idx = [i for i, t in enumerate(a["t"]) if t / 3600 <= hours]
    return m[idx[-1]] if idx else None


def leg_metrics(d, ds, cell, felix_h=None):
    """Every reportable metric of one leg digest."""
    levels, tgt = LEVELS[ds], cell_target(d, ds, cell)
    c = first_cross(d, tgt)
    h = c and {"h": c[0], "gb": c[1], "updates": c[2]}
    sm = smooth(d["acc"]["acc"], d["config"].get("window") or 5)
    fates = (d.get("fates") or {}).get("gb") or {}
    tot = sum(fates.values())
    stal = ((d.get("staleness") or {}).get("all")) or {}
    n = sum(stal.values())
    slots = d.get("slots") or {}
    cap = slots.get("capacity_s") or 0
    dep = d.get("departures") or {}
    out = {
        "status": "hit" if h else "miss", "target": tgt,
        "time_h": rnd(h and h["h"]), "gb": rnd(h and h["gb"], 1), "updates_to_target": h and h["updates"],
        "updates_per_h": rnd(h and h["updates"] / h["h"], 0) if h else None,
        "final_acc": rnd(100 * d["end"]["acc_last"], 2), "peak_acc5": rnd(100 * max(sm), 2),
        "end_h": rnd(d["end"]["h"]), "end_gb": rnd(d["end"]["gb"], 1), "total_updates": (d.get("updates") or {}).get("total"),
        "acc_at_h": {str(x): rnd(None if (a := acc_at(d, x)) is None else 100 * a) for x in HOURS},
        "time_h_to_acc": {str(x): rnd((c := first_cross(d, x / 100)) and c[0]) for x in levels},
        "gb_to_acc": {str(x): rnd((c := first_cross(d, x / 100)) and c[1], 1) for x in levels},
        "acc_at_felix_hit_h": rnd(None if felix_h is None or (a := acc_at(d, felix_h)) is None else 100 * a),
        "gb_by_fate": {k: rnd(v, 1) for k, v in fates.items()}, "paid_not_used_pct": rnd(100 * (1 - fates.get("aggregated", 0) / tot), 2) if tot else None,
        "mean_staleness": rnd(sum(int(k) * v for k, v in stal.items()) / n, 2) if n else None,
        "slot_busy_pct": rnd(100 * slots["busy_s"] / cap, 1) if cap and "busy_s" in slots else None,
        "slot_held_departed_pct": rnd(100 * slots["held_departed_s"] / cap, 1) if cap and "held_departed_s" in slots else None,
        "left_mid_task_pct": rnd(100 * dep["share"], 2) if "share" in dep else None,
        "slot_time_h": {k: rnd(v, 1) for k, v in pp.slot_split(d).items()} if "by_fate" in slots else None,
        "slot_time_lost_per_departure_s": rnd(pp.lost_per_departure_s(d), 0) if "by_fate" in slots else None,
        "timeouts": (d.get("fates") or {}).get("counts", {}).get("timeout", 0),
        "wire_msgs": (d.get("overhead") or {}).get("wire_msgs"), "diverged": d["end"].get("diverged", False),
    }
    return out


def ratio(a, b):
    return None if a is None or b is None or not b else round(a / b, 3)


def vs(fx, o):
    """Felix against another stack: speedup = their time / Felix's, gb_ratio = Felix's / theirs (< 1 = Felix cheaper)."""
    both = fx.get("time_h") is not None and o.get("time_h") is not None
    return {"speedup": ratio(o["time_h"], fx["time_h"]) if both else None, "gb_ratio": ratio(fx["gb"], o["gb"]) if both else None,
            "gb_saving_pct": rnd(100 * (1 - fx["gb"] / o["gb"]), 1) if both and o["gb"] else None,
            "acc_gain_pp_at_felix_hit_h": rnd(fx["target"] * 100 - o["acc_at_felix_hit_h"], 2) if o.get("acc_at_felix_hit_h") is not None else None}


def best(stacks, names):
    hit = [(stacks[n]["time_h"], n) for n in names if stacks.get(n, {}).get("status") == "hit"]
    return min(hit)[1] if hit else None


def cell_block(D, ds, cell, state):
    fxl = D.stack(ds, cell, "Felix")
    fxc = [first_cross(d, cell_target(d, ds, cell)) for d in fxl]
    felix_h = avg(c[0] for c in fxc) if fxc and all(fxc) else None
    stacks = {}
    for name in NAMES:
        legs = D.stack(ds, cell, name)
        if not legs:
            arm, b = next(k for k, v in pp.STACKS.items() if v == name)
            stacks[name] = {"status": "pending", "reason": state.get((ds, cell, arm, b), ("missing", ""))[1] or "missing"}
            continue
        ms = [leg_metrics(d, ds, cell, felix_h) for d in legs]
        m = ms[0] if len(ms) == 1 else {**ms[0], **{k: rnd(avg(x[k] for x in ms)) for k in ("time_h", "gb", "final_acc", "peak_acc5")},
                                        "status": "hit" if all(x["status"] == "hit" for x in ms) else "miss"}
        m["kind"] = "felix" if name == "Felix" else "oracle" if name in pp.ORACLE else "deployable"
        m["n_seeds"] = len(ms)
        stacks[name] = m
    fx = stacks["Felix"]
    if fx["status"] != "pending":
        for n, m in stacks.items():
            if n != "Felix" and m["status"] != "pending":
                m["felix_vs_this"] = vs(fx, m)
    bd, bo = best(stacks, pp.DEPLOY), best(stacks, pp.ORACLE)
    summary = {"target": fx.get("target"), "felix_first": None, "all_stacks_in": all(m["status"] != "pending" for m in stacks.values()),
               "best_deployable": bd, "best_oracle": bo}
    if fx["status"] == "hit":
        summary["felix_first"] = all(m["status"] != "hit" or m["time_h"] >= fx["time_h"] for n, m in stacks.items() if n != "Felix")
        for key, n in (("vs_best_deployable", bd), ("vs_best_oracle", bo), ("vs_fedbuff", "FedBuff")):
            if n and stacks[n]["status"] != "pending":
                summary[key] = {"stack": n, **{k: v for k, v in stacks[n]["felix_vs_this"].items() if k != "acc_gain_pp_at_felix_hit_h"}}
    return {"label": pp.CELL[cell], "target": summary["target"], "stacks": stacks, "summary": summary}


def ablations(D, ds, cell, state):
    """Arms of each ablation set on one ablation cell, vs Felix `fx` and FedBuff."""
    fx = D.stack(ds, cell, "Felix")
    if not fx:
        return {}
    ref = {"Felix": leg_metrics(fx[0], ds, cell)}
    out = {}
    for group, arms in ABL.items():
        rows = {}
        for a in arms:
            legs = D.get(ds, cell, a, "felix")
            if legs:
                m = leg_metrics(legs[0], ds, cell)
                rows[pp.ARMS[a]] = {"arm": a, **{k: m[k] for k in ("status", "time_h", "gb", "final_acc", "peak_acc5", "mean_staleness", "paid_not_used_pct")},
                                    "vs_felix": {"time_ratio": ratio(m["time_h"], ref["Felix"]["time_h"]), "gb_ratio": ratio(m["gb"], ref["Felix"]["gb"])}}
        if rows:
            out[group] = rows
    return out


def build(D, ds, root, seeds, key):
    state = {(d_, c, a, b): v for (rs, d_, (ph, c, a, b, tr), s), v in D.states.items() if rs and s == seeds[0]}
    cells = {}
    for c in pr.ALL:
        cells[c] = cell_block(D, ds, c, state)
        ab = ablations(D, ds, c, state) if c in pr.hp.FB2_ABL else {}
        if ab:
            cells[c]["ablations"] = ab
    allstacks = [m["status"] for c in cells.values() for m in c["stacks"].values()]
    done = sum(s != "pending" for s in allstacks)
    return {"meta": {"dataset": ds, "generated": datetime.datetime.now().isoformat(timespec="minutes"), "seeds": seeds, "run_root": root,
                     "code_key": key, "legs_done": done, "legs_total": len(allstacks), "complete": done == len(allstacks),
                     "units": {"time_h": "hours of simulated time to the cell's target (5-eval mean)", "gb": "cumulative wire GB at that time",
                               "acc": "percent", "speedup": "stack time / Felix time", "gb_ratio": "Felix GB / stack GB",
                               "slot_time_h": "slot-hours over the whole run (to the hit or 120 h): useful / discarded compute, held by a departed client, waiting at the barrier, idle"},
                     "notes": "status hit = reached target, miss = ran to the end without it (final_acc, peak_acc5), pending = leg not done"},
            "stack_kinds": {n: ("felix" if n == "Felix" else "oracle" if n in pp.ORACLE else "deployable") for n in NAMES},
            "cells": cells}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--root", default=pr.default_root() if hasattr(pr, "default_root") else None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1])
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--datasets", nargs="+", default=list(pr.DS))
    ap.add_argument("--no-build-root", action="store_true")
    a = ap.parse_args(argv)
    key = pr.hp.code_key()
    root = a.root
    if not a.no_build_root:
        root = os.path.join(a.root, "paper_root")
        build_root(a.root, root, key)
    D = pp.Data(root, a.seeds, False)
    os.makedirs(a.out, exist_ok=True)
    for ds in a.datasets:
        p = os.path.join(a.out, f"results_{ds}.json")
        j = build(D, ds, a.root, a.seeds, key)
        json.dump(j, open(p, "w"), indent=1)
        print(f"{p}: {j['meta']['legs_done']} / {j['meta']['legs_total']} stacks")


if __name__ == "__main__":
    main()
