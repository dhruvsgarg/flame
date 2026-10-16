#!/usr/bin/env python3
"""FELIX_INSIGHTS metrics per run: selection, waste, aggregation, overheads, wire bytes, Felix train/eval occupancy.

  insights.py RUN_DIR [RUN_DIR ...] [--json out.json] [--plot out.png]

One function per catalog ID (FELIX_INSIGHTS §3); `_real`/`_sim` twins are paired and their ratio printed (an insight
counts only if both sides agree). Reads `telemetry/*.jsonl` + `aggregator_config.json`; never writes into a run dir.
"""
import argparse
import collections
import glob
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor


def _jsonl(path):
    with open(path, errors="replace") as fh:
        for line in fh:
            try:
                yield json.loads(line)
            except ValueError:
                continue


def load(run_dir):
    agg = []
    for f in glob.glob(os.path.join(run_dir, "telemetry", "aggregator_*.jsonl")):
        agg.extend(_jsonl(f))
    trainers = collections.Counter()
    tr_wire = 0
    for f in glob.glob(os.path.join(run_dir, "telemetry", "trainer_*.jsonl")):
        for e in _jsonl(f):
            trainers[e.get("event")] += 1
            if e.get("event") == "wire":
                tr_wire += int(e.get("bytes") or 0)
    try:
        cfg = json.load(open(os.path.join(run_dir, "aggregator_config.json")))
    except (OSError, ValueError):
        cfg = {}
    return {"dir": run_dir, "agg": agg, "trainer_counts": trainers, "trainer_wire_bytes": tr_wire, "cfg": cfg}


def _ev(run, name, **match):
    return [e for e in run["agg"] if e.get("event") == name and all(e.get(k) == v for k, v in match.items())]


def _clock(e):
    v = e.get("vclock_now")
    return float(v) if v is not None else float(e["ts"])


def _gini(xs):
    xs = sorted(xs)
    n, s = len(xs), sum(xs)
    if n == 0 or s == 0:
        return None
    return sum((2 * i - n + 1) * x for i, x in enumerate(xs)) / (n * s)


def run_span(run):
    """Run clock span (s): sim vclock of the last commit, else wall since the trace origin."""
    rounds = _ev(run, "agg_round")
    if not rounds:
        return None
    if any(e.get("vclock_now") is not None for e in rounds):
        return max(float(e["vclock_now"]) for e in rounds if e.get("vclock_now") is not None)
    t0 = [e["ts"] for e in _ev(run, "trace_origin")] or [run["agg"][0]["ts"]]
    return float(rounds[-1]["ts"]) - float(t0[0])


def i_a2_coverage(run):
    per = collections.Counter()
    for e in _ev(run, "agg_round"):
        if e.get("task_to_perform") == "eval":
            continue
        per.update(e.get("contributing_trainers") or [])
    n = int(run["cfg"].get("hyperparameters", {}).get("min_trainers_to_start") or 0) or None
    counts = sorted(per.values(), reverse=True)
    top = counts[: max(1, len(counts) // 5)] if counts else []
    return {"contributors": len(per), "n": n, "coverage": (len(per) / n) if n else None,
            "gini_updates": _gini(counts + [0] * max(0, (n or 0) - len(counts))),
            "top20_share": (sum(top) / sum(counts)) if counts else None}


def i_a3_choice(run):
    out = {}
    for task in ("train", "eval"):
        sel = [e for e in _ev(run, "selection", task=task) if (e.get("num_chosen") or 0) > 0]
        r = [e["num_eligible"] / e["num_chosen"] for e in sel if e.get("num_eligible") is not None]
        out[task] = {"selections": len(sel), "chosen": sum(e["num_chosen"] for e in sel),
                     "choice_ratio_mean": (sum(r) / len(r)) if r else None,
                     "no_choice_share": (sum(1 for x in r if x <= 1.0) / len(r)) if r else None}
    return out


def i_a5_explore(run):
    sel = _ev(run, "selection", task="train")
    ex = sum(len(e.get("explore_ids") or []) for e in sel)
    ep = sum(len(e.get("exploit_ids") or []) for e in sel)
    return {"explore": ex, "exploit": ep, "explore_share": (ex / (ex + ep)) if ex + ep else None}


def i_b1_waste(run):
    dispatched = sum(e.get("num_chosen") or 0 for e in _ev(run, "selection", task="train"))
    aggregated = sum(len(e.get("contributing_trainers") or []) for e in _ev(run, "agg_round")
                     if e.get("task_to_perform") != "eval")
    abandons = collections.Counter(e.get("reason") for e in _ev(run, "abandon_timeout"))
    return {"train_dispatched": dispatched, "aggregated": aggregated,
            "waste_share": (1 - aggregated / dispatched) if dispatched else None, "abandons": dict(abandons)}


def i_c1_staleness(run):
    h = collections.Counter()
    for e in _ev(run, "agg_round"):
        if e.get("task_to_perform") != "eval":
            h.update(int(s) for s in (e.get("staleness") or []))
    n = sum(h.values())
    return {"updates": n, "stale_share": (sum(v for k, v in h.items() if k > 0) / n) if n else None,
            "gt5_share": (sum(v for k, v in h.items() if k > 5) / n) if n else None,
            "max": max(h) if h else None, "hist": dict(sorted(h.items()))}


def i_d1_server_wall(run):
    rows = _ev(run, "agg_timing")
    keys = ("recv_wait_s", "ingest_s", "commit_s", "other_s", "cycle_s")
    return {k: (sum(float(e.get(k) or 0) for e in rows) / len(rows)) if rows else None for k in keys} | {"commits": len(rows)}


def i_e1_wire(run):
    w = _ev(run, "wire")
    sent = sum(int(e["bytes"]) for e in w if e.get("dir") == "send")
    recv = sum(int(e["bytes"]) for e in w if e.get("dir") == "recv")
    agg = i_b1_waste(run)["aggregated"]
    return {"messages": len(w), "agg_sent_mb": sent / 1e6, "agg_recv_mb": recv / 1e6,
            "mb_per_aggregated_update": ((sent + recv) / 1e6 / agg) if (agg and w) else None,
            "trainer_side_mb": run["trainer_wire_bytes"] / 1e6}


def i_f1_occupancy(run):
    """Felix train/eval coupling (FELIX_INSIGHTS §1): rho = time share of train slots held by eval tasks, blocked share."""
    sel = sorted((e for e in _ev(run, "selection", task="train") if "in_flight_eval" in e), key=_clock)
    if not sel:
        return {"samples": 0}
    c = max(int(e.get("concurrency") or 0) for e in sel) or None
    held = blocked = span = 0.0
    for a, b in zip(sel, sel[1:]):
        dt = max(0.0, _clock(b) - _clock(a))
        span += dt
        held += dt * float(a.get("train_slots_held_by_eval") or 0)
        blocked += dt * (1.0 if a.get("train_blocked_by_eval") else 0.0)
    evals = sum(e.get("num_chosen") or 0 for e in _ev(run, "selection", task="eval"))
    return {"samples": len(sel), "c": c, "eval_tasks": evals,
            "rho": (held / (c * span)) if (c and span) else None,
            "blocked_time_share": (blocked / span) if span else None,
            "blocked_call_share": sum(1 for e in sel if e.get("train_blocked_by_eval")) / len(sel),
            "mean_in_flight_train": sum(e["in_flight_train"] for e in sel) / len(sel),
            "mean_in_flight_eval": sum(e["in_flight_eval"] for e in sel) / len(sel)}


def i_g1_rate(run):
    span = run_span(run)
    agg = i_b1_waste(run)["aggregated"]
    return {"span_s": span, "aggregated_per_h": (agg / span * 3600) if span else None}


METRICS = {"I-A2": i_a2_coverage, "I-A3": i_a3_choice, "I-A5": i_a5_explore, "I-B1": i_b1_waste, "I-C1": i_c1_staleness,
           "I-D1": i_d1_server_wall, "I-E1": i_e1_wire, "I-F1": i_f1_occupancy, "I-G1": i_g1_rate}


def analyze(run_dir):
    run = load(run_dir)
    out = {"run": os.path.basename(run_dir.rstrip("/"))}
    for k, f in METRICS.items():
        try:
            out[k] = f(run)
        except Exception as e:  # one broken metric must not hide the rest
            out[k] = {"error": repr(e)}
    return out


def _flat(d, p=""):
    for k, v in d.items():
        if isinstance(v, dict) and k != "hist" and k != "abandons":
            yield from _flat(v, f"{p}{k}.")
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            yield f"{p}{k}", v


def pairs(results):
    by = {}
    for r in results:
        m = re.match(r"run_\d+_\d+_(.*)_(real|sim)$", r["run"])
        if m:
            by.setdefault(m.group(1), {})[m.group(2)] = r
    return {k: v for k, v in by.items() if len(v) == 2}


def plot(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    names = [r["run"][16:60] for r in results]
    ax[0].barh(names, [r["I-B1"].get("waste_share") or 0 for r in results])
    ax[0].set_title("I-B1 train dispatches not aggregated")
    ax[1].barh(names, [r["I-C1"].get("stale_share") or 0 for r in results])
    ax[1].set_title("I-C1 stale share of aggregated updates")
    ax[2].barh(names, [r["I-F1"].get("rho") or 0 for r in results])
    ax[2].set_title("I-F1 rho: train slot-time held by eval")
    for a in ax[1:]:
        a.set_yticklabels([])
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--json")
    ap.add_argument("--plot")
    a = ap.parse_args(argv)
    with ProcessPoolExecutor(max_workers=min(16, len(a.runs))) as ex:
        results = list(ex.map(analyze, a.runs))
    for r in results:
        print(f"\n{r['run']}")
        for k in METRICS:
            print(f"  {k}: {json.dumps(r[k], default=str)[:300]}")
    for name, pr in pairs(results).items():
        rf, sf = dict(_flat(pr["real"])), dict(_flat(pr["sim"]))
        print(f"\nreal vs sim {name}")
        for k in sorted(set(rf) & set(sf)):
            if rf[k] or sf[k]:
                print(f"  {k:40s} real {rf[k]:10.4g}  sim {sf[k]:10.4g}")
    if a.json:
        json.dump(results, open(a.json, "w"), indent=1, default=str)
    if a.plot:
        plot(results, a.plot)
    return 0


if __name__ == "__main__":
    sys.exit(main())
