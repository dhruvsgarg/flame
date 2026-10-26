"""Run-telemetry extractors for the paper figures: one streaming pass per run, cached as JSON under data/runs/ (`--refresh` redoes).

Per run: eval curve (t, acc, GB), in-flight tasks over time, staleness split by source (held upload vs in-flight work, insights.py
I-C4), picks per client vs shard size, Felix-Pace's `dynamic_kc` trajectory. Run dirs are resolved from a pool's summary.tsv.
"""
import csv
import glob
import json
import os
import re
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
EXPTS = os.path.normpath(os.path.join(HERE, "..", "experiments"))
CACHE = os.path.join(HERE, "data", "runs")
F49 = os.path.join(EXPTS, "F49_1024_1530", "cifar")
HELD_LAG_S = 60.0  # insights.i_c4_staleness_source: a held upload waited > 60 s between ready and commit


def run_dir(pool, phase, baseline):
    """The pool's run for (phase dir, baseline), from its summary.tsv; None if absent."""
    f = os.path.join(pool, phase, "summary.tsv")
    if not os.path.exists(f):
        return None
    for r in csv.DictReader(open(f), delimiter="\t"):
        if r["baseline"] == baseline and r.get("sim_dir") not in (None, "", "-"):
            d = r["sim_dir"]
            d = d if os.path.isabs(d) else os.path.join(os.path.dirname(EXPTS), d)
            return d if os.path.isdir(d) else None
    return None


_RE = {k: re.compile(v) for k, v in {
    "bytes": rb'"bytes": (\d+)', "vclock": rb'"vclock_now": ([-0-9.e]+)', "acc": rb'"test-accuracy": ([-0-9.e]+)',
    "inflight": rb'"in_flight_train": (\d+)'}.items()}


def _scan(d):
    f = glob.glob(os.path.join(d, "telemetry", "aggregator_*.jsonl"))[0]
    t, nbytes, last_inf = 0.0, 0, -1e9
    out = {"curve": [], "inflight": [], "stale": {"held": [0, 0], "work": [0, 0]}, "picks": {}, "kc": []}
    with open(f, "rb") as fh:
        for line in fh:
            if b'"wire"' in line:
                nbytes += int(_RE["bytes"].search(line).group(1))
            elif b'"agg_round"' in line:
                m = _RE["vclock"].search(line)
                t = float(m.group(1)) if m else t
                if b'"task_to_perform": "eval"' in line:
                    continue
                e = json.loads(line)
                for s, lag in zip(e.get("staleness") or [], e.get("update_visibility_lag_s") or []):
                    k = "held" if float(lag or 0) > HELD_LAG_S else "work"
                    out["stale"][k][0] += 1
                    out["stale"][k][1] += int(s)
                for c in e.get("contributing_trainers") or []:
                    out["picks"][c] = out["picks"].get(c, 0) + 1
            elif b'"selection"' in line and b'"task": "train"' in line and t - last_inf >= 60 and (m := _RE["inflight"].search(line)):
                out["inflight"].append((t, int(m.group(1))))  # slots taken (incl. held by departed clients), one per vclock minute
                last_inf = t
            elif b'"agg_eval"' in line and (m := _RE["acc"].search(line)):
                out["curve"].append((t, float(m.group(1)), nbytes / 1e9))
            elif b'"dynamic_kc"' in line:
                e = json.loads(line)
                out["kc"].append((e.get("now"), e.get("c"), e.get("n_eligible_train"), e.get("loss_phase"), e.get("avg_staleness")))
    sizes, busy = {}, {}  # busy: vclock minute -> clients computing a train task (completion − duration, completion)
    for tf in glob.glob(os.path.join(d, "telemetry", "trainer_*.jsonl")):
        with open(tf, "rb") as fh:
            for line in fh:
                if b'"trainer_round"' in line and b'"task_to_perform": "eval"' not in line:
                    e = json.loads(line)
                    sizes.setdefault(e["end_id"], e.get("total_samples") or e.get("dataset_size"))
                    c, dur = e.get("sim_completion_ts"), e.get("sim_round_duration_s")
                    if c is not None and dur:
                        for m in range(int((c - dur) // 60), int(c // 60) + 1):
                            busy[m] = busy.get(m, 0) + min(c, 60 * (m + 1)) - max(c - dur, 60 * m)
    out["sizes"] = sizes
    out["busy"] = sorted((m, v / 60) for m, v in busy.items())
    out["kc"] = out["kc"][::max(1, len(out["kc"]) // 2000)]
    return out


def _cache_path(d):
    return os.path.join(CACHE, re.sub(r"[^\w.-]", "_", os.path.relpath(d, EXPTS)) + ".json")


def _load_one(args):
    d, refresh = args
    p = _cache_path(d)
    if not refresh and os.path.exists(p):
        return json.load(open(p))
    r = _scan(d)
    os.makedirs(CACHE, exist_ok=True)
    json.dump(r, open(p, "w"))
    return r


def load(dirs, refresh=False, procs=24):
    """{run dir: extracted dict} for every existing dir (parallel; cached)."""
    dirs = [d for d in dict.fromkeys(dirs) if d]
    with Pool(min(procs, max(1, len(dirs)))) as pool:
        return dict(zip(dirs, pool.map(_load_one, [(d, refresh) for d in dirs])))


# F49 cifar stacks: (label, phase arm, baseline). Felix = `fn_chr` (F49 decider); `*` = oracle availability.
F49_STACKS = (("Felix", "fn_chr", "felix"), ("FedBuff", "ctl_srv", "fedbuff"), ("REFL", "sync_srv", "refl"),
              ("FedDance", "sync_srv", "feddance"), ("Oort", "sync_srv", "oort"), ("FedAvg", "sync_srv", "fedavg"),
              ("REFL*", "rstar_srv", "refl"), ("FedDance*", "fdstar_srv", "feddance"), ("Oort*", "sync_srv", "oort_star"))
CELLS = {"s0u": "syn_0 · stream", "s50u": "syn_50 · stream", "smu": "MobiPerf · stream"}


def f49(cell, labels=None):
    """{label: run dir} for an F49 cell."""
    out = {}
    for lab, arm, b in F49_STACKS:
        if labels and lab not in labels:
            continue
        d = run_dir(F49, f"F49_{cell}_{arm}", b)
        if d:
            out[lab] = d
    return out
