#!/usr/bin/env python3
"""FX-N109: what Felix's selectors sense and pick, per sim leg (microbenchmark readout; telemetry only, no runs).

  selection_audit.py POOL... [--legs PHASE:BASELINE ...] [--json]

Per leg:
- picks: top-20% share and Gini of train tasks per client (never-picked count 0), coverage, mean local samples per task; Spearman of a client's train picks vs shard size, speed (−median duration) and per-sample
  loss (utility / size); mean samples per update vs the population mean.
- classes (cifar): total variation between the label mass of aggregated updates (one shard histogram per update,
  normalized) and the population's.
- belief: at each train return, |believed − actual| / actual of the utility, split by what set the belief (the last eval or
  the last train); believed / actual median (> 1 = the source over-states); Spearman believed ~ actual.
- eval: eval tasks per train task.
- dynamic K (I5 legs): K at 10 / 50 / 90% of the leg and the noise scale B there.
"""
import argparse
import collections
import glob
import json
import os
import pickle
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from accuracy_table import dataset_of, rows_from_pools  # noqa: E402

META = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_metadata", "dataset_splits")
CIFAR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "async_cifar10", "data", "cifar-10-batches-py")


def spearman(a, b):
    def rank(x):
        r = [0] * len(x)
        for k, i in enumerate(sorted(range(len(x)), key=lambda i: x[i])):
            r[i] = k
        return r
    if len(a) < 3:
        return None
    ra, rb = rank(a), rank(b)
    ma, mb = st.mean(ra), st.mean(rb)
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** .5
    return sum((x - ma) * (y - mb) for x, y in zip(ra, rb)) / den if den else None


def _jsonl(path, *keys):
    with open(path, "rb") as fh:
        for line in fh:
            if any(k in line for k in keys):
                yield json.loads(line)


def trainer_rows(run_dir):
    """end_id -> train-task trainer_round rows."""
    out = {}
    for f in glob.glob(os.path.join(run_dir, "telemetry", "trainer_*.jsonl")):
        rs = [r for r in _jsonl(f, b'"trainer_round"') if r.get("task_to_perform", "train") == "train"]
        if rs:
            out[os.path.basename(f)[len("trainer_"):-len(".jsonl")]] = rs
    return out


def _cifar_labels():
    import numpy as np
    lab = []
    for i in range(1, 6):
        with open(os.path.join(CIFAR, f"data_batch_{i}"), "rb") as fh:
            lab += pickle.load(fh, encoding="bytes")[b"labels"]
    return np.array(lab)


def class_tv(run_dir, picks, sizes):
    """TV(label mass of aggregated updates, population); trainers map to the reference split by matching shard size."""
    import numpy as np
    import yaml
    n = len(picks)
    split = yaml.safe_load(open(os.path.join(META, "cifar10_alpha0.1_n300.yaml")))["trainer_data_splits"]
    lab = _cifar_labels()
    by_suffix = {}
    for eid in picks:  # end ids end in a 3-digit index whose lowest value is trainer_001
        by_suffix[eid] = int(eid[-3:])
    lo = min(by_suffix.values())
    hist, pop = {}, np.zeros(10)
    for eid, k in by_suffix.items():
        idx = split.get(f"trainer_{k - lo + 1:03d}")
        if idx is None or len(idx) != sizes[eid]:
            return None  # mapping doesn't hold: don't guess
        h = np.bincount(lab[idx], minlength=10)
        hist[eid], pop = h / h.sum(), pop + h
    m = sum(picks[e] * hist[e] for e in picks)
    return 0.5 * float(abs(m / m.sum() - pop / pop.sum()).sum()) if n else None


def belief(run_dir):
    f = glob.glob(os.path.join(run_dir, "telemetry", "aggregator_*.jsonl"))
    if not f:
        return {}
    src, err, ratio, pairs, n_eval, n_train, dkc = {}, collections.defaultdict(list), collections.defaultdict(list), [], 0, 0, []
    for e in _jsonl(f[0], b'"utility_belief"', b'"agg_round"', b'"dynamic_kc"'):
        ev = e.get("event")
        if ev == "utility_belief":
            b, a = e.get("believed"), e.get("actual")
            if b is not None and a:
                s = src.get(e["end_id"], "train")
                err[s].append(abs(b - a) / a)
                ratio[s].append(b / a)
                pairs.append((b, a))
            src[e["end_id"]] = "train"
        elif ev == "agg_round":
            if e.get("task_to_perform") == "eval":
                n_eval += 1
                for t in e.get("contributing_trainers") or []:
                    src[t] = "eval"
            else:
                n_train += 1
        elif ev == "dynamic_kc":
            dkc.append((e.get("k"), e.get("noise_scale")))
    out = {"eval_per_train": n_eval / n_train if n_train else None,
           "belief_rank": spearman([p[0] for p in pairs], [p[1] for p in pairs])}
    for s in ("eval", "train"):
        out[f"err_{s}"] = st.median(err[s]) if err[s] else None
        out[f"bias_{s}"] = st.median(ratio[s]) if ratio[s] else None
        out[f"n_{s}"] = len(err[s])
    if dkc:
        out["k_traj"] = [dkc[int(q * (len(dkc) - 1))] for q in (0.1, 0.5, 0.9)]
    return out


def audit(run_dir):
    T = trainer_rows(run_dir)
    if not T:
        return None
    picks = {e: len(r) for e, r in T.items()}
    size = {e: r[0]["dataset_size"] for e, r in T.items()}
    dur = {e: st.median(x["sim_round_duration_s"] for x in r) for e, r in T.items()}
    rms = {e: st.mean(x["stat_utility"] / max(x["dataset_size"], 1) for x in r) for e, r in T.items()}
    ids = list(T)
    tot = sum(picks.values())
    top = sorted(picks.values(), reverse=True)[:max(1, len(ids) // 5)]
    p = [picks[e] for e in ids]
    n_pop = len(glob.glob(os.path.join(run_dir, "telemetry", "trainer_*.jsonl")))
    xs = sorted(p + [0] * (n_pop - len(ids)))  # never-picked clients count as 0
    gini = sum((2 * i - len(xs) + 1) * x for i, x in enumerate(xs)) / (len(xs) * tot)
    trained = [x.get("model_samples") for r in T.values() for x in r]
    out = {"trainers": len(ids), "top20": sum(top) / tot, "gini": gini, "coverage": len(ids) / max(n_pop, 1),
           "local_samples": st.mean(trained) if all(v is not None for v in trained) else None,
           "picks_size": spearman(p, [size[e] for e in ids]), "picks_speed": spearman(p, [-dur[e] for e in ids]),
           "picks_loss": spearman(p, [rms[e] for e in ids]), "dur_size": spearman([dur[e] for e in ids], [size[e] for e in ids]),
           "samples_per_update": sum(picks[e] * size[e] for e in ids) / tot, "pop_samples": st.mean(size.values())}
    if dataset_of(run_dir) == "cifar10":
        try:
            out["class_tv"] = class_tv(run_dir, picks, size)
        except (OSError, KeyError):
            out["class_tv"] = None
    out.update(belief(run_dir))
    return out


def fmt(v, nd=2):
    return "–" if v is None else (f"{v:+.{nd}f}" if isinstance(v, float) and -1 <= v <= 1 and nd == 2 else f"{v:.{nd}f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pools", nargs="+")
    ap.add_argument("--legs", nargs="*", default=[], help="PHASE:BASELINE filters (default: every sim leg)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    want = {tuple(x.split(":", 1)) for x in a.legs}
    rows = []
    for phase, b, _tr, _real, sim in rows_from_pools(a.pools):
        if sim and os.path.isdir(sim) and (not want or (phase, b) in want) and (r := audit(sim)):
            rows.append(dict(phase=phase, baseline=b, **r))
    if a.json:
        print(json.dumps(rows, indent=1))
        return 0
    print("| leg | top-20% | Gini · coverage | local samples | picks~size | ~speed | ~loss/sample | dur~size | samples/update (pop) | class TV "
          "| belief err eval / train | believed÷actual eval / train | belief rank | eval/train | K (B) at 10/50/90% |")
    print("|" + "---|" * 15)
    for r in rows:
        k = " · ".join(f"{kk} ({fmt(b, 1)})" for kk, b in r.get("k_traj") or []) or "–"
        print(f"| {r['phase']} {r['baseline']} | {r['top20']:.2f} | {r['gini']:.2f} · {r['coverage']:.2f} "
              f"| {fmt(r.get('local_samples'), 0)} | {fmt(r['picks_size'])} | {fmt(r['picks_speed'])} "
              f"| {fmt(r['picks_loss'])} | {fmt(r['dur_size'])} | {r['samples_per_update']:.0f} ({r['pop_samples']:.0f}) "
              f"| {fmt(r.get('class_tv'), 3)} | {fmt(r.get('err_eval'))} / {fmt(r.get('err_train'))} "
              f"| {fmt(r.get('bias_eval'))} / {fmt(r.get('bias_train'))} | {fmt(r.get('belief_rank'))} "
              f"| {fmt(r.get('eval_per_train'))} | {k} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
