#!/usr/bin/env python3
"""FX-N141 loop FL1: each arm vs FedBuff (averaged models, same batch) per cell; PASS = final >= FedBuff's - band and no slower at
every target up to the highest either reaches. Diversity beside it: top-20% share of train tasks and client coverage.

  fl1_verdict.py POOL... [--ceiling C.json ...] [--band-pp 1.0] [--json-out V.json]

Targets: static cells 50-70% every 5 pp; stream cells 40% up in 5 pp steps below 90% of the centralized ceiling at the cell's
horizon, plus 90% and 95% of that ceiling (operator 10-23, Q-TARGET a).
"""
import argparse
import glob
import json
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_report import cells, ceiling_at, label, stats  # noqa: E402

ANCHOR = ("fedbuff", "ctl_srv")


def diversity(run_dir):
    """(top-20% share, coverage) of train tasks per client."""
    counts = []
    for f in glob.glob(os.path.join(run_dir, "telemetry", "trainer_*.jsonl")):
        with open(f, "rb") as fh:
            counts.append(sum(1 for line in fh if b'"trainer_round"' in line and b'"task_to_perform": "eval"' not in line))
    if not counts or not sum(counts):
        return None, None
    counts.sort(reverse=True)
    return sum(counts[:max(1, len(counts) // 5)]) / sum(counts), sum(c > 0 for c in counts) / len(counts)


def targets(ds, trace, horizon_h, ceil):
    if trace.endswith(" static"):
        return [0.5, 0.55, 0.6, 0.65, 0.7]
    c = ceiling_at(ceil, ds, trace, horizon_h)
    if c is None:
        return [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
    return [t for t in (0.4, 0.45, 0.5, 0.55, 0.6, 0.65) if t < 0.9 * c - 1e-9] + [round(0.9 * c, 3), round(0.95 * c, 3)]


def judge(arm, fb, ths, band):
    """PASS / FAIL with the reason; arm and fb are stats() dicts (first seed: FL1 searches on one seed, R25)."""
    hit = [t for t in ths if arm[f"tta{t:g}"] is not None or fb[f"tta{t:g}"] is not None]
    if arm["final"] < fb["final"] - band:
        return "FAIL", f"final {100 * arm['final']:.1f} < FedBuff {100 * fb['final']:.1f} − {100 * band:.1f}"
    for t in hit:
        a, b = arm[f"tta{t:g}"], fb[f"tta{t:g}"]
        if a is None or (b is not None and a > b):
            return "FAIL", f"{100 * t:.1f}%: {a if a is None else round(a)} vs FedBuff {b if b is None else round(b)} min"
    return "PASS", f"highest {100 * max(hit):.1f}%" if hit else "no target reached"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pools", nargs="+")
    ap.add_argument("--ceiling", nargs="*", default=[])
    ap.add_argument("--band-pp", type=float, default=1.0)
    ap.add_argument("--json-out")
    a = ap.parse_args(argv)
    ceil = [r for c in a.ceiling for r in json.load(open(c))]
    cs = cells(a.pools)
    groups = {}
    for (ds, trace, arm, b), dirs in cs.items():
        groups.setdefault((ds, trace), {})[(b, arm)] = dirs[0]
    rows = []
    with Pool(min(32, len(cs) or 1)) as pool:
        div = dict(zip([d for g in groups.values() for d in g.values()], pool.map(diversity, [d for g in groups.values() for d in g.values()])))
    for (ds, trace), legs in sorted(groups.items()):
        if ANCHOR not in legs:
            print(f"\n## {ds} {trace}: no FedBuff anchor, skipped")
            continue
        horizon = stats(legs[ANCHOR], [], 1)["end_h"]
        ths = targets(ds, trace, horizon, ceil)
        smooth = 3 if ds == "google_speech" else 1
        st = {k: stats(d, ths, smooth) for k, d in legs.items()}
        fb = st[ANCHOR]
        print(f"\n## {ds} {trace} (horizon {horizon:.0f} h; targets {', '.join(f'{100 * t:.1f}' for t in ths)}%)\n"
              "| arm | final % | " + " | ".join(f"min→{100 * t:.1f}" for t in ths) + " | GB | top-20% | coverage | verdict |\n|"
              + "---|" * (len(ths) + 6))
        for (b, arm), s in sorted(st.items(), key=lambda kv: -(kv[1] or {}).get("final", 0)):
            if s is None:
                continue
            v = ("anchor", "") if (b, arm) == ANCHOR else judge(s, fb, ths, a.band_pp / 100)
            top, cov = div[legs[(b, arm)]]
            print(f"| {b} {label(arm)} | {100 * s['final']:.1f} | " + " | ".join("–" if s[f'tta{t:g}'] is None else f"{s[f'tta{t:g}']:.0f}"
                                                                          for t in ths)
                  + f" | {s['gb']:.0f} | {'–' if top is None else f'{100 * top:.0f}%'} | {'–' if cov is None else f'{100 * cov:.0f}%'}"
                  f" | {v[0]} {v[1]} |")
            rows.append(dict(dataset=ds, cell=trace, baseline=b, arm=arm, verdict=v[0], why=v[1], top20=top, coverage=cov,
                             **{k: s[k] for k in s}))
    if a.json_out:
        json.dump(rows, open(a.json_out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
