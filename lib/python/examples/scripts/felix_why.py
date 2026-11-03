#!/usr/bin/env python3
"""Where Felix's gains come from (FX-N131): per cell, Felix old / new / FedBuff on one seed (current models; averaged dashed).

  felix_why.py POOL... [--seed 1] [--out _metadata]     # bench or harness pools; one PNG per dataset

Prints, over every seed, each arm's hour-1 updates / h, samples per update, top-20% pick share and ρ(picks, shard).

Panels: accuracy vs hours · vs train samples (per-sample efficiency: a smarter pick shows here) · vs train updates · updates / h ·
samples per update · pick concentration (share of updates by the top x% of clients; ρ = Spearman of picks vs the client's shard).
"""
import argparse
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_report import cells as pool_cells, leg as eval_curve  # noqa: E402

ARMS = (("ctl_srv", "felix", "Felix old", "#eb6834"), ("l3b_srv", "felix", "Felix new", "#2a78d6"),
        ("ctl_srv", "fedbuff", "FedBuff", "#1baf7a"))
CELLS = (("s0u", "syn_0"), ("s50u", "syn_50_3st"))


def tasks(run):
    """[(send ts, samples, end)] of train tasks, by send time."""
    out = []
    for f in glob.glob(os.path.join(run, "telemetry", "trainer_*.jsonl")):
        for line in open(f, "rb"):
            if b'"trainer_round"' in line:
                e = json.loads(line)
                if e.get("task_to_perform", "train") == "train" and e.get("sim_send_ts") is not None:
                    out.append((float(e["sim_send_ts"]), float(e.get("model_samples") or 0), e["end_id"], int(e.get("total_samples") or 0)))
    return sorted(out)


def one(run):
    raw = eval_curve(run, b'"agg_eval_raw"')[0]
    avg = eval_curve(run)[0] if raw else []
    return [(t, a) for t, a, _ in raw or eval_curve(run)[0]], tasks(run), [(t, a) for t, a, _ in avg]


def seed_of(d):
    """Bench `<trace>_<b>_s<seed>/run_*`; harness: `_r<seed>` in the phase, else seed 1."""
    import re
    m = re.search(r"_s(\d+)/run_", d) or re.search(r"_r(\d+)_dbg_", d)
    return int(m.group(1)) if m else 1


def summary(tk):
    """(hour-1 updates, samples per update, top-20% share, ρ(picks, shard)) of one leg's train tasks."""
    picks, size = {}, {}
    for _, _, e, tot in tk:
        picks[e] = picks.get(e, 0) + 1
        size[e] = max(size.get(e, 0), tot)
    v = sorted(picks.values(), reverse=True)
    h1 = sum(1 for x in tk if x[0] - tk[0][0] < 3600) if tk else 0
    return (h1, sum(x[1] for x in tk) / max(len(tk), 1), sum(v[:max(1, len(v) // 5)]) / max(sum(v), 1),
            spearman([picks[e] for e in picks], [size[e] for e in picks]))


def spearman(x, y):
    rk = lambda v: {i: r for r, i in enumerate(sorted(range(len(v)), key=lambda i: v[i]))}
    rx, ry, n = rk(x), rk(y), len(x)
    return 1 - 6 * sum((rx[i] - ry[i]) ** 2 for i in range(n)) / (n * (n * n - 1)) if n > 2 else float("nan")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pools", nargs="+")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_metadata"))
    a = ap.parse_args(argv)
    lab_of = {(arm, b): lab for arm, b, lab, _ in ARMS}
    legs = [(ds, trace, lab_of[(arm, b)], d) for (ds, trace, arm, b), dirs in pool_cells(a.pools).items()
            if (arm, b) in lab_of for d in dirs]
    with ProcessPoolExecutor(16) as ex:
        res = dict(zip([x[3] for x in legs], ex.map(one, [x[3] for x in legs])))
    for ds in sorted({x[0] for x in legs}):
        print(f"\n{ds}: arm · cell · seeds · hour-1 updates · samples / update · top-20% share · ρ(picks, shard)")
        for trace in sorted({x[1] for x in legs if x[0] == ds}):
            for _, _, lab, _ in ARMS:
                sm = [summary(res[d][1]) for d_, t_, l_, d in legs if (d_, t_, l_) == (ds, trace, lab)]
                if sm:
                    m = [sum(x[k] for x in sm) / len(sm) for k in range(4)]
                    print(f"  {lab} · {trace} · {len(sm)} · {m[0]:.0f} · {m[1]:.0f} · {100 * m[2]:.0f}% · {m[3]:+.2f}")
        main_ds(a, ds, {(trace, lab): res[d] for d_, trace, lab, d in legs if d_ == ds and seed_of(d) == a.seed})
    return 0


def main_ds(a, ds, data):
    cells = [(c, t) for c, t in CELLS if any(k[0] == t for k in data)]
    fig, ax = plt.subplots(len(cells), 6, figsize=(30, 5.2 * len(cells)), squeeze=False)
    for i, (cell, trace) in enumerate(cells):
        for _, _, lab, col in ARMS:
            if (trace, lab) not in data:
                continue
            ev, tk, avg = data[(trace, lab)]
            ts = [t for t, *_ in tk]
            cs, cu = [], []
            s = 0.0
            for _, n, *_ in tk:
                s += n
                cs.append(s)
            import bisect
            at = lambda t: bisect.bisect_right(ts, t)
            acc = [100 * x for _, x in ev]
            ax[i, 0].plot([t / 3600 for t, _ in ev], acc, color=col, label=lab)
            if avg:
                ax[i, 0].plot([t / 3600 for t, _ in avg], [100 * x for _, x in avg], "--", color=col, lw=1, label=f"{lab}, averaged")
            ax[i, 1].plot([cs[at(t) - 1] / 1e6 if at(t) else 0 for t, _ in ev], acc, color=col)
            ax[i, 2].plot([at(t) / 1e3 for t, _ in ev], acc, color=col)
            hours = range(int(ts[-1] // 3600) + 1) if ts else []
            per_h = [[x for x in tk if h * 3600 <= x[0] < (h + 1) * 3600] for h in hours]
            ax[i, 3].plot(list(hours), [len(p) for p in per_h], color=col)
            ax[i, 4].plot(list(hours), [sum(x[1] for x in p) / len(p) if p else 0 for p in per_h], color=col)
            picks, size = {}, {}
            for _, _, e, tot in tk:
                picks[e] = picks.get(e, 0) + 1
                size[e] = max(size.get(e, 0), tot)
            v = sorted(picks.values(), reverse=True)
            tot_p = sum(v)
            ax[i, 5].plot([100 * (k + 1) / len(v) for k in range(len(v))], [100 * sum(v[:k + 1]) / tot_p for k in range(len(v))],
                          color=col, label=f"{lab}: top-20% {100 * sum(v[:max(1, len(v) // 5)]) / tot_p:.0f}%, "
                                           f"ρ(picks, shard) {spearman([picks[e] for e in picks], [size[e] for e in picks]):+.2f}")
        titles = ("accuracy vs hours", "accuracy vs samples trained (M)", "accuracy vs train updates (k)", "train updates / h",
                  "samples per update", "share of updates by top x% clients")
        for j, t in enumerate(titles):
            ax[i, j].set_title(f"{trace} · {t}", fontsize=11)
            ax[i, j].grid(alpha=0.25, lw=0.6)
            ax[i, j].spines[["top", "right"]].set_visible(False)
        ax[i, 5].legend(fontsize=8, frameon=False, loc="lower right")
    ax[0, 0].legend(fontsize=9, frameon=False)
    fig.suptitle(f"Where Felix's gain comes from · {ds} 12 h, seed {a.seed}, current models (averaged dashed) (FX-N131)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(a.out, f"felix_why_{ds}.png")
    fig.savefig(out, dpi=90)
    print(out)


if __name__ == "__main__":
    sys.exit(main())
