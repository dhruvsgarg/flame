#!/usr/bin/env python3
"""Where Felix's gains come from (FX-N131): per cell, Felix old / new / FedBuff (unserved models) on one seed.

  felix_why.py POOL [--seed 1] [--out _metadata]

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
from bench_report import leg as eval_curve  # noqa: E402

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
    raw = eval_curve(run, b'"agg_eval_raw"')[0] or eval_curve(run)[0]
    return [(t, a) for t, a, _ in raw], tasks(run)


def spearman(x, y):
    rk = lambda v: {i: r for r, i in enumerate(sorted(range(len(v)), key=lambda i: v[i]))}
    rx, ry, n = rk(x), rk(y), len(x)
    return 1 - 6 * sum((rx[i] - ry[i]) ** 2 for i in range(n)) / (n * (n * n - 1)) if n > 2 else float("nan")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pool")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_metadata"))
    a = ap.parse_args(argv)
    runs = {}
    for (cell, trace) in CELLS:
        for arm, b, lab, col in ARMS:
            g = glob.glob(os.path.join(a.pool, f"F42_{cell}_{arm}", "runs", f"{trace}_{b}_s{a.seed}", "run_*"))
            if g:
                runs[(cell, lab)] = g[0]
    with ProcessPoolExecutor(8) as ex:
        data = dict(zip(runs, ex.map(one, runs.values())))
    fig, ax = plt.subplots(len(CELLS), 6, figsize=(30, 5.2 * len(CELLS)), squeeze=False)
    for i, (cell, trace) in enumerate(CELLS):
        for _, _, lab, col in ARMS:
            if (cell, lab) not in data:
                continue
            ev, tk = data[(cell, lab)]
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
    fig.suptitle("Where Felix's gain comes from · cifar10 12 h bench, seed %d, unserved models (FX-N131)" % a.seed, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(a.out, "felix_why_cifar10.png")
    fig.savefig(out, dpi=90)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
