"""G1: FluxTune agg weights, `uniform` (P0 exp) vs `grad_aware` (DistilBERT, AG News, N=90). Acc vs commits and wall; ω spread."""
import bisect
import json
import os

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXPTS = os.path.join(HERE, "..", "experiments")
TEL = "_fluxtune_agnews_n100_smoke_syn_0_sim/telemetry/aggregator_fluxtune_agnews_n100_smoke_syn_0_sim.jsonl"
RUNS = {  # label -> (run prefix, color)
    "uniform": ("run_20261018_044202", "#2a78d6"),
    "grad_aware": ("run_20261018_170949", "#eb6834"),
}
SMOOTH = 10  # evals per rolling mean
KEEP = ('"agg_eval"', '"server_update"', '"agg_round"', '"run_meta"')


def load(prefix):
    t0, commits, evals, spread = None, [], [], []
    for line in open(os.path.join(EXPTS, prefix + TEL)):
        if not any(k in line for k in KEEP):
            continue
        d = json.loads(line)
        t0 = t0 or d["ts"]
        if d["event"] == "server_update":
            commits.append((d["ts"], d["commit_count"]))
        elif d["event"] == "agg_eval":
            evals.append((d["ts"], d["test-accuracy"]))
        elif d["event"] == "agg_round" and d.get("agg_weight") and len(d["agg_weight"]) > 1:
            w = np.asarray(d["agg_weight"], float)
            spread.append((d["ts"], w.max() / w.min()))
    cts = [t for t, _ in commits]

    def c_of(ts):  # latest commit at or before ts
        i = bisect.bisect_right(cts, ts)
        return commits[i - 1][1] if i else 0
    ev = np.array([(c_of(t), a, (t - t0) / 60) for t, a in evals])
    sp = np.array([(c_of(t), r) for t, r in spread])
    return ev, sp


data = {name: load(prefix) for name, (prefix, _) in RUNS.items()}

fig, (ax, axt, axw) = plt.subplots(3, 1, figsize=(8, 10), gridspec_kw={"height_ratios": [2, 2, 1.2]})
for name, (_, color) in RUNS.items():
    ev, sp = data[name]
    sm = np.convolve(ev[:, 1], np.ones(SMOOTH) / SMOOTH, mode="valid")
    for a_, x in ((ax, ev[:, 0]), (axt, ev[:, 2])):
        a_.plot(x, ev[:, 1], color=color, lw=0.8, alpha=0.25)
        a_.plot(x[SMOOTH - 1:], sm, color=color, lw=2, label=f"{name} (peak {ev[:, 1].max():.3f})")
        a_.annotate(name, (x[-1], sm[-1]), xytext=(6, 0), textcoords="offset points", color="#0b0b0b", va="center")
    # ω spread: median of max/min weight per commit
    cs = np.unique(sp[:, 0])
    med = np.array([np.median(sp[sp[:, 0] == c, 1]) for c in cs])
    k = min(25, len(med))  # rolling mean over commits
    axw.plot(cs[k - 1:], np.convolve(med, np.ones(k) / k, mode="valid"), color=color, lw=2, label=name)

for a_, xmax in ((ax, 1180), (axt, 135)):
    for y in (0.86, 0.87):
        a_.axhline(y, color="#52514e", lw=0.8, ls="--", alpha=0.6)
        a_.text(0.01 * xmax, y + 0.001, f"{y:.2f}", color="#52514e", fontsize=8, va="bottom")
    a_.set_ylim(0.78, 0.885)  # zoomed: both arms pass 0.78 by ~c150
    a_.set_xlim(0, xmax)
    a_.set_ylabel("Test accuracy")
    a_.legend(loc="lower right", frameon=False)
ax.set_xlabel("Server commits")
axt.set_xlabel("Wall time (min)")
ax.set_title(f"G1: FluxTune agg weights, uniform vs grad_aware (DistilBERT · AG News · N=90, 1 seed)\n"
             f"zoomed to acc ≥ 0.78; faint = per eval; bold = {SMOOTH}-eval rolling mean", fontsize=11)

axw.set_ylabel("ω max / min\n(25-commit mean)")
axw.set_xlabel("Server commits")
axw.set_xlim(0, 1180)
axw.legend(loc="lower right", frameon=False)
for a_ in (ax, axt, axw):
    a_.grid(axis="y", color="#e4e3df", lw=0.6)
    a_.spines[["top", "right"]].set_visible(False)

out = os.path.join(HERE, "..", "g1_uniform_vs_grad_aware.png")
fig.tight_layout()
fig.savefig(out, dpi=150)
print(os.path.abspath(out))
