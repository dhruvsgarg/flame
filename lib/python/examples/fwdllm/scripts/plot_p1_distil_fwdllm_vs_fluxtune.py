import csv, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "expt_scripts"))
from plotlib.style import ema
SMOOTH = 0.6  # make_paper_figs.py default; visual only, markers and numbers use raw
# P1 runs 1 + 2 (RUN_PLAN Done table): DistilBERT · AG News, FwdLLM vs FluxTune, acc vs wall and vclock.
d, target, bg = "p1_distil_20261025", 0.86, "#fcfcfb"
arms = [("FwdLLM", "fwdllm", "#D55E00", "o"), ("FluxTune", "fluxtune", "#08306b", "^")]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=120, sharey=True); fig.patch.set_facecolor(bg)
for name, key, c, m in arms:
    rows = list(csv.DictReader(open(f"{d}/acc_{key}.tsv"), delimiter="\t"))
    wall, sim, acc = ([float(r[k]) for r in rows] for k in ("wall_min", "vclock_h", "acc"))
    wall = [w / 60 for w in wall]
    best = max(range(len(acc)), key=acc.__getitem__)
    hit = next((i for i, a in enumerate(acc) if a >= target), None)
    print(f"{name}: evals {len(acc)} best {acc[best]:.4f} @ c{rows[best]['commit']} wall {wall[best]:.2f}h sim {sim[best]:.2f}h; "
          f"target {'c' + rows[hit]['commit'] + f' wall {wall[hit]:.2f}h sim {sim[hit]:.2f}h' if hit is not None else 'DNF'}; "
          f"end c{rows[-1]['commit']} wall {wall[-1]:.2f}h sim {sim[-1]:.2f}h")
    for ax, x in zip(axes, (wall, sim)):
        ax.plot(x, acc, color=c, lw=0.7, alpha=0.22)
        ax.plot(x, ema(acc, SMOOTH), color=c, lw=1.8, label=name)
        ax.plot(x[best], acc[best], m, color=c, ms=7, mec=bg, mew=1.5)
        ax.annotate(f"best {acc[best]:.3f}", (x[best], acc[best]), xytext=(0, 9), textcoords="offset points", ha="center", color=c, fontsize=9)
        if hit is not None:
            ax.axvline(x[hit], color=c, lw=1, ls=":")
            ax.annotate(f"{target:.2f} @ {x[hit]:.2f} h", (x[hit], 0.32), xytext=(4, 0), textcoords="offset points", color=c, fontsize=9)
for ax, lab in zip(axes, ("Wall clock (h)", "Simulated time (vclock h)")):
    ax.set_facecolor(bg)
    ax.axhline(target, color="#999", lw=1, ls="--"); ax.text(0, target + 0.006, f"target {target:.2f}", color="#666", fontsize=9)
    ax.set(xlabel=lab, ylim=(0.2, 0.92)); ax.set_xlim(left=0)
    ax.grid(axis="y", color="#e2e2e0"); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color("#bbb")
    ax.tick_params(colors="#555")
axes[0].set_ylabel("Test accuracy (AG News, full 7,600)")
axes[0].legend(frameon=False, loc="lower right")
fig.suptitle("DistilBERT · AG News · N=50 · FwdLLM vs FluxTune (P1 runs 1 + 2, 10-25, both stopped by hand)", x=0.01, ha="left", fontsize=12)
fig.tight_layout(); fig.savefig(sys.argv[1], facecolor=bg)
