"""Fig 1 teaser (fig:intro-teaser): SmolLM2-1.7B · Yahoo, SwiFT run 6c vs FwdLLM. One column, two panels.

Usage: python make_intro_fig.py   -> ../figs/intro/fig1_teaser.{pdf,png}
(a) accuracy vs simulated hours: peak gain, time and GB to FwdLLM's peak. (b) time a phone stays online per contribution:
FwdLLM's cohort hold per server update vs SwiFT's one task (dispatch -> upload), over MobiPerf session lengths.

Data: data/sm6c_fluxtune_e2e.json, data/sm_fwdllm_e2e.json (paper_expts_fluxtune/plots/make_ft_figs.py digests of jayne
paper_sm_20261026b/POOL/sm_paper and experiments/run_20261028_182751_fwdllm_yahoo_n60_smoke_syn_0_sim); holds from
data/sm_fwdllm.json (digest_motiv.py). Numbers use the 5-eval trailing mean for both runs; lines are display-smoothed.
"""
import json
import os

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import ft_style as S
from ft_style import FONT, INK, LINE, MARKER, Z, plt
from make_motiv_figs import cdf, holds_min, load, sessions_min

plt.rcParams.update(S.RC)  # column figure: guideline sizes (8 / 7 pt), not the Fig 2 panel sizes

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
OUT = os.path.join(HERE, "..", "figs", "intro")
OURS, BASE = S.STACK[S.SYS_NAME][0], S.STACK["FwdLLM"][0]
SMOOTH = 9


def trace(name):
    with open(os.path.join(DATA, f"{name}.json")) as fh:
        d = json.load(fh)
    e = np.array([x[:3] + [x[3] + x[4]] for x in d["evals"]], float)
    a = e[:, 2] * 100
    tm = np.array([a[max(0, i - 4):i + 1].mean() for i in range(len(a))])
    return dict(h=e[:, 0] / 3600, acc=a, tm=tm, gb=e[:, 3] / 1e9, sessions=np.array(d["sessions"]) / 60)


def bold(s):
    return f"$\\mathbf{{{s}}}$"


def main():
    ours, base = trace("sm6c_fluxtune_e2e"), trace("sm_fwdllm_e2e")
    jb = int(np.argmax(base["tm"]))
    lvl = base["tm"][jb]
    jo = int(np.argmax(ours["tm"] >= lvl))
    gain = ours["tm"].max() - lvl
    sooner, fewer = base["h"][jb] / ours["h"][jo], base["gb"][jb] / ours["gb"][jo]
    print(f"  FwdLLM peak {lvl:.1f}% at {base['h'][jb]:.0f} h / {base['gb'][jb]:.1f} GB; SwiFT reaches it at {ours['h'][jo]:.0f} h / "
          f"{ours['gb'][jo]:.1f} GB -> {sooner:.1f}x sooner, {fewer:.1f}x fewer GB; SwiFT peak {ours['tm'].max():.1f}% (+{gain:.1f})")

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(S.COL_W, 1.6), width_ratios=[1.25, 1])
    # (a) accuracy vs time
    for r, col, lw, z, nm in ((base, BASE, LINE["width"], Z["line"], "FwdLLM"), (ours, OURS, LINE["hero"], Z["hero"], S.SYS_NAME)):
        ax.plot(r["h"], S.smooth(r["acc"], SMOOTH), "-", color=col, lw=lw, zorder=z, label=nm)
    x_end = max(base["h"][-1], ours["h"][-1])
    ax.hlines(lvl, 0, base["h"][jb], colors=INK, lw=LINE["ref"], ls="--", zorder=Z["fill"], label="_ref")
    ax.plot([ours["h"][jo]], [lvl], "o", color=OURS, ms=MARKER["size"], zorder=Z["annot"], label="_hit")
    ax.plot([base["h"][jb]], [lvl], "o", color=BASE, ms=MARKER["size"], mfc="white", zorder=Z["annot"], label="_hit")
    ax.annotate("", xy=(ours["h"][jo] + 8, lvl), xytext=(base["h"][jb] - 8, lvl),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.7, mutation_scale=6, shrinkA=0, shrinkB=0), zorder=Z["annot"])
    xg = ours["h"][-1] + 25
    ax.annotate("", xy=(xg, ours["tm"].max()), xytext=(xg, lvl),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.7, mutation_scale=6, shrinkA=0, shrinkB=0), zorder=Z["annot"])
    ax.set_xlim(0, x_end)
    ax.set_ylim(30, 64)
    ax.set_yticks([30, 40, 50, 60])
    ax.set_xticks([0, 200, 400, 600, 800])
    ax.set_xlabel("Simulated time (hours)")
    ax.set_ylabel("Test accuracy (%)")
    # (b) time online per contribution
    sess = sessions_min()
    xs = np.sort(sess)
    bx.fill_between(xs, np.arange(1, len(xs) + 1) / len(xs), step="post", color=S.FILL, lw=0, zorder=Z["fill"], gid="region",
                    label="_sessions")
    hold = holds_min(load("sm_fwdllm"))
    task = ours["sessions"]
    cdf(bx, hold, color=BASE, lw=LINE["width"], zorder=Z["line"], label="FwdLLM hold")
    cdf(bx, task, color=OURS, lw=LINE["hero"], zorder=Z["hero"], label=f"{S.SYS_NAME} task")
    mh, mt = float(np.median(hold)), float(np.median(task))
    bx.plot([mt, mh], [0.5, 0.5], "o", color=INK, ms=MARKER["size"] - 1, zorder=Z["annot"], label="_med")
    print(f"  median: FwdLLM hold {mh:.0f} min (p90 {np.percentile(hold, 90):.0f}), SwiFT task {mt:.1f} min "
          f"(p90 {np.percentile(task, 90):.0f}) -> {mh / mt:.1f}x; sessions shorter than: hold {100 * (sess < mh).mean():.0f}%, "
          f"task {100 * (sess < mt).mean():.0f}%; max hold {hold.max():.0f} min")
    bx.set_xlim(0, 180)
    bx.set_xticks([0, 60, 120, 180])
    bx.set_ylim(0, 1)
    bx.set_yticks([0, 0.5, 1])
    bx.set_xlabel("Time online (min)")
    bx.set_ylabel("CDF")
    fig.legend(handles=[Line2D([], [], color=OURS, lw=LINE["hero"], label=S.SYS_NAME),
                        Line2D([], [], color=BASE, lw=LINE["width"], label="FwdLLM"),
                        Patch(facecolor=S.FILL, edgecolor="none", label="Phone sessions")],
               loc="outside upper center", ncol=3, columnspacing=1.0, borderaxespad=0.0)
    S.freeze(fig)
    S.place(ax, f"{bold(f'{sooner:.1f}' + r'\times')} sooner,\n{bold(f'{fewer:.1f}' + r'\times')} fewer GB",
            [(0.42, 0.40, "center", "top"), (0.42, 0.38, "center", "top"), (0.55, 0.40, "center", "top")])
    S.place(ax, f"{bold(f'+{gain:.1f}')} pt\npeak", [(0.62, 0.98, "left", "top"), (0.62, 0.90, "left", "top")])
    S.place(bx, f"{bold(f'{mh / mt:.1f}' + r'\times')}\nshorter", [(0.55, 0.62, "center", "bottom"), (0.45, 0.30, "center", "top")])
    r = fig.canvas.get_renderer()
    for a in (ax, bx):
        bb = a.xaxis.label.get_window_extent(r)
        if bb.x0 < 0 or bb.x1 > fig.bbox.x1:
            print(f"  CLIPPED x-label {a.get_xlabel()!r}")
    os.makedirs(OUT, exist_ok=True)
    for t, what in S.overlaps(fig):
        print(f"  OVERLAP fig1_teaser: {t!r} touches {what!r}")
    fig.savefig(os.path.join(OUT, "fig1_teaser.pdf"), metadata={"Subject": "runs 6c (paper_sm_20261026b), FwdLLM 1.7B (jayne 20261028_182751)"})
    fig.savefig(os.path.join(OUT, "fig1_teaser.png"))
    print(f"  -> {OUT}/fig1_teaser.pdf")


if __name__ == "__main__":
    main()
