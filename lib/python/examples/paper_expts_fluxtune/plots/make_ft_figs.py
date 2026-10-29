"""Every FluxTune paper figure that has data today (RUN_PLAN §1), from run digests (ft_digest.py) in digests/.

Usage: python make_ft_figs.py [--only NAME ...]      -> ../figs/<NAME>.{pdf,png}
Rebuild a digest: python ft_digest.py <run_dir> digests/<id>.json (AG News runs live on shepherd; digest there, copy back).
Numbers on the plots follow the paper's metric: a level is "reached" at the 2nd of 2 consecutive evals >= it.
"""
import argparse
import json
import os

from matplotlib.lines import Line2D

import ft_style as S
from ft_style import FONT, INK, LINE, Z, plt

HERE = os.path.dirname(os.path.abspath(__file__))
DIG, OUT = os.path.join(HERE, "digests"), os.path.join(HERE, "..", "figs")
K = 2      # evals in a row that define "reached" (figs_e2e.yaml tta_consecutive)
SMOOTH = 9  # display smoothing window (evals)

# run id -> (display name, RUN_PLAN run #, run dir relative to examples/fwdllm/)
RUNS = {
    "p1_fluxtune": ("FluxTune", 2, "p1_distil_20261025/bench/P1_fluxtune (shepherd)"),
    "p1_fwdllm": ("FwdLLM", 1, "p1_distil_20261025/bench/P1_fwdllm (shepherd)"),
    "rb_fluxtune": ("FluxTune", 4, "paper_rb_20261026b/POOL/rb_paper"),
    "su_fluxtune": ("FluxTune", "su", "paper_su_20261029/POOL/su_flux (running)"),
    "su_fwdllm": ("FwdLLM", "su", "paper_su_20261029/POOL/su_fwd (running)"),
    "dy_fluxtune": ("FluxTune", 29, "paper_dy_20261027/POOL_full_flux"),
    "dy_fwdllm": ("FwdLLM", 28, "paper_dy_20261027/POOL_full_fwdllm"),
    "a08": ("SGD step (no trust ratio)", 8, "paper_ab_20261028/POOL/a08 (shepherd)"),
    "a10": ("FluxTune without async", 10, "paper_ab_20261028/POOL/a10 (shepherd)"),
    "a26": ("Max-|JVP| probe", 26, "paper_ab_20261028b/POOL/a26 (shepherd)"),
    "a14": (r"$\rho_0$ 0.10", 14, "paper_ab_20261028b/POOL/a14 (shepherd)"),
    "a15": (r"$\rho_0$ 0.14", 15, "paper_ab_20261028/POOL/a15 (shepherd)"),
    "a16": (r"$\rho_0$ 0.28", 16, "paper_ab_20261028/POOL/a16 (shepherd)"),
    "a17": ("N 10", 17, "paper_ab_20261028b/POOL/a17 (shepherd)"),
    "a18": ("N 20", 18, "paper_ab_20261028b/POOL/a18 (shepherd)"),
    "a19": ("N 90", 19, "paper_ab_20261028b/POOL/a19 (shepherd)"),
    "a20": ("Adaptive N", 20, "paper_ab_20261028b/POOL/a20 (shepherd)"),
    "a21": (r"Constant $\rho$", 21, "paper_ab_20261028b/POOL/a21 (shepherd)"),
    "a22": ("Momentum 0.8", 22, "paper_ab_20261028b/POOL/a22 (shepherd)"),
    "a11": ("FwdLLM async", 11, "paper_ab_20261028b/POOL/a11 (shepherd)"),
    "a12": ("Felix(P)", 12, "paper_ab_20261028b/POOL/a12 (shepherd)"),
    "a13": ("FedBuff(P)", 13, "paper_ab_20261028b/POOL/a13 (shepherd)"),
}

# E2E cells: (title, ours, baseline or None, target, y floor %, status note or None)
CELLS = [
    ("DistilBERT · AG News", "p1_fluxtune", "p1_fwdllm", 0.86, 76, None),
    ("RoBERTa-L · Yelp-P", "rb_fluxtune", None, 0.82, 70, "FwdLLM run pending"),
    ("SmolLM2-1.7B · Yahoo", "su_fluxtune", "su_fwdllm", 0.65, 25, "running, near-IID"),
    ("DistilBERT · Yahoo", "dy_fluxtune", "dy_fwdllm", 0.65, 50, None),
]


class Run:
    def __init__(self, rid):
        d = json.load(open(os.path.join(DIG, f"{rid}.json")))
        self.rid, self.name = rid, RUNS[rid][0]
        ev = d["evals"]
        self.t = [e[0] / 3600 for e in ev]
        self.c = [e[1] for e in ev]
        self.acc = [e[2] for e in ev]
        self.gb = [(e[3] + e[4]) / 1e9 for e in ev]
        self.down = [e[3] / 1e9 for e in ev]
        self.up = [e[4] / 1e9 for e in ev]

    def reach(self, level):
        """Index of the eval where `level` is reached (2nd of K in a row), or None."""
        for i in range(len(self.acc) - K + 1):
            if min(self.acc[i:i + K]) >= level:
                return i + K - 1
        return None

    def peak(self):
        """Highest level this run reaches under the K-in-a-row rule."""
        return max(min(self.acc[i:i + K]) for i in range(len(self.acc) - K + 1))


def style(run, ours, arm_i=None):
    """(color, ls, marker, lw, z, filled) for a run."""
    if run is ours:
        c, ls, m = S.STACK["FluxTune"]
        return c, ls, m, LINE["hero"], Z["hero"], True
    if arm_i is not None:
        c, m = S.ARM[arm_i]
        return c, "-", m, LINE["width"], Z["line"], False
    c, ls, m = S.STACK[run.name]
    return c, ls, m, LINE["width"], Z["line"], False


def curve(ax, run, x, ours, arm_i=None):
    c, ls, m, lw, z, filled = style(run, ours, arm_i)
    ax.plot(x, S.smooth([a * 100 for a in run.acc], SMOOTH), color=c, ls=ls, lw=lw, zorder=z, marker=m, markevery=0.12,
            mfc=c if filled else "white", mec=c, mew=0.8, label=run.name)


def handle(run, ours, arm_i=None):
    c, ls, m, lw, z, filled = style(run, ours, arm_i)
    return Line2D([], [], color=c, ls=ls, lw=lw, marker=m, mfc=c if filled else "white", mec=c, mew=0.8, label=run.name)


def compare(ours, other, target):
    """(level, i_ours, i_other, both_hit_target): the fair comparison level (guideline §2)."""
    if ours.reach(target) is not None and other.reach(target) is not None:
        return target, ours.reach(target), other.reach(target), True
    level = min(ours.peak(), other.peak(), target)
    return level, ours.reach(level), other.reach(level), False


def ratio_text(a, b, word, worse):
    """'N× word' with N >= 1, or 'Same' within 5%; `worse` is the opposite word."""
    r = b / a
    if abs(r - 1) <= 0.05:
        return "Same commits" if "commit" in word else "Same time" if word == "faster" else f"Same {word.split()[-1]}"
    return rf"$\mathbf{{{r:.1f}{{\times}}}}$ {word}" if r > 1 else rf"$\mathbf{{{1 / r:.1f}{{\times}}}}$ {worse}"


def arrow(ax, x0, x1, y):
    ax.annotate("", xy=(x1, y), xytext=(x0, y), arrowprops=dict(arrowstyle="->", color=INK, lw=0.7, shrinkA=0, shrinkB=0),
                zorder=Z["annot"])


def target_line(ax, target):
    ax.axhline(target * 100, color=INK, ls="--", lw=LINE["ref"], zorder=Z["fill"])


def time_axis(ax, runs, ylo):
    ax.set_xscale("log")
    first = [r.t[i] for r in runs for i in range(len(r.t)) if r.acc[i] * 100 >= ylo and r.t[i] > 0]
    ax.set_xlim(min(first) * 0.7 if first else None, max(max(r.t) for r in runs) * 1.15)
    from matplotlib.ticker import LogLocator, NullFormatter
    lo, hi = ax.get_xlim()
    subs = (1, 2, 4, 6, 8) if hi / lo < 4 else (1, 3) if hi / lo < 30 else (1,)
    ax.xaxis.set_major_locator(LogLocator(base=10, subs=subs))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())


def callout_pair(ax, ours, other, target, xs_ours, xs_other, unit_word, worse_word, spots):
    """Arrow at the comparison level from ours to other, plus a black callout (guideline §2)."""
    level, io, ib, both = compare(ours, other, target)
    if io is None or ib is None:
        return
    xo, xb = xs_ours[io], xs_other[ib]
    if abs(xb / xo - 1) > 0.05:
        arrow(ax, xo, xb, level * 100)
    txt = ratio_text(xo, xb, unit_word, worse_word)
    if not both:
        d = (ours.peak() - other.peak()) * 100
        txt += "\n" + (rf"$\mathbf{{{d:+.1f}\%}}$ peak" if abs(d) >= 0.1 else "Same peak")
    S.place(ax, txt, spots)


# ------------------------------------------------------------------ E2E ------------------------------------------------------------------
def e2e(kind):
    fig, axes = plt.subplots(1, len(CELLS), figsize=(S.FULL_W, 1.95))
    for ax, (title, o, b, target, ylo, note) in zip(axes, CELLS):
        ours = Run(o)
        base = Run(b) if b else None
        runs = [ours] + ([base] if base else [])
        x = (lambda r: r.t) if kind == "time" else (lambda r: r.c)
        for r in runs[::-1]:
            curve(ax, r, x(r), ours)
        target_line(ax, target)
        top = max(max(r.acc) for r in runs) * 100
        ax.set_ylim(ylo, top + (top - ylo) * 0.38)  # empty band on top for the callout
        if kind == "time":
            time_axis(ax, runs, ylo)
        else:
            ax.set_xlim(0, max(max(r.c) for r in runs) * 1.02)
        ax.set_xlabel(("Time (hours, log scale)" if kind == "time" else "Server Commits") + f"\n{title}")
        if note:
            S.status(ax, "PRELIM", note)
    axes[0].set_ylabel("Test Accuracy (%)")
    fig.legend(handles=[Line2D([], [], color=S.STACK["FluxTune"][0], lw=LINE["hero"], marker="D", label="FluxTune"),
                        Line2D([], [], color=S.STACK["FwdLLM"][0], lw=LINE["width"], marker="o", mfc="white", label="FwdLLM"),
                        Line2D([], [], color=INK, ls="--", lw=LINE["ref"], label="Target")],
               loc="outside upper center", ncol=3)
    S.freeze(fig)
    for ax, (title, o, b, target, ylo, note) in zip(axes, CELLS):
        ours = Run(o)
        spots = [(0.97, 0.97, "right", "top"), (0.03, 0.97, "left", "top"), (0.97, 0.04, "right", "bottom")]
        if b:
            base = Run(b)
            xs = (lambda r: r.t) if kind == "time" else (lambda r: r.c)
            word = ("faster", "slower") if kind == "time" else ("fewer commits", "more commits")
            callout_pair(ax, ours, base, target, xs(ours), xs(base), *word, spots)
        else:
            i = ours.reach(target)
            if i is not None:
                v = f"{ours.t[i]:.0f} h" if kind == "time" else f"commit {ours.c[i]}"
                S.place(ax, f"Target {target * 100:.0f}% at {v}", spots)
    return S.save(fig, OUT, f"E2E_acc_vs_{'time' if kind == 'time' else 'commits'}")


def e2e_comm():
    fig, axes = plt.subplots(len(CELLS), 1, figsize=(S.COL_W, 2.9))
    for ax, (title, o, b, target, ylo, note) in zip(axes, CELLS):
        ours = Run(o)
        base = Run(b) if b else None
        if base:
            level, io, ib, both = compare(ours, base, target)
            rows = [(base, ib), (ours, io)]
        else:
            rows, level, both = [(ours, ours.reach(target))], target, True
        ends = []
        for y, (r, i) in enumerate(rows):
            c = S.STACK[r.name][0]
            if i is None:
                continue
            ax.barh(y, r.down[i], color=c, height=0.6, edgecolor=c, zorder=Z["bar"])
            ax.barh(y, r.up[i], left=r.down[i], color="white", height=0.6, edgecolor=c, hatch="xxxx", zorder=Z["bar"])
            ends.append((y, r.gb[i], r))
        ax.set_yticks(range(len(rows)), [r.name for r, _ in rows])
        ax.set_ylim(-0.5, len(rows) + 0.25)  # top strip holds the panel identity
        ax.grid(False)
        ax.grid(True, axis="x")
        xmax = max(g for _, g, _ in ends)
        ax.set_xlim(0, xmax * 1.75)
        lvl = f"{level * 100:.1f}%" + ("" if both else " (FwdLLM peak)")
        ax.text(0.99, 0.97, f"{title}, to {lvl}", transform=ax.transAxes, ha="right", va="top", fontsize=FONT["annot"])
        if note:
            S.status(ax, "PRELIM", note)
        og = next(g for _, g, r in ends if r is ours)
        bg = [g for _, g, r in ends if r is not ours]
        for y, g, r in ends:
            s = f"{g:.0f} GB" if g >= 10 else f"{g:.1f} GB"
            if r is ours and bg:
                s += ", " + ratio_text(og, bg[0], "fewer", "more")
            ax.text(g + xmax * 0.03, y, s, va="center", ha="left", fontsize=FONT["annot"], zorder=Z["annot"])
    axes[-1].set_xlabel("Data Moved (GB)")
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, fc=INK, ec=INK, label="Downlink"),
                        plt.Rectangle((0, 0), 1, 1, fc="white", ec=INK, hatch="xxxx", label="Uplink")],
               loc="outside upper center", ncol=2)
    return S.save(fig, OUT, "E2E_comm_to_target")


# ------------------------------------------------------- two-panel AG News comparisons -------------------------------------------------------
def pair_fig(name, ours_id, others, arms=False, ref_ids=(), target=0.86, ylo=78, callout=None):
    """Acc vs time (log) and vs commits; `others` drawn as ARM colors when arms=True, else as stacks.
    `callout`: run id compared against ours (arrow + ratio on the time panel)."""
    ours = Run(ours_id)
    refs = [Run(r) for r in ref_ids]
    oth = [Run(r) for r in others]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(S.COL_W, 1.75), sharey=True)
    for ax, xs in ((a1, lambda r: r.t), (a2, lambda r: r.c)):
        for j, r in list(enumerate(oth))[::-1]:
            curve(ax, r, xs(r), ours, j if arms else None)
        for r in refs:
            curve(ax, r, xs(r), ours)
        curve(ax, ours, xs(ours), ours)
        target_line(ax, target)
    allr = [ours] + oth + refs
    top = max(max(r.acc) for r in allr) * 100
    a1.set_ylim(ylo, top + (top - ylo) * 0.3)  # empty band on top for the callout
    a1.yaxis.set_major_locator(plt.MultipleLocator(2 if top - ylo < 12 else 5))
    time_axis(a1, allr, ylo)
    a2.set_xlim(0, max(max(r.c) for r in allr) * 1.02)
    a1.set_xlabel("Time (hours, log scale)")
    a2.set_xlabel("Server Commits")
    a1.set_ylabel("Test Accuracy (%)")
    hs = [handle(ours, ours)] + [handle(r, ours) for r in refs] + [handle(r, ours, j if arms else None) for j, r in enumerate(oth)]
    ours_label = {"tune_rho0": r"FluxTune ($\rho_0$ 0.07)", "tune_commit_rule": "FluxTune (N 50)",
                  "tune_rho_schedule": r"FluxTune (exp $\rho$)"}.get(name)
    if ours_label:
        hs[0].set_label(ours_label)
    nc = min(len(hs), 3)
    rows = -(-len(hs) // nc)
    hs = [hs[r * nc + c] for c in range(nc) for r in range(rows) if r * nc + c < len(hs)]  # read row by row
    fig.legend(handles=hs, loc="outside upper center", ncol=nc)
    S.freeze(fig)
    if callout:
        other = Run(callout)
        spots = [(0.97, 0.97, "right", "top"), (0.03, 0.97, "left", "top"), (0.97, 0.04, "right", "bottom")]
        callout_pair(a1, ours, other, target, ours.t, other.t, "faster", "slower", spots)
    return S.save(fig, OUT, name)


def tune_summary_text(name, ours_id, others, target):
    """One line per arm for the console: time to target or DNF + peak (numbers for the text / captions)."""
    for rid in [ours_id] + others:
        r = Run(rid)
        i = r.reach(target)
        hit = f"{r.t[i]:.2f} h, c{r.c[i]}" if i is not None else "DNF"
        print(f"    {name:20s} {RUNS[rid][0]:28s} run {RUNS[rid][1]!s:>3}: target {hit:16s} peak {r.peak() * 100:.2f}%")


FIGS = {
    "E2E_acc_vs_time": lambda: e2e("time"),
    "E2E_acc_vs_commits": lambda: e2e("commits"),
    "E2E_comm_to_target": e2e_comm,
    "Contri_probe_mean": lambda: pair_fig("Contri_probe_mean", "p1_fluxtune", ["a26"], arms=True, ref_ids=["p1_fwdllm"], callout="a26"),
    "Contri_trust_ratio": lambda: pair_fig("Contri_trust_ratio", "p1_fluxtune", ["a08"], arms=True, ref_ids=["p1_fwdllm"], callout="a08"),
    "Contri_async": lambda: pair_fig("Contri_async", "p1_fluxtune", ["a10"], arms=True, ref_ids=["p1_fwdllm", "a11"], callout="a10"),
    "tune_rho0": lambda: pair_fig("tune_rho0", "p1_fluxtune", ["a14", "a15", "a16"], arms=True),
    "tune_commit_rule": lambda: pair_fig("tune_commit_rule", "p1_fluxtune", ["a17", "a18", "a19", "a20"], arms=True),
    "tune_rho_schedule": lambda: pair_fig("tune_rho_schedule", "p1_fluxtune", ["a21", "a22"], arms=True, ylo=70, callout="a21"),
    "strategy_fwdllm_async": lambda: pair_fig("strategy_fwdllm_async", "p1_fluxtune", ["a11"], ref_ids=["p1_fwdllm"], callout="a11"),
    "strategy_fedbuff": lambda: pair_fig("strategy_fedbuff", "p1_fluxtune", ["a13"], ref_ids=["p1_fwdllm"], callout="a13"),
    "strategy_felix": lambda: pair_fig("strategy_felix", "p1_fluxtune", ["a12"], ref_ids=["p1_fwdllm"], callout="a12"),
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    for n, f in FIGS.items():
        if a.only and n not in a.only:
            continue
        print(f"{n} -> {os.path.relpath(f(), HERE)}", flush=True)
    for n, o, arms in (("tune_rho0", "p1_fluxtune", ["a14", "a15", "a16"]),
                       ("tune_commit_rule", "p1_fluxtune", ["a17", "a18", "a19", "a20"]),
                       ("tune_rho_schedule", "p1_fluxtune", ["a21", "a22"]),
                       ("contri", "p1_fluxtune", ["a08", "a10", "a26"]),
                       ("strategy", "p1_fwdllm", ["a11", "a12", "a13"])):
        if not a.only or n in a.only:
            tune_summary_text(n, o, arms, 0.86)
