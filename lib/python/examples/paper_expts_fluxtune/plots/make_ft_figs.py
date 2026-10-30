"""Every FluxTune paper figure that has data today (RUN_PLAN §1), from per-run digests in digests/.

Usage: python make_ft_figs.py [--only NAME ...]                 -> ../figs/<NAME>.{pdf,png}
       python make_ft_figs.py digest <run_dir> digests/<id>.json  (stdlib only: AG News runs live on shepherd; digest there, copy back)
Numbers on the plots follow the paper's metric: a level is "reached" at the 2nd of 2 consecutive evals >= it.

Digest = {"meta", "end", "evals": [[vclock_s, commit, acc, down_B, up_B]], "final": [vclock_s, acc] | null,
"part": [[commit, distinct trainers so far, distinct trainers in this commit]], "sessions": [s], "n_clients"}.
`commit` counts agg rounds with a commit_reason (server commits so far); bytes are cumulative aggregator wire bytes at that eval.
Session = one engagement: async one task (dispatch -> commit), sync a trainer's span within one model version (one round).
"""
import argparse
import glob
import json
import os
import sys

META_KEYS = ("p_trainable", "agg_goal", "concurrency", "train_batch_size", "commit_gate", "server_step_rule", "rho_schedule",
             "rho_star", "t_res", "probe_combine", "perturbation_count", "var_threshold")


def digest(run_dir):
    f = glob.glob(os.path.join(run_dir, "telemetry", "aggregator_*.jsonl"))[0]
    t, down, up, evals, final, meta, end, commits = 0.0, 0, 0, [], None, {}, {}, 0
    seen, cur, part, sessions, spans, n_clients = set(), set(), [], [], {}, None
    with open(f, "rb") as fh:
        for line in fh:
            if b'"wire"' in line:
                e = json.loads(line)
                if e.get("dir") == "send":
                    down += e["bytes"]
                else:
                    up += e["bytes"]
            elif b'"agg_round"' in line:
                e = json.loads(line)
                t = float(e["vclock_now"]) if e.get("vclock_now") is not None else t
                for ci in e.get("contributor_intervals") or []:
                    seen.add(ci["end"])
                    cur.add(ci["end"])
                    if e.get("is_async"):
                        sessions.append(round(ci["commit_ts"] - ci["dispatch_ts"], 1))
                    else:
                        k = (ci["end"], ci["dispatch_model_version"])
                        lo, hi = spans.get(k, (ci["dispatch_ts"], ci["commit_ts"]))
                        spans[k] = (min(lo, ci["dispatch_ts"]), max(hi, ci["commit_ts"]))
                if e.get("commit_reason") is not None:
                    commits += 1
                    part.append([commits, len(seen), len(cur)])
                    cur = set()
            elif n_clients is None and b'"selection"' in line:
                n_clients = json.loads(line).get("num_candidates")
            elif b'"agg_eval_final"' in line:
                e = json.loads(line)
                final = [float(e.get("vclock_now") or t), float(e["test-accuracy"])]
            elif b'"agg_eval"' in line:
                e = json.loads(line)
                if e.get("test-accuracy") is not None:
                    evals.append([round(t, 1), commits, float(e["test-accuracy"]), down, up])
            elif b'"run_meta"' in line:
                e = json.loads(line)
                meta = {k: e[k] for k in META_KEYS if k in e}
            elif b'"run_end"' in line:
                e = json.loads(line)
                end = {k: v for k, v in e.items() if isinstance(v, (int, float, str)) and k not in ("ts", "role", "event", "end_id")}
    sessions += [round(hi - lo, 1) for lo, hi in spans.values()]
    return dict(run_dir=os.path.abspath(run_dir), meta=meta, end=end, evals=evals, final=final, down=down, up=up, t_end=t, commits=commits,
                part=part, sessions=sessions, n_clients=n_clients)


if __name__ == "__main__" and sys.argv[1:2] == ["digest"]:  # before the matplotlib imports below
    with open(sys.argv[3], "w") as fh:
        json.dump(digest(sys.argv[2]), fh)
    sys.exit()

import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import ft_style as S  # noqa: E402
from ft_style import FONT, INK, LINE, MARKER, Z, plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DIG, OUT = os.path.join(HERE, "digests"), os.path.join(HERE, "..", "figs")
K = 2      # evals in a row that define "reached" (figs_e2e.yaml tta_consecutive)
SMOOTH = 9  # display smoothing window (evals)

# run id -> (display name, RUN_PLAN run #, run dir relative to examples/fwdllm/)
RUNS = {
    "p1_fluxtune": (S.SYS_NAME, 2, "p1_distil_20261025/bench/P1_fluxtune (shepherd)"),
    "p1_fwdllm": ("FwdLLM", 1, "p1_distil_20261025/bench/P1_fwdllm (shepherd)"),
    "rb_fluxtune": (S.SYS_NAME, 4, "paper_rb_20261026b/POOL/rb_paper"),
    "rb_fwdllm": ("FwdLLM", 3, "experiments/run_20261023_160644_rb60_fwdllm_n100_fwdllm_yelp-p_n100_smoke_syn_0_sim"),
    "sm6c_fluxtune": (S.SYS_NAME, 6, "paper_sm_20261026b/POOL/sm_paper"),
    "sm_fwdllm": ("FwdLLM", 5, "experiments/run_20261028_182751_fwdllm_yahoo_n60_smoke_syn_0_sim"),
    "su_fluxtune": (S.SYS_NAME, "su", "paper_su_20261029/POOL/su_flux (running)"),
    "su_fwdllm": ("FwdLLM", "su", "paper_su_20261029/POOL/su_fwd (running)"),
    "dy_fluxtune": (S.SYS_NAME, 29, "paper_dy_20261027/POOL_full_flux"),
    "dy_fwdllm": ("FwdLLM", 28, "paper_dy_20261027/POOL_full_fwdllm"),
    "a08": ("SGD step (no trust ratio)", 8, "paper_ab_20261028/POOL/a08 (shepherd)"),
    "a10": (f"{S.SYS_NAME} without async", 10, "paper_ab_20261028/POOL/a10 (shepherd)"),
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
    ("RoBERTa-L · Yelp-P", "rb_fluxtune", "rb_fwdllm", 0.82, 45, None),
    ("SmolLM2-1.7B · Yahoo", "sm6c_fluxtune", "sm_fwdllm", 0.65, 25, None),  # non-IID α 1; near-IID su_* still running
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
        self.part, self.sessions, self.n_clients = d.get("part", []), sorted(d.get("sessions", [])), d.get("n_clients")

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
        c, ls, m = S.STACK[S.SYS_NAME]
        return c, ls, m, LINE["hero"], Z["hero"], True
    if arm_i is not None:
        c, m = S.ARM[arm_i]
        return c, "-", m, LINE["width"], Z["line"], False
    c, ls, m = S.STACK[run.name]
    return c, ls, m, LINE["width"], Z["line"], False


def curve(ax, run, x, ours, arm_i=None, y=None):
    """y defaults to the smoothed accuracy (%)."""
    c, ls, m, lw, z, filled = style(run, ours, arm_i)
    ax.plot(x, S.smooth([a * 100 for a in run.acc], SMOOTH) if y is None else y, color=c, ls=ls, lw=lw, zorder=z, marker=m, markevery=0.12,
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
        return "Same time" if word == "faster" else f"Same {word.split()[-1]}"
    return rf"$\mathbf{{{r:.1f}{{\times}}}}$ {word}" if r > 1 else rf"$\mathbf{{{1 / r:.1f}{{\times}}}}$ {worse}"


def arrow(ax, x0, x1, y):
    ax.annotate("", xy=(x1, y), xytext=(x0, y), arrowprops=dict(arrowstyle="->", color=INK, lw=0.7, shrinkA=0, shrinkB=0),
                zorder=Z["annot"])


def target_line(ax, target):
    ax._target = ax.axhline(target * 100, color=INK, ls="--", lw=LINE["ref"], zorder=Z["fill"])


def time_axis(ax, runs, ylo):
    """Linear hours (log would shrink the gap we want seen), from 0 to the longest run's end; peak_gain may widen it."""
    ax._t_end = max(max(r.t) for r in runs) * 1.02
    ax.set_xlim(0, ax._t_end)
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax.set_xticks([x for x in ax.get_xticks() if 0 <= x <= ax._t_end])


def gain_text(ours, other):
    """Bold '+X% peak' (ours over other's peak), or '' within 0.1 pt."""
    d = (ours.peak() - other.peak()) * 100
    return rf"$\mathbf{{{d:+.1f}\%}}$ peak" if abs(d) >= 0.1 else ""


def peak_gain(ax, ours, other):
    """Right strip (widens x): peak ticks in run colors, arrow other -> ours, bold '+X%' label. Call after freeze."""
    po, pb = ours.peak() * 100, other.peak() * 100
    if abs(po - pb) < 0.1:
        return
    r = ax.figure.canvas.get_renderer()
    t = ax.text(0.99, (po + pb) / 2, gain_text(ours, other).replace(" ", "\n"), ha="right", va="center", fontsize=FONT["annot"],
                transform=ax.get_yaxis_transform(), zorder=Z["annot"], linespacing=1.0)
    fa = 0.99 - t.get_window_extent(r).width / ax.get_window_extent(r).width - 0.03  # arrow x, axes fraction
    fd = fa - 0.07  # data ends here; the ticks span fd..fa
    ax.set_xlim(0, ax._t_end / fd)
    if getattr(ax, "_target", None):
        ax._target.set_xdata([0, fd])
    xa = ax._t_end * fa / fd
    for run, p in ((ours, po), (other, pb)):
        c = next(ln.get_color() for ln in ax.lines if ln.get_label() == run.name)  # the run's curve color (stack or arm)
        ax.plot([ax._t_end, xa], [p, p], color=c, lw=LINE["width"], solid_capstyle="butt", zorder=Z["annot"])
    ax.annotate("", xy=(xa, po), xytext=(xa, pb), arrowprops=dict(arrowstyle="->", color=INK, lw=0.7, shrinkA=0, shrinkB=0),
                zorder=Z["annot"])


def callout_pair(ax, ours, other, target, xs_ours, xs_other, unit_word, worse_word, spots):
    """Speedup gap (shaded span, drop lines, hit dots, arrow) with the ratio nearest its center, plus the peak-gain strip."""
    level, io, ib, both = compare(ours, other, target)
    if io is None or ib is None:
        return
    peak_gain(ax, ours, other)  # first: it may widen the x axis
    xo, xb, y = xs_ours[io], xs_other[ib], level * 100
    if abs(xb / xo - 1) > 0.05:
        ylo = ax.get_ylim()[0]
        ax.fill_between([min(xo, xb), max(xo, xb)], ylo, y, color=INK, alpha=0.07, lw=0, zorder=Z["fill"])
        for x, run in ((xo, ours), (xb, other)):
            c = next(ln.get_color() for ln in ax.lines if ln.get_label() == run.name)
            ax.plot([x, x], [ylo, y], color=INK, ls=":", lw=LINE["ref"], zorder=Z["fill"])
            ax.plot(x, y, "o", ms=MARKER["size"] + 0.5, mfc=c, mec="white", mew=0.6, zorder=Z["annot"] + 1)
        arrow(ax, xo, xb, y)
        mx, my = ax.transAxes.inverted().transform(ax.transData.transform(((xo + xb) / 2, y)))
        spots = [(mx, my - 0.03, "center", "top"), (mx, my + 0.03, "center", "bottom")]
        ax.figure.canvas.draw()  # annotation arrows get their path at draw time; place() needs it
    S.place(ax, ratio_text(xo, xb, unit_word, worse_word), spots)


# ------------------------------------------------------------------ E2E ------------------------------------------------------------------
def e2e_legend(fig, ref=None):
    hs = [Line2D([], [], color=S.STACK[S.SYS_NAME][0], lw=LINE["hero"], marker="D", label=S.SYS_NAME),
          Line2D([], [], color=S.STACK["FwdLLM"][0], lw=LINE["width"], marker="o", mfc="white", label="FwdLLM")]
    if ref:
        hs.append(Line2D([], [], color=INK, ls="--", lw=LINE["ref"], label=ref))
    fig.legend(handles=hs, loc="outside upper center", ncol=len(hs))


def e2e():
    fig, axes = plt.subplots(1, len(CELLS), figsize=(S.FULL_W, 1.95))
    for ax, (title, o, b, target, ylo, note) in zip(axes, CELLS):
        ours = Run(o)
        base = Run(b) if b else None
        runs = [ours] + ([base] if base else [])
        for r in runs[::-1]:
            curve(ax, r, r.t, ours)
        target_line(ax, target)
        top = max(max(r.acc) for r in runs) * 100
        ax.set_ylim(ylo, top + (top - ylo) * 0.38)  # empty band on top for the callout
        time_axis(ax, runs, ylo)
        ax.set_xlabel(f"Time (hours)\n{title}")
        if note:
            S.status(ax, "PRELIM", note)
    axes[0].set_ylabel("Test Accuracy (%)")
    e2e_legend(fig, "Target")
    S.freeze(fig)
    for ax, (title, o, b, target, ylo, note) in zip(axes, CELLS):
        ours = Run(o)
        spots = [(0.97, 0.97, "right", "top"), (0.03, 0.97, "left", "top"), (0.97, 0.04, "right", "bottom")]
        if b:
            base = Run(b)
            callout_pair(ax, ours, base, target, ours.t, base.t, "faster", "slower", spots)
        else:
            i = ours.reach(target)
            if i is not None:
                S.place(ax, f"Target {target * 100:.0f}% at {ours.t[i]:.0f} h", spots)
    return S.save(fig, OUT, "E2E_acc_vs_time")


def first_commit_at(run, n):
    """First commit by which `n` distinct trainers have contributed, or None."""
    return next((p[0] for p in run.part if p[1] >= n), None)


def e2e_participation():
    """Distinct trainers that have contributed so far (% of all clients) vs server commits."""
    fig, axes = plt.subplots(1, len(CELLS), figsize=(S.FULL_W, 1.95))
    for ax, (title, o, b, target, ylo, note) in zip(axes, CELLS):
        ours = Run(o)
        runs = [ours] + ([Run(b)] if b else [])
        n = ours.n_clients
        for r in runs[::-1]:
            curve(ax, r, [p[0] for p in r.part], ours, y=[100 * p[1] / n for p in r.part])
        ax.axhline(100, color=INK, ls="--", lw=LINE["ref"], zorder=Z["fill"])
        full = [first_commit_at(r, n) or r.part[-1][0] for r in runs]
        ax.set_xlim(0, max(full) * 1.6)
        ax.set_ylim(0, 135)  # empty band on top for the callout
        ax.set_yticks(np.linspace(0, 100, 5))
        ax.set_xlabel(f"Server Commits\n{title}")
        if note:
            S.status(ax, "PRELIM", note)
    axes[0].set_ylabel("Clients Used (%)")
    e2e_legend(fig, "All clients")
    S.freeze(fig)
    for ax, (title, o, b, target, ylo, note) in zip(axes, CELLS):
        ours = Run(o)
        n = ours.n_clients
        spots = [(0.97, 0.97, "right", "top"), (0.97, 0.04, "right", "bottom")]
        co = first_commit_at(ours, n)
        if b and co is not None:
            cb = first_commit_at(Run(b), n)
            if cb is not None:
                if abs(cb / co - 1) > 0.05:
                    arrow(ax, co, cb, 100)
                S.place(ax, "All clients:\n" + ratio_text(co, cb, "fewer commits", "more commits"), spots)
        elif co is not None:
            S.place(ax, f"All {n} clients by commit {co}", spots)
    return S.save(fig, OUT, "E2E_participation")


def e2e_participation_per_commit():
    """Distinct trainers contributing to each server commit, % of all clients (display smoothed)."""
    fig, axes = plt.subplots(1, len(CELLS), figsize=(S.FULL_W, 1.95))
    for ax, (title, o, b, target, ylo, note) in zip(axes, CELLS):
        ours = Run(o)
        runs = [ours] + ([Run(b)] if b else [])
        for r in runs[::-1]:
            curve(ax, r, [p[0] for p in r.part], ours, y=S.smooth([100 * p[2] / ours.n_clients for p in r.part], 3 * SMOOTH))
        ax.set_xlim(0, max(r.part[-1][0] for r in runs) * 1.02)
        ax.set_ylim(0, 100)  # same scale in every cell; the top band stays free for the callout
        ax.set_yticks(np.linspace(0, 100, 5))
        ax.set_xlabel(f"Server Commits\n{title}")
        if note:
            S.status(ax, "PRELIM", note)
    axes[0].set_ylabel("Clients per Commit (%)")
    e2e_legend(fig)
    S.freeze(fig)
    for ax, (title, o, b, target, ylo, note) in zip(axes, CELLS):
        mo = float(np.median([p[2] for p in Run(o).part]))
        spots = [(0.97, 0.97, "right", "top"), (0.03, 0.97, "left", "top")]
        if b:
            mb = float(np.median([p[2] for p in Run(b).part]))
            S.place(ax, "Median " + ratio_text(mb, mo, "more clients", "fewer clients"), spots)
        else:
            S.place(ax, f"Median {100 * mo / Run(o).n_clients:.0f}% of clients", spots)
    return S.save(fig, OUT, "E2E_participation_per_commit")


def e2e_session_length():
    """CDF of one engagement: FluxTune one task (dispatch -> commit), FwdLLM a client's span in one round. Linear axis, so the gap
    reads at its true size; x cut at the longer run's 99th percentile."""
    fig, axes = plt.subplots(1, len(CELLS), figsize=(S.FULL_W, 1.95), sharey=True)
    for ax, (title, o, b, target, ylo, note) in zip(axes, CELLS):
        ours = Run(o)
        runs = [ours] + ([Run(b)] if b else [])
        hi = max(np.quantile(r.sessions, 0.99) for r in runs)
        unit, div = ("s", 1) if hi < 180 else ("min", 60) if hi < 3 * 3600 else ("h", 3600)
        q = np.linspace(0, 1, 400)
        for r in runs[::-1]:
            curve(ax, r, np.quantile(r.sessions, q) / div, ours, y=q)
        ax.set_xlim(0, hi * 1.05 / div)
        ax.set_ylim(0, 1.3)  # empty band on top for the callout
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_xlabel(f"Session Length ({unit})\n{title}")
        ax.unit_div = div
        if note:
            S.status(ax, "PRELIM", note)
    axes[0].set_ylabel("Fraction of Sessions (CDF)")
    e2e_legend(fig)
    S.freeze(fig)
    for ax, (title, o, b, target, ylo, note) in zip(axes, CELLS):
        ours = Run(o)
        mo = float(np.median(ours.sessions))
        spots = [(0.97, 0.97, "right", "top"), (0.03, 0.97, "left", "top"), (0.97, 0.04, "right", "bottom")]
        if b:
            mb = float(np.median(Run(b).sessions))
            if abs(mb / mo - 1) > 0.05:
                arrow(ax, mo / ax.unit_div, mb / ax.unit_div, 0.5)
            S.place(ax, "Median\n" + ratio_text(mo, mb, "shorter", "longer"), spots)
        else:
            S.place(ax, f"Median {mo / 60:.0f} min" if mo >= 60 else f"Median {mo:.0f} s", spots)
    return S.save(fig, OUT, "E2E_session_length")


def e2e_speedup():
    """Time to the fair level (target, else FwdLLM's peak): FwdLLM vs FluxTune bars, one panel per cell (own linear hours axis,
    cells span 3 h to 400 h); speedup above the pair."""
    fig, axes = plt.subplots(1, len(CELLS), figsize=(S.FULL_W, 1.75))
    for ax, (title, o, b, target, ylo, note) in zip(axes, CELLS):
        ours = Run(o)
        if b:
            base = Run(b)
            level, io, ib, both = compare(ours, base, target)
            bars = [(base, ib), (ours, io)]
        else:
            level, both = target, True
            bars = [(None, None), (ours, ours.reach(target))]
        tops = []
        for x, (r, i) in enumerate(bars):
            if r is None or i is None:
                tops.append((x, 0, "pending" if r is None else "DNF"))
                continue
            c = S.STACK[r.name][0]
            ax.bar(x, r.t[i], width=0.6, color=c, edgecolor=c, zorder=Z["bar"])
            tops.append((x, r.t[i], f"{r.t[i]:.1f} h" if r.t[i] < 10 else f"{r.t[i]:.0f} h"))
        ax.set_xticks([0, 1], ["FwdLLM", S.SYS_NAME])
        ax.set_xlim(-0.6, 1.6)
        ax.set_ylim(0, max(h for _, h, _ in tops) * 1.15)
        lvl = f"{level * 100:.0f}%" if both else f"{level * 100:.1f}% (FwdLLM peak)"
        ax.set_xlabel(f"{title}\nto {lvl}")
        if note:
            S.status(ax, "PRELIM", note)
        ax.tops = tops
        ax.ratio = (bars[0][0].t[bars[0][1]] / ours.t[bars[1][1]]) if b and None not in (bars[0][1], bars[1][1]) else None
        ax.gain = gain_text(ours, base) if b else ""
    axes[0].set_ylabel("Time (hours)")
    S.freeze(fig)
    for ax in axes:
        ts = []
        for _ in range(4):  # widen ylim until every bar-top label fits
            for t in ts:
                t.remove()
            y0 = ax.get_ylim()[1] * 0.03
            ts = [ax.text(x, h + y0, s, ha="center", va="bottom", fontsize=FONT["annot"], zorder=Z["annot"]) for x, h, s in ax.tops]
            if ax.ratio:
                ts.append(ax.text(0.5, max(h for _, h, _ in ax.tops) + y0 * 6,
                                  rf"$\mathbf{{{ax.ratio:.1f}{{\times}}}}$ faster" + ("\n" + ax.gain if ax.gain else ""),
                                  ha="center", va="bottom", fontsize=FONT["annot"], zorder=Z["annot"]))
            fig.canvas.draw()
            top = max(ax.transData.inverted().transform(t.get_window_extent().corners()[1])[1] for t in ts)
            if top < ax.get_ylim()[1] * 0.98:
                break
            ax.set_ylim(0, top * 1.08)
    return S.save(fig, OUT, "E2E_speedup")


def e2e_comm():
    """Communication saved: FwdLLM GB / SwiFT GB to the fair level (target, else FwdLLM's peak), one bar per cell; GB at bar end."""
    rows = []
    for title, o, b, target, ylo, note in CELLS:
        ours, base = Run(o), Run(b)
        level, io, ib, both = compare(ours, base, target)
        rows.append((title, base.gb[ib] / ours.gb[io], ours.gb[io], base.gb[ib]))
    fig, ax = plt.subplots(figsize=(S.COL_W, 1.6))
    ys = range(len(rows))[::-1]  # first cell on top
    ax.barh(list(ys), [r for _, r, _, _ in rows], height=0.55, color=S.STACK[S.SYS_NAME][0], zorder=Z["bar"])
    gb = lambda g: f"{g:.0f}" if g >= 10 else f"{g:.1f}"  # noqa: E731
    for y, (title, r, go, gbase) in zip(ys, rows):
        ax.text(r + 0.15, y, rf"$\mathbf{{{r:.1f}{{\times}}}}$ ({gb(go)} vs {gb(gbase)} GB)", va="center", ha="left",
                fontsize=FONT["annot"], zorder=Z["annot"])
    ax.set_yticks(list(ys), [t for t, _, _, _ in rows])
    ax.set_xlim(0, max(r for _, r, _, _ in rows) * 2.1)  # room for the bar-end labels
    ax.set_xticks([x for x in ax.get_xticks() if x <= max(r for _, r, _, _ in rows) * 1.05])
    ax.grid(False)
    ax.grid(True, axis="x")
    ax.set_xlabel("Less Communication than FwdLLM (×)\nto reach FwdLLM's peak accuracy")
    return S.save(fig, OUT, "E2E_comm_to_target")


# ------------------------------------------------------- two-panel AG News comparisons -------------------------------------------------------
def pair_fig(name, ours_id, others, arms=False, ref_ids=(), target=0.86, ylo=78, callout=None):
    """Acc vs time (log); `others` drawn as ARM colors when arms=True, else as stacks.
    `callout`: run id compared against ours (arrow + ratio on the time panel)."""
    ours = Run(ours_id)
    refs = [Run(r) for r in ref_ids]
    oth = [Run(r) for r in others]
    fig, a1 = plt.subplots(figsize=(S.COL_W, 2.0))
    for j, r in list(enumerate(oth))[::-1]:
        curve(a1, r, r.t, ours, j if arms else None)
    for r in refs:
        curve(a1, r, r.t, ours)
    curve(a1, ours, ours.t, ours)
    target_line(a1, target)
    allr = [ours] + oth + refs
    top = max(max(r.acc) for r in allr) * 100
    a1.set_ylim(ylo, top + (top - ylo) * 0.3)  # empty band on top for the callout
    a1.yaxis.set_major_locator(plt.MultipleLocator(2 if top - ylo < 12 else 5))
    time_axis(a1, allr, ylo)
    a1.set_xlabel("Time (hours)")
    a1.set_ylabel("Test Accuracy (%)")
    hs = [handle(ours, ours)] + [handle(r, ours) for r in refs] + [handle(r, ours, j if arms else None) for j, r in enumerate(oth)]
    ours_label = {"tune_rho0": rf"{S.SYS_NAME} ($\rho_0$ 0.07)", "tune_commit_rule": f"{S.SYS_NAME} (N 50)",
                  "tune_rho_schedule": rf"{S.SYS_NAME} (exp $\rho$)"}.get(name)
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
    "E2E_acc_vs_time": e2e,
    "E2E_speedup": e2e_speedup,
    "E2E_comm_to_target": e2e_comm,
    "E2E_participation": e2e_participation,
    "E2E_participation_per_commit": e2e_participation_per_commit,
    "E2E_session_length": e2e_session_length,
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
