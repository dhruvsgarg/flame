"""Fig 2 (Motivation, fluxtune_paper_outline.md §12): five panels, one PDF each, from per-run digests in ../data.

Usage: python make_motiv_figs.py [--only motiv_collapse ...]   -> ../figs/background/<name>.{pdf,png}
Digests: code/digest_motiv.py on the node that holds the run (shepherd: q*, r*, run1; jayne: sm_fwdllm), copied to ../data.

Runs (ids stay out of the image): q1 constant rho 0.07 (paper_pitch_20261029b), q4 FwdLLM shipped (paper_pitch_20261029a),
r4a FwdLLM lr 0.03 (paper_runs_20261029), run1 FwdLLM DistilBERT (p1_distil_20261025), sm_fwdllm FwdLLM 1.7B (jayne
experiments/run_20261028_182751_fwdllm_yahoo_n60_smoke_syn_0_sim), q2 q3 q5a q5b r2s r3a r3b r8a r8b (law only).

Definitions: measured Phi_t = ||theta_t|| / ||theta_0|| (initial trainable norm, run_meta); rho_t = ||dtheta_t|| / ||theta_{t-1}||
(logged update norm over the pre-step norm; the logged `rho` divides by the post-step norm and under-reads large steps);
predicted Phi_t = prod_{s<=t} sqrt(1 + rho_s^2). Accuracy line = 5-eval trailing mean; raw evals as faint dots.
"""
import argparse
import json
import math
import os
import statistics as st

import numpy as np
import yaml

import ft_style as S
from ft_style import FONT, INK, LINE, MARKER, Z, plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
OUT = os.path.join(HERE, "..", "figs", "background")
TRACE = os.path.join(HERE, "..", "..", "_metadata", "availability_traces", "mobiperf_traces.yaml")

W, H = 1.35, 1.75  # one fifth of the 6.75 in text width; legend above the axes
# 8 / 7 pt (guideline §1) does not fit a 1.35 in panel: labels 7 pt, ticks / legend / annotations 6.5 pt (floor 6)
FONT = dict(FONT, label=7, tick=6.5, legend=6.5, annot=6.5)
plt.rcParams.update({"axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5})
GATED = S.STACK["FwdLLM"][0]
GATED3 = "#999933"  # variant of the gated system: own color, parent's marker, dashed (guideline §2)
NOBOUND = S.ARM[0][0]
CONTEXT = "#8a8985"  # de-emphasized context series (Fig 2b only)
LBL = {"q4": "Gated system", "r4a": "Gated, 3× lr", "q1": "Relative step,\nno bound"}
LAW_RUNS = "q1 q2 q3 q4 q5a q5b sm_fwdllm r2s r3a r3b r8a r8b r4a".split()
GATED_RUNS = {"q4", "r4a", "sm_fwdllm", "run1"}


def load(r):
    with open(os.path.join(DATA, f"{r}.json")) as fh:
        return json.load(fh)


def series(d):
    """Per server update: commit, measured Phi, rho (pre-step), predicted Phi."""
    th0 = d["th0"] or d["su"][0][1]
    prev, lsum, out = th0, 0.0, []
    for s in d["su"]:
        rho = s[2] / prev
        lsum += math.log1p(rho * rho)
        out.append((s[0], s[1] / th0, rho, math.exp(0.5 * lsum)))
        prev = s[1]
    return np.array(out)


def acc_vs_commit(d):
    """(commit, raw acc %, 5-eval trailing mean %)."""
    c = np.array([e[0] for e in d["evals"]], float)
    a = np.array([e[1] for e in d["evals"]], float) * 100
    sm = np.array([a[max(0, i - 4):i + 1].mean() for i in range(len(a))])
    return c, a, sm


def phi_at(ser, commits):
    return np.interp(commits, ser[:, 0], ser[:, 1])


def roll_median(y, k):
    y = np.asarray(y, float)
    return np.array([np.median(y[max(0, i - k // 2):i + k // 2 + 1]) for i in range(len(y))])


def new_fig():
    fig, ax = plt.subplots(figsize=(W, H))
    return fig, ax


def log_ticks(ax, axis, ticks, fmt=lambda v: f"{v:g}"):
    from matplotlib.ticker import FixedLocator, NullLocator
    a = ax.xaxis if axis == "x" else ax.yaxis
    a.set_major_locator(FixedLocator(ticks))
    a.set_minor_locator(NullLocator())
    (ax.set_xticklabels if axis == "x" else ax.set_yticklabels)([fmt(v) for v in ticks])


def legend_top(fig, handles, ncol=1):
    fig.legend(handles=handles, loc="outside upper center", ncol=ncol, fontsize=FONT["legend"], handlelength=1.6,
               columnspacing=0.6, borderaxespad=0.0)


def finish(fig, name, meta):
    fig.canvas.draw()
    path = S.save_meta(fig, OUT, name, meta) if hasattr(S, "save_meta") else None
    if path is None:
        for t, what in S.overlaps(fig):
            print(f"  OVERLAP {name}: {t!r} touches {what!r}", flush=True)
        os.makedirs(OUT, exist_ok=True)
        fig.savefig(os.path.join(OUT, f"{name}.pdf"), metadata={"Subject": meta}, transparent=True)
        fig.savefig(os.path.join(OUT, f"{name}.png"), transparent=False)
        plt.close(fig)
    print(f"  -> {os.path.join(OUT, name)}.pdf")


# ---------------------------------------------------------------- 2a
def motiv_collapse():
    fig, ax = new_fig()
    sty = {"q1": (NOBOUND, "-", "o"), "q4": (GATED, "-", "o"), "r4a": (GATED3, "--", "o")}
    for r in ("r4a", "q4", "q1"):
        d = load(r)
        ser, (c, a, sm) = series(d), acc_vs_commit(d)
        x = phi_at(ser, c)
        col, ls, _ = sty[r]
        ax.plot(x, a, ".", ms=1.2, color=col, alpha=0.25, zorder=Z["fill"], label="_raw")
        ax.plot(x, sm, ls, color=col, lw=LINE["width"], zorder=Z["line"], label=LBL[r].replace("\n", " "))
        i = int(np.argmax(sm))
        print(f"  {r}: peak {sm[i]:.1f}% at Phi {x[i]:.2f} (c{int(c[i])}); last {sm[-1]:.1f}% at Phi {x[-1]:.2f}"
              + "".join(f"; Phi {p}: {np.interp(p, x, sm):.1f}%" for p in (5, 8, 14.6) if x[-1] >= p))
    ax.axhline(25, color=INK, lw=LINE["ref"], ls=":", zorder=Z["fill"])
    ax.axvline(3.7, color=INK, lw=LINE["ref"], ls=":", zorder=Z["fill"])
    ax.set_xscale("log")
    ax.set_xlim(1, 40)
    ax.set_ylim(10, 90)
    log_ticks(ax, "x", [1, 2, 4, 8, 16, 32])
    ax.set_yticks([25, 50, 75])
    ax.set_xlabel("Weight growth Φ (log)")
    ax.set_ylabel("Test accuracy (%)")
    from matplotlib.lines import Line2D
    legend_top(fig, [Line2D([], [], ls=sty[r][1], color=sty[r][0], lw=LINE["width"], label=LBL[r].replace("\n", " "))
                     for r in ("q4", "r4a", "q1")] + [Line2D([], [], ls=":", color=INK, lw=LINE["ref"], label="Φ ≈ 3.7")])
    S.freeze(fig)
    S.place(ax, "Chance", [(x, 0.165, "center", "top") for x in (0.74, 0.70, 0.78, 0.60, 0.50)])
    finish(fig, "motiv_collapse", "runs q1 (paper_pitch_20261029b), q4 (paper_pitch_20261029a), r4a (paper_runs_20261029)")


# ---------------------------------------------------------------- 2b
def motiv_growth():
    fig, ax = new_fig()
    worst, rows = {}, []
    for r in LAW_RUNS:
        ser = series(load(r))
        pred, meas = ser[:, 3], ser[:, 1]
        k = np.r_[0:len(ser):10, len(ser) - 1]
        col = {"q1": NOBOUND, "q4": GATED, "r4a": GATED3}.get(r, CONTEXT)
        ls = "--" if r == "r4a" else "-"
        hero = r in ("q1", "q4", "r4a")
        ax.plot(pred[k], meas[k], ls, color=col, lw=LINE["width"] if hero else LINE["muted"], alpha=1 if hero else 0.5,
                zorder=Z["line"] if hero else Z["bar"], label=f"_{r}")
        mk = "^" if r in GATED_RUNS else "o"
        ax.plot(pred[-1], meas[-1], mk, color=col, ms=MARKER["size"], mfc=col if r != "r4a" else "white",
                mew=0.8, zorder=Z["hero"], label=f"_{r}_end")
        err = np.abs(np.log(meas / pred))
        worst[r] = err.max()
        rows.append((r, pred[-1], meas[-1], 100 * (meas[-1] / pred[-1] - 1), err.max(), ser[:, 2].max(), ser[:, 2].sum()))
    lo, hi = 1, 40
    ax.plot([lo, hi], [lo, hi], "--", color=INK, lw=LINE["ref"], zorder=Z["fill"], label="_diag")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    for a in "xy":
        log_ticks(ax, a, [1, 2, 4, 8, 16, 32])
    ax.set_xlabel("Predicted Φ (log)")
    ax.set_ylabel("Measured Φ (log)")
    ax.grid(False)
    w = 100 * (math.exp(max(worst.values())) - 1)
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], ls=ls, color=c, lw=LINE["width"], marker=m, mfc=f, ms=MARKER["size"], label=l) for ls, c, m, f, l in
          (("-", NOBOUND, "o", NOBOUND, "Relative step, no bound"), ("-", GATED, "^", GATED, "Gated system"),
           ("--", GATED3, "^", "white", "Gated, 3× lr"), ("-", CONTEXT, "o", CONTEXT, "Other runs"))]
    legend_top(fig, hs, ncol=1)
    S.freeze(fig)
    S.place(ax, f"{len(LAW_RUNS)} runs within\n$\\mathbf{{{w:.1f}\\%}}$", [(0.96, 0.04, "right", "bottom"), (0.96, 0.25, "right", "bottom")])
    print("  run       pred   meas   end err  max|ln|  max rho  sum rho")
    for r, p, m, e, x, mr, sr in rows:
        print(f"  {r:9s} {p:6.2f} {m:6.2f} {e:+6.2f}%  {x:.4f}  {mr:.3f}  {sr:6.1f}")
    q1 = next(row for row in rows if row[0] == "q1")
    print(f"  worst |ln| q-runs {max(v for r, v in worst.items() if r.startswith('q')):.4f}; all {max(worst.values()):.4f} ({w:.1f}%); "
          f"q1 end |ln| / sum rho = {abs(math.log(q1[2] / q1[1])):.4f} / {q1[6]:.0f} = {abs(math.log(q1[2] / q1[1])) / q1[6]:.1e} (mean cos bound)")
    finish(fig, "motiv_growth", "runs " + " ".join(LAW_RUNS))


# ---------------------------------------------------------------- 2c
def motiv_gate():
    d = load("q4")
    g = np.array([[x[0], x[1], x[2], x[3]] for x in d["gate"]], float)
    c = g[:, 0]
    norm = lambda y: roll_median(y / np.median(y[:10]), 25)  # noqa: E731
    var, gsq, it = norm(g[:, 1]), norm(g[:, 2]), norm(g[:, 3])
    ratio = roll_median((g[:, 1] / g[:, 2]) / np.median(g[:10, 1] / g[:10, 2]), 25)
    fig, ax = new_fig()
    ax.plot(c, var, "-", color=GATED, lw=LINE["width"], zorder=Z["line"], label="Gate variance")
    ax.plot(c, gsq, "-", color=INK, lw=LINE["muted"], zorder=Z["line"], label="Squared update norm")
    ax.plot(c, it, ":", color=GATED, lw=LINE["width"], zorder=Z["line"], label="Iterations per update")
    ax.plot(c, ratio, "--", color=CONTEXT, lw=LINE["muted"], zorder=Z["line"], label="Variance ÷ squared norm")
    ax.set_yscale("log")
    ax.set_ylim(0.9, 12)  # data 0.99-10.5x; ratio line sits just above the floor
    log_ticks(ax, "y", [1, 2, 5, 10], lambda v: f"{v:g}×")
    ax.set_xlim(0, c[-1])
    ax.set_xticks([0, 200, 400, 600])
    ax.set_xlabel("Server update")
    ax.set_ylabel("Relative to start")
    legend_top(fig, ax.get_lines(), ncol=1)
    S.freeze(fig)
    k = max(1, len(g) // 10)
    m = lambda y: float(np.median(y))  # noqa: E731
    print(f"  first-iter var {m(g[:k, 1]):.2f} -> {m(g[-k:, 1]):.2f} (x{m(g[-k:, 1]) / m(g[:k, 1]):.2f}); "
          f"||g|| {m(g[:k, 2]) ** .5:.0f} -> {m(g[-k:, 2]) ** .5:.0f} (||g||^2 x{m(g[-k:, 2]) / m(g[:k, 2]):.2f}); "
          f"iterations {m(g[:k, 3]):.0f} -> {m(g[len(g) // 2 - k // 2:len(g) // 2 + k // 2 + 1, 3]):.0f} -> {m(g[-k:, 3]):.0f}; "
          f"ratio range {ratio.min():.2f}-{ratio.max():.2f}")
    fp = g[:, 3] * 10 * 1 * 2  # 10 clients x 1 picked perturbation (top cosine, chosen on CPU) x 2 passes
    print(f"  forward passes per server update (iterations x 10 clients x 1 picked perturbation x 2): median {m(fp):.0f}, "
          f"first 50 {m(fp[:50]):.0f}, last 50 {m(fp[-50:]):.0f}; buffer of 50 client updates: {50 * 10 * 2}")
    cos = [s[8] for s in d["su"] if s[8] is not None]
    h = len(cos) // 2
    print(f"  cos audits n={len(cos)} median {np.median(cos):.4f}; first half {np.median(cos[:h]):.4f}, second half {np.median(cos[h:]):.4f}")
    finish(fig, "motiv_gate", "run q4 (paper_pitch_20261029a)")


# ---------------------------------------------------------------- 2d
def motiv_rho():
    fig, ax = new_fig()
    sty = {"q4": (GATED, "-"), "r4a": (GATED3, "--")}
    for r in ("q4", "r4a"):
        d = load(r)
        ser, (c, a, sm) = series(d), acc_vs_commit(d)
        rho = roll_median(ser[:, 2], 10)
        col, ls = sty[r]
        ax.plot(ser[:, 0], rho, ls, color=col, lw=LINE["width"], zorder=Z["line"], label=LBL[r])
        cp = c[int(np.argmax(sm))]
        ax.plot(cp, np.interp(cp, ser[:, 0], rho), "o", color=col, ms=MARKER["size"], mfc="white", mew=0.9, zorder=Z["hero"],
                label="_peak")
        at = lambda p: float(np.interp(p, ser[:, 1], ser[:, 2]))  # noqa: E731
        print(f"  {r}: rho first {ser[0, 2]:.3f}, last {ser[-1, 2]:.3f} (rolled {rho[0]:.3f} -> {rho[-1]:.3f}); "
              f"acc peak c{int(cp)}; rho at Phi 2: {at(2):.3f}, Phi 4: {at(4):.3f}")
        if r == "r4a":
            ch = next(int(c[i]) for i in range(len(c)) if c[i] > cp and sm[i] <= 25.0)
            ax.plot(ch, np.interp(ch, ser[:, 0], rho), "x", color=INK, ms=MARKER["size"] + 0.5, mew=1.0, zorder=Z["hero"],
                    label="_chance")
            print(f"  r4a reaches chance (smoothed <= 25.0%) at c{ch}, Phi {np.interp(ch, ser[:, 0], ser[:, 1]):.2f}")
    ax.set_yscale("log")
    ax.set_ylim(0.01, 1)
    log_ticks(ax, "y", [0.01, 0.1, 1])
    ax.set_xlim(0, 605)
    ax.set_xticks([0, 200, 400, 600])
    ax.set_xlabel("Server update")
    ax.set_ylabel("Relative step ρ (log)")
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], ls=sty[r][1], color=sty[r][0], lw=LINE["width"], label=LBL[r]) for r in ("q4", "r4a")]
    hs += [Line2D([], [], ls="", marker="o", color=INK, mfc="white", ms=MARKER["size"], label="Best accuracy"),
           Line2D([], [], ls="", marker="x", color=INK, ms=MARKER["size"], label="Reaches chance")]
    legend_top(fig, hs, ncol=1)
    S.freeze(fig)
    finish(fig, "motiv_rho", "runs q4 (paper_pitch_20261029a), r4a (paper_runs_20261029)")


# ---------------------------------------------------------------- 2e
def sessions_min():
    with open(TRACE) as fh:
        tr = yaml.safe_load(fh)["traces"]
    out = []
    for dev in tr.values():
        st_ = dev.get("states_2st") or []
        for (t0, s0), (t1, _) in zip(st_, st_[1:]):
            if s0 == "AVL_TRAIN" and t1 > t0:
                out.append((t1 - t0) / 60)
    return np.array(out)


def holds_min(d):
    vc = [s[9] for s in d["su"]]
    return np.diff([0.0] + vc) / 60


def cdf(ax, x, **kw):
    """Explicit step vertices (not drawstyle), so the overlap check sees the drawn path."""
    x = np.sort(x)
    y = np.arange(1, len(x) + 1) / len(x)
    return ax.plot(np.repeat(x, 2)[1:], np.repeat(y, 2)[:-1], **kw)


def motiv_hold():
    fig, ax = new_fig()
    sess = sessions_min()
    xs = np.sort(sess)
    ax.fill_between(xs, np.arange(1, len(xs) + 1) / len(xs), step="post", color=S.FILL, lw=0, zorder=Z["fill"], gid="region",
                    label="_sessions")
    runs = {"run1": ("DistilBERT", "-"), "sm_fwdllm": ("1.7B", "--")}
    for r, (name, ls) in runs.items():
        d = load(r)
        h = holds_min(d)
        cdf(ax, h, color=GATED, ls=ls, lw=LINE["width"], zorder=Z["line"], label=name)
        med = float(np.median(h))
        ax.plot(med, 0.5, "o", color=GATED, ms=MARKER["size"] - 0.5, zorder=Z["hero"], label="_med")
        task = float(np.median([x[4] for x in d["rounds"] if x[4]])) / 60
        ax.plot([task, task], [0, 0.08], "-", color=INK, lw=LINE["width"], zorder=Z["hero"], label="_tick")
        iters = float(np.median([s[7] for s in d["su"] if s[7]]))
        print(f"  {name}: hold median {med:.1f} min, p90 {np.percentile(h, 90):.0f} min; one task {task * 60:.0f} s; "
              f"sessions shorter than the median hold {100 * (sess < med).mean():.0f}%; "
              f"active compute per phone ~{iters:.0f} iterations x {task:.1f} min = {iters * task:.0f} min vs hold {med:.0f} min")
    print(f"  sessions: n {len(sess)}, median {np.median(sess):.1f} min, {100 * (sess <= 10).mean():.0f}% <= 10 min")
    ax.set_xscale("log")
    ax.set_xlim(1 / 60, 300)
    log_ticks(ax, "x", [1 / 60, 1, 60], lambda v: {1 / 60: "1 s", 1: "1 min", 60: "1 h"}[v])
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel("Time (log)")
    ax.set_ylabel("CDF")
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    hs = [Line2D([], [], ls=runs[r][1], color=GATED, lw=LINE["width"], label=f"Hold, {runs[r][0]}") for r in runs]
    hs += [Patch(facecolor=S.FILL, edgecolor="none", label="Phone sessions"),
           Line2D([], [], ls="", color=INK, marker="|", ms=5, mew=LINE["width"], label="One task (compute)")]
    legend_top(fig, hs, ncol=1)
    S.freeze(fig)
    for r, (name, _) in runs.items():
        task = float(np.median([x[4] for x in load(r)["rounds"] if x[4]])) / 60
        x = math.log(task / (1 / 60)) / math.log(300 * 60)
        S.place(ax, name, [(x + dx, y, "left", "bottom") for dx in (0.03, 0.05) for y in (0.10, 0.14)], rotation=90)
    finish(fig, "motiv_hold", "runs run1 (p1_distil_20261025/bench/P1_fwdllm), sm_fwdllm (jayne 20261028_182751); trace mobiperf states_2st")


FIGS = {"motiv_collapse": motiv_collapse, "motiv_growth": motiv_growth, "motiv_gate": motiv_gate, "motiv_rho": motiv_rho,
        "motiv_hold": motiv_hold}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--only", nargs="*", default=list(FIGS))
    for n in p.parse_args().only:
        print(n)
        FIGS[n]()
