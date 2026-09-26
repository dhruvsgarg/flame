#!/usr/bin/env python3
# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""FX-N13 figures from oracle_misselection.py outputs (<run>/analysis/*.csv).

Arms are `label=run_dir`, or `--campaign DIR` picks every P7 (B) and P7o (B+oracle) row, real and sim.
Writes to --out (default: <campaign>/P7_figures):
  belief_vs_true[_<B>].png  believed-vs-true Spearman per checkpoint round, per arm (one per baseline)
  pick_rank[_<B>].png       true-utility percentile of each pick (1 = best pickable), per arm
  utility_heatmap_<arm>.png  true utility, trainers x round
  utility_trajectories.png / loss_trajectories.png   4 widest-ranging trainers of the first arm
  summary.txt              per-arm means
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics as st

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # categorical slots 1-4, fixed order
BLUES = LinearSegmentedColormap.from_list(
    "blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281", "#0d366b"])
INK, MUTED, GRID = "#1f1f1e", "#6b6a66", "#e4e3df"


def _rows(path: str) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def _by_round(rows: list[dict], key: str) -> tuple[list[int], list[float]]:
    acc: dict[int, list[float]] = {}
    for r in rows:
        if r.get(key) not in ("", None):
            acc.setdefault(int(r["round"]), []).append(float(r[key]))
    xs = sorted(acc)
    return xs, [st.mean(acc[x]) for x in xs]


def campaign_arms(root: str) -> list[tuple[str, str]]:
    arms = []
    for phase, suffix in (("P7", ""), ("P7o", "+oracle")):
        p = os.path.join(root, phase, "summary.tsv")
        if not os.path.exists(p):
            continue
        for row in _rows_tsv(p):
            for leg in ("real", "sim"):
                d = row.get(f"{leg}_dir", "")
                if d and os.path.isdir(d):
                    arms.append((f"{row['baseline']}{suffix} {leg}", d))
    return arms


def by_baseline(arms) -> dict[str, list[tuple[str, str]]]:
    """Group arms on the label's baseline ("felix+oracle sim" -> felix)."""
    groups: dict[str, list[tuple[str, str]]] = {}
    for label, d in arms:
        groups.setdefault(label.split()[0].split("+")[0], []).append((label, d))
    return groups


def _rows_tsv(p: str) -> list[dict]:
    with open(p) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(color=GRID, lw=0.8)
    ax.tick_params(colors=MUTED)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)


def line_per_arm(arms, key, title, ylabel, out, ylim=None):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (label, d) in enumerate(arms):
        xs, ys = _by_round(_rows(os.path.join(d, "analysis", "oracle_misselection.csv")), key)
        if not xs:
            continue
        ax.plot(xs, ys, color=SERIES[i % len(SERIES)], lw=2, marker="o", ms=4, label=label)
        ax.annotate(label, (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK)
    _style(ax, title, "aggregator round (model version)", ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def heatmap(label, d, out):
    rows = _rows(os.path.join(d, "analysis", "oracle_true_utility.csv"))
    rounds = sorted({int(r["round"]) for r in rows})
    last = {r["trainer"]: float(r["true_util"]) for r in rows if int(r["round"]) == rounds[-1]}
    trainers = sorted(last, key=last.get)
    idx_r = {x: i for i, x in enumerate(rounds)}
    idx_t = {t: i for i, t in enumerate(trainers)}
    grid = [[float("nan")] * len(rounds) for _ in trainers]
    for r in rows:
        if r["trainer"] in idx_t:
            grid[idx_t[r["trainer"]]][idx_r[int(r["round"])]] = float(r["true_util"])
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(grid, aspect="auto", cmap=BLUES, interpolation="nearest",
                   extent=(rounds[0], rounds[-1], 0, len(trainers)), origin="lower")
    fig.colorbar(im, ax=ax, label="true utility")
    _style(ax, f"True utility per trainer — {label}", "aggregator round",
           "trainer (sorted by final utility)")
    ax.grid(False)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def trajectories(label, d, out, n=4, key="true_util", ylabel="true utility"):
    rows = _rows(os.path.join(d, "analysis", "oracle_true_utility.csv"))
    per: dict[str, list[tuple[float, float]]] = {}
    for r in rows:
        per.setdefault(r["trainer"], []).append((float(r["stream_s"]), float(r[key])))
    spread = {t: max(v for _, v in pts) - min(v for _, v in pts) for t, pts in per.items()}
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, t in enumerate(sorted(spread, key=spread.get, reverse=True)[:n]):
        pts = sorted(per[t])
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=SERIES[i], lw=2, marker="o",
                ms=4, label=f"…{t[-4:]}")
    _style(ax, f"{ylabel.capitalize()} over stream time — {label}", "stream time (s)", ylabel)
    ax.legend(frameon=False, fontsize=8, title="trainer")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def summary(arms) -> str:
    lines = [f"{'arm':<22} {'n_sel':>6} {'spearman':>9} {'pick_rank':>9} {'regret':>7} {'hit':>6}"]
    for label, d in arms:
        rows = _rows(os.path.join(d, "analysis", "oracle_misselection.csv"))

        def m(k):
            v = [float(r[k]) for r in rows if r[k] != ""]
            return f"{st.mean(v):.3f}" if v else "-"
        lines.append(f"{label:<22} {len(rows):>6} {m('spearman'):>9} {m('rank_pct'):>9} "
                     f"{m('regret_rel'):>7} {m('hit_rate'):>6}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arms", nargs="*", help="label=run_dir")
    ap.add_argument("--campaign")
    ap.add_argument("--out")
    a = ap.parse_args()
    arms = [tuple(x.split("=", 1)) for x in a.arms]
    if a.campaign:
        arms += campaign_arms(a.campaign)
    arms = [(l, d) for l, d in arms if os.path.exists(os.path.join(d, "analysis", "oracle_misselection.csv"))]
    if not arms:
        raise SystemExit("no arm has analysis/oracle_misselection.csv; run oracle_misselection.py first")
    out = a.out or os.path.join(a.campaign or ".", "P7_figures")
    os.makedirs(out, exist_ok=True)
    groups = by_baseline(arms)
    for b, g in groups.items():
        tag = f"_{b}" if len(groups) > 1 else ""
        line_per_arm(g, "spearman", f"Believed vs true utility (Spearman) — {b}", "rank correlation",
                     os.path.join(out, f"belief_vs_true{tag}.png"), ylim=(-1.05, 1.05))
        line_per_arm(g, "rank_pct", f"Pick quality: true-utility percentile of each pick — {b}",
                     "percentile (1 = best pickable)", os.path.join(out, f"pick_rank{tag}.png"), ylim=(0, 1.05))
    for label, d in arms:
        heatmap(label, d, os.path.join(out, f"utility_heatmap_{label.replace(' ', '_').replace('+', '')}.png"))
    trajectories(*arms[0], os.path.join(out, "utility_trajectories.png"))
    trajectories(*arms[0], os.path.join(out, "loss_trajectories.png"), key="rms_loss",
                 ylabel="per-sample RMS loss")
    s = summary(arms)
    with open(os.path.join(out, "summary.txt"), "w") as f:
        f.write(s + "\n")
    print(s)
    print(f"figures: {out}")


if __name__ == "__main__":
    main()
