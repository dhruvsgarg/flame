#!/usr/bin/env python3
# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Preview `plotlib/baselines.py`'s color/marker/linestyle registry on synthetic
data — line, scatter, and bar charts styled exactly like the real paper figures,
but with no telemetry I/O. Eyeball a palette change (a new hue, a re-ordered
ramp, a renamed baseline) in seconds instead of waiting on `make_paper_figs.py`
against real run dirs (~4 min per manifest, streaming multi-GB telemetry).

Data is synthetic and deterministic (seeded per baseline key) — it exists only
to give each baseline a distinguishable-but-plausible curve/scatter/bar so
color/marker/style choices can be judged the way they'll actually be read in
the paper, not to represent any real experiment.

    python preview_palette.py                          # every registered baseline
    python preview_palette.py --baselines fwdllm,fluxtune
    python preview_palette.py --manifest figs_main_v2.yaml   # that manifest's set
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from plotlib import baselines as B          # noqa: E402
from plotlib import style as S              # noqa: E402


def _load_manifest_keys(path: str) -> list[str]:
    if not os.path.exists(path):
        print(f"  [preview-palette] manifest not found: {path}", file=sys.stderr)
        return []
    import yaml
    with open(path) as fh:
        manifest = yaml.safe_load(fh) or {}
    return list((manifest.get("runs") or {}).keys())


# --------------------------------------------------------------------------- #
# synthetic per-baseline data — deterministic (seeded on the key), no telemetry
# --------------------------------------------------------------------------- #
def _dummy_curve(key: str, n: int = 40):
    """Saturating learning-curve-shaped line: rises to a key-dependent plateau
    with a bit of noise, so overlapping curves still look plausible."""
    rng = np.random.default_rng(abs(hash(key)) % (2**32))
    x = np.linspace(0, 10, n)
    plateau = 70 + (abs(hash(key)) % 20)
    rate = 0.4 + (abs(hash(key + "r")) % 100) / 250.0
    y = plateau * (1 - np.exp(-rate * x)) + rng.normal(0, 1.2, n)
    return x, y


def _dummy_scatter(key: str, n: int = 30):
    rng = np.random.default_rng(abs(hash(key + "s")) % (2**32))
    cx = 2 + (abs(hash(key + "cx")) % 60) / 10.0
    cy = 2 + (abs(hash(key + "cy")) % 60) / 10.0
    return rng.normal(cx, 0.8, n), rng.normal(cy, 0.8, n)


def _dummy_bar(key: str) -> float:
    rng = np.random.default_rng(abs(hash(key + "b")) % (2**32))
    return float(rng.uniform(1.0, 9.0))


# --------------------------------------------------------------------------- #
# chart builders — reuse the same rcParams/column geometry/legend-emphasis
# pipeline as the real figures, so what you see here is what the paper shows
# --------------------------------------------------------------------------- #
def _finish(fig, ax, title):
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        leg = ax.legend(loc="best")
        B.apply_legend_emphasis(leg)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(title, fontsize=8, color="#888888", style="italic")
    return fig


def preview_line(keys):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=S.column_figsize(1.0, S.DEFAULT_ASPECT))
    for k in keys:
        st = B.style_for(k)
        x, y = _dummy_curve(k)
        ax.plot(x, y, color=st.color, linestyle=st.linestyle, marker=st.marker,
                markevery=0.15, markersize=4.5, label=st.label,
                zorder=3 + st.order, **st.marker_fill_kwargs())
    ax.set_xlabel("synthetic x")
    ax.set_ylabel("synthetic y")
    return _finish(fig, ax, "line — dummy data, palette preview only")


def preview_scatter(keys):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=S.column_figsize(1.0, S.DEFAULT_ASPECT))
    for k in keys:
        st = B.style_for(k)
        x, y = _dummy_scatter(k)
        ax.scatter(x, y, marker=st.marker, s=22, label=st.label,
                   zorder=3 + st.order, **st.scatter_fill_kwargs())
    ax.set_xlabel("synthetic x")
    ax.set_ylabel("synthetic y")
    return _finish(fig, ax, "scatter — dummy data, palette preview only")


def preview_bar(keys):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=S.column_figsize(1.0, 0.72))
    xs = np.arange(len(keys))
    vals = [_dummy_bar(k) for k in keys]
    styles = [B.style_for(k) for k in keys]
    ax.bar(xs, vals, 0.62, color=[s.color for s in styles],
           hatch=[s.bar_hatch() for s in styles],
           edgecolor="white", linewidth=0.6, zorder=3)
    ax.set_xticks(xs)
    ax.set_xticklabels([B.style_for(k).label for k in keys], rotation=35, ha="right")
    for t, k in zip(ax.get_xticklabels(), keys):
        w, s = B.style_for(k).legend_font()
        t.set_fontweight(w); t.set_fontstyle(s)
    for xi, v in zip(xs, vals):
        ax.text(xi, v, f"{v:.1f}", ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("synthetic value")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("bar — dummy data, palette preview only", fontsize=8,
                color="#888888", style="italic")
    return fig


CHART_BUILDERS = {
    "line": preview_line,
    "scatter": preview_scatter,
    "bar": preview_bar,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baselines", default=None,
                    help="comma list of baseline keys (default: every key in "
                         "plotlib.baselines.BASELINES)")
    ap.add_argument("--manifest", default=None,
                    help="baseline set = this figs manifest's `runs:` keys, "
                         "instead of the full registry (e.g. figs_main_v2.yaml)")
    ap.add_argument("--charts", default="line,scatter,bar",
                    help="comma list of chart types (default: all three)")
    ap.add_argument("--out-root", default=os.path.join(HERE, "palette_preview"),
                    help="output dir (flat, overwritten each render)")
    args = ap.parse_args()

    if args.baselines:
        keys = [k.strip() for k in args.baselines.split(",") if k.strip()]
    elif args.manifest:
        keys = _load_manifest_keys(_resolve_manifest(args.manifest))
        if not keys:
            print("  [preview-palette] manifest had no baselines — falling back "
                  "to the full registry", file=sys.stderr)
            keys = list(B.BASELINES.keys())
    else:
        keys = list(B.BASELINES.keys())
    keys = B.ordered(keys)
    if not keys:
        print("  [preview-palette] no baselines to preview", file=sys.stderr)
        return 1

    which = [c.strip() for c in args.charts.split(",") if c.strip()]
    S.use_paper_style()
    out_dir = S.render_outdir(args.out_root)
    written = []
    for name in which:
        builder = CHART_BUILDERS.get(name)
        if builder is None:
            print(f"  [preview-palette] unknown chart '{name}' — skipping ("
                  f"choices: {list(CHART_BUILDERS)})", file=sys.stderr)
            continue
        fig = builder(keys)
        S.save_pdf(fig, out_dir, f"palette_{name}")
        written.append(name)

    print(f"  [preview-palette] previewed {len(keys)} baseline(s): {', '.join(keys)}")
    print(f"  wrote {len(written)} chart(s) → {out_dir}")
    return 0


def _resolve_manifest(path: str) -> str:
    if os.path.isabs(path) or os.path.exists(path):
        return path
    return os.path.join(HERE, path)


if __name__ == "__main__":
    sys.exit(main())
