# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Plot SNR and throughput loss vs. elevation for one satellite at one
ground station, computed directly from ecef.npz (no sweep CSV needed).

Runs against exactly one ground station per invocation -- pass
--ground-station or --lat/--lon (see _series.add_ground_station_args). To
compare stations, run this script once per station.

The satellite is a name (PLANET-SIM-00-00) or an integer index into
ecef.npz's sat_names.

Usage (from repo root):
    python lib/python/flame/link/scripts/plot_link_vs_elevation.py \\
        PLANET-SIM-00-00 --ground-station svalbard
    python lib/python/flame/link/scripts/plot_link_vs_elevation.py \\
        17 --lat 51.5 --lon -0.1 --name london
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _series import add_ground_station_args, make_config, resolve_ground_station, satellite_series

# Categorical slot 1 of the dataviz reference palette.
SERIES_COLOR = "#2a78d6"
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"


def resolve_satellite(sat, sat_names):
    if not sat.lstrip("-").isdigit():
        if sat not in sat_names:
            raise SystemExit(f"satellite {sat!r} not found in ecef.npz")
        return sat
    idx = int(sat)
    if not 0 <= idx < len(sat_names):
        raise SystemExit(f"satellite index {idx} out of range (0..{len(sat_names) - 1})")
    return str(sat_names[idx])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("satellite", help="satellite name (e.g. PLANET-SIM-00-00) or index")
    add_ground_station_args(p)
    p.add_argument("--ecef", default="lib/python/examples/_metadata/leo/ecef.npz")
    p.add_argument("--min-elevation-deg", type=float, default=5.0)
    p.add_argument("--out", help="output PNG (default: results/link_layer/plots/<gs>_<satellite>.png)")
    args = p.parse_args()
    gs_name, lat, lon = resolve_ground_station(args, p)

    d = np.load(args.ecef)
    ecef_km, time_s, sat_names = d["ecef_km"], d["time_s"], d["sat_names"]
    sat = resolve_satellite(args.satellite, sat_names)
    sat_idx = int(np.where(sat_names == sat)[0][0])

    cfg = make_config(lat, lon, args.min_elevation_deg)
    df = satellite_series(cfg, ecef_km[:, sat_idx, :], time_s)
    if df.empty:
        raise SystemExit(f"{sat} never above the horizon at {gs_name}")
    df = df.sort_values("elevation_deg")
    vis = df[df.visible]
    if vis.empty:
        raise SystemExit(f"{sat} never reaches {args.min_elevation_deg:g}° elevation at {gs_name}")

    plt.rcParams.update({"font.family": "sans-serif", "text.color": INK, "axes.labelcolor": INK_2})
    fig, (ax_snr, ax_loss) = plt.subplots(2, 1, figsize=(9, 7), sharex=True, facecolor=SURFACE)

    for ax in (ax_snr, ax_loss):
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=INK_2, length=0)
        ax.axvspan(0, args.min_elevation_deg, color=GRID, alpha=0.6, linewidth=0)
        ax.axvline(args.min_elevation_deg, color=INK_2, linewidth=1, linestyle=(0, (4, 3)))

    ax_snr.plot(vis.elevation_deg, vis.snr_db, color=SERIES_COLOR, linewidth=2)
    ax_loss.plot(df.elevation_deg, df.throughput_loss_pct, color=SERIES_COLOR, linewidth=2)

    ax_snr.set_ylabel("SNR (dB)")
    ax_loss.set_ylabel("Throughput loss vs. zenith (%)")
    ax_loss.set_xlabel("Elevation (deg)")
    ax_loss.set_ylim(-3, 105)
    ax_loss.set_xlim(0, df.elevation_deg.max() * 1.02)
    ax_loss.annotate(
        f"below {args.min_elevation_deg:g}° mask: link down, 100% loss",
        xy=(args.min_elevation_deg, 100),
        xytext=(args.min_elevation_deg + 0.6, 92),
        fontsize=9,
        color=INK_2,
    )
    fig.suptitle(f"{sat} @ {gs_name}: SNR and throughput loss vs. elevation", x=0.07, ha="left", fontsize=13, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out = Path(args.out) if args.out else Path("results/link_layer/plots") / f"{gs_name}_{sat}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
