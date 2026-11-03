# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Per-satellite-pass downlink link budget for one ground station, over
every satellite in ecef.npz.

A "pass" is one contiguous stretch of a satellite's track at or above
--min-elevation-deg (AOS to LOS at the elevation mask) -- the usable
contact window. Writes one row per pass: the AOS/LOS window, duration, and
best/worst/mean SNR, RTT and throughput loss over that pass, plus the mean
FSPL / atmospheric / ionospheric loss terms.

Runs against exactly one ground station per invocation -- pass
--ground-station or --lat/--lon (see _series.add_ground_station_args). To
compare stations, run this script once per station.

Usage (from repo root):
    python lib/python/flame/link/scripts/link_budget_sweep.py \\
        --ground-station svalbard --out-dir results/link_layer
    python lib/python/flame/link/scripts/link_budget_sweep.py \\
        --lat 51.5 --lon -0.1 --name london --out-dir results/link_layer
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from _series import add_ground_station_args, make_config, resolve_ground_station, satellite_series
from flame.link.budget import compute_link_budget


def detect_passes(df, satellite):
    """Group one satellite's per-timestep rows (elevation > 0, one row per
    second) into contiguous visible (elevation >= min_elevation_deg)
    passes, and summarize each into one row.
    """
    time_s = df.time_s.to_numpy()
    vis = df.visible.to_numpy()
    # A new pass starts at a gap in time_s or a not-visible -> visible edge.
    gap = np.diff(time_s, prepend=time_s[0] - 2) > 1
    edge = np.diff(vis.astype(int), prepend=1 - vis[0]) != 0
    pass_id = np.cumsum(gap | edge)

    rows = []
    for pid, g in df.assign(_pass=pass_id)[vis].groupby(pass_id[vis]):
        rows.append(
            {
                "satellite": satellite,
                "pass_id": int(pid),
                "start_time_s": float(g.time_s.min()),
                "end_time_s": float(g.time_s.max()),
                "duration_s": float(g.time_s.max() - g.time_s.min() + 1),
                "num_samples": int(len(g)),
                "max_elevation_deg": float(g.elevation_deg.max()),
                "min_slant_range_km": float(g.slant_range_km.min()),
                "min_rtt_ms": float(g.rtt_ms.min()),
                "max_rtt_ms": float(g.rtt_ms.max()),
                "mean_fspl_db": float(g.fspl_db.mean()),
                "mean_atmospheric_loss_db": float(g.atmospheric_loss_db.mean()),
                "ionospheric_loss_db": float(g.ionospheric_loss_db.iloc[0]),
                "max_snr_db": float(g.snr_db.max()),
                "min_snr_db": float(g.snr_db.min()),
                "mean_snr_db": float(g.snr_db.mean()),
                "max_throughput_mbps": float(g.throughput_mbps.max()),
                "min_throughput_mbps": float(g.throughput_mbps.min()),
                "mean_throughput_mbps": float(g.throughput_mbps.mean()),
                "min_throughput_loss_pct": float(g.throughput_loss_pct.min()),
                "max_throughput_loss_pct": float(g.throughput_loss_pct.max()),
                "mean_throughput_loss_pct": float(g.throughput_loss_pct.mean()),
            }
        )
    return pd.DataFrame(rows)


def _self_check(passes_df, cfg, ecef_km, sat_names, n=25):
    """Spot-check pass boundaries against the scalar compute_link_budget."""
    if passes_df.empty:
        return
    idx_of = {name: i for i, name in enumerate(sat_names)}
    for _, r in passes_df.sample(min(n, len(passes_df)), random_state=0).iterrows():
        s = idx_of[r.satellite]
        for t in (int(r.start_time_s), int(r.end_time_s)):
            ref = compute_link_budget(ecef_km[t, s], cfg, "downlink")
            assert ref.visible, (r.satellite, t, "pass boundary not visible")
        assert r.max_elevation_deg >= cfg.min_elevation_deg - 1e-6
        assert r.duration_s >= 1


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_ground_station_args(p)
    p.add_argument("--ecef", default="lib/python/examples/_metadata/leo/ecef.npz")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--min-elevation-deg", type=float, default=5.0)
    p.add_argument(
        "--write-timesteps",
        action="store_true",
        help="also write the full per-second CSV (below-mask rows included, 100%% loss)",
    )
    args = p.parse_args()
    name, lat, lon = resolve_ground_station(args, p)

    d = np.load(args.ecef)
    ecef_km, time_s, sat_names = d["ecef_km"], d["time_s"], d["sat_names"]
    cfg = make_config(lat, lon, args.min_elevation_deg)

    pass_frames, ts_frames = [], []
    for s, sat_name in enumerate(sat_names):
        df = satellite_series(cfg, ecef_km[:, s, :], time_s)
        if df.empty:
            continue
        passes = detect_passes(df, sat_name)
        if not passes.empty:
            passes.insert(0, "ground_station", name)
            pass_frames.append(passes)
        if args.write_timesteps:
            df = df.copy()
            df.insert(0, "satellite", sat_name)
            df.insert(0, "ground_station", name)
            ts_frames.append(df)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    passes_df = pd.concat(pass_frames, ignore_index=True) if pass_frames else pd.DataFrame()
    _self_check(passes_df, cfg, ecef_km, sat_names)

    passes_path = out / f"link_budget_passes_{name}.csv"
    passes_df.to_csv(passes_path, index=False, float_format="%.6g")
    n_sats = passes_df.satellite.nunique() if not passes_df.empty else 0
    print(f"{name}: {len(passes_df)} passes across {n_sats} satellites -> {passes_path}")

    if args.write_timesteps:
        ts_df = pd.concat(ts_frames, ignore_index=True) if ts_frames else pd.DataFrame()
        ts_path = out / f"link_budget_timesteps_{name}.csv"
        ts_df.to_csv(ts_path, index=False, float_format="%.6g")
        print(f"{name}: {len(ts_df)} per-second rows -> {ts_path}")


if __name__ == "__main__":
    main()
