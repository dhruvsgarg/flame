from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

EARTH_RADIUS_KM = 6371.0

def latlon_to_ecef(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    x = EARTH_RADIUS_KM * np.cos(lat) * np.cos(lon)
    y = EARTH_RADIUS_KM * np.cos(lat) * np.sin(lon)
    z = EARTH_RADIUS_KM * np.sin(lat)
    return np.stack([x, y, z], axis=1)

def schedule_image_capture(config: dict) -> None:
    fmow_root = Path(config.dataset.root_dir)
    leo_dir = Path(config.satellites.leo_dir)
    radius_km = config.capture.radius
    out_path = leo_dir / "captures.npz"

    print(f"[setup_fmow] Indexing training images...")
    fmow = pd.read_csv(fmow_root / "rgb_metadata.csv")
    train_data = (fmow.split == "train").to_numpy()
    image_idx = np.arange(len(fmow))[train_data]
    lat = fmow.lat.to_numpy()[train_data]
    lon = fmow.lon.to_numpy()[train_data]
    tree = cKDTree(latlon_to_ecef(lat, lon))

    geo = np.load(leo_dir / "geodetic.npz")
    coords = geo["coords"] # (timesteps, satellites, corrdinates[lat, lon])
    time_s = geo["time_s"] # timesteps
    sat_names = geo["sat_names"] # satellites

    print(f"[setup_fmow] Simulating satellite image captures...")
    total_sats = len(sat_names)
    all_events, offsets = [], [0]

    for sat_pos in range(total_sats):
        lat = coords[:, sat_pos, 0]
        lon = coords[:, sat_pos, 1]
        ground_ecef = latlon_to_ecef(lat, lon)

        already_captured = set()
        prev_visible = set()
        events = []

        for t in range(len(time_s)):
            nearby = tree.query_ball_point(ground_ecef[t], r=radius_km)
            visible_images = set(int(i) for i in nearby)

            for i in (visible_images - prev_visible):
                curr_image = int(image_idx[i])
                if curr_image in already_captured:
                    continue
                events.append((int(time_s[t]), curr_image))
                already_captured.add(curr_image)

            prev_visible = visible_images
        
        all_events.extend(events)
        offsets.append(offsets[-1] + len(events))

        print(f"[setup_fmow] Satellite {sat_pos}: {len(events)} capture events")

    events_arr = np.array(all_events, dtype=np.int64).reshape(-1, 2)
    offsets_arr = np.array(offsets, dtype=np.int64)

    np.savez(
        out_path,
        events=events_arr,
        offsets=offsets_arr,
        sat_names=sat_names
    )

    print(f"[setup_fmow] Done: {total_sats} satellites, {offsets[-1]} total capture events, radius={radius_km}km -> {out_path}")
        


