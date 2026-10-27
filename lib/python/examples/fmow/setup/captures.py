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


def capture_footprint(satellite_ecef: np.ndarray, off_nadir_deg: float):
    """Ground footprint center and chord radius on the spherical Earth.

    The limb caps the footprint even when the angle includes the horizon.
    This chooses the near ray/sphere intersection, never the far side.
    """
    distance = np.linalg.norm(satellite_ecef, axis=-1)
    if np.any(distance <= EARTH_RADIUS_KM):
        raise ValueError("Satellite ECEF positions must be above the Earth's surface")
    horizon = np.arcsin(EARTH_RADIUS_KM / distance)
    angle = np.minimum(np.radians(off_nadir_deg), horizon)
    central_angle = np.arcsin(np.clip(distance / EARTH_RADIUS_KM * np.sin(angle), 0, 1)) - angle
    centers = satellite_ecef * (EARTH_RADIUS_KM / distance)[..., None]
    return centers, 2 * EARTH_RADIUS_KM * np.sin(central_angle / 2)

def schedule_image_capture(config: dict) -> None:
    fmow_root = Path(config.dataset.root_dir)
    leo_dir = Path(config.satellites.leo_dir)
    off_nadir_deg = config.capture.off_nadir_deg
    out_path = leo_dir / "captures.npz"

    print(f"[setup_fmow] Indexing training images...")
    fmow = pd.read_csv(fmow_root / "rgb_metadata.csv")
    train_data = (fmow.split == "train").to_numpy()
    image_idx = np.arange(len(fmow))[train_data]
    lat = fmow.lat.to_numpy()[train_data]
    lon = fmow.lon.to_numpy()[train_data]
    tree = cKDTree(latlon_to_ecef(lat, lon))

    geo = np.load(leo_dir / "geodetic.npz")
    time_s = geo["time_s"] # timesteps
    sat_names = geo["sat_names"] # satellites
    with np.load(leo_dir / "ecef.npz") as ecef:
        positions = ecef["ecef_km"]
        if not np.array_equal(ecef["time_s"], time_s) or not np.array_equal(ecef["sat_names"], sat_names):
            raise ValueError("ECEF and geodetic timestamps and satellite ordering must match")
    if positions.shape != (len(time_s), len(sat_names), 3) or not np.isfinite(positions).all():
        raise ValueError("ECEF positions must be finite with shape (timesteps, satellites, 3)")

    print(f"[setup_fmow] Simulating satellite image captures...")
    total_sats = len(sat_names)
    all_events, offsets = [], [0]

    for sat_pos in range(total_sats):
        ground_ecef, chord_radius = capture_footprint(positions[:, sat_pos], off_nadir_deg)

        already_captured = set()
        prev_visible = set()
        events = []

        for t in range(len(time_s)):
            nearby = tree.query_ball_point(ground_ecef[t], r=chord_radius[t] + 1e-8)
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
        sat_names=sat_names,
        off_nadir_deg=off_nadir_deg,
    )

    print(f"[setup_fmow] Done: {total_sats} satellites, {offsets[-1]} total capture events, off_nadir_deg={off_nadir_deg} -> {out_path}")
        


