import argparse
import sys
import numpy as np
import yaml

from pathlib import Path

EARTH_RADIUS_KM = 6371.0
FMOW_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FMOW_DIR.parent))
from fmow.simulation.contacts import ContactSchedule
METADATA_DIR = FMOW_DIR / "metadata"
LEO_DIR = METADATA_DIR / "leo"
OUTPUT_DIR = METADATA_DIR / "availability_traces" 
FILENAME = "satellite_traces.yaml"
GS_FILE = "ground_stations.yaml"

def geo_to_ecef(lat_deg: float, lon_deg: float) -> np.ndarray:
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    return np.array([
        EARTH_RADIUS_KM * np.cos(lat) * np.cos(lon),
        EARTH_RADIUS_KM * np.cos(lat) * np.sin(lon),
        EARTH_RADIUS_KM * np.sin(lat)
    ])

def find_runs(available: np.ndarray):
    # Find runs of availability and record timesteps where they start/end
    padded = np.concatenate(([False], available, [False]))
    diff = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1) - 1
    return list(zip(starts, ends))

def elevation_angles_deg(station_ecef: np.ndarray, sat_ecef: np.ndarray) -> np.ndarray:
    # Calculate availability for every satellite based on a single ground station
    d = sat_ecef - station_ecef
    d_norm = np.linalg.norm(d, axis=-1)
    up = station_ecef / np.linalg.norm(station_ecef)
    sin_elev = np.sum(d*up, axis=-1) / d_norm
    return np.degrees(np.arcsin(np.clip(sin_elev, -1.0, 1.0)))

def find_satellite_events(
    visible: np.ndarray, min_window: float
) -> np.ndarray:
    events = np.zeros_like(visible)

    for sat_idx in range(visible.shape[1]):
        for start, end in find_runs(visible[:, sat_idx]):
            # Find all times which are visible for at 
            # least min_window
            avail_end = end - int(min_window)
            if avail_end < start:
                continue
            events[start:avail_end+1, sat_idx] = True

    return events
    

def convert_to_trace(
    visible: np.ndarray
) -> list:
    events = [(0, "UN_AVL")]
    for start, end in find_runs(visible):
        events.append((int(start), "AVL_TRAIN"))
        events.append((int(end + 1), "UN_AVL"))


    if len(events) > 1 and events[1][0] == events[0][0]:
        events = events[1:]
    return events

def build_contact_schedule(ground_stations, sat_ecef, time_s, satellite_names, min_elevation):
    visibility = {}
    for name, station in ground_stations.items():
        ecef = geo_to_ecef(station["lat"], station["lon"])
        visibility[name] = elevation_angles_deg(ecef, sat_ecef) >= min_elevation
    return ContactSchedule.from_visibility(time_s, satellite_names, visibility)


def project_schedule(schedule, time_s, min_window):
    events = schedule.availability_events(min_window)
    counts = np.zeros(len(time_s), dtype=int)
    for trace in events.values():
        for (start, state), (end, _) in zip(trace, trace[1:]):
            if state == "AVL_TRAIN":
                counts += (time_s >= start) & (time_s < end)
    stats = {
        "num_satellites": len(schedule.satellite_names),
        "eligible_min": int(counts.min()), "eligible_max": int(counts.max()),
        "eligible_mean": float(counts.mean()),
        "ghost_satellites": sum(not any(state == "AVL_TRAIN" for _, state in t) for t in events.values()),
    }
    return events, stats


def generate_trace(ground_stations, sat_ecef, min_elevation, min_window):
    # Compatibility entrypoint for callers with one-second orbital samples.
    times = np.arange(sat_ecef.shape[0], dtype=float)
    schedule = build_contact_schedule(ground_stations, sat_ecef, times,
                                      [str(i) for i in range(sat_ecef.shape[1])], min_elevation)
    return project_schedule(schedule, times, min_window)


def write_satellite_availability(
    gs_file: Path, ecef_path: Path, out_file: Path,
    elevation_angle: float, min_window: float, contacts_path: Path | None = None,
) -> None:
    with open(gs_file) as f:
        gs_data = yaml.safe_load(f)
    ground_stations = gs_data["stations"]

    ecef = np.load(ecef_path)
    sat_ecef, time_s = ecef["ecef_km"], ecef["time_s"]

    schedule = build_contact_schedule(ground_stations, sat_ecef, time_s, ecef["sat_names"], elevation_angle)
    satellite_events, stats = project_schedule(schedule, time_s, min_window)
    schedule.save(contacts_path or out_file.with_name("contact_schedule.npz"))

    print(f"Ground stations: {len(ground_stations)}, satellites: {stats['num_satellites']}")
    print(f"Eligible at once: min={stats['eligible_min']}, mean={stats['eligible_mean']}, max={stats['eligible_max']}")
    print(f"Satellites with zero usable passes: {stats['ghost_satellites']}")

    trainers = {}
    for sat_idx, event in enumerate(satellite_events.values()):
        events = satellite_events.get(sat_idx, [(0, "UN_AVL")])
        trainer_key = f"trainer_{(sat_idx + 1):03d}"
        trainers[trainer_key] = [list(e) for e in events]
        
    output = {
        "description": (
            f"Geometric availability trace: {len(ground_stations)} ground stations, "
            f"min_elevation_deg={elevation_angle}, min_window_s={min_window}"
        ),
        "generated_from": {
            # "ground_stations": str(gs_file),
            "min_elevation_angle": elevation_angle,
            "min_window": min_window,
        },
        "trainers": trainers,
    }
    
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        yaml.safe_dump(output, f, sort_keys=False, default_flow_style=None)

    print(f"Wrote geometric availability trace to {out_file}")
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ecef-file", type=Path, default=LEO_DIR / "ecef.npz")
    parser.add_argument("--gs-file", type=Path, default=LEO_DIR / GS_FILE)
    parser.add_argument("--elevation-angle", type=float, default=10.0)
    parser.add_argument("--min-window", type=float, default=90.0)
    parser.add_argument("--dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--filename", type=Path, default=FILENAME)
    args = parser.parse_args()

    write_satellite_availability(
        args.gs_file, args.ecef_file, args.dir / args.filename,
        args.elevation_angle, args.min_window,
    )

if __name__ == "__main__":
    main()
