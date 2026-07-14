import numpy as np
import pandas as pd

from fmow.setup.captures import schedule_image_capture, latlon_to_ecef
from fmow.setup.config import FMoWConfig, DatasetConfig, SatellitesConfig, CaptureConfig

def dist_km(lat1, lon1, lat2, lon2):
    p1 = latlon_to_ecef(np.array([lat1]), np.array([lon1]))[0]
    p2 = latlon_to_ecef(np.array([lat2]), np.array([lon2]))[0]
    return float(np.linalg.norm(p1-p2))

def test_schedule_image_capture(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    leo_dir = tmp_path / "leo"
    leo_dir.mkdir()

    # Create mockup FMoW dataset
    # Images at (0,0) and (0,5)
    rgb_metadata = pd.DataFrame({
        "split": ["train", "train", "test"],
        "lat": [0.0, 0.0, 0.0],
        "lon": [0.0, 5.0, 0.0]
    })
    rgb_metadata.to_csv(data_dir / "rgb_metadata.csv", index=False)

    # Make sure the two points are far enough apart
    radius_km = 5.0
    assert dist_km(0.0, 0.0, 0.0, 5.0) > radius_km * 10

    # Create mockup geodetic.npz
    # Satellite goes through (0.001,0.001) at 100s and (0,5.001) at 400s
    # Trajectory should be within image radius
    coords = np.array([
        [[0.001, 0.001]],
        [[80.0, 80.0]],
        [[0.001, 0.001]],
        [[0.0, 5.001]]
    ])
    time_s = np.array([100, 200, 300, 400])
    sat_names = np.array(["sat0"])
    np.savez(leo_dir / "geodetic.npz", coords=coords, time_s=time_s, sat_names=sat_names)

    # Use mockup directories
    config = FMoWConfig(
        dataset=DatasetConfig(root_dir=str(data_dir)),
        satellites=SatellitesConfig(leo_dir=str(leo_dir)),
        capture=CaptureConfig(radius=radius_km)
    )

    # Run schedule
    schedule_image_capture(config)

    # Assertions
    # Only two offsets (one beginning and one end for the one satellite)
    # Only two events (Image 0 at 100s and Image 1 at 400s)
    out = np.load(leo_dir / "captures.npz")
    assert list(out["sat_names"] == ["sat0"])
    assert out["offsets"].tolist() == [0, 2]
    assert out["events"][0].tolist() == [100, 0]
    assert out["events"][1].tolist() == [400, 1]
    assert 2 not in out["events"][:, 1]

