import numpy as np
import pandas as pd
import pytest

from fmow.setup.captures import schedule_image_capture, latlon_to_ecef, capture_footprint, EARTH_RADIUS_KM
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
    positions = latlon_to_ecef(coords[:, 0, 0], coords[:, 0, 1]) * ((EARTH_RADIUS_KM + 550) / EARTH_RADIUS_KM)
    np.savez(leo_dir / "ecef.npz", ecef_km=positions[:, None, :], time_s=time_s, sat_names=sat_names)

    # Use mockup directories
    config = FMoWConfig(
        dataset=DatasetConfig(root_dir=str(data_dir)),
        satellites=SatellitesConfig(leo_dir=str(leo_dir)),
        capture=CaptureConfig(off_nadir_deg=1.0)
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
    assert float(out["off_nadir_deg"]) == 1.0


def test_off_nadir_geometry_and_horizon():
    r = EARTH_RADIUS_KM
    satellite = np.array([[r + 550, 0, 0]])
    center, radius = capture_footprint(satellite, 0)
    np.testing.assert_allclose(center, [[r, 0, 0]])
    assert radius[0] == 0
    _, radius = capture_footprint(satellite, 30)
    # Independent ray/sphere intersection for a ray 30 degrees off nadir.
    angle = np.radians(30)
    ray = np.array([-np.cos(angle), np.sin(angle), 0])
    s = satellite[0]
    distance = -np.dot(s, ray) - np.sqrt(np.dot(s, ray) ** 2 - (np.dot(s, s) - r*r))
    intersection = s + distance * ray
    assert np.linalg.norm(intersection - center[0]) == pytest.approx(radius[0])
    _, horizon_radius = capture_footprint(satellite, 90)
    horizon_angle = np.arccos(r / (r + 550))
    assert horizon_radius[0] == pytest.approx(2 * r * np.sin(horizon_angle / 2))
    assert horizon_radius[0] < 2 * r  # the far side cannot be captured


def test_positions_below_surface_rejected():
    with pytest.raises(ValueError, match="above"):
        capture_footprint(np.array([[EARTH_RADIUS_KM - 1, 0, 0]]), 30)

