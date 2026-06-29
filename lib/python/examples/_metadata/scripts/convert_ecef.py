from pathlib import Path

import numpy as np
from pyproj import Transformer

script_dir = Path(__file__).parent
leo_dir = script_dir.parent / "leo"
file_path = leo_dir / "ecef.npz"
out_path = leo_dir / "geodetic.npz"

data = np.load(file_path)
ecef_km = data["ecef_km"]

print("Transforming ecef to geodetic coordinates...")

transformer = Transformer.from_crs("epsg:4978", "epsg:4326", always_xy=True)
ecef_m = ecef_km * 1000
x = ecef_m[:, :, 0]
y = ecef_m[:, :, 1]
z = ecef_m[:, :, 2]
lon, lat, alt_m = transformer.transform(x, y, z)
coords = np.stack((lat, lon), axis=-1)

np.savez_compressed(
    out_path,
    coords=coords,
    time_s=data["time_s"],
    sat_names=data["sat_names"],
    epoch_iso=data["epoch_iso"],
)

print("Saved to geodetic.npz")
