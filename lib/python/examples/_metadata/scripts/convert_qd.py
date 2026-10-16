from pathlib import Path

import apexpy
import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0
QD_HEIGHT_KM = 300.0

REGION_FILES = {
    "north": "NorthernEIA_PB.csv",
    "equatorial": "EquatorialQD_PB.csv",
    "south": "SouthernEIA_PB.csv",
}

def load_bubbles(data_dir: Path) -> pd.DataFrame:
    frames = []
    for region, filename in REGION_FILES.items():
        df = pd.read_csv(data_dir / filename)
        df["region"] = region
        frames.append(df)
    return pd.concat(frames, ignore_index=True)

def add_utc_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    hh = df["Scan0 time"] // 100
    mm = df["Scan0 time"] % 100
    base = pd.to_datetime(df["Year"], format="%Y") + pd.to_timedelta(df["Day of Year"] - 1, unit="D")
    df["time_utc"] = base + pd.to_timedelta(hh, unit="h") + pd.to_timedelta(mm, unit="m")
    return df

def convert_qd_to_geo(df: pd.DataFrame) -> pd.DataFrame:
    geo_lat = np.empty(len(df))
    geo_lon = np.empty(len(df))

    for year, idx in df.groupby("Year").groups.items():
        apex = apexpy.Apex(date=int(year))
        lat, lon = apex.convert(
            df.loc[idx, "QD Latitude"].to_numpy(),
            df.loc[idx, "QD Longitude"].to_numpy(),
            "qd", "geo", height=QD_HEIGHT_KM,
        )
        pos = df.index.get_indexer(idx)
        geo_lat[pos] = lat
        geo_lon[pos] = lon

    df["geo_lat"] = geo_lat
    df["geo_lon"] = geo_lon
    return df

def great_circle_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))

def add_radius_km(df: pd.DataFrame) -> pd.DataFrame:
    # Same approach the dataset's own methods paper uses for bubble
    # separations (Paper.md Sec 4.6): convert both edges to geographic
    # coordinates and take the great-circle distance between them, rather
    # than inventing a QD-degree-to-km scale factor. West/East FW75M are
    # QD longitudes at the bubble's own QD latitude, so that latitude is
    # reused for both edge conversions.
    radius_km = np.empty(len(df))

    for year, idx in df.groupby("Year").groups.items():
        apex = apexpy.Apex(date=int(year))
        west_lat, west_lon = apex.convert(
            df.loc[idx, "QD Latitude"].to_numpy(),
            df.loc[idx, "West FW75M"].to_numpy(),
            "qd", "geo", height=QD_HEIGHT_KM,
        )
        east_lat, east_lon = apex.convert(
            df.loc[idx, "QD Latitude"].to_numpy(),
            df.loc[idx, "East FW75M"].to_numpy(),
            "qd", "geo", height=QD_HEIGHT_KM,
        )
        pos = df.index.get_indexer(idx)
        radius_km[pos] = great_circle_km(west_lat, west_lon, east_lat, east_lon) / 2

    df["radius_km"] = radius_km
    return df

def convert_plasma_bubbles(data_dir: Path, out_path: Path) -> None:
    print(f"[convert_qd] Loading plasma bubble CSVs from {data_dir}...")
    bubbles = load_bubbles(data_dir)
    print(f"[convert_qd] Loaded {len(bubbles)} detections across {len(REGION_FILES)} regions")

    print(f"[convert_qd] Attaching UTC timestamps...")
    bubbles = add_utc_timestamp(bubbles)

    print(f"[convert_qd] Converting QD -> geographic coordinates via apexpy...")
    bubbles = convert_qd_to_geo(bubbles)

    print(f"[convert_qd] Computing bubble radius (great-circle distance between west/east FW75M edges)...")
    bubbles = add_radius_km(bubbles)

    time_utc_s = bubbles["time_utc"].astype("int64").to_numpy() // 10**9  # ns -> s since epoch

    np.savez(
        out_path,
        region=bubbles["region"].to_numpy(dtype=str),
        track_id=bubbles["Bubble Track Count"].to_numpy(),
        year=bubbles["Year"].to_numpy(),
        day_of_year=bubbles["Day of Year"].to_numpy(),
        time_utc_s=time_utc_s,
        qd_lat=bubbles["QD Latitude"].to_numpy(),
        qd_lon=bubbles["QD Longitude"].to_numpy(),
        geo_lat=bubbles["geo_lat"].to_numpy(),
        geo_lon=bubbles["geo_lon"].to_numpy(),
        radius_km=bubbles["radius_km"].to_numpy(),
        eia_radiance=bubbles["EIA Radiance"].to_numpy(),
        gaussian_coef0=bubbles["Gaussian coef0"].to_numpy(),
    )

    print(f"[convert_qd] Done: {len(bubbles)} detections -> {out_path}")

if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent.parent / "plasma_bubbles"
    out_path = data_dir / "pb_coordinates.npz"
    convert_plasma_bubbles(data_dir, out_path)
