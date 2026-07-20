import pytest
import numpy as np

from fmow.setup.captures import latlon_to_ecef, EARTH_RADIUS_KM

# Test lat/lon to ecef conversions
def test_latlon_to_ecef_equator():
    lat = np.array([0.0])
    lon = np.array([0.0])

    ecef = latlon_to_ecef(lat, lon)
    
    assert np.allclose(ecef[0, 0], EARTH_RADIUS_KM)
    assert np.allclose(ecef[0, 1], 0.0)
    assert np.allclose(ecef[0, 2], 0.0)

def test_latlon_to_ecef_north_pole():
    lat = np.array([90.0])
    lon = np.array([0.0])

    ecef = latlon_to_ecef(lat, lon)
    
    assert np.allclose(ecef[0, 0], 0.0)
    assert np.allclose(ecef[0, 1], 0.0)
    assert np.allclose(ecef[0, 2], EARTH_RADIUS_KM)


