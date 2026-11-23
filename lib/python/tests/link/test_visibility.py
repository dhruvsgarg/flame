# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
import numpy as np

from flame.link.budget import compute_elevation_deg
from flame.link.config import LinkLayerConfig
from flame.link.propagation import geodetic_to_ecef_km
from flame.link.visibility import invisible_ends, registry_satellite_map, visibility_matrix


def _gs_ecef(cfg):
    gs = cfg.ground_station
    return geodetic_to_ecef_km(gs.lat_deg, gs.lon_deg, gs.alt_m, cfg.geometry.earth_radius_km)


def test_hides_only_ends_not_visible_now():
    vis = np.array([[True, False, True], [False, False, True]])
    sat = {"a": 0, "b": 1, "c": 2}
    assert invisible_ends(vis, sat, sat.get, 0) == {"b"}
    assert invisible_ends(vis, sat, sat.get, 1) == {"a", "b"}


def test_never_visible_satellite_is_always_hidden():
    vis = np.array([[True, False]] * 3)
    for t in range(3):
        assert "never" in invisible_ends(vis, ["up", "never"], {"up": 0, "never": 1}.get, t)


def test_unknown_satellite_stays_eligible():
    assert invisible_ends(np.array([[False]]), ["x"], lambda e: None, 0) == set()


def test_time_wraps_past_trajectory_end():
    vis = np.array([[True], [False], [True]])  # n=3
    assert invisible_ends(vis, ["a"], {"a": 0}.get, 4) == {"a"}  # 4 % 3 == 1
    assert invisible_ends(vis, ["a"], {"a": 0}.get, 5) == set()  # 5 % 3 == 2


def test_visibility_matrix_overhead_vs_far_side():
    cfg = LinkLayerConfig.from_dict({})
    g = _gs_ecef(cfg)
    overhead = g * (1 + 550.0 / np.linalg.norm(g))  # 550 km straight up
    far_side = -overhead
    ecef = np.array([[overhead, far_side]])  # [1 step, 2 sats, 3]
    assert visibility_matrix(ecef, cfg).tolist() == [[True, False]]


def test_visibility_matrix_matches_link_budget_elevation():
    cfg = LinkLayerConfig.from_dict({})
    g = _gs_ecef(cfg)
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(4000, 3))  # ~0.5% of the sky is above a 25 deg mask
    ecef = (pts / np.linalg.norm(pts, axis=1, keepdims=True) * (np.linalg.norm(g) + 550.0))[None]
    vis = visibility_matrix(ecef, cfg)[0]
    expected = [compute_elevation_deg(p, g) >= cfg.min_elevation_deg for p in ecef[0]]
    assert vis.tolist() == expected
    assert 0 < vis.sum() < len(vis)  # the sample has both visible and hidden satellites


def test_mask_override_changes_result():
    cfg = LinkLayerConfig.from_dict({})
    g = _gs_ecef(cfg)
    up = g / np.linalg.norm(g)
    east = np.cross(up, [0.0, 0.0, 1.0]); east /= np.linalg.norm(east)
    # a satellite ~15 degrees above the horizon: seen with a 10 deg mask, not a 25 deg one
    el = np.radians(15.0)
    sat = g + 1500.0 * (np.cos(el) * east + np.sin(el) * up)
    ecef = np.array([[sat]])
    assert visibility_matrix(ecef, cfg, 10.0).tolist() == [[True]]
    assert visibility_matrix(ecef, cfg, 25.0).tolist() == [[False]]


def test_registry_maps_task_id_to_trainer_id_minus_one(tmp_path):
    f = tmp_path / "reg.yaml"
    f.write_text(
        "trainers:\n"
        "  trainer_001: {trainer_id: 1, task_id: aaa}\n"
        "  trainer_200: {trainer_id: 200, task_id: bbb}\n"
    )
    assert registry_satellite_map(str(f)) == {"aaa": 0, "bbb": 199}


def test_missing_registry_gives_empty_map(tmp_path):
    assert registry_satellite_map(str(tmp_path / "nope.yaml")) == {}


def test_config_parses_new_options():
    cfg = LinkLayerConfig.from_dict({
        "routing": "direct",
        "direct": {"visibleOnly": True},
        "trainerRegistryPath": "x.yaml",
        "transfer": {"unreachableCommS": 86400},
    })
    assert cfg.direct.visible_only is True
    assert cfg.trainer_registry_path == "x.yaml"
    assert cfg.transfer.unreachable_comm_s == 86400.0
    default = LinkLayerConfig.from_dict({})
    assert default.direct.visible_only is False
    assert default.transfer.unreachable_comm_s == 0.0
