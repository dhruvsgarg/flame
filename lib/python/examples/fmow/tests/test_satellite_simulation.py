from types import SimpleNamespace

import pytest

from fmow.simulation import SatelliteSimulation


class FlameParent:
    def read_trainer_unavailability(self, trace=None, base_dir=None):
        return (trace, base_dir)


class Role(SatelliteSimulation, FlameParent):
    @property
    def satellite_registry_path(self):
        return self.registry

    def _avail_now(self):
        return 37.0


def make_role(tmp_path, role):
    path = tmp_path / "fmow.yaml"
    path.write_text("{}")
    instance = Role()
    instance._init_satellite_simulation(SimpleNamespace(hyperparameters=SimpleNamespace(
        fmow_config_path=str(path), satellite_index=2)), role)
    return instance


def test_roles_share_helpers_without_creating_an_independent_clock(tmp_path):
    agg = make_role(tmp_path, "aggregator")
    trainer = make_role(tmp_path, "trainer")
    trainer.simulated = True
    trainer._sim_send_ts = 12.0
    assert agg.satellite_now_s() == 37
    assert trainer.satellite_now_s() == 12
    assert trainer.satellite_index == 2
    assert not hasattr(trainer, "_vclock")
    assert trainer._satellite_contacts is None  # old runs do not require new artifacts
    assert agg.read_trainer_unavailability("syn_0", "base") == ("syn_0", "base")


def test_registry_identity_available_before_first_update(tmp_path):
    agg = make_role(tmp_path, "aggregator")
    agg.registry = tmp_path / "registry.yaml"
    agg.registry.write_text("trainers:\n  trainer_003: {task_id: endpoint-three}\n")
    assert agg.satellite_index_for_endpoint("endpoint-three") == 2
    with pytest.raises(KeyError):
        agg.satellite_index_for_endpoint("unknown")


def test_duplicate_registry_endpoint_rejected(tmp_path):
    agg = make_role(tmp_path, "aggregator")
    agg.registry = tmp_path / "registry.yaml"
    agg.registry.write_text("trainers:\n  trainer_001: {task_id: same}\n  trainer_002: {task_id: same}\n")
    with pytest.raises(ValueError, match="duplicate"):
        agg.satellite_index_for_endpoint("same")


def test_real_trainer_uses_aggregator_time_origin(tmp_path, monkeypatch):
    import fmow.simulation.runtime as runtime
    trainer = make_role(tmp_path, "trainer")
    trainer.simulated = False
    trainer.trainer_start_ts = 90
    trainer._agg_start_origin = 100
    monkeypatch.setattr(runtime.time, "time", lambda: 150)
    assert trainer.satellite_now_s() == 50
