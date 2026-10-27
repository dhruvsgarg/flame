"""Shared satellite helpers for role entrypoints, independent of FL algorithms."""
from pathlib import Path
import time

import yaml

from fmow.setup.config import load_config
from fmow.dependencies.satellite_availability import SatelliteAvailability
from .contacts import ContactSchedule


class SatelliteSimulation:
    """Opt-in helpers; the Flame parent retains its own role lifecycle.

    Aggregators own the global virtual clock. Trainers read assignment times.
    No dataset, model, optimizer, RF implementation, or worker pool lives here.
    """

    def _init_satellite_simulation(self, config, role):
        if role not in {"trainer", "aggregator"}:
            raise ValueError("Satellite role must be trainer or aggregator")
        self.config = config
        self.fmow_cfg = load_config(config.hyperparameters.fmow_config_path)
        self._satellite_role = role
        self._satellite_contacts = None
        self._satellite_availability = None
        self._satellite_endpoints = None
        if role == "trainer":
            self.satellite_index = int(config.hyperparameters.satellite_index)

    @property
    def satellite_contacts(self):
        if self._satellite_contacts is None:
            self._satellite_contacts = ContactSchedule.load(self.fmow_cfg.availability.contacts_path)
        return self._satellite_contacts

    @property
    def satellite_availability(self):
        if self._satellite_availability is None:
            self._satellite_availability = SatelliteAvailability(self.fmow_cfg.availability.trace_path)
        return self._satellite_availability

    @property
    def satellite_registry_path(self):
        return Path(__file__).resolve().parents[1] / "metadata/trainer_registry.yaml"

    def satellite_index_for_endpoint(self, endpoint):
        if self._satellite_endpoints is None:
            with self.satellite_registry_path.open() as f:
                trainers = yaml.safe_load(f)["trainers"]
            self._satellite_endpoints = {}
            for key, trainer in trainers.items():
                task_id = trainer["task_id"]
                if task_id in self._satellite_endpoints:
                    raise ValueError("Satellite registry contains duplicate endpoint identities")
                self._satellite_endpoints[task_id] = int(key.removeprefix("trainer_")) - 1
        return self._satellite_endpoints[endpoint]

    def read_trainer_unavailability(self, trace=None, base_dir=None):
        if trace != "satellite":
            return super().read_trainer_unavailability(trace, base_dir=base_dir)
        return self.satellite_availability.events_by_endpoint(self.satellite_registry_path)

    def satellite_now_s(self):
        if self._satellite_role == "aggregator":
            return self._avail_now()
        if self.simulated:
            return float(self._sim_send_ts) if self._sim_send_ts is not None else 0.0
        origin = self._agg_start_origin if self._agg_start_origin is not None else self.trainer_start_ts
        return time.time() - origin
