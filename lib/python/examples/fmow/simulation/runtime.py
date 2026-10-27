"""Shared satellite helpers for role entrypoints, independent of FL algorithms."""
from pathlib import Path
from dataclasses import asdict
import importlib
import logging
import math
import time
import uuid

import cloudpickle
import yaml

from flame.mode.message import MessageType
from fmow.setup.config import load_config
from fmow.dependencies.satellite_availability import SatelliteAvailability
from .contacts import ContactSchedule
from .links import Direction, TransferRequest
from .transfers import TransferPlanner

logger = logging.getLogger(__name__)


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
        self._satellite_planner = None
        self._satellite_provider_spec = getattr(config.hyperparameters, "satellite_link_provider", None)
        self._satellite_assignments = {}
        self._satellite_assignment = None
        self._satellite_seen_assignments = set()
        self._satellite_failed_ends = set()
        self._satellite_uploads = {}
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
            if (not getattr(self, "_work_done", False) and self._satellite_assignment is not None
                    and self._satellite_assignment["complete"]):
                return self._satellite_assignment["model_arrival_s"]
            return float(self._sim_send_ts) if self._sim_send_ts is not None else 0.0
        origin = self._agg_start_origin if self._agg_start_origin is not None else self.trainer_start_ts
        return time.time() - origin

    def configure_satellite_link_provider(self, provider):
        """Inject any provider satisfying the pure sample(request) contract."""
        if self._satellite_role != "aggregator" or not self.simulated:
            raise ValueError("Link delivery integration currently supports simulated aggregators only")
        if not callable(getattr(provider, "sample", None)):
            raise TypeError("Satellite link provider must implement sample(request)")
        if any(getattr(self, name, None) is not None for name in ("link_runtime_uplink", "link_runtime_downlink")):
            raise ValueError("Configure one link timing system; legacy Flame link runtime is already enabled")
        limit = getattr(self.config.hyperparameters, "max_experiment_runtime_s", None)
        contacts = self.satellite_contacts.until(limit) if limit else self.satellite_contacts
        self._satellite_planner = TransferPlanner(contacts, provider)

    def _get_satellite_planner(self):
        if self._satellite_planner is None and self._satellite_provider_spec is not None:
            spec = self._satellite_provider_spec
            module, separator, name = spec["factory"].partition(":")
            if not separator:
                raise ValueError("satellite_link_provider.factory must be module:factory")
            factory = getattr(importlib.import_module(module), name)
            self.configure_satellite_link_provider(factory(**spec.get("kwargs", {})))
        return self._satellite_planner

    def _prepare_training_dispatch(self, channel, ends, msg):
        msg = super()._prepare_training_dispatch(channel, ends, msg)
        planner = self._get_satellite_planner()
        cohort = uuid.uuid4().hex
        manifest = {}
        # One shared manifest preserves serialize-once model distribution.
        nbytes = len(cloudpickle.dumps(msg[MessageType.WEIGHTS])) if planner else 0
        for end in ends:
            index = self.satellite_index_for_endpoint(end)
            assignment_id = f"{cohort}:{index}"
            start = float(msg.get(MessageType.SIM_SEND_TS, self.satellite_now_s()))
            outcome = planner.estimate(TransferRequest(assignment_id + ":model", index,
                Direction.MODEL_DOWNLOAD, start, nbytes)) if planner else None
            header = {
                "id": assignment_id, "satellite_index": index,
                "model_version": msg[MessageType.MODEL_VERSION],
                "model_arrival_s": outcome.arrival_s if outcome else start,
                "complete": outcome.complete if outcome else True,
                "download": asdict(outcome) if outcome else None,
            }
            manifest[end] = self._satellite_assignments[end] = header
        return {**msg, MessageType.SATELLITE_ASSIGNMENTS: manifest}

    def _assignment_header(self, msg):
        return msg.get(MessageType.SATELLITE_ASSIGNMENTS, {}).get(self.config.task_id)

    def _allow_model_reuse(self, msg):
        header = self._assignment_header(msg)
        return bool(header and header["id"] not in self._satellite_seen_assignments
                    and self._round == self._updates_returned_upto_round)

    def _assignment_has_model(self, msg):
        header = self._assignment_header(msg)
        return header is None or header["complete"]

    def _on_training_assignment(self, msg):
        super()._on_training_assignment(msg)
        header = self._assignment_header(msg)
        if header is None:
            return
        if header["satellite_index"] != self.satellite_index or header["model_version"] != self._round:
            raise ValueError("Satellite assignment identity/version mismatch")
        self._satellite_assignment = dict(header)
        self._satellite_seen_assignments.add(header["id"])

    def satellite_assignment_delivered(self):
        return self._satellite_assignment is None or self._satellite_assignment["complete"]

    def satellite_training_start_s(self):
        return self.satellite_now_s()

    def satellite_link_delivery_enabled(self):
        return self._satellite_assignment is not None and self._satellite_assignment["download"] is not None

    def _prepare_training_result(self, msg):
        msg = super()._prepare_training_result(msg)
        if self._satellite_assignment is not None:
            msg[MessageType.SATELLITE_ASSIGNMENT] = dict(self._satellite_assignment)
        return msg

    def _prepare_simulated_update(self, channel, end, msg, metadata):
        expected = self._satellite_assignments.get(end)
        header = msg.get(MessageType.SATELLITE_ASSIGNMENT)
        if expected is not None:
            if header is None or header["id"] != expected["id"] or msg.get(MessageType.MODEL_VERSION) != expected["model_version"]:
                raise ValueError(f"Unexpected satellite assignment result from {end}")
            if not expected["complete"]:
                self._satellite_failed_ends.add(end)
                self._avail_free_slot_ledger(channel, end)
                return None
            planner = self._get_satellite_planner()
            if planner is not None:
                compute_end = float(msg[MessageType.SIM_COMPLETION_TS])
                if not math.isfinite(compute_end) or compute_end < expected["model_arrival_s"]:
                    raise ValueError("Satellite compute completion must follow model arrival")
                if MessageType.WEIGHTS_BYTES in msg:
                    size = len(msg[MessageType.WEIGHTS_BYTES])
                elif MessageType.WEIGHTS in msg:
                    size = len(cloudpickle.dumps(msg[MessageType.WEIGHTS]))
                else:
                    size = 0
                outcome = planner.estimate(TransferRequest(expected["id"] + ":update",
                    expected["satellite_index"], Direction.UPDATE_UPLOAD, compute_end, size))
                self._satellite_uploads[end] = outcome
                logger.info("[SATELLITE_TRANSFER] end=%s assignment=%s outcome=%s", end, expected["id"], asdict(outcome))
                if not outcome.complete:
                    self._satellite_failed_ends.add(end)
                    self._avail_free_slot_ledger(channel, end)
                    return None
                msg[MessageType.SIM_COMPLETION_TS] = outcome.arrival_s
        return super()._prepare_simulated_update(channel, end, msg, metadata)

    def _sim_withhold_if_unavail(self, channel, end, sct, msgmd):
        outcome = self._satellite_uploads.get(end)
        expected = self._satellite_assignments.get(end)
        header = msgmd[0].get(MessageType.SATELLITE_ASSIGNMENT)
        if (outcome is not None and outcome.complete and expected is not None
                and header is not None and header["id"] == expected["id"]
                and outcome.transfer_id == expected["id"] + ":update"
                and sct == outcome.arrival_s):
            # The full station contacts already gated this payload. Eligibility
            # is only a selector projection, not a second delivery constraint.
            self._record_commit_belief(end, sct)
            return False
        return super()._sim_withhold_if_unavail(channel, end, sct, msgmd)

    def _validate_simulated_quorum(self, channel, buffered, first_k):
        super()._validate_simulated_quorum(channel, buffered, first_k)
        if self._get_satellite_planner() is not None and len(buffered) < first_k:
            raise RuntimeError(
                f"Selected cohort cannot meet aggregation goal: {len(buffered)}/{first_k} deliverable; "
                f"undeliverable endpoints={sorted(self._satellite_failed_ends)}. "
                "No replacement policy is configured."
            )

    def _extra_unavailable_trainers(self):
        return super()._extra_unavailable_trainers() | self._satellite_failed_ends

    def _next_avail_vclock(self):
        nxt = super()._next_avail_vclock()
        planner = self._get_satellite_planner()
        if planner is not None:
            horizon = planner.contacts.horizon_s
            return min(nxt, horizon) if nxt is not None else horizon
        return nxt
