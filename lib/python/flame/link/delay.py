# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Trainer-side communication legs of one FL round.

A round is: model download (ground station -> satellite, RF uplink), local
compute, update upload (satellite -> ground station, RF downlink). Each leg
is a `LinkRuntime.transfer` evaluated at the time it actually starts, so the
upload sees the satellite where it is after compute, not at round start.

Active only when `linkLayer.enabled` and `linkLayer.affectsDelay` are both
true; otherwise `from_link_layer_dict` returns None and callers keep their
compute-only delay.
"""

import logging
from dataclasses import dataclass
from typing import Optional

from flame.link.runtime import LinkRuntime
from flame.link.transfer import TransferEstimate

logger = logging.getLogger(__name__)


@dataclass
class RoundLegs:
    download: TransferEstimate
    upload: TransferEstimate

    @property
    def complete(self) -> bool:
        return self.download.complete and self.upload.complete

    @property
    def comm_s(self) -> float:
        """Download + upload time; 0 if either leg never finds a path."""
        return self.download.total_s + self.upload.total_s if self.complete else 0.0

    def as_telemetry(self) -> dict:
        return {
            **self.download.as_telemetry("link_download"),
            **self.upload.as_telemetry("link_upload"),
            "link_comm_s": self.comm_s,
        }


class LinkDelayModel:
    def __init__(self, uplink: LinkRuntime, downlink: LinkRuntime):
        self.uplink = uplink  # ground station -> satellite (model download)
        self.downlink = downlink  # satellite -> ground station (update upload)
        self.cfg = uplink.cfg
        self._warned_unreachable = set()

    @classmethod
    def from_link_layer_dict(cls, raw: Optional[dict]) -> Optional["LinkDelayModel"]:
        if not raw or not raw.get("enabled", False) or not raw.get("affectsDelay", False):
            return None
        uplink = LinkRuntime.from_link_layer_dict(raw, direction="uplink")
        downlink = LinkRuntime(uplink.cfg, uplink.ecef_km, direction="downlink")
        return cls(uplink, downlink)

    def payload_bytes(self, model) -> int:
        """Configured transfer size, else the model's in-memory weight size."""
        if self.cfg.transfer.payload_bytes:
            return int(self.cfg.transfer.payload_bytes)
        return int(sum(v.numel() * v.element_size() for v in model.state_dict().values()))

    def round_legs(
        self, satellite_index: int, start_s: float, compute_s: float, nbytes: int
    ) -> RoundLegs:
        download = self.uplink.transfer(satellite_index, start_s, nbytes)
        upload_start = start_s + (download.total_s if download.complete else 0.0) + compute_s
        upload = self.downlink.transfer(satellite_index, upload_start, nbytes)
        legs = RoundLegs(download, upload)
        if not legs.complete and satellite_index not in self._warned_unreachable:
            self._warned_unreachable.add(satellite_index)
            logger.warning(
                f"[LINK_UNREACHABLE] satellite {satellite_index} never has a "
                f"{self.cfg.routing} path to the ground station; comm delay set to 0."
            )
        return legs
