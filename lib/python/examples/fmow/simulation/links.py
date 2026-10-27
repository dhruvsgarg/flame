"""Provider-neutral link inputs and outputs. No RF implementation lives here."""
from dataclasses import dataclass
from enum import Enum
import math
from typing import Protocol


class Direction(str, Enum):
    MODEL_DOWNLOAD = "ground_to_satellite"
    UPDATE_UPLOAD = "satellite_to_ground"


@dataclass(frozen=True)
class LinkRequest:
    satellite_index: int
    station_id: str
    direction: Direction
    time_s: float


@dataclass(frozen=True)
class LinkState:
    """Piecewise constant potential rate, not shared bandwidth allocation.

    Units are bits/second and seconds. State applies up to valid_until_s.
    Providers may look up a precomputed trace or calculate this state on demand.
    """
    rate_bps: float
    propagation_s: float
    valid_until_s: float

    def __post_init__(self):
        if not all(math.isfinite(v) for v in (self.rate_bps, self.propagation_s, self.valid_until_s)):
            raise ValueError("Link state values must be finite")
        if self.rate_bps < 0 or self.propagation_s < 0:
            raise ValueError("Link rate and propagation delay must be nonnegative")


class LinkProvider(Protocol):
    def sample(self, request: LinkRequest) -> LinkState:
        """Return state without sleeping, selecting participants or changing clocks."""
        ...


@dataclass(frozen=True)
class TransferRequest:
    transfer_id: str
    satellite_index: int
    direction: Direction
    start_s: float
    payload_bytes: int

    def __post_init__(self):
        if not self.transfer_id or not math.isfinite(self.start_s) or self.start_s < 0:
            raise ValueError("Transfer requires an identity and a finite nonnegative start")
        if not isinstance(self.payload_bytes, int) or self.payload_bytes < 0:
            raise ValueError("Payload size must be a nonnegative byte count")
        Direction(self.direction)


@dataclass(frozen=True)
class TransferOutcome:
    transfer_id: str
    complete: bool
    station_id: str | None
    start_s: float
    transmission_start_s: float | None
    transmission_end_s: float | None
    arrival_s: float | None
    transmitted_bytes: float
    remaining_bytes: float
    wait_s: float
    transmit_s: float
    reason: str
