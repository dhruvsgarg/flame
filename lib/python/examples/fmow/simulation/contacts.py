"""Finite station-specific contacts and their legacy eligibility projection."""
from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Contact:
    station_id: str
    start_s: float
    end_s: float


class ContactSchedule:
    def __init__(self, satellite_names, station_names, start_s, horizon_s, contacts):
        self.satellite_names = tuple(str(x) for x in satellite_names)
        self.station_names = tuple(str(x) for x in station_names)
        self.start_s, self.horizon_s = float(start_s), float(horizon_s)
        if not (math.isfinite(self.start_s) and math.isfinite(self.horizon_s)
                and 0 <= self.start_s < self.horizon_s):
            raise ValueError("Contact schedule requires a finite increasing time range")
        if len(set(self.satellite_names)) != len(self.satellite_names) or len(set(self.station_names)) != len(self.station_names):
            raise ValueError("Satellite and station names must be unique")
        self._contacts = {}
        for index in range(len(self.satellite_names)):
            windows = tuple(sorted(contacts.get(index, ()), key=lambda c: (c.start_s, c.station_id)))
            previous = {}
            for window in windows:
                if (window.station_id not in self.station_names
                        or not self.start_s <= window.start_s < window.end_s <= self.horizon_s
                        or window.start_s < previous.get(window.station_id, self.start_s)):
                    raise ValueError("Invalid, overlapping, or out-of-range contact")
                previous[window.station_id] = window.end_s
            self._contacts[index] = windows
        if set(contacts) - set(self._contacts):
            raise ValueError("Contact references an unknown satellite")

    def for_satellite(self, index):
        return self._contacts[index]

    def at(self, index, time_s):
        return tuple(c for c in self.for_satellite(index) if c.start_s <= time_s < c.end_s)

    def availability_events(self, min_window_s=0):
        """Shorten each station's start-eligibility window, then union stations.

        Endpoints are half-open. Full physical contact windows are never trimmed.
        """
        if not math.isfinite(min_window_s) or min_window_s < 0:
            raise ValueError("min_window_s must be finite and nonnegative")
        result = {}
        for index, contacts in self._contacts.items():
            merged = []
            for c in contacts:
                end = c.end_s - min_window_s
                if end <= c.start_s:
                    continue
                if merged and c.start_s <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], end))
                else:
                    merged.append((c.start_s, end))
            events = [(0.0, "UN_AVL")]
            for start, end in merged:
                if start == events[-1][0]:
                    events[-1] = (start, "AVL_TRAIN")
                else:
                    events.append((start, "AVL_TRAIN"))
                events.append((end, "UN_AVL"))
            result[index] = events
        return result

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [(i, self.station_names.index(c.station_id), c.start_s, c.end_s)
                for i, contacts in self._contacts.items() for c in contacts]
        rows = np.asarray(rows, dtype=float).reshape(-1, 4)
        np.savez(path, format_version=1, sat_names=self.satellite_names,
                 station_names=self.station_names, trace_start_s=self.start_s,
                 horizon_s=self.horizon_s, contacts=rows)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            if int(data["format_version"]) != 1:
                raise ValueError("Unsupported contact schedule format")
            names = tuple(str(x) for x in data["station_names"])
            contacts = {}
            for index, station, start, end in data["contacts"]:
                if index != int(index) or station != int(station) or not 0 <= station < len(names):
                    raise ValueError("Invalid contact identity")
                contacts.setdefault(int(index), []).append(Contact(names[int(station)], float(start), float(end)))
            return cls(data["sat_names"], names, float(data["trace_start_s"]), float(data["horizon_s"]), contacts)

    @classmethod
    def from_visibility(cls, time_s, satellite_names, visibility):
        times = np.asarray(time_s, dtype=float)
        if times.ndim != 1 or not len(times) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError("Contact timestamps must be finite and increasing")
        horizon = times[-1] + (times[-1] - times[-2] if len(times) > 1 else 1.0)
        boundaries = np.append(times, horizon)
        contacts = {}
        for station, values in visibility.items():
            values = np.asarray(values, dtype=bool)
            if values.shape != (len(times), len(satellite_names)):
                raise ValueError("Visibility shape does not match the timeline and satellites")
            for index in range(len(satellite_names)):
                diff = np.diff(np.r_[False, values[:, index], False].astype(np.int8))
                for start, end in zip(np.flatnonzero(diff == 1), np.flatnonzero(diff == -1)):
                    contacts.setdefault(index, []).append(Contact(str(station), float(boundaries[start]), float(boundaries[end])))
        return cls(satellite_names, tuple(visibility), times[0], horizon, contacts)
