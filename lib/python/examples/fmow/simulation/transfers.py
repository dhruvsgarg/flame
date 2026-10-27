"""Finite-trace transfer planning, independent of training and RF models.

Initial policy: independent links, one station per transfer, pause/resume on
later contacts with that same station. Different assignment legs may choose
different stations. No automatic orbit replay or real socket shaping occurs.
"""
from .links import LinkRequest, TransferOutcome


class TransferPlanner:
    def __init__(self, contacts, provider):
        self.contacts, self.provider = contacts, provider

    def estimate(self, request):
        windows = self.contacts.for_satellite(request.satellite_index)
        if request.payload_bytes == 0 and request.start_s <= self.contacts.horizon_s:
            return TransferOutcome(request.transfer_id, True, None, request.start_s,
                                   request.start_s, request.start_s, request.start_s,
                                   0, 0, 0, 0, "empty_payload")
        stations = sorted({w.station_id for w in windows if w.end_s > request.start_s})
        outcomes = [self._through_station(request, station, windows) for station in stations]
        completed = [o for o in outcomes if o.complete]
        if completed:
            return min(completed, key=lambda o: (o.arrival_s, o.station_id))
        if outcomes:
            return min(outcomes, key=lambda o: (o.remaining_bytes, o.station_id))
        return TransferOutcome(request.transfer_id, False, None, request.start_s,
                               None, None, None, 0, request.payload_bytes,
                               max(0, self.contacts.horizon_s - request.start_s), 0, "no_contact_in_trace")

    def _through_station(self, request, station, windows):
        bits_left = float(request.payload_bytes * 8)
        cursor = request.start_s
        first = last = None
        transmit_s = wait_s = 0.0
        for window in windows:
            if window.station_id != station or window.end_s <= cursor:
                continue
            t = max(cursor, window.start_s)
            wait_s += t - cursor
            while t < window.end_s:
                state = self.provider.sample(LinkRequest(request.satellite_index, station,
                                                         request.direction, t))
                if state.valid_until_s <= t:
                    raise ValueError("Link provider must return state valid beyond the requested time")
                end = min(window.end_s, state.valid_until_s)
                duration = end - t
                if state.rate_bps == 0:
                    wait_s += duration
                else:
                    first = t if first is None else first
                    needed = bits_left / state.rate_bps
                    used = min(needed, duration)
                    transmit_s += used
                    last = t + used
                    bits_left = max(0, bits_left - state.rate_bps * used)
                    if needed <= duration:
                        arrival = last + state.propagation_s
                        complete = arrival <= self.contacts.horizon_s
                        return TransferOutcome(request.transfer_id, complete, station, request.start_s,
                                               first, last, arrival if complete else None,
                                               request.payload_bytes, 0, wait_s, transmit_s,
                                               "delivered" if complete else "arrival_beyond_trace")
                t = end
            cursor = window.end_s
        wait_s += max(0, self.contacts.horizon_s - cursor)
        return TransferOutcome(request.transfer_id, False, station, request.start_s,
                               first, last, None, request.payload_bytes - bits_left / 8,
                               bits_left / 8, wait_s, transmit_s, "insufficient_capacity_in_trace")
