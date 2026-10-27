import pytest

from fmow.simulation.contacts import Contact, ContactSchedule
from fmow.simulation.links import Direction, LinkState, TransferRequest
from fmow.simulation.transfers import TransferPlanner


class ConstantLink:
    def __init__(self, rate=8e6, delay=0, horizon=1000):
        self.rate, self.delay, self.horizon = rate, delay, horizon
        self.calls = []

    def sample(self, request):
        self.calls.append(request)
        return LinkState(self.rate, self.delay, self.horizon)


def plan(windows, size, start=0, horizon=1000, provider=None):
    stations = sorted({w.station_id for w in windows})
    contacts = ContactSchedule(["sat"], stations, 0, horizon, {0: windows})
    request = TransferRequest("assignment:upload", 0, Direction.UPDATE_UPLOAD, start, size)
    return TransferPlanner(contacts, provider or ConstantLink(horizon=horizon)).estimate(request)


def test_waits_and_resumes_without_restarting_payload():
    result = plan([Contact("a", 0, 5), Contact("a", 10, 15)], 8_000_000,
                  provider=ConstantLink(delay=.002))
    assert result.complete and result.arrival_s == pytest.approx(13.002)
    assert result.wait_s == 5 and result.transmit_s == 8
    assert result.remaining_bytes == 0


def test_five_minute_windows_can_require_two_contacts():
    result = plan([Contact("a", 0, 300), Contact("a", 600, 900)], 600_000_000)
    assert result.complete and result.arrival_s == 900
    assert result.wait_s == 300 and result.transmit_s == 600


def test_horizon_is_finite_and_incomplete_delivery_is_not_zero_delay():
    result = plan([Contact("a", 0, 5)], 10_000_000, horizon=10)
    assert not result.complete and result.arrival_s is None
    assert result.transmitted_bytes == result.remaining_bytes == 5_000_000


def test_never_visible_and_empty_payload():
    result = plan([], 1_000_000, horizon=10)
    assert not result.complete and result.remaining_bytes == 1_000_000
    assert result.reason == "no_contact_in_trace"
    result = plan([], 0, start=2, horizon=10)
    assert result.complete and result.arrival_s == 2


def test_earliest_station_is_chosen_without_combining_partial_transfers():
    windows = [Contact("a", 0, 1), Contact("b", 2, 10)]
    result = plan(windows, 2_000_000)
    assert result.station_id == "b" and result.arrival_s == 4
    result = plan([Contact("a", 0, 1), Contact("b", 2, 3)], 2_000_000, horizon=4)
    assert not result.complete and result.remaining_bytes == 1_000_000


def test_changing_rates_and_direction_reach_provider():
    class Varying:
        def sample(self, request):
            assert request.direction == Direction.UPDATE_UPLOAD
            return LinkState(8e6 if request.time_s < 2 else 16e6, 0, 2 if request.time_s < 2 else 10)
    result = plan([Contact("a", 0, 10)], 4_000_000, provider=Varying())
    assert result.arrival_s == 3


def test_invalid_provider_cannot_pin_planning_loop():
    class Invalid:
        def sample(self, request):
            return LinkState(8e6, 0, request.time_s)
    with pytest.raises(ValueError, match="beyond"):
        plan([Contact("a", 0, 10)], 1, provider=Invalid())
    with pytest.raises(ValueError):
        LinkState(float("inf"), 0, 10)


def test_propagation_after_contact_end_is_not_a_second_contact_wait():
    result = plan([Contact("a", 0, 1)], 1_000_000, horizon=5,
                  provider=ConstantLink(delay=.1, horizon=5))
    assert result.complete and result.arrival_s == 1.1
    result = plan([Contact("a", 0, 1)], 1_000_000, horizon=1,
                  provider=ConstantLink(delay=.1, horizon=1))
    assert not result.complete and result.reason == "arrival_beyond_trace"
