import numpy as np
import pytest

from fmow.simulation.contacts import Contact, ContactSchedule


def test_station_contacts_survive_any_station_projection(tmp_path):
    schedule = ContactSchedule(["sat"], ["a", "b"], 0, 900,
        {0: [Contact("a", 0, 400), Contact("b", 200, 800)]})
    assert schedule.availability_events(300)[0] == [
        (0, "AVL_TRAIN"), (100, "UN_AVL"), (200, "AVL_TRAIN"), (500, "UN_AVL")]
    assert schedule.at(0, 350) == (Contact("a", 0, 400), Contact("b", 200, 800))
    assert schedule.at(0, 600) == (Contact("b", 200, 800),)
    path = tmp_path / "contacts.npz"
    schedule.save(path)
    loaded = ContactSchedule.load(path)
    assert loaded.for_satellite(0) == schedule.for_satellite(0)
    assert loaded.horizon_s == 900
    assert not loaded.at(0, 900)


def test_short_station_contacts_are_not_combined_to_meet_minimum():
    schedule = ContactSchedule(["sat"], ["a", "b"], 0, 600,
        {0: [Contact("a", 0, 250), Contact("b", 250, 500)]})
    assert schedule.availability_events(300)[0] == [(0, "UN_AVL")]
    assert schedule.availability_events(0)[0] == [(0, "AVL_TRAIN"), (500, "UN_AVL")]


def test_non_unit_timestamps_are_seconds_not_sample_indices():
    schedule = ContactSchedule.from_visibility([10, 20, 30], ["sat"], {"a": [[True], [True], [False]]})
    assert schedule.for_satellite(0) == (Contact("a", 10, 30),)
    assert schedule.availability_events(5)[0] == [(0, "UN_AVL"), (10, "AVL_TRAIN"), (25, "UN_AVL")]
    assert schedule.horizon_s == 40


@pytest.mark.parametrize("contacts", [
    {0: [Contact("unknown", 0, 1)]},
    {0: [Contact("a", 0, 11)]},
    {0: [Contact("a", 0, 5), Contact("a", 4, 7)]},
    {1: [Contact("a", 0, 1)]},
])
def test_invalid_contact_identity_or_window_rejected(contacts):
    with pytest.raises(ValueError):
        ContactSchedule(["sat"], ["a"], 0, 10, contacts)


def test_empty_contact_schedule_roundtrips(tmp_path):
    path = tmp_path / "empty.npz"
    schedule = ContactSchedule(["sat"], ["a"], 0, 10, {})
    schedule.save(path)
    assert ContactSchedule.load(path).for_satellite(0) == ()
