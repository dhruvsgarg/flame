# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""FX-D148: wait-K version start dispatches to the eligible ends when fewer than c (FedScale resampleClients)."""

import types

import pytest

from flame.mode.horizontal.oort.top_aggregator import TopAggregator as OortTopAggregator
from flame.mode.horizontal.syncfl.top_aggregator import TopAggregator


class _Agg(TopAggregator):
    check_and_sleep = evaluate = initialize = load_data = train = lambda self: None


class _OortAgg(OortTopAggregator):
    check_and_sleep = evaluate = initialize = load_data = train = lambda self: None


def _agg(partial, fresh=(), accepted=()):
    agg = _Agg.__new__(_Agg)
    agg.config = types.SimpleNamespace(hyperparameters=types.SimpleNamespace(sync_partial_dispatch=partial))
    agg._sync_version_inflight = lambda ch: set(fresh)
    agg._sync_accepted_ends = lambda: set(accepted)
    return agg


def _channel(n=50, c=6):
    return types.SimpleNamespace(_ends={f"e{i}": None for i in range(n)}, _selector=types.SimpleNamespace(num_of_ends=c))


@pytest.mark.parametrize("eligible", [0, 3, 5, 6, 9])
def test_off_waits_for_c_at_version_start(eligible):
    assert _agg(False)._sync_wait_k_plan(_channel(), eligible, 5) == (6, 6)


@pytest.mark.parametrize("eligible,plan", [(1, (1, 1)), (5, (5, 5)), (6, (6, 6)), (9, (6, 6))])
def test_on_dispatches_to_whoever_is_eligible(eligible, plan):
    assert _agg(True)._sync_wait_k_plan(_channel(), eligible, 5) == plan


def test_on_with_none_eligible_still_starves():
    need, threshold = _agg(True)._sync_wait_k_plan(_channel(), 0, 5)
    assert need == 0 and threshold == 1  # threshold > eligible: the starvation wake runs, no empty selection


def test_mid_round_top_up_unchanged():
    for partial in (False, True):
        assert _agg(partial, fresh={"e1", "e2"}, accepted={"e3"})._sync_wait_k_plan(_channel(), 2, 5) == (2, 2)
        assert _agg(partial, fresh={"e1"})._sync_wait_k_plan(_channel(), 0, 5) is None


def test_nothing_missing_returns_none():
    assert _agg(True, fresh={f"e{i}" for i in range(4)}, accepted={"e9", "e8"})._sync_wait_k_plan(_channel(), 9, 5) is None


def _oort(partial, fresh=(), accepted=()):
    agg = _OortAgg.__new__(_OortAgg)
    agg.config = types.SimpleNamespace(hyperparameters=types.SimpleNamespace(sync_partial_dispatch=partial))
    agg._sync_version_inflight = lambda ch: set(fresh)
    agg._sync_accepted_ends = lambda: set(accepted)
    return agg


@pytest.mark.parametrize("eligible", [0, 3, 13, 20])
def test_oort_off_needs_the_full_cohort(eligible):
    assert _oort(False)._wait_k_need(None, eligible, 13) == 13


@pytest.mark.parametrize("eligible,need", [(0, 13), (1, 1), (8, 8), (13, 13), (20, 13)])
def test_oort_on_dispatches_to_whoever_is_eligible(eligible, need):
    assert _oort(True)._wait_k_need(None, eligible, 13) == need  # 0 eligible keeps 13: the starvation gate waits


def test_oort_mid_round_top_up_unchanged():
    for partial in (False, True):
        assert _oort(partial, fresh={"e1", "e2"}, accepted={"e3"})._wait_k_need(None, 4, 13) == 4
        assert _oort(partial, fresh={"e1"})._wait_k_need(None, 0, 13) is None
        assert _oort(partial, fresh={f"e{i}" for i in range(10)}, accepted={"a", "b", "c"})._wait_k_need(None, 9, 13) is None


@pytest.mark.parametrize("baseline", ["feddance", "refl", "oort", "oort_star"])
def test_fedscale_baselines_enable_it(baseline):
    import pathlib
    import yaml
    cat = yaml.safe_load((pathlib.Path(__file__).parents[2] / "examples/_metadata/baselines.yaml").read_text())
    root = cat.get("baselines", cat)
    assert root[baseline]["aggregator"]["hyperparameters"]["syncPartialDispatch"] is True
