# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Async sim commit-ordering gate (PARITY.md §3c).

In sim the trainer does not sleep its budget — it computes fast and stamps a
FUTURE sim_completion_ts — so a physically-delayed message could be committed
AFTER the virtual clock advanced past its sct, out of completion order, which
inflates staleness (current_version - trained_version). Real waits implicitly
(the trainer actually takes its budget). The gate makes the sim hold the commit
of the buffered earliest update while an un-arrived in-flight trainer is expected
to complete earlier, so commits stay in true completion order.

This pins the pure decision predicate; the loop/timeout that drives it lives in
_sim_recv_min and is exercised end-to-end by test_async_sim_ordering + the smoke.
"""

from flame.mode.horizontal.asyncfl.top_aggregator import TopAggregator as Agg

SLACK = 2.0


def test_commit_when_nothing_expected_earlier():
    # buffered update at sct=100; no un-arrived trainer has a known expected sct
    assert Agg._safe_to_commit(100.0, None, SLACK) is True


def test_commit_when_buffered_is_the_earliest():
    # buffered min (100) completes before the earliest outstanding (130) -> commit
    assert Agg._safe_to_commit(100.0, 130.0, SLACK) is True


def test_wait_when_an_outstanding_trainer_completes_earlier():
    # an un-arrived trainer is expected at 80, before the buffered min (100):
    # committing now would be out of order -> wait
    assert Agg._safe_to_commit(100.0, 80.0, SLACK) is False


def test_slack_absorbs_estimate_noise():
    # outstanding expected only marginally earlier (within slack) -> commit anyway
    assert Agg._safe_to_commit(100.0, 99.0, SLACK) is True   # 100 <= 99 + 2
    assert Agg._safe_to_commit(100.0, 97.0, SLACK) is False  # 100 > 97 + 2


def test_never_commit_empty_buffer():
    assert Agg._safe_to_commit(None, 50.0, SLACK) is False
    assert Agg._safe_to_commit(None, None, SLACK) is False
