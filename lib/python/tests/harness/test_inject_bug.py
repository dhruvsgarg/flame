# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""S1: FLAME_INJECT_BUG re-enables a known sim bug only when named; unset is production."""

import pytest

from flame import harness


def test_unset_is_off(monkeypatch):
    monkeypatch.delenv("FLAME_INJECT_BUG", raising=False)
    assert not any(harness.injected(b) for b in harness.INJECTABLE_BUGS)


def test_named_bug_only(monkeypatch):
    monkeypatch.setenv("FLAME_INJECT_BUG", "order_by_sct,freeze_trainer_clock")
    assert harness.injected("order_by_sct") and harness.injected("freeze_trainer_clock")
    assert not harness.injected("no_busy_hold")


def test_unknown_bug_rejected():
    with pytest.raises(AssertionError):
        harness.injected("typo")
