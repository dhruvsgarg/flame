# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Channel.one_end must not crash once the peer left (ends() -> None at shutdown)."""

from flame.channel import Channel


def _chan(ends_result):
    ch = Channel.__new__(Channel)
    ch.ends = lambda state=None, allow_recv_bootstrap=False: ends_result
    return ch


def test_none_ends_returns_none():
    assert _chan(None).one_end("send") is None


def test_empty_ends_returns_none():
    assert _chan([]).one_end("send") is None


def test_first_end_returned():
    assert _chan(["a", "b"]).one_end("send") == "a"
