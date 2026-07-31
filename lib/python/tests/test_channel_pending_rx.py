# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""``Channel.ends_with_pending_rx`` -- the cheap "has this end's update already
ARRIVED?" probe real's capacity read subtracts.

Real clears an end from ``_trainer_inflight_dispatch_version`` when the drain
loop PROCESSES its message, not when the message lands. A trainer that has sent
has an idle GPU either way, so the interval between the two belongs to the
aggregator's drain lag, not to trainer concurrency. Charging it to the cap made
``felix_it`` real read up to 55 outstanding against c=30 on 49.1% of dispatches
while its trainers' own ``train_with_data_id`` spans peaked at exactly 30.

The probe runs on the dispatch path, so the load-bearing property is that it
stays CHEAP: queue depth only, never a deserialize (§F-19). ``Channel.peek``
cloudpickle-loads the payload and must never be used for this.
"""

import asyncio

from flame.channel import Channel


class _FakeEnd:
    def __init__(self, depth=0, peek_buf=None):
        self.rxq = asyncio.Queue()
        for _ in range(depth):
            self.rxq.put_nowait(b"payload")
        self.peek_buf = peek_buf


def _chan(ends: dict) -> Channel:
    ch = Channel.__new__(Channel)  # no backend/loop needed for the probe
    ch._ends = ends
    return ch


class TestPendingRxProbe:
    def test_empty_queues_report_nothing_pending(self):
        ch = _chan({"a": _FakeEnd(), "b": _FakeEnd()})
        assert ch.ends_with_pending_rx() == set()

    def test_queued_message_marks_the_end_arrived(self):
        ch = _chan({"a": _FakeEnd(depth=1), "b": _FakeEnd()})
        assert ch.ends_with_pending_rx() == {"a"}

    def test_peek_buf_counts_as_arrived(self):
        """A peeked-but-not-consumed payload has left the queue; the trainer is
        just as idle, so it must still count."""
        ch = _chan({"a": _FakeEnd(peek_buf=b"payload"), "b": _FakeEnd()})
        assert ch.ends_with_pending_rx() == {"a"}

    def test_multiple_queued_messages_count_the_end_once(self):
        ch = _chan({"a": _FakeEnd(depth=4)})
        assert ch.ends_with_pending_rx() == {"a"}

    def test_no_ends_is_empty_not_an_error(self):
        assert _chan({}).ends_with_pending_rx() == set()


class TestProbeIsCheapAndSafe:
    def test_never_deserializes_the_payload(self):
        """Undeserializable bytes must not raise -- proof the probe reads depth
        only. `peek()` would blow up on this exact input."""
        ch = _chan({"a": _FakeEnd(depth=1)})
        ch._ends["a"].rxq.put_nowait(b"\x00not-a-pickle\xff")
        assert ch.ends_with_pending_rx() == {"a"}

    def test_end_without_queue_attrs_is_skipped_not_fatal(self):
        ch = _chan({"a": _FakeEnd(depth=1), "weird": object()})
        assert ch.ends_with_pending_rx() == {"a"}

    def test_snapshot_is_not_a_live_view(self):
        """Returned set must not mutate under the caller as ends drain."""
        ends = {"a": _FakeEnd(depth=1)}
        ch = _chan(ends)
        got = ch.ends_with_pending_rx()
        ends["a"].rxq.get_nowait()
        assert got == {"a"}
        assert ch.ends_with_pending_rx() == set()

    def test_does_not_consume_the_message(self):
        """The drain loop still has to receive it -- the probe must not pop."""
        ch = _chan({"a": _FakeEnd(depth=2)})
        ch.ends_with_pending_rx()
        assert ch._ends["a"].rxq.qsize() == 2
