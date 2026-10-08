# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""FX-N77 MQTT wire: zero-copy paho reads/writes, in-place Any(Data) parse, chunk reassembly, tx flush before LEAVE."""

import asyncio
import os
import socket
import time

import paho.mqtt.client as mqtt
import pytest
from google.protobuf.any_pb2 import Any
from paho.mqtt.client import MQTTv5

from flame.backend.chunk_store import ChunkStore
from flame.backend.mqtt import _ANY_DATA_URL, _DataView, _encode_chunk, _split_any
from flame.backend.paho_fast import BIG_PACKET, FastClient
from flame.proto import backend_msg_pb2 as msg_pb2


def _varint(n):
    out = bytearray()
    while True:
        b, n = n & 0x7F, n >> 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _publish_packet(topic: bytes, mid: int, payload: bytes) -> bytes:
    body = len(topic).to_bytes(2, "big") + topic + mid.to_bytes(2, "big") + b"\x00" + payload  # v5, no properties
    return b"\x34" + _varint(len(body)) + body  # PUBLISH, QoS 2


def _client(cls):
    c = cls(mqtt.CallbackAPIVersion.VERSION1, "c", protocol=MQTTv5)
    a, b = socket.socketpair()
    a.setblocking(False)
    b.setblocking(False)
    c._sock = a
    return c, b


@pytest.mark.parametrize("size", [100, BIG_PACKET + 1, 4 * 1024 * 1024 + 77])
@pytest.mark.parametrize("cls", [mqtt.Client, FastClient])
def test_large_publish_reads_like_stock(cls, size):
    payload = os.urandom(size)
    c, peer = _client(cls)
    raw = _publish_packet(b"t/x", 4242, payload)
    sent = 0
    while 4242 not in c._in_messages:  # feed in pieces: partial reads must resume
        if sent < len(raw):
            try:
                sent += peer.send(raw[sent:sent + 300_000])
            except BlockingIOError:
                pass
        assert c._packet_read() in (mqtt.MQTT_ERR_SUCCESS, mqtt.MQTT_ERR_AGAIN)
    m = c._in_messages[4242]
    assert (m.topic, m.mid, m.qos, bytes(m.payload)) == ("t/x", 4242, 2, payload)
    if cls is FastClient and size >= BIG_PACKET:
        assert isinstance(m.payload, memoryview)
    peer.close()


def test_fast_client_writes_exact_bytes():
    c, peer = _client(FastClient)
    data = os.urandom(3 * 1024 * 1024)
    c._packet_queue(0xC0, bytearray(data), 0, 0)  # raw bytes, not a PUBLISH
    got = bytearray()
    while len(got) < len(data):
        c.loop_write()
        try:
            got += peer.recv(1 << 20)
        except BlockingIOError:
            pass
    assert bytes(got) == data


def test_data_view_parses_any_encoding():
    for payload, seqno, eom in ((b"x" * 300, 7, True), (b"", 0, False), (bytes(range(256)) * 5000, 129, False)):
        d = msg_pb2.Data(end_id="end-1", channel_name="param-channel", seqno=seqno, eom=eom, payload=payload)
        packed = Any()
        packed.Pack(d)
        for raw in (_encode_chunk("end-1", "param-channel", memoryview(payload), seqno, eom), packed.SerializeToString()):
            url, value = _split_any(memoryview(raw))
            v = _DataView(value)
            assert url == _ANY_DATA_URL
            assert (v.end_id, v.channel_name, v.seqno, v.eom, bytes(v.payload)) == ("end-1", "param-channel", seqno, eom,
                                                                                    payload)


def _chunk(seqno, eom, payload):
    return _DataView(_split_any(memoryview(_encode_chunk("s", "ch", memoryview(payload), seqno, eom)))[1])


def test_chunk_store_new_message_drops_partial():
    cs = ChunkStore()
    cs.assemble(_chunk(0, False, b"old0"))
    cs.assemble(_chunk(1, False, b"old1"))  # sender abandoned this message
    for i, part in enumerate((b"a", b"b", b"c")):
        cs.assemble(_chunk(i, i == 2, part))
    assert cs.eom and cs.get_data() == b"abc"


def test_flush_tx_waits_for_queued_sends():
    from flame.channel import Channel
    from flame.common.util import background_thread_loop

    with background_thread_loop() as loop:
        ch = object.__new__(Channel)
        ch._name, ch._ends = "ch", {}
        ch._backend = type("B", (), {"loop": lambda self: loop})()

        async def _mk():
            return asyncio.Queue()

        ch._bcast_queue = asyncio.run_coroutine_threadsafe(_mk(), loop).result()
        loop.call_soon_threadsafe(ch._bcast_queue.put_nowait, b"eot")

        async def _consume():
            await ch._bcast_queue.get()
            await asyncio.sleep(0.3)
            ch._bcast_queue.task_done()

        asyncio.run_coroutine_threadsafe(_consume(), loop)
        t0 = time.time()
        assert ch.flush_tx(timeout=5)
        assert time.time() - t0 >= 0.25  # waited for the send to finish
        loop.call_soon_threadsafe(ch._bcast_queue.put_nowait, b"stuck")
        assert not ch.flush_tx(timeout=0.2)  # bounded


def test_leave_applies_after_earlier_chunks_of_that_end():
    """A LEAVE handled on the loop removed the end while its EOT sat in the chunk thread (trainers hung in await_join)."""
    from flame.backend.chunk_manager import ChunkManager
    from flame.common.util import background_thread_loop

    with background_thread_loop() as loop:
        events = []

        class _Ch:
            def __init__(self):
                self.q = asyncio.run_coroutine_threadsafe(self._mk(), loop).result()
                self.ends = {"agg"}

            @staticmethod
            async def _mk():
                return asyncio.Queue()

            def get_rxq(self, end_id):
                return self.q if end_id in self.ends else None

            def note_arrival(self):
                events.append("data")

            async def remove(self, end_id):
                self.ends.discard(end_id)
                events.append("remove")

        backend = type("B", (), {"loop": lambda self: loop, "set_cleanup_ready": lambda self, e: None,
                                 "set_cleanup_ready_async": lambda self, e: None})()
        ch, mgr = _Ch(), ChunkManager(backend)
        big = os.urandom(3 << 20)
        for i in range(3):  # a 3-chunk message, then the LEAVE right behind it
            mgr.handle(_DataView(_split_any(memoryview(_encode_chunk("agg", "ch", memoryview(big)[i << 20:(i + 1) << 20],
                                                                     i, i == 2)))[1]), ch)
        asyncio.run_coroutine_threadsafe(mgr.in_order("agg", lambda: ch.remove("agg")), loop).result()
        deadline = time.time() + 5
        while len(events) < 2 and time.time() < deadline:
            time.sleep(0.01)
        assert events == ["data", "remove"]
        assert ch.q.get_nowait()[0] == big


def test_trainer_finishes_on_departed_eot():
    """An EOT left unread when the aggregator's end was removed ends the trainer instead of an await_join hang."""
    from flame.mode.horizontal.syncfl.trainer import Trainer

    t = type("T", (), {"_aggregator_left": Trainer._aggregator_left})()
    t._work_done, t.fetch_success = False, False
    ch = type("C", (), {"departed_eot": None, "all_ends": lambda self: []})()
    assert not t._aggregator_left(ch)
    ch.departed_eot = True
    assert t._aggregator_left(ch) and t._work_done and t.fetch_success
    ch.all_ends = lambda: ["agg"]  # rejoined: keep working
    t._work_done = False
    assert not t._aggregator_left(ch)
