"""Transfer identity, channel isolation and reordered chunk regressions."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from flame.backend.chunk_manager import ChunkManager, ChunkThread
from flame.backend.chunk_store import ChunkStore
from flame.backend.mqtt import MqttBackend
from flame.backend.p2p import PointToPointBackend
from flame.proto.backend_msg_pb2 import Data


def chunk(transfer, seq, payload, eom=False):
    return Data(end_id="trainer", channel_name="models", transfer_id=transfer,
                seqno=seq, payload=payload, eom=eom)


def thread():
    return ChunkThread(kwargs={"backend": None, "channel": None, "end_id": "trainer"})


def test_interleaved_transfers_cannot_mix():
    t = thread()
    assert t._consume(chunk("old", 0, b"old-")) is None
    assert t._consume(chunk("new", 1, b"end", True)) is None
    assert t._consume(chunk("new", 0, b"new-")) == b"new-end"
    assert t._consume(chunk("old", 1, b"end", True)) == b"old-end"
    assert t._consume(chunk("new", 0, b"new-")) is None
    assert t._consume(chunk("new", 1, b"end", True)) is None


def test_invalid_transfer_does_not_break_next_transfer():
    t = thread()
    assert t._consume(chunk("bad", -1, b"bad", True)) is None
    assert t._consume(chunk("good", 0, b"good", True)) == b"good"
    assert t._consume(chunk("", 0, b"legacy", True)) is None


def test_reordering_duplicates_and_conflicting_eom():
    s = ChunkStore()
    assert s.assemble(chunk("a", 2, b"c", True))
    assert s.assemble(chunk("a", 0, b"a"))
    assert s.assemble(chunk("a", 0, b"a"))
    assert not s.eom
    assert not s.assemble(chunk("different", 1, b"b"))
    assert not s.assemble(chunk("a", 1, b"b", True))
    assert s.assemble(chunk("a", 1, b"b"))
    assert s.get_data() == b"abc"


def test_active_transfer_can_last_longer_than_idle_timeout(monkeypatch):
    import flame.backend.chunk_store as stores
    import flame.backend.chunk_manager as managers
    clock = [0.0]
    monkeypatch.setattr(stores.time, "monotonic", lambda: clock[0])
    t = thread()
    t._consume(chunk("active", 0, b"a"))
    clock[0] = 20
    t._consume(chunk("active", 1, b"b"))
    clock[0] = 40
    assert t._consume(chunk("active", 2, b"c", True)) == b"abc"
    t._consume(chunk("stalled", 0, b"x"))
    clock[0] = 71
    t._expire()
    assert not t.stores
    assert "stalled" in t.retired


def test_channels_for_same_peer_have_separate_workers(monkeypatch):
    monkeypatch.setattr(ChunkThread, "start", lambda self: None)
    manager = ChunkManager(None)
    for name in ["models", "metrics"]:
        msg = chunk("a", 0, b"data")
        msg.channel_name = name
        manager.handle(msg, SimpleNamespace(name=lambda n=name: n))
    assert set(manager._chunk_threads) == {("models", "trainer"), ("metrics", "trainer")}
    manager.stop("trainer")
    assert not manager._chunk_threads


def test_senders_keep_identity_across_chunks_and_change_it_per_message(monkeypatch):
    import flame.backend.chunk_store as stores
    monkeypatch.setattr(stores, "DEFAULT_CHUNK_SIZE", 2)
    mqtt = MqttBackend.__new__(MqttBackend)
    mqtt.send_chunk_async = AsyncMock()
    asyncio.run(mqtt.send_chunks_async("topic", "models", b"abcdef"))
    ids = [call.args[-1] for call in mqtt.send_chunk_async.call_args_list]
    assert len(ids) == 3 and len(set(ids)) == 1
    asyncio.run(mqtt.send_chunks_async("topic", "models", b"x"))
    assert mqtt.send_chunk_async.call_args_list[-1].args[-1] != ids[0]
    p2p = PointToPointBackend.__new__(PointToPointBackend)
    p2p._id = "trainer"
    messages = list(p2p._generate_data_messages("models", b"abcdef"))
    assert len(messages) == 3 and len({m.transfer_id for m in messages}) == 1
    assert all(m.transfer_id for m in messages)
