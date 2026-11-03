# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""FX-N77 weight wire: flat tensor codec round-trips, materialize_weights reads both formats, one-copy MQTT chunks parse."""

import cloudpickle
import pytest
import torch

from flame.common import tensor_codec
from flame.common.util import materialize_weights, pack_weights
from flame.mode.message import MessageType


def _weights(device="cpu"):
    g = torch.Generator().manual_seed(0)
    return {
        "conv.weight": torch.randn(3, 5, 7, generator=g),
        "bn.num_batches_tracked": torch.tensor(17),  # 0-dim int64
        "half": torch.randn(3, generator=g).half(),  # 6 bytes: next tensor needs padding
        "mask": torch.tensor([True, False, True]),  # 3 bytes
        "empty": torch.empty(0, 4),
        "d": torch.randn(2, 2, generator=g, dtype=torch.float64),
    }


def _same(a, b):
    assert list(a) == list(b)
    for k in a:
        assert a[k].dtype == b[k].dtype and a[k].shape == b[k].shape and torch.equal(a[k].cpu(), b[k].cpu()), k


def test_round_trip_cpu_is_writable_copy():
    w = _weights()
    blob = tensor_codec.encode(w)
    assert tensor_codec.is_encoded(blob)
    out = tensor_codec.decode(blob)
    _same(w, out)
    out["conv.weight"].add_(1.0)  # owns its memory: writing must not touch the blob or raise
    _same(w, tensor_codec.decode(blob))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_round_trip_cuda_both_ways():
    w = {k: v.cuda() for k, v in _weights().items()}
    out = tensor_codec.decode(tensor_codec.encode(w), device="cuda")
    assert all(t.is_cuda for t in out.values())
    _same(w, out)


def test_materialize_reads_both_formats(monkeypatch):
    w = _weights()
    msg = {MessageType.WEIGHTS_BYTES: pack_weights(w)}
    _same(w, materialize_weights(msg))
    assert MessageType.WEIGHTS_BYTES not in msg and materialize_weights(msg) is msg[MessageType.WEIGHTS]
    monkeypatch.setenv(tensor_codec.ENV_CODEC, "pickle")  # the revert knob
    blob = pack_weights(w)
    assert not tensor_codec.is_encoded(blob) and isinstance(cloudpickle.loads(blob), dict)
    _same(w, materialize_weights({MessageType.WEIGHTS_BYTES: blob}))


def test_mqtt_chunk_encoding_parses_like_any_pack():
    """FX-N77: the one-copy chunk encoder is wire-compatible with Any.Pack(Data(...))."""
    from google.protobuf.any_pb2 import Any

    from flame.backend.mqtt import _encode_chunk
    from flame.proto import backend_msg_pb2 as msg_pb2

    for payload, seqno, eom in ((b"x" * 300, 7, True), (b"", 0, False), (bytes(range(256)) * 5000, 129, False)):
        raw = _encode_chunk("end-1", "param-channel", memoryview(payload), seqno, eom)
        a = Any().FromString(raw)
        assert a.Is(msg_pb2.Data.DESCRIPTOR)
        d = msg_pb2.Data()
        a.Unpack(d)
        assert (d.end_id, d.channel_name, d.seqno, d.eom, d.payload) == ("end-1", "param-channel", seqno, eom, payload)


def test_channel_frame_round_trip_out_of_band():
    """FX-N77: large bytes travel out-of-band and decode as zero-copy views; small messages stay plain cloudpickle."""
    import cloudpickle as cp

    from flame.channel import decode_message, encode_message

    blob = pack_weights(_weights()) * 20000  # > 1 MiB
    msg = {MessageType.WEIGHTS_BYTES: blob, MessageType.MODEL_VERSION: 3}
    frame = encode_message(msg)
    out = decode_message(frame)
    assert isinstance(out[MessageType.WEIGHTS_BYTES], memoryview) and bytes(out[MessageType.WEIGHTS_BYTES]) == blob
    assert out[MessageType.MODEL_VERSION] == 3
    small = {MessageType.MODEL_VERSION: 1}
    assert encode_message(small) == cp.dumps(small) and decode_message(cp.dumps(small)) == small
    one = {MessageType.WEIGHTS_BYTES: pack_weights(_weights())}  # the codec decodes a view of a frame too
    _same(_weights(), materialize_weights(decode_message(encode_message({**one, "pad": b"x" * (2 << 20)}))))
