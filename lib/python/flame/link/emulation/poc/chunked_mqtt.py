""" since model weights are large packs of data, this file breaks up large data into smaller chunks to send,
and can recieve smaller chunks and put them back together. Uses flame's already existign chunking method"""



import time
import uuid

import paho.mqtt.client as mqtt
from google.protobuf import any_pb2

from flame.backend.chunk_store import ChunkStore
from flame.backend.mqtt import MqttQoS
from flame.common.constants import MQTT_TOPIC_PREFIX
from flame.proto import backend_msg_pb2

NETPOC_PREFIX = f"{MQTT_TOPIC_PREFIX}/netpoc"


def make_topic(run_id: str, suffix: str) -> str:
    """e.g. make_topic('lan_1mb_123', 'data/down') -> '/flame/netpoc/lan_1mb_123/data/down'"""
    return f"{NETPOC_PREFIX}/{run_id}/{suffix}"


class ChunkedSender:
    """Sends a payload over MQTT using flame's real chunking + protobuf wire format."""

    def __init__(self, client: mqtt.Client, topic: str, channel_name: str = "netpoc"):
        self.client = client
        self.topic = topic
        self.channel_name = channel_name

    def send(self, payload: bytes) -> None:
        end_id = str(uuid.uuid4())
        store = ChunkStore()
        store.set_data(payload)
        while True:
            chunk_bytes, seqno, eom = store.get_chunk()
            if chunk_bytes is None:
                break
            data_msg = backend_msg_pb2.Data(
                end_id=end_id,
                channel_name=self.channel_name,
                payload=chunk_bytes,
                seqno=seqno,
                eom=eom,
            )
            any_msg = any_pb2.Any()
            any_msg.Pack(data_msg)

            self.client.publish(
                self.topic,
                payload=any_msg.SerializeToString(),
                qos=MqttQoS.EXACTLY_ONCE,
            )


class ChunkedReceiver:
    """Subscribes to a topic and assembles incoming chunks back into the original payload."""

    def __init__(self, client: mqtt.Client, topic: str):
        self.client = client
        self.topic = topic
        self.store = ChunkStore()
        self._done_payload = None
        self.client.message_callback_add(topic, self._on_message)
        self.client.subscribe(topic, qos=MqttQoS.EXACTLY_ONCE)

    def _on_message(self, client, userdata, msg):
        any_msg = any_pb2.Any()
        any_msg.ParseFromString(msg.payload)
        data_msg = backend_msg_pb2.Data()
        any_msg.Unpack(data_msg)

        accepted = self.store.assemble(data_msg)
        if not accepted:
            raise RuntimeError(f"out-of-order chunk seqno={data_msg.seqno}")
        if data_msg.eom:
            self._done_payload = self.store.data

    def wait_for_payload(self, timeout: float = 120.0) -> bytes:
        """Block (polling) until eom is received or timeout elapses."""
        start = time.time()
        while self._done_payload is None:
            if time.time() - start > timeout:
                raise TimeoutError(f"no complete payload received within {timeout}s")
            time.sleep(0.01)
        return self._done_payload


def make_client(client_id: str) -> mqtt.Client:
    """One client factory so peer.py doesn't duplicate paho setup."""
    client = mqtt.Client(
        client_id=client_id,
        protocol=mqtt.MQTTv5,
        callback_api_version=mqtt.CallbackAPIVersion.VERSION1,
    )
    return client