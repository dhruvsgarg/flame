import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from flame.backend.mqtt import MqttBackend


@pytest.mark.parametrize("ack", [2, SimpleNamespace(value=128)])
def test_join_announcement_follows_subscription_acknowledgements(ack):
    async def run():
        backend = MqttBackend.__new__(MqttBackend)
        backend._is_connected = True
        backend._pending_subscriptions = {}
        backend._joined_channels = set()
        backend._topics_for_notify = lambda channel: ["a", "b"]
        backend.notify = Mock(return_value=True)
        mids = []
        def subscribe(topic, qos):
            assert not backend.notify.called
            mid = len(mids) + 1
            mids.append(mid)
            asyncio.get_running_loop().call_soon(backend.on_subscribe, None, None, mid, [ack])
            return 0, mid
        backend._mqtt_client = SimpleNamespace(subscribe=subscribe)
        channel = SimpleNamespace(name=lambda: "models")
        if isinstance(ack, int):
            await backend._join_channel(channel)
            assert backend.notify.call_count == 1
            assert backend._joined_channels == {"models"}
        else:
            with pytest.raises(ConnectionError):
                await backend._join_channel(channel)
            assert not backend.notify.called
        assert not backend._pending_subscriptions
    asyncio.run(run())


def test_connection_timeout_never_announces_membership(monkeypatch):
    import flame.backend.mqtt as mqtt
    monkeypatch.setattr(mqtt, "DEFAULT_RUN_ASYNC_WAIT_TIME", 0.01)
    async def run():
        backend = MqttBackend.__new__(MqttBackend)
        backend._is_connected = False
        backend.notify = Mock()
        with pytest.raises(asyncio.TimeoutError):
            await backend._join_channel(SimpleNamespace(name=lambda: "models"))
        assert not backend.notify.called
    asyncio.run(run())
