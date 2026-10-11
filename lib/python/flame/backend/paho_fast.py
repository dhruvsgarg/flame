# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""paho Client with linear-cost large PUBLISH I/O (FX-N77).

Stock paho 2.1 copies a received 4 MB chunk ~6 times (recv(4 MB) allocs, `+=`, four struct.unpack slices) and re-slices a
partially written packet on every send (O(n^2)). Here a packet is read once into its own buffer (`recv_into`), a large
PUBLISH payload is a memoryview into it, and writes slice a memoryview.
"""

import socket
import ssl

import paho.mqtt.client as mqtt
from paho.mqtt.client import MQTTv5, PUBLISH, MQTTErrorCode, MQTTMessage, mqtt_ms_wait_for_pubrel, time_func
from paho.mqtt.properties import Properties

BIG_PACKET = 64 * 1024  # smaller packets take paho's own path (payload stays bytes)


def _varint(buf, pos: int) -> tuple:
    """(value, next pos) of an MQTT/protobuf base-128 varint."""
    value = shift = 0
    while True:
        b = buf[pos]
        pos += 1
        value |= (b & 0x7F) << shift
        if not b & 0x80:
            return value, pos
        shift += 7


class FastClient(mqtt.Client):
    """mqtt.Client whose large packets cost one copy each way; behaviour is otherwise stock."""

    def _packet_queue(self, command, packet, mid, qos, info=None):
        return super()._packet_queue(command, memoryview(packet), mid, qos, info)

    def _packet_read(self) -> MQTTErrorCode:
        """paho's read state machine; the remaining bytes go straight into one preallocated buffer."""
        if not isinstance(self._sock, socket.socket):  # websocket transport: stock path
            return super()._packet_read()
        ip = self._in_packet
        try:
            if ip["command"] == 0:
                command = self._sock_recv(1)
                if not command:
                    return MQTTErrorCode.MQTT_ERR_CONN_LOST
                ip["command"] = command[0]
            if ip["have_remaining"] == 0:
                while True:
                    byte = self._sock_recv(1)
                    if not byte:
                        return MQTTErrorCode.MQTT_ERR_CONN_LOST
                    ip["remaining_count"].append(byte[0])
                    if len(ip["remaining_count"]) > 4:
                        return MQTTErrorCode.MQTT_ERR_PROTOCOL
                    ip["remaining_length"] += (byte[0] & 127) * ip["remaining_mult"]
                    ip["remaining_mult"] *= 128
                    if not byte[0] & 128:
                        break
                ip["have_remaining"] = 1
                ip["to_process"] = ip["remaining_length"]
                ip["packet"] = bytearray(ip["remaining_length"])
            with memoryview(ip["packet"]) as view:
                for _ in range(100):  # as paho: don't hog the loop on a huge packet
                    if ip["to_process"] == 0:
                        break
                    n = self._sock.recv_into(view[ip["remaining_length"] - ip["to_process"]:])
                    if n == 0:
                        return MQTTErrorCode.MQTT_ERR_CONN_LOST
                    ip["to_process"] -= n
        except (BlockingIOError, ssl.SSLWantReadError, ssl.SSLWantWriteError):
            return MQTTErrorCode.MQTT_ERR_AGAIN
        except OSError as err:
            self._easy_log(mqtt.MQTT_LOG_ERR, "failed to receive on socket: %s", err)
            return MQTTErrorCode.MQTT_ERR_CONN_LOST
        if ip["to_process"] > 0:
            with self._msgtime_mutex:
                self._last_msg_in = time_func()
            return MQTTErrorCode.MQTT_ERR_AGAIN
        return self._finish_packet()

    def _finish_packet(self) -> MQTTErrorCode:
        self._in_packet["pos"] = 0
        rc = self._packet_handle()
        self._in_packet = {"command": 0, "have_remaining": 0, "remaining_count": [], "remaining_mult": 1,
                           "remaining_length": 0, "packet": bytearray(b""), "to_process": 0, "pos": 0}
        with self._msgtime_mutex:
            self._last_msg_in = time_func()
        return rc

    def _handle_publish(self) -> MQTTErrorCode:
        packet = self._in_packet["packet"]
        if len(packet) < BIG_PACKET:
            return super()._handle_publish()
        header = self._in_packet["command"]
        mv = memoryview(packet)
        message = MQTTMessage()
        message.dup = ((header & 0x08) >> 3) != 0
        message.qos = (header & 0x06) >> 1
        message.retain = (header & 0x01) != 0
        slen = int.from_bytes(mv[0:2], "big")
        message.topic = bytes(mv[2:2 + slen])
        off = 2 + slen
        if message.qos > 0:
            message.mid = int.from_bytes(mv[off:off + 2], "big")
            off += 2
        if self._protocol == MQTTv5:
            plen, pstart = _varint(mv, off)
            message.properties = Properties(PUBLISH >> 4)
            message.properties.unpack(bytes(mv[off:pstart + plen]))
            off = pstart + plen
        message.payload = mv[off:]
        message.timestamp = time_func()
        if message.qos == 0:
            self._handle_on_message(message)
            return MQTTErrorCode.MQTT_ERR_SUCCESS
        if message.qos == 1:
            self._handle_on_message(message)
            return MQTTErrorCode.MQTT_ERR_SUCCESS if self._manual_ack else self._send_puback(message.mid)
        if message.qos == 2:
            rc = self._send_pubrec(message.mid)
            message.state = mqtt_ms_wait_for_pubrel
            with self._in_message_mutex:
                self._in_messages[message.mid] = message
            return rc
        return MQTTErrorCode.MQTT_ERR_PROTOCOL
