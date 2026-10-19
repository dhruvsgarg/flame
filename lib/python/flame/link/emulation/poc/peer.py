"""peer.py -- implements the client/server data flow

Implements the handshake + timed down/up data transfer state machine over MQTT.
"""

import argparse
import json
import random
import sys
import threading
import time

from chunked_mqtt import make_client, make_topic, ChunkedSender, ChunkedReceiver


class CtrlEvents:
    def __init__(self):
        self.hello = threading.Event()
        self.ready = threading.Event()
        self.start_down = threading.Event()
        self.down_ack = threading.Event()
        self.start_up = threading.Event()
        self.done = threading.Event()

    def on_ctrl_message(self, client, userdata, msg):
        text = msg.payload.decode("utf-8")
        {
            "HELLO": self.hello,
            "READY": self.ready,
            "START_DOWN": self.start_down,
            "DOWN_ACK": self.down_ack,
            "START_UP": self.start_up,
            "DONE": self.done,
        }.get(text, threading.Event()).set()


def wait_or_die(event: threading.Event, name: str, timeout: float):
    if not event.wait(timeout):
        print(f"TIMEOUT waiting for {name} after {timeout}s", file=sys.stderr)
        sys.exit(1)


def publish_ctrl(client, ctrl_topic, message: str):
    client.publish(ctrl_topic, payload=message.encode("utf-8"), qos=1)


def run_server(args):
    events = CtrlEvents()
    client = make_client(f"netpoc_server_{args.run_id}")
    client.connect(args.host, args.port)
    client.loop_start()

    ctrl_topic = make_topic(args.run_id, "ctrl")
    down_topic = make_topic(args.run_id, "data/down")
    up_topic = make_topic(args.run_id, "data/up")

    client.message_callback_add(ctrl_topic, events.on_ctrl_message)
    client.subscribe(ctrl_topic, qos=1)
    up_receiver = ChunkedReceiver(client, up_topic)

    wait_or_die(events.hello, "HELLO", args.timeout)
    publish_ctrl(client, ctrl_topic, "READY")

    wait_or_die(events.start_down, "START_DOWN", args.timeout)
    down_payload = random.randbytes(args.down_bytes)
    down_sender = ChunkedSender(client, down_topic)
    down_sender.send(down_payload)

    wait_or_die(events.down_ack, "DOWN_ACK", args.timeout)
    t0 = time.time()
    publish_ctrl(client, ctrl_topic, "START_UP")

    up_payload = up_receiver.wait_for_payload(timeout=args.timeout)
    t1 = time.time()
    assert len(up_payload) == args.up_bytes, (
        f"expected {args.up_bytes} bytes, got {len(up_payload)}"
    )
    elapsed_up = t1 - t0
    throughput_up_mbps = (args.up_bytes * 8) / elapsed_up / 1e6

    publish_ctrl(client, ctrl_topic, "DONE")

    result = {
        "role": "server",
        "run_id": args.run_id,
        "elapsed_up_s": elapsed_up,
        "bytes_up": args.up_bytes,
        "throughput_up_mbps": throughput_up_mbps,
    }
    with open(args.result_file, "a") as f:
        f.write(json.dumps(result) + "\n")

    client.loop_stop()
    client.disconnect()
    sys.exit(0)


def run_client(args):
    events = CtrlEvents()
    client = make_client(f"netpoc_client_{args.run_id}")
    client.connect(args.host, args.port)
    client.loop_start()

    ctrl_topic = make_topic(args.run_id, "ctrl")
    down_topic = make_topic(args.run_id, "data/down")
    up_topic = make_topic(args.run_id, "data/up")

    client.message_callback_add(ctrl_topic, events.on_ctrl_message)
    client.subscribe(ctrl_topic, qos=1)
    down_receiver = ChunkedReceiver(client, down_topic)

    publish_ctrl(client, ctrl_topic, "HELLO")

    wait_or_die(events.ready, "READY", args.timeout)
    t0 = time.time()
    publish_ctrl(client, ctrl_topic, "START_DOWN")

    down_payload = down_receiver.wait_for_payload(timeout=args.timeout)
    t1 = time.time()
    assert len(down_payload) == args.down_bytes, (
        f"expected {args.down_bytes} bytes, got {len(down_payload)}"
    )
    elapsed_down = t1 - t0
    throughput_down_mbps = (args.down_bytes * 8) / elapsed_down / 1e6
    publish_ctrl(client, ctrl_topic, "DOWN_ACK")

    wait_or_die(events.start_up, "START_UP", args.timeout)
    up_payload = random.randbytes(args.up_bytes)
    up_sender = ChunkedSender(client, up_topic)
    up_sender.send(up_payload)

    wait_or_die(events.done, "DONE", args.timeout)

    result = {
        "role": "client",
        "run_id": args.run_id,
        "elapsed_down_s": elapsed_down,
        "bytes_down": args.down_bytes,
        "throughput_down_mbps": throughput_down_mbps,
    }
    with open(args.result_file, "a") as f:
        f.write(json.dumps(result) + "\n")

    client.loop_stop()
    client.disconnect()
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description="network_poc peer process")
    parser.add_argument("--role", choices=["server", "client"], required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--down-bytes", type=int, default=1048576)
    parser.add_argument("--up-bytes", type=int, default=1048576)
    parser.add_argument("--result-file", required=True)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    if args.role == "server":
        run_server(args)
    else:
        run_client(args)


if __name__ == "__main__":
    main()