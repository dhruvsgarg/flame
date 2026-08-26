"""Checkpoint A validation: prove Toxiproxy actually shapes traffic,
not just that the HTTP API accepts our JSON.

Run echo_server.py in one terminal first, then run this in another. Should show some toxicity (increased latency).
"""
import socket
import time

from toxiproxy_ctl import ToxiproxyClient, NetworkProfile

PROXY_NAME = "echo_test"
PROXY_LISTEN = "127.0.0.1:22222"   # what our client connects to
UPSTREAM = "127.0.0.1:9999"         # the real echo_server.py

MESSAGE = b"hello toxiproxy" * 100  # a small payload to send/receive


def send_and_time(host, port):
    """Connect, send MESSAGE, wait for the echo back, return elapsed seconds."""
    start = time.time()
    with socket.create_connection((host, port), timeout=10) as sock:
        sock.sendall(MESSAGE)
        received = sock.recv(65536)
    elapsed = time.time() - start
    assert received == MESSAGE, "echo didn't match what we sent!"
    return elapsed


def main():
    client = ToxiproxyClient()

    # clean slate: delete the proxy if it already exists from a previous run
    try:
        client.delete_proxy(PROXY_NAME)
    except RuntimeError:
        pass  # didn't exist yet, that's fine

    client.create_proxy(PROXY_NAME, PROXY_LISTEN, UPSTREAM)
    listen_host, listen_port = PROXY_LISTEN.split(":")

    # 1. baseline: no toxics, should be fast (a few ms)
    baseline = send_and_time(listen_host, int(listen_port))
    print(f"baseline (no toxic):      {baseline*1000:.1f} ms")

    # 2. apply a 500ms latency profile on the downstream direction
    profile = NetworkProfile(latency_ms=500, jitter_ms=0, bandwidth_mbps=1000)
    client.apply_profile(PROXY_NAME, profile, "downstream")

    shaped = send_and_time(listen_host, int(listen_port))
    print(f"with 500ms latency toxic: {shaped*1000:.1f} ms")
    print(f"difference:               {(shaped-baseline)*1000:.1f} ms  (expect ~500ms)")

    # cleanup
    client.delete_proxy(PROXY_NAME)
    print("cleaned up proxy, done.")


if __name__ == "__main__":
    main()