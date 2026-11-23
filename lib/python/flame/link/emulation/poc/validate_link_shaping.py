"""Parity check: Toxiproxy shaped from a link-layer estimate vs. the estimate.

Builds a narrow link (localhost MQTT tops out at a few tens of Mbps, far
below a full Ku-band channel), shapes a ToxiproxyShaper from the satellite's
download/upload TransferEstimates, runs peer.py through the proxy, and
compares measured transfer time to transmit_s + propagation_s.

Needs a running MQTT broker; starts toxiproxy-server if its API is down.
Run from the repo root:
  python lib/python/flame/link/emulation/poc/validate_link_shaping.py
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import uuid

from flame.link.emulation.shaper import ToxiproxyShaper
from flame.link.emulation.toxiproxy import ToxiproxyClient
from flame.link.runtime import LinkRuntime

PEER_PY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "peer.py")


def ensure_toxiproxy(client: ToxiproxyClient):
    try:
        client._request("GET", "/version")
        return None
    except Exception:
        proc = subprocess.Popen(["toxiproxy-server"], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        time.sleep(1)
        return proc


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--broker-host", default="127.0.0.1")
    parser.add_argument("--broker-port", type=int, default=1883)
    parser.add_argument("--satellite-index", type=int, default=289)
    parser.add_argument("--start-s", type=float, default=3214.0)
    parser.add_argument("--bytes", type=int, default=1048576)
    parser.add_argument("--bandwidth-mhz", type=float, default=1.0)
    parser.add_argument("--efficiency", type=float, default=0.25)
    parser.add_argument("--ecef", default="lib/python/examples/_metadata/leo/ecef.npz")
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    raw = {
        "enabled": True,
        "satelliteEcefPath": args.ecef,
        "frequency": {"bandwidthMhz": args.bandwidth_mhz},
        "transfer": {"efficiency": args.efficiency},
    }
    uplink = LinkRuntime.from_link_layer_dict(raw, direction="uplink")
    downlink = LinkRuntime(uplink.cfg, uplink.ecef_km, direction="downlink")
    est_down = uplink.transfer(args.satellite_index, args.start_s, args.bytes)
    est_up = downlink.transfer(args.satellite_index, args.start_s, args.bytes)

    sidecar = ensure_toxiproxy(ToxiproxyClient())
    shaper = ToxiproxyShaper(uplink.cfg, args.satellite_index, f"{args.broker_host}:{args.broker_port}")
    shaper.shape("downstream", est_down)
    shaper.shape("upstream", est_up)
    proxy_host, proxy_port = shaper.broker_address.split(":")

    run_id = f"linkshape_{uuid.uuid4().hex[:8]}"
    out = tempfile.mkdtemp(prefix="linkshape_")
    common = ["--run-id", run_id, "--down-bytes", str(args.bytes), "--up-bytes", str(args.bytes),
              "--timeout", str(args.timeout)]
    try:
        server = subprocess.Popen([sys.executable, PEER_PY, "--role", "server", "--host", args.broker_host,
                                   "--port", str(args.broker_port),
                                   "--result-file", os.path.join(out, "server.jsonl"), *common])
        client = subprocess.Popen([sys.executable, PEER_PY, "--role", "client", "--host", proxy_host,
                                   "--port", proxy_port,
                                   "--result-file", os.path.join(out, "client.jsonl"), *common])
        rc = (server.wait(timeout=args.timeout + 10), client.wait(timeout=args.timeout + 10))
    finally:
        shaper.close()
        if sidecar is not None:
            sidecar.terminate()
    if rc != (0, 0):
        sys.exit(f"peer failed: server/client exit codes {rc}")

    with open(os.path.join(out, "client.jsonl")) as f:
        measured_down = json.loads(f.readline())["elapsed_down_s"]
    with open(os.path.join(out, "server.jsonl")) as f:
        measured_up = json.loads(f.readline())["elapsed_up_s"]

    print(f"{'leg':9s} {'rate_mbps':>9s} {'expected_s':>10s} {'measured_s':>10s} {'error':>7s}")
    for leg, est, meas in (("download", est_down, measured_down), ("upload", est_up, measured_up)):
        expected = est.transmit_s + est.propagation_s
        print(f"{leg:9s} {est.throughput_mbps:9.2f} {expected:10.3f} {meas:10.3f} "
              f"{(meas - expected) / expected * 100:6.1f}%")


if __name__ == "__main__":
    main()
