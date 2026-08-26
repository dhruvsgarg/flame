"""run_sweep.py -- orchestrates the full profile x payload_size sweep."""

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
import uuid

import yaml

from toxiproxy_ctl import ToxiproxyClient, NetworkProfile

CSV_COLUMNS = [
    "profile", "payload_label", "down_bytes", "up_bytes",
    "latency_ms", "jitter_ms", "bw_down_mbps", "bw_up_mbps",
    "elapsed_down_s", "expected_down_s", "throughput_down_mbps",
    "elapsed_up_s", "expected_up_s", "throughput_up_mbps",
    "status",
]

PROXY_NAME = "netpoc_sweep"
PROXY_LISTEN_PORT = 22220


def check_broker_alive():
    try:
        alive = subprocess.run(["pgrep", "mosquitto"], capture_output=True).returncode == 0
        if not alive:
            print("WARNING: mosquitto not detected via pgrep -- continuing anyway", file=sys.stderr)
    except FileNotFoundError:
        print("WARNING: couldn't check broker liveness (pgrep missing)", file=sys.stderr)


def start_toxiproxy_sidecar():
    log_file = open("toxiproxy_sidecar.log", "w")
    proc = subprocess.Popen(["toxiproxy-server"], stdout=log_file, stderr=subprocess.STDOUT)
    time.sleep(1)
    return proc


def stop_toxiproxy_sidecar(proc):
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def resolve_direction(profile_cfg, direction_key):
    cfg = profile_cfg.get(direction_key, profile_cfg["down"])
    return NetworkProfile(
        latency_ms=cfg["latency_ms"],
        jitter_ms=cfg["jitter_ms"],
        bandwidth_mbps=cfg["bandwidth_mbps"],
    )


def expected_seconds(latency_ms, bandwidth_mbps, num_bytes):
    latency_s = latency_ms / 1000.0
    transmission_s = (num_bytes * 8) / (bandwidth_mbps * 1e6)
    return latency_s + transmission_s


def run_one_combo(client: ToxiproxyClient, args, profile_cfg, size_cfg):
    down_profile = resolve_direction(profile_cfg, "down")
    up_profile = resolve_direction(profile_cfg, "up")

    client.create_proxy(PROXY_NAME, f"127.0.0.1:{PROXY_LISTEN_PORT}",
                         f"{args.broker_host}:{args.broker_port}")
    client.apply_profile(PROXY_NAME, down_profile, "downstream")
    client.apply_profile(PROXY_NAME, up_profile, "upstream")

    run_id = f"{profile_cfg['name']}_{size_cfg['label']}_{uuid.uuid4().hex[:8]}"
    server_result_file = os.path.join(args.out_dir, f"{run_id}_server.jsonl")
    client_result_file = os.path.join(args.out_dir, f"{run_id}_client.jsonl")

    server_proc = subprocess.Popen([
        sys.executable, "peer.py", "--role", "server", "--run-id", run_id,
        "--host", args.broker_host, "--port", str(args.broker_port),
        "--down-bytes", str(size_cfg["bytes"]), "--up-bytes", str(size_cfg["bytes"]),
        "--result-file", server_result_file, "--timeout", str(args.timeout),
    ])
    client_proc = subprocess.Popen([
        sys.executable, "peer.py", "--role", "client", "--run-id", run_id,
        "--host", "127.0.0.1", "--port", str(PROXY_LISTEN_PORT),
        "--down-bytes", str(size_cfg["bytes"]), "--up-bytes", str(size_cfg["bytes"]),
        "--result-file", client_result_file, "--timeout", str(args.timeout),
    ])

    try:
        server_rc = server_proc.wait(timeout=args.timeout + 10)
        client_rc = client_proc.wait(timeout=args.timeout + 10)
    except subprocess.TimeoutExpired:
        server_proc.kill()
        client_proc.kill()
        server_rc = client_rc = -1

    client.delete_proxy(PROXY_NAME)

    row = {col: "" for col in CSV_COLUMNS}
    row.update({
        "profile": profile_cfg["name"],
        "payload_label": size_cfg["label"],
        "down_bytes": size_cfg["bytes"],
        "up_bytes": size_cfg["bytes"],
        "latency_ms": down_profile.latency_ms,
        "jitter_ms": down_profile.jitter_ms,
        "bw_down_mbps": down_profile.bandwidth_mbps,
        "bw_up_mbps": up_profile.bandwidth_mbps,
    })

    if server_rc != 0 or client_rc != 0:
        row["status"] = "FAILED"
        return row

    try:
        with open(server_result_file) as f:
            server_result = json.loads(f.readline())
        with open(client_result_file) as f:
            client_result = json.loads(f.readline())
    except (FileNotFoundError, json.JSONDecodeError):
        row["status"] = "FAILED"
        return row

    row["elapsed_down_s"] = client_result["elapsed_down_s"]
    row["throughput_down_mbps"] = client_result["throughput_down_mbps"]
    row["expected_down_s"] = expected_seconds(
        down_profile.latency_ms, down_profile.bandwidth_mbps, size_cfg["bytes"])

    row["elapsed_up_s"] = server_result["elapsed_up_s"]
    row["throughput_up_mbps"] = server_result["throughput_up_mbps"]
    row["expected_up_s"] = expected_seconds(
        up_profile.latency_ms, up_profile.bandwidth_mbps, size_cfg["bytes"])

    row["status"] = "OK"
    return row


def main():
    parser = argparse.ArgumentParser(description="network_poc sweep orchestrator")
    parser.add_argument("--broker-host", default="127.0.0.1")
    parser.add_argument("--broker-port", type=int, default=1883)
    parser.add_argument("--profiles-file", default="profiles.yaml")
    parser.add_argument("--profiles", nargs="*", default=None)
    parser.add_argument("--sizes", nargs="*", default=None)
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    check_broker_alive()

    with open(args.profiles_file) as f:
        config = yaml.safe_load(f)

    profiles = config["profiles"]
    sizes = config["payload_sizes_bytes"]
    if args.profiles:
        profiles = [p for p in profiles if p["name"] in args.profiles]
    if args.sizes:
        sizes = [s for s in sizes if s["label"] in args.sizes]

    sidecar = start_toxiproxy_sidecar()
    client = ToxiproxyClient()
    rows = []

    try:
        for profile_cfg in profiles:
            for size_cfg in sizes:
                print(f"running {profile_cfg['name']} x {size_cfg['label']} ...")
                row = run_one_combo(client, args, profile_cfg, size_cfg)
                rows.append(row)
                print(f"  -> {row['status']}")
    finally:
        stop_toxiproxy_sidecar(sidecar)

    csv_path = os.path.join(args.out_dir, f"netpoc_sweep_{int(time.time())}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print("\n--- summary ---")
    for row in rows:
        print(f"{row['profile']:15s} {row['payload_label']:18s} {row['status']:8s} "
              f"down={row['elapsed_down_s']} expected={row['expected_down_s']}")
    print(f"\nCSV written to: {csv_path}")


if __name__ == "__main__":
    main()