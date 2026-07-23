#!/usr/bin/env python3
# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Offline MQTT transport-leg profiler for fwdllm sim calibration.

The sim skips real waits, so it cannot measure MQTT transport at runtime (the
publish is async and the two legs cross process boundaries). But a REAL run logs
genuine wall-clock timestamps on BOTH sides of each message, and both processes
share the host clock -- so the send->receive delta IS the transport time. This
extracts the two legs so `sim_completion_leg_s` can be set in the sim yaml
(sct = sim_send_ts + compute + leg):

  leg_i  (agg -> trainer, weights): trainer `task_recv.ts` - agg `agg_to_trainer.ts`
  leg_ii (trainer -> agg, grads):   agg "received gradients" wall - trainer `trainer_to_agg.ts`

Each receive is paired with its most-recent preceding send to the same peer
(FIFO per topic), robust to off-by-one and drops. Reports median/mean/p90 per
leg and the suggested `sim_completion_leg_s = median(leg_i) + median(leg_ii)`.

Usage:  python profile_transport_leg.py <real_run_dir> [--cap-s 300]
"""
import argparse
import glob
import json
import os
import re
import statistics
from datetime import datetime

_LOGTS = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d\d\d)")


def _load_jsonl(path):
    with open(path) as f:
        for line in f:
            try:
                yield json.loads(line)
            except (ValueError, json.JSONDecodeError):
                continue


def _agg_sends(agg_tel):
    """peer_id -> sorted [(ts, size_bytes, payload_kind)] (agg_to_trainer).
    payload_kind distinguishes the big WEIGHTS payload from the tiny VAR=bad
    'retry' message -- transport is size-dependent, so we keep it per-message."""
    d = {}
    for e in _load_jsonl(agg_tel):
        if e.get("event") == "comm" and e.get("direction") == "agg_to_trainer":
            d.setdefault(e.get("peer_id"), []).append(
                (e["ts"], e.get("size_bytes"), e.get("payload_kind")))
    for k in d:
        d[k].sort()
    return d


def _trainer_events(tr_tel):
    """(trainer_id, sorted weight-receive ts, sorted [(grad-send ts, size)])."""
    recv, gsend, tid = [], [], None
    for e in _load_jsonl(tr_tel):
        ev = e.get("event")
        if ev == "task_recv":
            recv.append(e["ts"])
            tid = tid or e.get("trainer_id")
        elif ev == "comm" and e.get("direction") == "trainer_to_agg":
            gsend.append((e["ts"], e.get("size_bytes")))
            tid = tid or e.get("trainer_id") or e.get("end_id")
    recv.sort()
    gsend.sort()
    return tid, recv, gsend


def _agg_grad_recvs(agg_log):
    """trainer_id -> sorted wall-unix ts of the agg's grad-receipt log line."""
    d = {}
    for line in open(agg_log):
        if "received gradients from " not in line:
            continue
        m = _LOGTS.match(line)
        mm = re.search(r"received gradients from (\S+)", line)
        if not (m and mm):
            continue
        ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S,%f").timestamp()
        d.setdefault(mm.group(1), []).append(ts)
    for k in d:
        d[k].sort()
    return d


def _nearest_preceding(sends, recv_ts, cap_s):
    """For each receive ts, the (delta, *meta) from its most-recent preceding
    send (same peer). `sends` is [(ts, *meta)]. Keeps only sane 0<d<cap_s."""
    out, j = [], 0
    sends = sorted(sends, key=lambda x: x[0])
    for r in sorted(recv_ts):
        while j + 1 < len(sends) and sends[j + 1][0] <= r:
            j += 1
        if j < len(sends) and sends[j][0] <= r:
            d = r - sends[j][0]
            if 0.0 < d < cap_s:
                out.append((d, *sends[j][1:]))
    return out


def _stats(ds):
    """Summary of transport deltas (first element of each tuple)."""
    xs = sorted(d[0] for d in ds)
    if not xs:
        return None
    p90 = xs[min(len(xs) - 1, int(0.9 * len(xs)))]
    return {"n": len(xs), "median_s": round(statistics.median(xs), 4),
            "mean_s": round(statistics.fmean(xs), 4), "p90_s": round(p90, 4)}


def _fit_base_bw(ds):
    """OLS fit transport_s = base_s + bytes/bandwidth over (delta, size) pairs.
    Returns (base_s, mbps) or None. `ds` = [(delta_s, size_bytes, ...)]."""
    pts = [(float(sz), d) for d, sz, *_ in ds if sz]
    n = len(pts)
    if n < 2:
        return None
    sx = sum(p[0] for p in pts); sy = sum(p[1] for p in pts)
    sxx = sum(p[0] * p[0] for p in pts); sxy = sum(p[0] * p[1] for p in pts)
    denom = n * sxx - sx * sx
    if denom == 0:
        return None
    slope = (n * sxy - sx * sy) / denom          # seconds per byte
    base = (sy - slope * sx) / n
    mbps = (1.0 / slope / 1e6) if slope > 0 else float("inf")
    return round(max(base, 0.0), 5), round(mbps, 1)


def _by_kind(ds):
    """median transport_s + median MB, per payload_kind (kind is 3rd tuple elt)."""
    groups = {}
    for d in ds:
        kind = d[2] if len(d) > 2 else None
        groups.setdefault(kind, []).append(d)
    out = {}
    for k, g in groups.items():
        sizes = [x[1] for x in g if x[1]]
        out[k] = {"n": len(g),
                  "median_s": round(statistics.median([x[0] for x in g]), 4),
                  "median_MB": round(statistics.median(sizes) / 1e6, 3) if sizes else None}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir")
    ap.add_argument("--cap-s", type=float, default=300.0,
                    help="discard paired deltas >= this (clock-skew guard)")
    args = ap.parse_args()

    agg_tel = glob.glob(os.path.join(args.run_dir, "telemetry", "aggregator_*.jsonl"))
    agg_log = glob.glob(os.path.join(args.run_dir, "*aggregator.log"))
    tr_tels = glob.glob(os.path.join(args.run_dir, "telemetry", "trainer_*.jsonl"))
    if not (agg_tel and agg_log and tr_tels):
        raise SystemExit(f"missing agg telemetry/log or trainer telemetry in {args.run_dir}")

    sends = _agg_sends(agg_tel[0])
    grecv = _agg_grad_recvs(agg_log[0])
    leg_i, leg_ii = [], []
    for tt in tr_tels:
        tid, recv, gsend = _trainer_events(tt)
        if not tid:
            continue
        leg_i += _nearest_preceding(sends.get(tid, []), recv, args.cap_s)
        leg_ii += _nearest_preceding(gsend, grecv.get(tid, []), args.cap_s)

    print(f"run: {os.path.basename(args.run_dir.rstrip('/'))}\n")
    print(f"  leg_i  (agg->trainer): {_stats(leg_i)}")
    for k, v in sorted(_by_kind(leg_i).items(), key=lambda x: -(x[1]['median_MB'] or 0)):
        print(f"      kind={k}: {v}")
    print(f"      size fit -> base_s, MB/s = {_fit_base_bw(leg_i)}")
    print(f"  leg_ii (trainer->agg): {_stats(leg_ii)}")
    print(f"      size fit -> base_s, MB/s = {_fit_base_bw(leg_ii)}")

    bk = _by_kind(leg_i)
    gii = _stats(leg_ii)
    if bk and gii:
        # A cycle's leg = agg->trainer delivery (WEIGHTS on a fresh model_version,
        # else the tiny VAR=bad retry) + trainer->agg grad delivery. Report both
        # so the yaml can use the size fit (preferred) or a per-cycle constant.
        w = next((v["median_s"] for k, v in bk.items() if k == "weights"), None)
        vb = next((v["median_s"] for k, v in bk.items() if k and k != "weights"), None)
        g = gii["median_s"]
        print("\n  per-cycle sim_completion_leg_s (median):")
        if w is not None:
            print(f"      WEIGHTS cycle: {round(w + g, 4)}  (weights {w} + grad {g})")
        if vb is not None:
            print(f"      VAR=bad cycle: {round(vb + g, 4)}  (var_bad {vb} + grad {g})")
        print("  PREFERRED: size-based leg (base_s + bytes/bandwidth) from the fits "
              "above -- the sim knows each payload's size, so it can charge the "
              "right leg per message instead of one scalar.")


if __name__ == "__main__":
    main()
