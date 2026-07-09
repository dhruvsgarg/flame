#!/usr/bin/env python3
# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Audit intra-databin weight-download redundancy from a run's telemetry.

Charter §5c Opt-1: within a data-bin the model_version is constant and the full
WEIGHTS payload is byte-identical across iterations, so a trainer should receive
it AT MOST ONCE per bin (later dispatches get the tiny VAR=bad "keep training"
message). Measures, per data-bin, how many full-weight sends each trainer got.

Robust to the `data_id` cycling footgun (data_id repeats within a round): the
true data-bin id is reconstructed by counting COMMITS (`agg_round` with
`var_good_enough=True`) in stream order; weight-sends for data-bin N precede
commit N.

Usage:
  python audit_weight_redundancy.py <run_dir_or_agg_jsonl> [--max-redundant-frac F]
Exit 1 if the redundant fraction exceeds --max-redundant-frac (default off), so
this doubles as a CI regression check over a smoke run.
"""
import argparse
import glob
import json
import os
import statistics as st
import sys
from collections import defaultdict


def _agg_jsonl(path):
    if os.path.isdir(path):
        hits = glob.glob(os.path.join(path, "telemetry", "aggregator_*.jsonl"))
        if not hits:
            sys.exit(f"no aggregator_*.jsonl under {path}/telemetry/")
        return hits[0]
    return path


def audit(path):
    f = _agg_jsonl(path)
    databin = 0                       # running commit count == true data-bin id
    # databin -> peer -> [n_weights, n_varbad]
    sends = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    wbytes = defaultdict(int)
    with open(f) as fh:
        for line in fh:
            # cheap prefilter before json.loads (files run >1 GB)
            is_comm = '"comm"' in line and "agg_to_trainer" in line
            is_round = '"agg_round"' in line
            if not (is_comm or is_round):
                continue
            try:
                e = json.loads(line)
            except Exception:
                continue
            ev = e.get("event")
            if ev == "agg_round":
                if e.get("var_good_enough"):
                    databin += 1      # commit -> next data-bin starts
            elif ev == "comm" and e.get("direction") == "agg_to_trainer":
                pk = e.get("payload_kind")
                peer = e.get("peer_id")
                if pk == "weights":
                    sends[databin][peer][0] += 1
                    wbytes[databin] += e.get("size_bytes", 0)
                elif pk == "var_bad":
                    sends[databin][peer][1] += 1
    return f, sends, wbytes


def report(path):
    f, sends, wbytes = audit(path)
    if not sends:
        print(f"{path}: no agg_to_trainer comm events found (WS3-a telemetry missing?)")
        return None
    uniq, wsent, redun_sends, redun_bytes = [], [], 0, 0
    tot_w = tot_vb = 0
    n_bins_with_repeat = 0
    for db, peers in sends.items():
        u = len(peers)
        w = sum(p[0] for p in peers.values())
        vb = sum(p[1] for p in peers.values())
        r = sum(max(0, p[0] - 1) for p in peers.values())   # sends beyond first/peer
        uniq.append(u); wsent.append(w); tot_w += w; tot_vb += vb
        redun_sends += r
        if r > 0:
            n_bins_with_repeat += 1
            if w:
                redun_bytes += (wbytes[db] / w) * r
    n = len(sends)
    frac = redun_sends / tot_w if tot_w else 0.0
    print(f"===== {os.path.basename(os.path.dirname(os.path.dirname(f))) or f} =====")
    print(f"data-bins (reconstructed via commit count): {n}")
    print(f"unique trainers/bin: median={st.median(uniq):.0f} max={max(uniq)}")
    print(f"weight-sends/bin:    median={st.median(wsent):.0f} max={max(wsent)}")
    print(f"total weight-sends={tot_w}  var_bad-sends={tot_vb}")
    print(f"data-bins with a trainer sent weights 2+ times: {n_bins_with_repeat}/{n} "
          f"({100*n_bins_with_repeat/n:.1f}%)")
    print(f"REDUNDANT weight-sends (same trainer 2+ in a bin): {redun_sends} "
          f"({100*frac:.1f}% of weight-sends)")
    print(f"REDUNDANT weight-bytes: {redun_bytes/1e9:.2f} GB")
    return frac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="run dir or aggregator_*.jsonl")
    ap.add_argument("--max-redundant-frac", type=float, default=None,
                    help="exit 1 if redundant weight-send fraction exceeds this")
    a = ap.parse_args()
    frac = report(a.path)
    if a.max_redundant_frac is not None and frac is not None and frac > a.max_redundant_frac:
        print(f"FAIL: redundant fraction {frac:.3f} > {a.max_redundant_frac}")
        sys.exit(1)


if __name__ == "__main__":
    main()
