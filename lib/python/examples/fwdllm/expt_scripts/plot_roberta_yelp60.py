#!/usr/bin/env python3
# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Render the RoBERTa-L · Yelp-P to-0.60 figures (run_node_roberta_yelp_to60_queue.sh) from its queue dir.

    python plot_roberta_yelp60.py <fwdllm/rb60_<ts>> [--target 0.60] [--n 100] [--synthetic]

Writes an e2e manifest (row + B1-B3, plots #2, 6, 9, 12) to <qdir>/manifests/ and renders it with
make_paper_figs.py into <qdir>/paper_figs/. Only runs at the final n are used; works mid-queue.
"""

import argparse
import csv
import os
import subprocess
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("qdir")
    ap.add_argument("--target", type=float, default=0.60)
    ap.add_argument("--n", type=int, default=None, help="trainer count to plot (default: last row's)")
    ap.add_argument("--synthetic", action="store_true", help="layout preview (make_paper_figs --synthetic)")
    a = ap.parse_args()
    qdir = os.path.abspath(a.qdir)

    rows = []
    tsv = os.path.join(qdir, "runs.tsv")
    if os.path.exists(tsv):
        with open(tsv) as fh:
            rows = [r for r in csv.reader(fh, delimiter="\t") if len(r) >= 4 and r[2] != "none"]
    n = a.n or (int(rows[-1][1]) if rows else 100)
    runs = {arm: [os.path.join("experiments", run)] for arm, rn, run, _ in rows if int(rn) == n}  # last wins
    print(f"[plot_rb60] n {n}: {' '.join(sorted(runs)) or '-'}")

    man = {"kind": "e2e", "prefix": "e2e", "baseline": "fwdllm", "champion": "fluxtune",
           "tta_consecutive": 1, "figures": ["row", "b1", "b2", "b3"],
           "cells": [{"id": "roberta_large_yelp", "label": f"RoBERTa-L · Yelp-P (to {a.target:.2f})",
                      "n_clients": n, "target_acc": a.target,
                      "arms": {"fwdllm": runs.get("fwdllm", []), "fluxtune": runs.get("fluxtune", [])}}]}
    mdir = os.path.join(qdir, "manifests")
    os.makedirs(mdir, exist_ok=True)
    path = os.path.join(mdir, "figs_e2e_rb60.yaml")
    with open(path, "w") as fh:
        yaml.safe_dump(man, fh, sort_keys=False, allow_unicode=True)
    cmd = [sys.executable, os.path.join(HERE, "make_paper_figs.py"), "--manifest", path,
           "--out-root", os.path.join(qdir, "paper_figs"), "--png"] + (["--synthetic"] if a.synthetic else [])
    rc = subprocess.call(cmd, cwd=HERE)
    print(f"[plot_rb60] figures in {os.path.join(qdir, 'paper_figs')}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
