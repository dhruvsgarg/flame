#!/usr/bin/env python3
# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Render the ag60 preview figures (run_node_agnews_to60_queue.sh) from its queue dir.

    python plot_ag60.py <fwdllm/ag60_<ts>> [more queue dirs ...] [--target 0.60] [--synthetic]

Builds e2e / ablation / availability manifests from <qdir>/runs.tsv into <qdir>/manifests/ and renders each
with make_paper_figs.py into <qdir>/paper_figs/<manifest>/. Arms not yet run are skipped, so it works mid-queue.
Several queue dirs (a resumed queue) merge their runs.tsv in order; output goes to the last one.
"""

import argparse
import csv
import os
import subprocess
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))

# (group id, title, figures, [(arm label in runs.tsv, manifest key, legend, sync)])
GROUPS = [
    ("phase_a", "FwdLLM + one FluxTune concept (C1)", ["group"],
     [("A0", "fwdllm", "FwdLLM (A0)", True), ("A1", "a1", "+ mean of 10 probes", True),
      ("A2", "a2", "+ trust-ratio step", True), ("A3", "a3", "+ fixed pool", True),
      ("A4", "a4", "+ async (FedBuff-IT)", False), ("A6", "fluxtune", "FluxTune (A6)", False)]),
    ("phase_a_pairs", "FwdLLM + two concepts (C1 pairs)", ["group"],
     [("A0", "fwdllm", "FwdLLM", True), ("A12", "a12", "+ E1 + E2", True),
      ("A13", "a13", "+ E1 + E3", True), ("A23", "a23", "+ E2 + E3", True),
      ("A6", "fluxtune", "FluxTune", False)]),
    ("phase_b", "Cumulative: FwdLLM → FluxTune (C2)", ["group"],
     [("A0", "fwdllm", "FwdLLM", True), ("A1", "b1", "+ mean probes", True),
      ("A12", "b2", "+ trust ratio", True), ("A123", "b3", "+ fixed pool", True),
      ("A6", "fluxtune", "+ async = FluxTune", False)]),
    ("agg_criteria", "Aggregation criterion (C3)", ["group"],
     [("Cnt", "ac_snr", "SNR target (n_target)", False), ("CN10", "ac_n10", "fixed N 10", False),
      ("CN20", "ac_n20", "fixed N 20", False), ("A6", "fluxtune", "fixed N 50", False),
      ("CN90", "ac_n90", "fixed N 90", False)]),
    ("update_utility", "Update combination (C3)", ["group"],
     [("A0", "fwdllm", "grad-aligned 1-of-10 (FwdLLM)", True),
      ("A6", "fluxtune", "mean of 10 probes", False)]),
    ("step_size", "Server step size (C3)", ["group", "step_size"],
     [("Spool", "ss_const", "trust ratio, const ρ", False), ("Srm", "ss_rm", "trust ratio, ρ0·t^-0.11", False),
      ("A6", "fluxtune", "trust ratio, ρ0·exp(−t/2T_res)", False),
      ("Smom", "ss_mom", "exp + momentum 0.5", False)]),
    ("agg_weights", "Aggregation weights (C3)", ["group"],
     [("A6", "fluxtune", "uniform ω", False), ("Wnew", "aw_new", "staleness ω (Felix)", False),
      ("Wga", "aw_ga", "staleness × alignment (grad_aware)", False)]),
    ("rho_max", "ρ0 at N 90 (R1/R2, gating)", ["group", "step_size"],
     [("CN90", "fluxtune", "ρ0 0.07, T_res 283", False), ("R1", "r1", "ρ0 0.10, T_res 139", False),
      ("R2", "r2", "ρ0 0.14, T_res 71", False)]),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("qdirs", nargs="+")
    ap.add_argument("--target", type=float, default=0.60)
    ap.add_argument("--synthetic", action="store_true", help="layout preview (make_paper_figs --synthetic)")
    a = ap.parse_args()
    qdir = os.path.abspath(a.qdirs[-1])

    runs = {}
    for d in a.qdirs:
        tsv = os.path.join(d, "runs.tsv")
        if os.path.exists(tsv):
            with open(tsv) as fh:
                for row in csv.reader(fh, delimiter="\t"):
                    if len(row) >= 4 and row[3]:
                        runs[row[0]] = [os.path.join("experiments", row[3])]  # last run of a label wins
    print(f"[plot_ag60] {len(runs)} arm(s) with runs: {' '.join(sorted(runs)) or '-'}")

    cell = {"id": "distilbert_agnews", "label": "DistilBERT · AG News (preview, to 0.60)",
            "n_clients": 100, "target_acc": a.target}
    r = lambda k: runs.get(k, [])  # noqa: E731
    manifests = {
        "figs_e2e_ag60": {"kind": "e2e", "prefix": "e2e", "baseline": "fwdllm", "champion": "fluxtune",
                          "tta_consecutive": 1, "figures": ["row", "b1", "b2", "b3"],
                          "cells": [{**cell, "arms": {"fwdllm": r("A0"), "fluxtune": r("A6")}}]},
        "figs_ablation_ag60": {"kind": "ablation", "baseline": "fwdllm", "champion": "fluxtune",
                               "tta_consecutive": 1, "cell": cell,
                               "groups": [{"id": g, "title": t, "figures": f,
                                           "arms": [{"key": k, "label": lab, "sync": s, "runs": r(arm)}
                                                    for arm, k, lab, s in arms]}
                                          for g, t, f, arms in GROUPS]},
        "figs_availability_ag60": {"kind": "e2e", "prefix": "c_avail", "baseline": "felix_round",
                                   "champion": "fluxtune", "tta_consecutive": 1, "figures": ["row"],
                                   "cells": [{**cell, "id": "syn_0", "label": "100% available (syn_0)",
                                              "arms": {"fwdllm": r("A0"), "fedbuff_round": r("C4fb"),
                                                       "felix_round": r("C4fx"), "fluxtune": r("A6")}}]},
    }

    mdir = os.path.join(qdir, "manifests")
    os.makedirs(mdir, exist_ok=True)
    rc = 0
    for stem, man in manifests.items():
        path = os.path.join(mdir, f"{stem}.yaml")
        with open(path, "w") as fh:
            yaml.safe_dump(man, fh, sort_keys=False, allow_unicode=True)
        cmd = [sys.executable, os.path.join(HERE, "make_paper_figs.py"), "--manifest", path,
               "--out-root", os.path.join(qdir, "paper_figs", stem), "--png"] + (["--synthetic"] if a.synthetic else [])
        print(f"[plot_ag60] {stem}")
        rc |= subprocess.call(cmd, cwd=HERE)
    print(f"[plot_ag60] figures in {os.path.join(qdir, 'paper_figs')}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
