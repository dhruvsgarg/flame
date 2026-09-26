# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""FX-N13: the streaming figures pick up every P7/P7o baseline row, labelled by baseline."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../examples/async_cifar10/scripts"))
import felix_streaming_figures as fig  # noqa: E402


def _tsv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("trace\tbaseline\treal_dir\tsim_dir\n")
        for r in rows:
            f.write("\t".join(r) + "\n")


def test_campaign_arms_label_by_baseline(tmp_path):
    d = {k: str(tmp_path / k) for k in ("fr", "fs", "or", "os")}
    for v in d.values():
        os.makedirs(v)
    _tsv(str(tmp_path / "P7" / "summary.tsv"), [("syn_0", "felix", d["fr"], d["fs"])])
    _tsv(str(tmp_path / "P7o" / "summary.tsv"), [("syn_0", "oort", d["or"], d["os"])])
    arms = fig.campaign_arms(str(tmp_path))
    assert [a[0] for a in arms] == ["felix real", "felix sim", "oort+oracle real", "oort+oracle sim"]
    assert set(fig.by_baseline(arms)) == {"felix", "oort"}
    assert [a[0] for a in fig.by_baseline(arms)["oort"]] == ["oort+oracle real", "oort+oracle sim"]
