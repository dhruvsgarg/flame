# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""PARITY C5 readiness score: phase -> scenario mapping and worst-pair-wins scoring."""

import importlib.util
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "examples" / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("readiness_score", SCRIPTS / "readiness_score.py")
rs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rs)


def test_scenario_mapping():
    assert rs.scenario("T3_syn_0b", "syn_0") == "syn_0"
    assert rs.scenario("gs_G0U_mobiperf_3st", "mobiperf_3st") == "mobiperf"
    assert rs.scenario("G0T_lin_syn_50", "syn_50") == "stream_lin"
    assert rs.scenario("G0T_eve_syn_0", "syn_0") == "stream_eve"
    assert rs.scenario("gs_P7", "syn_0") == "stream_cpu"
    for skipped in ("P11a", "P7o", "G0To_lin_syn_0", "G1AS", "T3C_syn_20s", "gs_G0UC_syn_50"):
        assert rs.scenario(skipped, "syn_50") is None, skipped


def test_worst_pair_wins_and_na(monkeypatch):
    C = rs.pl.Cell
    monkeypatch.setattr(rs.pl, "grade_pool", lambda root, s: [C("T3_syn_50", "syn_50", "felix", "green", ""),
                                                               C("G0U_syn_50", "syn_50", "felix", "red", "x")])
    cells = rs.score(["p"])
    assert cells[("felix", "cifar10", "syn_50")] == "red"
    assert cells[("oort", "google_speech", "mobiperf")] == "n/a"
    assert "0 / 82 green" in rs.render(cells)
