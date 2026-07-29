# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for analyze_run.py's redispatch_decomp/vclock_charge
readers (simulate_fwdllm.md §F-8 dark-data fix: these two events had no
analyze_run.py reader despite being emitted since 07-27/07-28)."""

import os
import sys

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..",
                 "scripts", "analysis"),
)

from analyze_run import redispatch_decomp_plots, vclock_charge_plots  # noqa: E402


def _redispatch_event(time_mode, payload_kind, post_close, peer_wait, gap):
    return {
        "event": "redispatch_decomp",
        "time_mode": time_mode,
        "payload_kind": payload_kind,
        "post_close_overhead_wall_s": post_close,
        "peer_wait_wall_s": peer_wait,
        "redispatch_gap_wall_s": gap,
    }


def _charge_event(time_mode, label, payload_kind, span_s, charged_s):
    return {
        "event": "vclock_charge",
        "time_mode": time_mode,
        "label": label,
        "payload_kind": payload_kind,
        "span_s": span_s,
        "charged_s": charged_s,
    }


def test_redispatch_decomp_plots_no_data_is_a_placeholder(tmp_path):
    paths = redispatch_decomp_plots([], str(tmp_path), "stamp", str(tmp_path))
    assert len(paths) == 1
    assert os.path.exists(paths[0])


def test_redispatch_decomp_plots_writes_cdf_and_bar(tmp_path):
    records = [
        _redispatch_event("real", "weights", 0.46, 0.1, 0.56),
        _redispatch_event("real", "weights", 0.52, 0.05, 0.57),
        _redispatch_event("sim", "weights", 0.03, 0.1, 0.13),
        _redispatch_event("sim", "weights", 0.04, 0.08, 0.12),
    ]
    paths = redispatch_decomp_plots(records, str(tmp_path), "stamp", str(tmp_path))
    assert len(paths) == 4  # 3 CDFs (post_close/peer_wait/gap) + 1 mean bar
    for p in paths:
        assert os.path.exists(p)
    assert any("post_close_overhead_cdf" in p for p in paths)
    assert any("post_close_mean_bar" in p for p in paths)


def test_vclock_charge_plots_no_data_is_a_placeholder(tmp_path):
    paths = vclock_charge_plots([], str(tmp_path), "stamp", str(tmp_path))
    assert len(paths) == 1
    assert os.path.exists(paths[0])


def test_vclock_charge_plots_writes_cdf_and_gap_bar(tmp_path):
    records = [
        _charge_event("real", "redispatch_turnaround", "weights", 0.50, 0.0),
        _charge_event("real", "redispatch_turnaround", "weights", 0.46, 0.0),
        _charge_event("sim", "redispatch_turnaround", "weights", 0.03, 0.4365),
        _charge_event("sim", "redispatch_turnaround", "weights", 0.04, 0.4365),
    ]
    paths = vclock_charge_plots(records, str(tmp_path), "stamp", str(tmp_path))
    assert len(paths) == 2  # span CDF + uncharged-gap bar
    for p in paths:
        assert os.path.exists(p)
    assert any("span_cdf" in p for p in paths)
    assert any("uncharged_gap_bar" in p for p in paths)


def test_vclock_charge_plots_skips_gap_bar_without_both_modes(tmp_path):
    """Real-only or sim-only telemetry (e.g. a real-only smoke run) still
    yields the span CDF but no gap bar -- nothing to diff against."""
    records = [_charge_event("real", "drain_tail", None, 0.33, 0.0)]
    paths = vclock_charge_plots(records, str(tmp_path), "stamp", str(tmp_path))
    assert not any("uncharged_gap_bar" in p for p in paths)
