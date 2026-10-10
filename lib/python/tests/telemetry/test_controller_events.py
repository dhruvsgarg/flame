# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Schema tests for the fwdllm controller telemetry (cheatsheet concept table)."""

import math

import pytest

from flame.telemetry.events import (
    EVENT_BMAX_PROBE,
    EVENT_RUN_META,
    EVENT_SAT_STATE,
    KNOWN_EVENTS,
    build_bmax_probe,
    build_run_meta,
    build_sat_state,
    build_server_update,
)

_BASE = dict(round_num=1, data_id=3, iteration=1, model_version=9,
             update_delta_norm=0.5, weight_norm=4.0, learning_rate=1e-3)


def test_new_events_registered():
    assert {EVENT_RUN_META, EVENT_BMAX_PROBE, EVENT_SAT_STATE} <= KNOWN_EVENTS


def test_run_meta_drops_none_and_keeps_scope():
    ev, f = build_run_meta(scope="aggregator", config={"rho_star": 0.06, "t_res": None})
    assert ev == EVENT_RUN_META
    assert f == {"scope": "aggregator", "rho_star": 0.06}


def test_server_update_controller_fields():
    _, f = build_server_update(
        **_BASE, budget_b=0.1, budget_b_max=math.log(2), rho_star=0.06, n_req=19.0,
        commit_count=12, g_norm=3.0, step_skipped=False, rho_max=0.1,
        pool_size=20, pool_mean_sq=2.0, var_dim=768,
        g_rule=10.0, p_trainable=118_348, safety_s=1.5,
        cos_ground_truth=0.05,
    )
    assert f["commit_count"] == 12 and f["g_norm"] == 3.0 and f["step_skipped"] is False
    assert f["progress_lambda"] == pytest.approx(2 * 0.1 / 1.5)
    cos_th = math.sqrt(10.0 * 20 / 118_348)
    assert f["cos_theory"] == pytest.approx(cos_th)
    assert f["aim_d"] == pytest.approx(0.05 / cos_th)


def test_server_update_omits_absent_controller_fields():
    _, f = build_server_update(**_BASE)
    for k in ("commit_count", "g_norm", "cos_theory", "aim_d", "progress_lambda"):
        assert k not in f


def test_bmax_probe_record():
    ev, f = build_bmax_probe(
        commit_count=150, base_acc=0.8, phis=(1.5, 2.0), accs=[0.7, 0.4],
        phi_knee=1.25, b_rem=0.22, b_max_before=0.69, b_max_after=0.9,
        budget_b=0.68, policy="anchor",
    )
    assert ev == EVENT_BMAX_PROBE
    assert f["phis"] == [1.5, 2.0] and f["b_max_after"] == 0.9


def test_sat_state_record():
    ev, f = build_sat_state(
        commit_count=40, acc=0.85, smoothed=0.84, best=0.86, gl=0.023,
        progress=0.001, stalls=3, breaches=0, fired_at=None, fired_reason=None,
    )
    assert ev == EVENT_SAT_STATE
    assert f["gl"] == 0.023 and f["stalls"] == 3


def test_detector_exposes_smoothed_and_gl():
    from examples.fwdllm.expts.saturation_stop import SaturationDetector
    d = SaturationDetector(warmup=0, window=1, slope_horizon=1)
    d.update(0, 0.8)
    d.update(1, 0.6)
    assert d.last_smoothed == pytest.approx(0.6)
    assert d.last_gl == pytest.approx(0.25)
