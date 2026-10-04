# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""FX-N10/FX-N22: debug_run.sh --dataset applies _metadata/datasets.yaml; --agg-goal/--concurrency set the
test shape through the launcher's single-source agg_goal. Runs the real make_debug_yaml heredoc."""

import pytest

from tests.launch.test_debug_run_trace_substitution import _run_generator, generator_source  # noqa: F401


def _one(generator_source, tmp_path, monkeypatch, baseline, **env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    return _run_generator(generator_source, tmp_path, baseline, "", mode="sim")[0]


def test_no_dataset_is_the_cifar_template(generator_source, tmp_path, monkeypatch):
    monkeypatch.delenv("DATASET", raising=False)
    e = _one(generator_source, tmp_path, monkeypatch, "felix")
    assert e["trainer"]["num_trainers"] == 300 and e["trainer"]["dataset"]["name"] == "cifar10"
    assert "dataset_name" not in e["aggregator"]["config_overrides"]["hyperparameters"]


@pytest.mark.parametrize("baseline,fedbuff_opt", [("felix", True), ("fedbuff", True), ("oort", False)])
def test_google_speech_profile(generator_source, tmp_path, monkeypatch, baseline, fedbuff_opt):
    e = _one(generator_source, tmp_path, monkeypatch, baseline, DATASET="google_speech")
    assert e["trainer"]["num_trainers"] == 100 and e["trainer"]["dataset"]["name"] == "google_speech"
    assert e["trainer"]["config_overrides"]["hyperparameters"]["dataset_name"] == "google_speech"
    agg = e["aggregator"]["config_overrides"]
    assert agg["hyperparameters"]["dataset_name"] == "google_speech"
    assert (agg.get("optimizer", {}).get("kwargs", {}).get("dataset_name") == "google-speech") == fedbuff_opt
    assert e["name"].startswith("dbg_google_speech_") and "_n100_" in e["name"]


def test_shape_goes_through_agg_goal_and_only_async_gets_c(generator_source, tmp_path, monkeypatch):
    f = _one(generator_source, tmp_path, monkeypatch, "felix", AGG_GOAL="3", CONC="5")
    assert f["aggregator"]["agg_goal"] == 3
    assert f["aggregator"]["config_overrides"]["selector"]["kwargs"]["c"] == 5
    o = _one(generator_source, tmp_path, monkeypatch, "oort", AGG_GOAL="3", CONC="5")
    assert o["aggregator"]["agg_goal"] == 3 and "c" not in o["aggregator"]["config_overrides"]["selector"]["kwargs"]


def test_speech_device_time_scale_scales_d_and_timeout(generator_source, tmp_path, monkeypatch):
    """FX-D46: speech D x5 (divisor 0.2) and the per-dispatch timeout 450s; a harness --delay-factor wins; cifar untouched."""
    monkeypatch.delenv("DELAY_FACTOR", raising=False)
    e = _one(generator_source, tmp_path, monkeypatch, "fedbuff", DATASET="google_speech")
    assert e["trainer"]["hyperparameters"]["training_delay_factor"] == pytest.approx(0.2)
    assert e["aggregator"]["config_overrides"]["hyperparameters"]["send_timeout_wait_s"] == 450.0
    h = _one(generator_source, tmp_path, monkeypatch, "fedbuff", DATASET="google_speech", DELAY_FACTOR="4")
    assert h["trainer"]["hyperparameters"]["training_delay_factor"] == 4.0
    assert "send_timeout_wait_s" not in h["aggregator"]["config_overrides"]["hyperparameters"]
    monkeypatch.delenv("DELAY_FACTOR", raising=False)
    c = _one(generator_source, tmp_path, monkeypatch, "fedbuff", DATASET="cifar10")
    assert "training_delay_factor" not in c["trainer"].get("hyperparameters", {})
    assert "send_timeout_wait_s" not in c["aggregator"]["config_overrides"]["hyperparameters"]
