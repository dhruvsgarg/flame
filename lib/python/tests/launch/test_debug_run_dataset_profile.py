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
    assert "targetAccuracy" not in e["aggregator"]["config_overrides"]["hyperparameters"]


@pytest.mark.parametrize("baseline,fedbuff_opt", [("felix", True), ("fedbuff", True), ("oort", False)])
def test_google_speech_profile(generator_source, tmp_path, monkeypatch, baseline, fedbuff_opt):
    e = _one(generator_source, tmp_path, monkeypatch, baseline, DATASET="google_speech")
    assert e["trainer"]["num_trainers"] == 100 and e["trainer"]["dataset"]["name"] == "google_speech"
    assert e["trainer"]["config_overrides"]["hyperparameters"]["dataset_name"] == "google_speech"
    agg = e["aggregator"]["config_overrides"]
    assert agg["hyperparameters"]["dataset_name"] == "google_speech"
    assert agg["hyperparameters"]["targetAccuracy"] == 0.6  # FX-N10 stop rule
    assert (agg.get("optimizer", {}).get("kwargs", {}).get("dataset_name") == "google-speech") == fedbuff_opt
    assert e["name"].startswith("dbg_google_speech_") and "_n100_" in e["name"]


@pytest.mark.parametrize("baseline", ["felix", "fedbuff"])
def test_speech_async_server_lr_and_cifar_fedbuff_batch(generator_source, tmp_path, monkeypatch, baseline):
    """FX-N74/FX-D185: felix and fedbuff run server lr 1.0 on both datasets; speech SGD 0.04 b16, cifar fedbuff SGD 0.04 b32."""
    e = _one(generator_source, tmp_path, monkeypatch, baseline, DATASET="google_speech")
    assert e["aggregator"]["config_overrides"]["optimizer"]["kwargs"]["learning_rate"] == 1.0
    hp = e["trainer"]["hyperparameters"]
    assert (hp["trainerOptimizer"], hp["learningRate"], hp["batchSize"]) == ("sgd", 0.04, 16)
    monkeypatch.delenv("DATASET", raising=False)
    c = _one(generator_source, tmp_path, monkeypatch, baseline)
    server_lr = c["aggregator"]["config_overrides"].get("optimizer", {}).get("kwargs", {}).get("learning_rate")
    assert server_lr == 1.0  # FX-D185: felix = fedbuff
    if baseline == "fedbuff":
        assert (c["trainer"]["hyperparameters"]["learningRate"], c["trainer"]["hyperparameters"]["batchSize"]) == (0.04, 32)


def test_shape_goes_through_agg_goal_and_only_async_gets_c(generator_source, tmp_path, monkeypatch):
    f = _one(generator_source, tmp_path, monkeypatch, "felix", AGG_GOAL="3", CONC="5")
    assert f["aggregator"]["agg_goal"] == 3
    assert f["aggregator"]["config_overrides"]["selector"]["kwargs"]["c"] == 5
    o = _one(generator_source, tmp_path, monkeypatch, "oort", AGG_GOAL="3", CONC="5")
    assert o["aggregator"]["agg_goal"] == 3 and "c" not in o["aggregator"]["config_overrides"]["selector"]["kwargs"]


def test_dataset_timeout_scales_with_delay_factor(generator_source, tmp_path, monkeypatch):
    """FX-D197: per-dataset abandon timeout (2 x longest FedScale task), compressed by a harness --delay-factor; no D scale."""
    monkeypatch.delenv("DELAY_FACTOR", raising=False)
    e = _one(generator_source, tmp_path, monkeypatch, "fedbuff", DATASET="google_speech")
    assert "training_delay_factor" not in e["trainer"].get("hyperparameters", {})
    assert e["aggregator"]["config_overrides"]["hyperparameters"]["send_timeout_wait_s"] == 2560.0
    h = _one(generator_source, tmp_path, monkeypatch, "fedbuff", DATASET="google_speech", DELAY_FACTOR="4")
    assert h["trainer"]["hyperparameters"]["training_delay_factor"] == 4.0
    assert h["aggregator"]["config_overrides"]["hyperparameters"]["send_timeout_wait_s"] == 640.0
    monkeypatch.delenv("DELAY_FACTOR", raising=False)
    c = _one(generator_source, tmp_path, monkeypatch, "fedbuff", DATASET="cifar10")
    assert c["aggregator"]["config_overrides"]["hyperparameters"]["send_timeout_wait_s"] == 903.0


def test_trainer_hp_beats_by_baseline(generator_source, tmp_path, monkeypatch):
    # Run 20: --trainer-hp learningRate=0.05 lost to speech refl's by_baseline 0.005 (the launcher prefers trainer.hyperparameters).
    e = _one(generator_source, tmp_path, monkeypatch, "refl", DATASET="google_speech", TRAINER_HP="learningRate=0.05")
    assert e["trainer"]["hyperparameters"]["learningRate"] == 0.05
