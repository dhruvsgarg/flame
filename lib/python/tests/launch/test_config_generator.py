# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Tests for ConfigGenerator with the real shared metadata bundle."""

from pathlib import Path

import pytest
import yaml


SHARED_METADATA = Path(__file__).resolve().parents[2] / "examples" / "_metadata"
TRAINER_BASE = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "feddance_cifar10"
    / "configs"
    / "trainer_base.yaml"
)


pytestmark = pytest.mark.skipif(
    not SHARED_METADATA.is_dir() or not TRAINER_BASE.is_file(),
    reason="shared metadata or feddance_cifar10 trainer_base not present",
)


@pytest.fixture
def loader():
    from flame.launch.spawner import MetadataLoader

    return MetadataLoader(SHARED_METADATA)


@pytest.fixture
def gen(loader):
    from flame.launch.spawner import ConfigGenerator

    return ConfigGenerator(loader, TRAINER_BASE)


class TestMetadataLoader:
    def test_registry_loaded(self, loader):
        assert len(loader.trainer_registry) > 0

    def test_dataset_splits_loaded(self, loader):
        assert len(loader.dataset_splits) > 0


class TestConfigGenerator:
    def test_generate_basic(self, gen):
        cfg = gen.generate_trainer_config(
            trainer_id=1, alpha=0.1, availability_mode="syn_0"
        )
        assert "taskid" in cfg
        assert isinstance(cfg["taskid"], str) and len(cfg["taskid"]) > 0
        assert cfg["hyperparameters"]["trainer_indices_list"]
        assert cfg["selector"]["sort"] == "feddance"

    def test_overrides_applied(self, gen):
        cfg = gen.generate_trainer_config(
            trainer_id=2,
            alpha=0.1,
            availability_mode="syn_0",
            **{"job.id": "experiment-xyz", "hyperparameters.batchSize": 64},
        )
        assert cfg["job"]["id"] == "experiment-xyz"
        assert cfg["hyperparameters"]["batchSize"] == 64


FWDLLM_TRAINER_BASE = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "fwdllm"
    / "configs"
    / "trainer_base.yaml"
)
FWDLLM_AGGREGATOR_MAIN = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "fwdllm"
    / "aggregator"
    / "main_fedfwd_agg.py"
)


@pytest.mark.skipif(
    not SHARED_METADATA.is_dir() or not FWDLLM_TRAINER_BASE.is_file(),
    reason="shared metadata or fwdllm trainer_base not present",
)
class TestFwdllmEndToEndConfigGeneration:
    """Phase 7 step P7: for each of the four fwdllm baselines, generate the
    aggregator + a trainer config end-to-end (mirrors Smoke Test D, extended
    from the single retired fedfwd_async_random_dynkc to all four)."""

    @pytest.fixture
    def baselines(self):
        from flame.launch.baselines import load_baselines

        return load_baselines(SHARED_METADATA)

    @pytest.fixture
    def gen(self, loader):
        from flame.launch.spawner import ConfigGenerator

        return ConfigGenerator(loader, FWDLLM_TRAINER_BASE)

    @pytest.mark.parametrize(
        "baseline_name", ["fwdllm", "fwdllm_plus", "fluxtune", "fluxtune_dynkc"]
    )
    def test_trainer_config_generates_without_keyerror(
        self, gen, baselines, baseline_name
    ):
        gen.set_baseline_overrides(baselines[baseline_name].get("trainer", {}))
        cfg = gen.generate_trainer_config(
            trainer_id=1,
            alpha=0.1,
            availability_mode="syn_0",
            dataset_name="agnews",
            num_trainers=10,
            skip_index_splits=True,
            **{"hyperparameters.client_idx": 0},
        )
        assert cfg["hyperparameters"]["client_idx"] == 0
        assert "trainer_indices_list" not in cfg["hyperparameters"]

    @pytest.mark.parametrize(
        "baseline_name,expected",
        [
            ("fwdllm", {"client_notify.enabled": "False"}),
            ("fwdllm_plus", {"client_notify.enabled": "False"}),
            (
                "fluxtune",
                {
                    "select_perturbation_using_jvp": True,
                    "client_notify.enabled": "True",
                },
            ),
            ("fluxtune_dynkc", {"client_notify.enabled": "False"}),
        ],
    )
    def test_trainer_config_matches_matrix(
        self, gen, baselines, baseline_name, expected
    ):
        gen.set_baseline_overrides(baselines[baseline_name].get("trainer", {}))
        cfg = gen.generate_trainer_config(
            trainer_id=1,
            alpha=0.1,
            availability_mode="syn_0",
            dataset_name="agnews",
            num_trainers=10,
            skip_index_splits=True,
        )
        hp = cfg["hyperparameters"]
        for dotted_key, want in expected.items():
            cur = hp
            for part in dotted_key.split(".")[:-1]:
                cur = cur[part]
            assert cur[dotted_key.split(".")[-1]] == want

    @pytest.mark.parametrize(
        "baseline_name,expected",
        [
            (
                "fwdllm",
                {
                    "selector.sort": "random",
                    "optimizer.sort": "fedavg",
                    "hyperparameters.reselect_each_iteration": False,
                },
            ),
            (
                "fwdllm_plus",
                {
                    "selector.sort": "random",
                    "optimizer.sort": "fedavg",
                    "hyperparameters.reselect_each_iteration": True,
                },
            ),
            (
                "fluxtune",
                {"selector.sort": "async_oort", "optimizer.sort": "fedbuff"},
            ),
            (
                "fluxtune_dynkc",
                {"selector.sort": "async_random", "optimizer.sort": "fedbuff"},
            ),
        ],
    )
    def test_aggregator_config_matches_matrix_and_validates(
        self, baselines, baseline_name, expected
    ):
        import json

        from flame.launch.baselines import deep_merge
        from flame.launch.runner import ExperimentRunner

        tmpl = json.load(open(SHARED_METADATA / "aggregator_base.json"))
        merged = deep_merge(tmpl, baselines[baseline_name]["aggregator"])
        for dotted_key, want in expected.items():
            cur = merged
            for part in dotted_key.split(".")[:-1]:
                cur = cur[part]
            assert cur[dotted_key.split(".")[-1]] == want

        # Real entrypoint, not a synthetic fixture -- locks in the
        # main_fedfwd_agg.py marker import that makes the stack-detection
        # regex find the fwdllm stack on this file.
        runner = ExperimentRunner(FWDLLM_AGGREGATOR_MAIN.parents[1])
        runner._validate_stack(FWDLLM_AGGREGATOR_MAIN, merged)
