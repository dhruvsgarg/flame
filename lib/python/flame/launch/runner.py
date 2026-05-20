# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""ExperimentRunner: orchestrates a single experiment end-to-end.

Paths are resolved as follows (lowest precedence → highest):
1. Defaults relative to example_dir.
2. ExampleConfig fields in the experiment YAML.
3. MetadataPaths fields in the experiment YAML.
4. Constructor overrides.
"""

import json
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from flame.launch.aggregator_spawner import AggregatorSpawner
from flame.launch.execution_config_generator import (
    create_execution_config,
    save_execution_config,
)
from flame.launch.experiment_config import ExperimentConfig
from flame.launch.snapshot import ExperimentSnapshot

try:
    from flame.launch.resource_monitor import create_monitor_from_config
except ImportError:
    create_monitor_from_config = None  # type: ignore
from flame.launch.spawner import ConfigGenerator, MetadataLoader, TrainerSpawner


REPO_ROOT_HINT = "examples"


def _resolve(base: Path, rel: Optional[str]) -> Optional[Path]:
    if rel is None:
        return None
    p = Path(rel)
    if p.is_absolute():
        return p
    return (base / p).resolve()


class ExperimentRunner:
    """Run experiments end-to-end against a generic example layout."""

    def __init__(
        self,
        example_dir: Path,
        metadata_dir: Optional[Path] = None,
    ):
        self.example_dir = Path(example_dir).resolve()
        self.metadata_dir_default = (
            Path(metadata_dir).resolve()
            if metadata_dir is not None
            else self.example_dir / "metadata"
        )
        self.experiments_dir = self.example_dir / "experiments"

        self.current_exp_dir: Optional[Path] = None
        self.aggregator_spawner: Optional[AggregatorSpawner] = None
        self.trainer_spawner: Optional[TrainerSpawner] = None
        self.resource_monitor = None

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _resolve_example_paths(self, exp: ExperimentConfig) -> dict:
        """Resolve trainer_main/aggregator_main/trainer_base/metadata_dir."""
        ex_dir = (
            Path(exp.example.dir).resolve()
            if exp.example.dir
            else self.example_dir
        )

        meta_dir = (
            Path(exp.metadata.dir).resolve()
            if exp.metadata.dir
            else self.metadata_dir_default
        )
        registry = _resolve(meta_dir, exp.metadata.registry) or (
            meta_dir / "trainer_registry.yaml"
        )

        return {
            "example_dir": ex_dir,
            "trainer_main": ex_dir / exp.example.trainer_main,
            "aggregator_main": ex_dir / exp.example.aggregator_main,
            "trainer_base": ex_dir / exp.example.trainer_base,
            "metadata_dir": meta_dir,
            "registry_path": registry,
        }

    def run_experiment(self, exp: ExperimentConfig) -> None:
        print(f"\n{'=' * 70}\nRUNNING EXPERIMENT: {exp.name}\n{'=' * 70}")

        paths = self._resolve_example_paths(exp)

        try:
            self.current_exp_dir = self._create_experiment_directory(exp)
            print(f"  exp dir: {self.current_exp_dir}")

            agg_config_path = paths["example_dir"] / exp.aggregator.config_template
            if not agg_config_path.exists():
                raise FileNotFoundError(f"aggregator config not found: {agg_config_path}")
            with open(agg_config_path) as f:
                agg_cfg = json.load(f)
                agg_job_id = agg_cfg.get("job", {}).get("id")
                agg_job_name = agg_cfg.get("job", {}).get("name")
            if not agg_job_id:
                raise ValueError(f"aggregator config missing job.id: {agg_config_path}")

            metadata_loader = MetadataLoader(paths["metadata_dir"])
            config_gen = ConfigGenerator(metadata_loader, paths["trainer_base"])

            log_prefix = exp.get_log_prefix()
            agg_log = self.current_exp_dir / f"{log_prefix}_aggregator.log"
            trainers_log = self.current_exp_dir / f"{log_prefix}_trainers.log"
            monitor_log = self.current_exp_dir / f"{log_prefix}_resources.log"

            self.aggregator_spawner = AggregatorSpawner(log_file=agg_log)
            self.trainer_spawner = TrainerSpawner(
                config_gen,
                num_gpus=exp.execution.num_gpus,
                sleep_between_spawns=exp.execution.sleep_between_spawns,
                log_file=trainers_log,
            )

            if exp.execution.monitoring.enabled and create_monitor_from_config is not None:
                self.resource_monitor = create_monitor_from_config(
                    monitor_log,
                    {
                        "check_interval_seconds": exp.execution.monitoring.check_interval_seconds,
                        "ram_warning_percent": exp.execution.monitoring.ram_warning_percent,
                        "ram_critical_percent": exp.execution.monitoring.ram_critical_percent,
                        "gpu_warning_percent": exp.execution.monitoring.gpu_warning_percent,
                        "gpu_critical_percent": exp.execution.monitoring.gpu_critical_percent,
                    },
                )

            self.aggregator_spawner.spawn(
                paths["aggregator_main"],
                agg_config_path,
                log_to_wandb=exp.aggregator.log_to_wandb,
                wandb_run_name=exp.aggregator.wandb_run_name,
            )
            if not self.aggregator_spawner.wait_until_ready(
                exp.execution.aggregator_warmup_time
            ):
                raise RuntimeError("aggregator failed to start")

            if self.resource_monitor:
                self.resource_monitor.start()

            agg_spawn_cmd = [sys.executable, str(paths["aggregator_main"]), str(agg_config_path)]
            trainer_spawn_cmd = self._build_trainer_spawn_command(exp)

            exec_config = create_execution_config(
                exp,
                agg_config_path.relative_to(paths["example_dir"]),
                spawn_commands={
                    "aggregator": [str(c) for c in agg_spawn_cmd],
                    "trainers": [str(c) for c in trainer_spawn_cmd],
                },
            )
            save_execution_config(exec_config, self.current_exp_dir / "execution_config.yaml")

            snapshot = ExperimentSnapshot(self.current_exp_dir)
            snapshot.create_snapshot(
                exp, paths["metadata_dir"], agg_config_path, trainer_spawn_cmd, agg_spawn_cmd
            )

            trainer_ids = list(
                range(exp.trainer.start_id, exp.trainer.start_id + exp.trainer.num_trainers)
            )
            config_overrides = {
                "job.id": agg_job_id,
                "job.name": agg_job_name,
                "hyperparameters.training_delay_enabled": str(exp.trainer.enable_training_delays),
            }
            if exp.trainer.hyperparameters:
                for key, value in exp.trainer.hyperparameters.items():
                    config_overrides[f"hyperparameters.{key}"] = value

            self.trainer_spawner.spawn_all(
                trainer_ids,
                alpha=exp.trainer.dataset.dirichlet_alpha,
                availability_mode=exp.trainer.availability.mode,
                trainer_main_path=paths["trainer_main"],
                **config_overrides,
            )

            print(f"\nexperiment running. logs: {agg_log}, {trainers_log}")
            self.trainer_spawner.wait_all()
            print("\nexperiment completed.")

        except Exception as e:
            import traceback

            print(f"\nexperiment failed: {e}")
            traceback.print_exc()
            raise
        finally:
            self._cleanup()

    def run_experiment_batch(self, config_file: Path) -> None:
        from flame.launch.experiment_config import load_experiment_config

        batch = load_experiment_config(config_file)
        print(f"loaded {len(batch.experiments)} experiments from {config_file}")
        for i, exp in enumerate(batch.experiments, 1):
            print(f"\n[{i}/{len(batch.experiments)}] {exp.name}")
            try:
                self.run_experiment(exp)
            except Exception as e:
                print(f"experiment {exp.name} failed: {e}")
                resp = input("continue? (y/n): ")
                if resp.lower() != "y":
                    break

    def _create_experiment_directory(self, exp: ExperimentConfig) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        d = self.experiments_dir / f"run_{timestamp}_{exp.name}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _build_trainer_spawn_command(self, exp: ExperimentConfig) -> list:
        return [
            "python3",
            "-m",
            "flame.launch.spawn_trainer_cli",
            "--alpha", str(exp.trainer.dataset.dirichlet_alpha),
            "--availability", exp.trainer.availability.mode,
            "--num-trainers", str(exp.trainer.num_trainers),
            "--start-id", str(exp.trainer.start_id),
            "--num-gpus", str(exp.execution.num_gpus),
        ]

    def _signal_handler(self, signum, frame):
        print("\ntermination signal received")
        self._cleanup()
        sys.exit(130)

    def _cleanup(self):
        if self.resource_monitor:
            self.resource_monitor.stop()
        if self.trainer_spawner:
            self.trainer_spawner.terminate_all()
        if self.aggregator_spawner:
            self.aggregator_spawner.terminate()
