# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""FMoW horizontal FL, AsyncFL (FedBuff/FELIX) aggregator for PyTorch.

Reuses the FedAvg aggregator's FMoW model/data/eval/availability logic on
the asyncfl stack. MRO: FMoW methods -> asyncfl TopAggregator -> syncfl.
"""

import logging

from flame.config import Config
from flame.mode.horizontal.asyncfl.top_aggregator import TopAggregator as AsyncTopAggregator

from main_fedavg_agg import PyTorchFMoWAggregator

logger = logging.getLogger(__name__)


class PyTorchFMoWAsyncAggregator(PyTorchFMoWAggregator, AsyncTopAggregator):
    """PyTorch FMoW async aggregator."""

    def __init__(self, config: Config, log_to_wandb: bool = False) -> None:
        super().__init__(config, log_to_wandb)
        self.reject_stale_updates = self.config.hyperparameters.reject_stale_updates or False


if __name__ == "__main__":
    import argparse

    from flame.launch.cli import load_config_from_argv

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--log_to_wandb", action="store_true")
    parser.add_argument("--wandb_run_name", type=str)
    args, _ = parser.parse_known_args()

    config = load_config_from_argv()

    a = PyTorchFMoWAsyncAggregator(config, args.log_to_wandb)

    from flame import telemetry

    telemetry.configure(role="aggregator", end_id=config.job.job_id)

    a.compose()
    a.run()
