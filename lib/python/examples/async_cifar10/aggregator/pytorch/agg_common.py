# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Shared body of the three example aggregators (asyncfl, oort sync, fedavg sync): model, test data, background eval.

FX-N77: the global model and the aggregation math live on CPU (updates decode there zero-copy, no per-update H2D);
the GPU serves only eval and oracle forwards, so ingest never queues behind an eval on the device.
`FLAME_AGG_MODEL_DEVICE=eval` reverts to the model on the eval device.
"""

import logging
import os
import threading
import time

import torch
import torch.nn.functional as F

from flame import harness
from flame.dataset import Dataset

import fl_data

logger = logging.getLogger(__name__)

ENV_AGG_MODEL_DEVICE = "FLAME_AGG_MODEL_DEVICE"


class ExampleAggregatorMixin:
    """initialize / load_data / evaluate for the example aggregators; mix in ahead of the stack's TopAggregator."""

    EVAL_ROUND_ONE = False  # sync stacks also evaluate round 1 (their baseline point)

    @property
    def data_spec(self) -> "fl_data.DatasetSpec":
        """FX-N10: this run's dataset (hyperparameters.dataset_name)."""
        return fl_data.spec_for(self.config.hyperparameters)

    def initialize(self):
        """Initialize role."""
        self.eval_device = harness.device_for(self.harness_mode)
        on_eval = os.environ.get(ENV_AGG_MODEL_DEVICE, "cpu").lower() == "eval"
        self.device = self.eval_device if on_eval else torch.device("cpu")
        self.model = self.data_spec.model().to(self.device)
        self._replica = None
        self._init_oracle_util(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data"))

    def eval_replica(self):
        """The current global model on the eval device, refreshed once per round (oracle forwards)."""
        if self.device == self.eval_device:
            return self.model
        if self._replica is None or self._replica[0] != self._round:
            m = self._replica[1] if self._replica else self.data_spec.model().to(self.eval_device)
            m.load_state_dict(self.model.state_dict())
            self._replica = (self._round, m)
        return self._replica[1]

    def load_data(self) -> None:
        """Load a test dataset."""
        n_test = harness.harness_test_samples(self.config.hyperparameters, self.harness_mode)
        if self.harness_mode == "stub":
            dataset = harness.synthetic_dataset(n_test, self.data_spec.stub_shape, self.data_spec.num_classes,
                                                seed_key="agg_test", label_skew=0.0)
        else:
            dataset = self.data_spec.test()
            if self.harness_mode == "tiny_cpu":
                dataset = torch.utils.data.Subset(dataset, list(range(n_test)))
            dataset = fl_data.in_memory(dataset)  # FX-D64: per-sample decode in the eval thread stalled real ingest
        self.test_loader = torch.utils.data.DataLoader(
            dataset,
            # FX-N76 R3: eval batch is independent of the train batch (16 for speech: 690 synced batches, 10 s per eval).
            batch_size=int(getattr(self.config.hyperparameters, "eval_batch_size", None) or 512),
            shuffle=False,
            num_workers=0,
            pin_memory=False,  # FX-D64: pinning an in-memory test set cost 2 s per speech eval
        )
        self.dataset = Dataset(dataloader=self.test_loader)

    def train(self) -> None:
        pass

    def check_and_sleep(self) -> None:
        pass

    def evaluate(self) -> None:
        """Test the model every evalEveryNRounds rounds (default 10), off the critical path in a daemon thread.

        Backgrounding is why eval needs no sim_model_*_compute_time vclock fold (flame/config.py); a synchronous
        eval would need one or sim would under-count wall time.
        """
        eval_every = getattr(self.config.hyperparameters, "eval_every_n_rounds", 10) or 10
        if self._round % eval_every != 0 and not (self.EVAL_ROUND_ONE and self._round == 1):
            return
        self._eval_every_n_commits = 1  # FX-D63: the round gate is the cadence; a commit stride on top halved it
        eval_model = self._eval_snapshot_model()  # on self.eval_device
        if eval_model is None:
            return  # prior async eval still running
        round_num, test_loader, device = self._round, self.test_loader, self.eval_device

        def _job():
            try:
                t0 = time.time()
                eval_model.eval()
                loss = torch.zeros((), device=device)
                correct = torch.zeros((), device=device, dtype=torch.long)
                with torch.no_grad():
                    for data, target in test_loader:
                        data, target = data.to(device), target.to(device)
                        output = eval_model(data)
                        loss += F.nll_loss(output, target, reduction="sum")
                        correct += (output.argmax(dim=1) == target).sum()
                loss, correct = loss.item(), correct.item()  # one device sync per eval (FX-N76 R3)
                total = len(test_loader.dataset)
                logger.info(f"[ASYNC_EVAL_TIMING] round={round_num} wall_s={time.time() - t0:.2f} device={device}")  # FX-N70
                self._eval_emit(round_num, loss / total, correct / total)
            except Exception as e:  # eval must never break training
                logger.warning(f"[ASYNC_EVAL] failed (non-fatal): {e}")
                self._eval_inflight = False

        threading.Thread(target=_job, daemon=True).start()
