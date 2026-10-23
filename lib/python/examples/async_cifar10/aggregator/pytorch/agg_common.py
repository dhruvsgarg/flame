# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Shared body of the three example aggregators (asyncfl, oort sync, fedavg sync): model, test data, background eval.

FX-N77: the global model and the aggregation math live on CPU (updates decode there zero-copy, no per-update H2D);
the GPU serves only eval and oracle forwards, so ingest never queues behind an eval on the device.
`FLAME_AGG_MODEL_DEVICE=eval` reverts to the model on the eval device.
"""

import copy
import logging
import os
import threading
import time

import torch
import torch.nn.functional as F

from flame import harness, telemetry
from flame.dataset import Dataset

import fl_data

logger = logging.getLogger(__name__)

ENV_AGG_MODEL_DEVICE = "FLAME_AGG_MODEL_DEVICE"
ENV_AGG_MALLOC_TRIM = "FLAME_AGG_MALLOC_TRIM"


def trim_host_heap() -> float:
    """FX-N84: return freed glibc heap (decoded updates) to the OS; 0.4-0.9 GB per speech agg. Returns seconds taken."""
    t0 = time.time()
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass
    return time.time() - t0


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
        # FX-D142: eval copy before the trace origin; built lazily, its CUDA init delayed round 2 in real only
        self._eval_model = copy.deepcopy(self.model).to(self.eval_device)
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

    def _serve_beta(self) -> float:
        """FX-N128 serveEma=sweep: average over ~one population sweep of versions (K updates each): b = 1 - K / n, n = clients
        that have contributed so far (grows from K as the population joins). FX-N133: sync stacks, one version per round."""
        n = len(getattr(self, "_trainer_participation_in_round_count", None) or getattr(self, "_contributors_seen", None) or {})
        k = getattr(self, "_agg_goal", None) or getattr(self.config.hyperparameters, "aggregation_goal", None) or 1
        return max(0.0, 1.0 - float(k) / max(n, 1))

    def _ema_step(self) -> None:
        """FX-N128: per committed version, ema_b <- b ema_b + (1 - b) model for each b in `evalEma` ("0.5,0.9"; scored beside the
        model as `agg_eval_ema`) and, with `serveEma` (sweep | b), the served model (scored as `agg_eval`). Training unchanged."""
        betas = getattr(self, "_ema_betas", None)
        if betas is None:
            raw = getattr(self.config.hyperparameters, "evalEma", None) or ""
            betas = self._ema_betas = [float(x) for x in str(raw).replace(",", " ").split()]
            self._emas = {}
            self._serve = getattr(self.config.hyperparameters, "serveEma", None)
        if not betas and not self._serve:
            return
        sd = self.model.state_dict()
        for key in betas + (["serve"] if self._serve else []):
            b = key if key != "serve" else self._serve_beta() if str(self._serve) == "sweep" else float(self._serve)
            e = self._emas.get(key)
            if e is None:
                self._emas[key] = {k: v.detach().clone() for k, v in sd.items()}
                continue
            for k, v in sd.items():
                if torch.is_floating_point(v):
                    e[k].mul_(b).add_(v.detach().to(e[k].device), alpha=1 - b)
                else:
                    e[k].copy_(v)
        if self._serve and "serve" in self._emas and telemetry.is_enabled():
            self._emit_drift(sd)

    def _emit_drift(self, sd) -> None:
        """FX-D225: per version, d = |w - w̄| (served EMA), step = |Δw|, cos of consecutive steps (`agg_drift`)."""
        from flame.optimizer.bn_buffers import is_bn_stat
        from flame.selector.update_stats import flatten
        w = flatten(sd, is_bn_stat)
        prev, last = getattr(self, "_drift_prev", None), getattr(self, "_drift_step", None)
        self._drift_prev = w.clone()
        if prev is None:
            return
        step = w - prev
        dist = float((w - flatten(self._emas["serve"], is_bn_stat).to(w.device)).norm())
        sn = float(step.norm())
        cos = float(step.dot(last) / (sn * last.norm())) if last is not None and sn > 0 and last.norm() > 0 else None
        self._drift_step = step
        telemetry.emit("agg_drift", round=self._round, now=self._avail_now(), dist=dist, step=sn, step_cos=cos)

    def evaluate(self) -> None:
        """Test the model every evalEveryNRounds rounds (default 10), off the critical path in a daemon thread.

        Backgrounding is why eval needs no sim_model_*_compute_time vclock fold (flame/config.py); a synchronous
        eval would need one or sim would under-count wall time.
        """
        if getattr(self, "_round_committed", None) is False:  # FX-D128: eval only on the committing pass
            return
        self._ema_step()
        every_s = getattr(self.config.hyperparameters, "eval_every_s", None)
        if every_s:  # FX-N112: first commit at or after each mark of the leg's clock
            mark = int(self._avail_now() // float(every_s))
            if mark <= getattr(self, "_eval_last_mark", -1):
                return
        else:
            eval_every = getattr(self.config.hyperparameters, "eval_every_n_rounds", 10) or 10
            if self._round % eval_every != 0 and not (self.EVAL_ROUND_ONE and self._round == 1):
                return
        self._eval_every_n_commits = 1  # FX-D63: the round gate is the cadence; a commit stride on top halved it
        eval_model = self._eval_snapshot_model()  # on self.eval_device
        if eval_model is None:
            return  # prior async eval still running
        if every_s:
            self._eval_last_mark = mark
        round_num, test_loader, device = self._round, self.test_loader, self.eval_device
        emas = [(b, {k: v.clone() for k, v in sd.items()}) for b, sd in getattr(self, "_emas", {}).items() if b != "serve"]
        served = getattr(self, "_emas", {}).get("serve")
        if served is not None:  # FX-N128: agg_eval scores the served model; the raw one goes to agg_eval_raw
            emas.append(("raw", None))
            raw_sd = {k: v.clone() for k, v in eval_model.state_dict().items()}
            eval_model.load_state_dict(served)

        def _test(m):
            m.eval()
            loss = torch.zeros((), device=device)
            correct = torch.zeros((), device=device, dtype=torch.long)
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = fl_data.model_input(data.to(device)), target.to(device)
                    output = m(data)
                    loss += F.nll_loss(output, target, reduction="sum")
                    correct += (output.argmax(dim=1) == target).sum()
            total = len(test_loader.dataset)
            return loss.item() / total, correct.item() / total  # one device sync per eval (FX-N76 R3)

        def _job():
            try:
                t0 = time.time()
                loss, acc = _test(eval_model)
                logger.info(f"[ASYNC_EVAL_TIMING] round={round_num} wall_s={time.time() - t0:.2f} device={device}")  # FX-N70
                if emas:  # before the emit frees the eval slot
                    from flame import telemetry
                    m = self._ema_model = getattr(self, "_ema_model", None) or copy.deepcopy(eval_model)
                    for b, sd in emas:
                        m.load_state_dict(raw_sd if b == "raw" else sd)
                        el, ea = _test(m)
                        telemetry.emit("agg_eval_raw" if b == "raw" else "agg_eval_ema", round=round_num,
                                       **({} if b == "raw" else {"beta": b}), **{"test-loss": el, "test-accuracy": ea})
                self._eval_emit(round_num, loss, acc)
                if os.environ.get(ENV_AGG_MALLOC_TRIM, "1") != "0":  # once per eval, off the commit path
                    logger.info(f"[AGG_MALLOC_TRIM] round={round_num} s={trim_host_heap():.3f}")
            except Exception as e:  # eval must never break training
                logger.warning(f"[ASYNC_EVAL] failed (non-fatal): {e}")
                self._eval_inflight = False

        threading.Thread(target=_job, daemon=True).start()
