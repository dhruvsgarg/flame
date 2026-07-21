# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""eval_model() used to run synchronously on the aggregator's critical path,
stalling dispatch and trainer idle time on every data_id boundary. It's now
backgrounded on a daemon thread (mirroring async_cifar10's evaluate()), which
required fixing a real race: eval_model() used to reassign
self.fmodel/self.params/self.buffers via fc.make_functional_with_buffers() --
the SAME three attributes the main training path reassigns right before
self.aggregate() every cycle. Two threads racing on those risked silent
gradient corruption, not just a crash.

Drives the real eval_model()/`_eval_snapshot_model` (not stubs) to prove:

1. eval_model() no longer touches self.fmodel/self.params/self.buffers at all
   (that assignment was dead code -- its output was never read).
2. A background eval running concurrently with the main thread's reassignment
   of those same attributes does not corrupt it.
3. The snapshot `_eval_snapshot_model()` hands to the background thread
   reflects the model as of the snapshot call, even if self.model mutates
   further while eval is still running.

Only CUDA-memory-logging plumbing (log_memory/_force_cuda_memory_cleanup) is
stubbed -- not portable to a GPU-less CI runner.
"""

import threading

import torch
import torch.nn as nn

from flame.mode.horizontal.syncfl.fwdllm_aggregator import TopAggregator


class _FakeEvalDataset:
    """Minimal stand-in for FedSGD's test_global.dataset: eval_model() calls
    len(self.test_global.dataset) directly (not len(tensors[i]))."""

    def __init__(self, tensors, length):
        self.tensors = tensors
        self._length = length

    def __len__(self):
        return self._length


class _FakeEvalAggregator:
    """Minimal REAL-model harness for eval_model()/`_eval_snapshot_model`."""

    eval_model = TopAggregator.eval_model
    _eval_snapshot_model = TopAggregator._eval_snapshot_model
    compute_metrics = TopAggregator.compute_metrics

    def _force_cuda_memory_cleanup(self):
        pass  # heavy/GPU-only plumbing, irrelevant to the race under test

    def log_memory(self, tag, device):
        pass  # ditto

    def __init__(self, n=8, in_features=4, num_labels=2):
        self.device = torch.device("cpu")
        self.model = nn.Linear(in_features, num_labels)
        self.num_labels = num_labels

        class _Args:
            eval_batch_size = 4
            fp16 = False

        self.args = _Args()

        input_ids = torch.randn(n, in_features)
        labels = torch.randint(0, num_labels, (n,))
        # Only indices 1 (input_ids) and 4 (labels) are read by eval_model();
        # the rest just need to be real tensors since _cached_test_data's
        # construction calls .to(device) on every element unconditionally.
        filler = torch.zeros(n)
        tensors = [filler, input_ids, filler, filler, labels]

        class _TestGlobal:
            """eval_model() passes self.test_global (not .dataset) to
            compute_metrics_with_logging, which iterates it for a debug log
            line -- empty iteration is enough to no-op that, unrelated to
            this file's race/snapshot behavior."""

            dataset = _FakeEvalDataset(tensors, n)
            examples = None

            def __iter__(self):
                return iter([])

        self.test_global = _TestGlobal()
        self._cached_test_data = None
        self._eval_inflight = False
        self._eval_model = None
        self.iteration_per_data_id = 0  # eval_model() logs this, unrelated to the race
        # Sentinel "training-path" state -- eval_model() must never touch these
        # (the actual race: the main thread assigns these same three names
        # right before self.aggregate() on every cycle).
        self.fmodel = "SENTINEL_FMODEL"
        self.params = "SENTINEL_PARAMS"
        self.buffers = "SENTINEL_BUFFERS"


class TestEvalModelNoLongerTouchesSharedState:
    def test_fmodel_params_buffers_untouched(self):
        agg = _FakeEvalAggregator()
        result, model_outputs, wrong = agg.eval_model()
        assert "eval_loss" in result
        assert agg.fmodel == "SENTINEL_FMODEL"
        assert agg.params == "SENTINEL_PARAMS"
        assert agg.buffers == "SENTINEL_BUFFERS"

    def test_default_model_param_is_self_model(self):
        """No `model=` kwarg -> falls back to self.model, so any other
        existing (non-backgrounded) caller stays byte-identical."""
        agg = _FakeEvalAggregator()
        # A snapshot with different weights than self.model; eval_model()'s
        # default path must use self.model, not this snapshot.
        snapshot = agg._eval_snapshot_model()
        with torch.no_grad():
            for p in snapshot.parameters():
                p.fill_(12345.0)
        result_default, _, _ = agg.eval_model()
        result_snapshot, _, _ = agg.eval_model(model=snapshot)
        # Wildly different weights -> wildly different loss; the two calls
        # must not coincidentally agree.
        assert result_default["eval_loss"] != result_snapshot["eval_loss"]


class TestBackgroundedEvalRaceSafety:
    """A backgrounded eval must never corrupt the main thread's concurrent
    self.fmodel/self.params/self.buffers assignment."""

    def test_concurrent_main_thread_assignment_survives_background_eval(self):
        agg = _FakeEvalAggregator(n=64)  # a few batches, so the thread runs a beat
        snapshot = agg._eval_snapshot_model()
        assert snapshot is not None

        def _eval_job():
            agg.eval_model(model=snapshot)

        t = threading.Thread(target=_eval_job)
        t.start()
        # Main thread does exactly what the training path does right before
        # self.aggregate(): reassign fmodel/params/buffers from the LIVE model.
        main_fmodel = object()
        main_params = object()
        main_buffers = object()
        agg.fmodel, agg.params, agg.buffers = main_fmodel, main_params, main_buffers
        t.join(timeout=10)
        assert not t.is_alive(), "background eval did not finish in time"
        # The main thread's own assignment must survive completely untouched.
        assert agg.fmodel is main_fmodel
        assert agg.params is main_params
        assert agg.buffers is main_buffers


class TestEvalSnapshotCorrectness:
    """Snapshot correctness: the backgrounded eval's results must reflect the
    model AS SNAPSHOTTED, even if self.model mutates further (via a subsequent
    aggregate()) before the eval thread finishes."""

    def test_snapshot_weights_frozen_at_snapshot_time(self):
        agg = _FakeEvalAggregator()
        with torch.no_grad():
            agg.model.weight.fill_(0.0)
            agg.model.bias.fill_(0.0)

        snapshot = agg._eval_snapshot_model()

        # Mutate the LIVE model AFTER snapshotting -- simulates a subsequent
        # aggregate() cycle running while the background eval is still in
        # flight.
        with torch.no_grad():
            agg.model.weight.fill_(999.0)
            agg.model.bias.fill_(999.0)

        assert torch.equal(snapshot.weight, torch.zeros_like(snapshot.weight))
        assert torch.equal(snapshot.bias, torch.zeros_like(snapshot.bias))

    def test_eval_result_reflects_snapshot_not_live_model(self):
        agg = _FakeEvalAggregator()
        with torch.no_grad():
            agg.model.weight.fill_(0.0)
            agg.model.bias.fill_(0.0)

        snapshot = agg._eval_snapshot_model()
        with torch.no_grad():
            agg.model.weight.fill_(999.0)
            agg.model.bias.fill_(999.0)

        # Must not raise / not silently pick up the mutated live weights.
        result, _, _ = agg.eval_model(model=snapshot)
        assert "eval_loss" in result
        for p in snapshot.parameters():
            assert torch.equal(p, torch.zeros_like(p))
