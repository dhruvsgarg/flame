# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""self.grad's per-cycle FedAvg accumulation is not associative, and for
grad_aware not even commutative (each contribution's rate is
`cos(trainer_grad, self.grad)` against whatever partial sum is already
merged) -- so accumulation ORDER changes the result.

`_process_single_trainer_message` now only buffers into
`_pending_cohort_contribs`; `_process_aggregation_goal_met` merges by
replaying the buffer in `_canonicalize_cohort_commit_order`'s canonical
(D, trainer_id) order, not receipt order.

These tests pin the fix: two arrival orders that canonicalize to the same
sequence must merge to a bit-identical `self.grad`, for fedavg (rate=1.0)
and grad_aware (rate reads running self.grad, the harder case).
"""

import torch

from flame.mode.horizontal.syncfl.fwdllm_aggregator import TopAggregator


class _MergeAgg:
    """Binds the real merge + canonicalization methods onto a minimal
    stand-in, driven directly (not through the channel/telemetry machinery
    _process_aggregation_goal_met also touches)."""

    aggregate_grads_from_trainers = TopAggregator.aggregate_grads_from_trainers
    canon = TopAggregator._canonicalize_cohort_commit_order
    _cosine_flat = staticmethod(TopAggregator._cosine_flat)
    _grad_aware_rate = staticmethod(TopAggregator._grad_aware_rate)
    _flat_grad_norm = staticmethod(TopAggregator._flat_grad_norm)

    def __init__(self, model, weighted, agg_rate_conf=None):
        self.model = model
        self.device = torch.device("cpu")
        self.params = list(model.parameters())
        self.grad = [torch.zeros_like(p) for p in self.params]
        self._weighted_aggregation_enabled = weighted
        self._model_version = 0
        self.grad_for_var_check_list = []
        self.jvp_for_snr_check_list = []
        self._cycle_grad_norms = []

        class _Opt:
            pass

        self.optimizer = _Opt()
        self.optimizer.agg_rate_conf = agg_rate_conf or {}

    def log_memory(self, *a, **k):
        pass  # avoid a real CUDA call in a CPU-only test env

    def print_trainable_params_stats(self, *a, **k):
        pass

    def replay(self, ends, delays, contribs):
        """Buffer -> canonicalize -> merge, mirroring the real collect path's
        new deferred-merge order (_process_single_trainer_message buffers;
        _process_aggregation_goal_met canonicalizes then replays)."""
        self._per_agg_trainer_list = list(ends)
        self._commit_key_by_end = {
            e: (float(d), str(e)) for e, d in zip(ends, delays)
        }
        self._pending_cohort_contribs = list(contribs)
        self.canon()
        for grad, version_for_rate, stat_utility, var_check, jvp_check in (
            self._pending_cohort_contribs
        ):
            self.aggregate_grads_from_trainers(
                grad,
                version_for_rate=version_for_rate,
                stat_utility=stat_utility,
                grad_for_var_check=var_check,
                jvp_for_snr_check=jvp_check,
            )


def _model():
    torch.manual_seed(0)
    return torch.nn.Linear(4, 1, bias=False)


def _contribs(model, seed_offset=0):
    """4 distinct per-trainer gradient dicts, keyed by the model's only param
    name; fixed (not re-randomized per replay) so merges across orderings
    are directly comparable."""
    (name, _param) = next(model.named_parameters())
    torch.manual_seed(100 + seed_offset)
    grads = [torch.randn(1, 4) for _ in range(4)]
    return [
        ({name: g}, 0, 0.5, None, None) for g in grads
    ]


class TestFedAvgOrderIndependence:
    """optimizer.sort=fedavg -> rate=1.0 unconditionally (no self.grad read
    mid-merge): a pure float-associativity case."""

    def test_two_arrival_orders_converge_to_canonical_merge(self):
        model = _model()
        ends = ["379", "378", "372", "371"]
        delays = [5.5, 6.5, 6.5, 8.0]  # 378/372 tie at D=6.5
        contribs = _contribs(model)

        real_order = [0, 1, 2, 3]  # 379,378,372,371 (real tie-break: arrival)
        sim_order = [0, 2, 1, 3]  # 379,372,378,371 (sim tie-break: sct-sort)

        real = _MergeAgg(model, weighted=False)
        sim = _MergeAgg(model, weighted=False)
        real.replay(
            [ends[i] for i in real_order], [delays[i] for i in real_order],
            [contribs[i] for i in real_order],
        )
        sim.replay(
            [ends[i] for i in sim_order], [delays[i] for i in sim_order],
            [contribs[i] for i in sim_order],
        )

        assert real._per_agg_trainer_list == sim._per_agg_trainer_list
        for r, s in zip(real.grad, sim.grad):
            assert torch.equal(r, s)


class TestGradAwareOrderDependence:
    """grad_aware reads self.grad's running partial sum per contribution's
    rate -- order-sensitive. Canonical replay must converge; uncanonicalized
    replay must diverge, proving the test catches regressions."""

    _conf = {"type": "grad_aware", "base": "neutral", "align_gate": True,
             "align_floor": 0.0, "inverse_var": False}

    def test_two_arrival_orders_converge_after_canonicalization(self):
        model = _model()
        ends = ["379", "378", "372", "371"]
        delays = [5.5, 6.5, 6.5, 8.0]
        contribs = _contribs(model)
        real_order = [0, 1, 2, 3]
        sim_order = [0, 2, 1, 3]

        real = _MergeAgg(model, weighted=True, agg_rate_conf=self._conf)
        sim = _MergeAgg(model, weighted=True, agg_rate_conf=self._conf)
        real.replay(
            [ends[i] for i in real_order], [delays[i] for i in real_order],
            [contribs[i] for i in real_order],
        )
        sim.replay(
            [ends[i] for i in sim_order], [delays[i] for i in sim_order],
            [contribs[i] for i in sim_order],
        )

        for r, s in zip(real.grad, sim.grad):
            assert torch.equal(r, s)

    def test_uncanonicalized_replay_actually_diverges(self):
        """Negative control: merging the same two orderings without
        canonicalizing first gives a different self.grad -- confirms the
        canon-then-replay fix above does real work."""
        model = _model()
        contribs = _contribs(model)
        real_order = [0, 1, 2, 3]
        sim_order = [0, 2, 1, 3]

        real = _MergeAgg(model, weighted=True, agg_rate_conf=self._conf)
        sim = _MergeAgg(model, weighted=True, agg_rate_conf=self._conf)
        for i in real_order:
            grad, version, util, vc, jc = contribs[i]
            real.aggregate_grads_from_trainers(
                grad, version_for_rate=version, stat_utility=util,
                grad_for_var_check=vc, jvp_for_snr_check=jc,
            )
        for i in sim_order:
            grad, version, util, vc, jc = contribs[i]
            sim.aggregate_grads_from_trainers(
                grad, version_for_rate=version, stat_utility=util,
                grad_for_var_check=vc, jvp_for_snr_check=jc,
            )

        diverges = any(
            not torch.equal(r, s) for r, s in zip(real.grad, sim.grad)
        )
        assert diverges


class TestVarCheckPoolIsRateIndependent:
    """The variance gate measures noise, not model-update rate -- staleness-
    scaling the var-check pool (as `self.grad` legitimately is) understates
    its variance and let round-cadence's stale carried-surplus (§F-17) fake
    convergence."""

    _old_conf = {"type": "old"}

    def test_var_check_pool_ignores_rate_but_grad_still_uses_it(self):
        model = _model()
        (name, _param) = next(model.named_parameters())
        torch.manual_seed(7)
        grad = {name: torch.randn(1, 4)}
        var_check = [torch.randn_like(p) for p in model.parameters()]

        fresh = _MergeAgg(model, weighted=True, agg_rate_conf=self._old_conf)
        fresh._model_version = 0
        fresh.aggregate_grads_from_trainers(
            dict(grad), version_for_rate=0, stat_utility=0.5,
            grad_for_var_check=var_check, jvp_for_snr_check=None,
        )

        stale = _MergeAgg(model, weighted=True, agg_rate_conf=self._old_conf)
        stale._model_version = 50  # staleness=50 -> rate = 1/sqrt(51), far from 1.0
        stale.aggregate_grads_from_trainers(
            dict(grad), version_for_rate=0, stat_utility=0.5,
            grad_for_var_check=var_check, jvp_for_snr_check=None,
        )

        # Variance-check pool: identical regardless of staleness/rate.
        assert torch.equal(
            fresh.grad_for_var_check_list[0], stale.grad_for_var_check_list[0]
        )
        raw_stacked = torch.stack(list(var_check))
        assert torch.equal(fresh.grad_for_var_check_list[0], raw_stacked)

        # Model-update merge: still rate-weighted, so it DOES differ.
        assert not torch.equal(fresh.grad[0], stale.grad[0])
