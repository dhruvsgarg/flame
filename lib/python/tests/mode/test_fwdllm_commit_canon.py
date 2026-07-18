# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Canonical (D, trainer_id) commit ordering.

When two trainers share a registry delay (equal D), real breaks the sct tie by
physical arrival and sim by sct-sort, so the receive ORDER can swap even though
`var` and the cohort SET match -- and the EXACT-order `cohort_sequence` rung
flags it. `_canonicalize_cohort_commit_order` reorders THIS cycle's cohort by
(D, str(end)) so equal-D ties break by trainer_id IDENTICALLY in real and sim.

`_pending_cohort_contribs` (buffered per-contribution material, not yet merged
into self.grad) is 1:1 with `_per_agg_trainer_list` -- both share one append
site in `_process_single_trainer_message` -- so canon() must permute them in
lockstep: the eventual self.grad summation (in `_process_aggregation_goal_met`)
replays `_pending_cohort_contribs` in whatever order canon() leaves it, so an
un-permuted contrib list would merge in the wrong order even though the
trainer-id bookkeeping looked canonical (simulate_fwdllm.md P0-1).

These tests pin: (1) two different input orders (real-physical vs sim-sct)
canonicalize to the SAME sequence, contribs included; (2) the tie-break is by
trainer_id; (3) no-op when keys are missing (delays off) or already canonical;
(4) a tie-swap never crosses the split-half boundary (calculate_var invariant).
"""

from flame.mode.horizontal.syncfl.fwdllm_aggregator import TopAggregator


class _CanonAgg:
    """Minimal stand-in exposing only what _canonicalize_cohort_commit_order
    touches. `end` values are the last-3-hex trainer tokens for readability."""

    def __init__(self, ends, delays, contribs=None):
        self._per_agg_trainer_list = list(ends)
        # end -> (D, str(end)) key, mirroring aggregate_weights' capture; a None
        # delay models a contributor that did not stamp D (delays off).
        self._commit_key_by_end = {
            e: ((float(d), str(e)) if d is not None else None)
            for e, d in zip(ends, delays)
        }
        # Buffered per-contribution material (see fwdllm_aggregator.py's
        # _pending_cohort_contribs) -- 1:1 with _per_agg_trainer_list at the
        # same index; defaults to the end tokens themselves so a permutation
        # is directly comparable to the reordered trainer-id list.
        self._pending_cohort_contribs = list(contribs if contribs is not None else ends)

    canon = TopAggregator._canonicalize_cohort_commit_order


# D-values keyed by the last-3-hex token; trainers 372 & 378 both drew
# training_delay_s=13.0 -> D=6.5, the tie.
_D = {
    "370": 2.0, "375": 2.5, "373": 3.5, "376": 5.0, "379": 5.5,
    "378": 6.5, "372": 6.5, "371": 8.0, "377": 8.5, "374": 9.0,
}
_CANON_ORDER = ["370", "375", "373", "376", "379", "372", "378", "371", "377", "374"]


def _mk(ends):
    return _CanonAgg(ends, [_D[e] for e in ends])


class TestCanonicalizesToOneOrder:
    def test_real_physical_and_sim_sct_orders_converge(self):
        # real broke the 6.5 tie 378-before-372; sim broke it 372-before-378.
        real_order = ["370", "375", "373", "376", "379", "378", "372", "371", "377", "374"]
        sim_order = ["370", "375", "373", "376", "379", "372", "378", "371", "377", "374"]
        ra, sa = _mk(real_order), _mk(sim_order)
        ra.canon()
        sa.canon()
        assert ra._per_agg_trainer_list == _CANON_ORDER
        assert sa._per_agg_trainer_list == _CANON_ORDER
        assert ra._per_agg_trainer_list == sa._per_agg_trainer_list
        # contribs (the actual to-be-merged material) converge too, not just
        # the trainer-id bookkeeping -- this is what makes self.grad's later
        # replay order real/sim-identical.
        assert ra._pending_cohort_contribs == _CANON_ORDER
        assert sa._pending_cohort_contribs == _CANON_ORDER

    def test_tie_breaks_by_trainer_id_not_arrival(self):
        # only the two tied (D=6.5) members may move, and by id order (372<378)
        a = _mk(["379", "378", "372", "371"])
        a.canon()
        assert a._per_agg_trainer_list == ["379", "372", "378", "371"]

    def test_pending_contribs_reorder_in_lockstep(self):
        # contribs need not equal the end tokens (real code buffers gradient
        # tensors, not ids) -- pin that canon() permutes by INDEX, not value.
        a = _CanonAgg(
            ["379", "378", "372", "371"],
            [_D[e] for e in ["379", "378", "372", "371"]],
            contribs=["c_379", "c_378", "c_372", "c_371"],
        )
        a.canon()
        assert a._per_agg_trainer_list == ["379", "372", "378", "371"]
        assert a._pending_cohort_contribs == ["c_379", "c_372", "c_378", "c_371"]


class TestNoOps:
    def test_already_canonical_is_unchanged(self):
        a = _mk(_CANON_ORDER)
        before = list(a._per_agg_trainer_list)
        before_contribs = list(a._pending_cohort_contribs)
        a.canon()
        assert a._per_agg_trainer_list == before
        assert a._pending_cohort_contribs == before_contribs

    def test_missing_delay_key_falls_back_to_arrival_order(self):
        # a contributor without a stamped D (delays off) -> arrival order kept
        ends = ["379", "378", "372", "371"]
        a = _CanonAgg(ends, [5.5, None, 6.5, 8.0])
        a.canon()
        assert a._per_agg_trainer_list == ends  # no reorder
        assert a._pending_cohort_contribs == ends

    def test_single_contributor_is_noop(self):
        a = _mk(["370"])
        a.canon()
        assert a._per_agg_trainer_list == ["370"]


class TestVarInvariance:
    def test_tie_swap_stays_within_split_half(self):
        # the tie (372/378) sits at indices 5-6 of the 10-cohort; the split-half
        # boundary is n//2 = 5, so both are in the SECOND half in BOTH orders ->
        # the reorder cannot move a grad across the boundary -> calculate_var
        # (mean of first-half vs second-half) is invariant.
        n = len(_CANON_ORDER)
        real_order = ["370", "375", "373", "376", "379", "378", "372", "371", "377", "374"]
        first_ids = lambda seq: set(seq[: n // 2])
        second_ids = lambda seq: set(seq[n // 2:])
        # half-MEMBERSHIP is identical before and after canonicalization
        a = _mk(real_order)
        pre_first, pre_second = first_ids(real_order), second_ids(real_order)
        a.canon()
        assert first_ids(a._per_agg_trainer_list) == pre_first
        assert second_ids(a._per_agg_trainer_list) == pre_second
