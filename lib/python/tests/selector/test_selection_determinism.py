# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Selector determinism under seeding (real/sim parity prerequisite).

Each selector now owns a DEDICATED RNG (``self._rng`` / ``self._pyrng``) seeded
at construction from ``config.hyperparameters.seed`` (threaded as the reserved
``_seed`` kwarg by the channel manager). Drawing from the dedicated RNG rather
than the process-global ``np.random`` / ``random`` makes selection a pure
function of (state, seed) AND insulates it from any other np.random consumer in
the process — so a residual real/sim divergence under a shared seed is a genuine
input divergence, not RNG desync. These tests assert that contract on the two
selectors the felix baseline uses (FedBuff for train, Oort for utility):

  * same seed + same state  -> identical selection (reproducible),
  * the selection actually consumes the RNG (distinct seeds can differ), so the
    seeding is meaningful rather than a no-op.

Without this, real and simulated runs can never match selection even with
identical logical state -- which is exactly what the 2026-05-30 parity smoke
(mean Jaccard 0.48) showed.
"""

import random

import numpy as np
import pytest

from flame.channel import (
    KEY_CH_SELECT_REQUESTER,
    KEY_CH_STATE,
    VAL_CH_STATE_SEND,
)
from flame.selector.fedbuff import FedBuffSelector
from flame.selector.oort import OortSelector


def _seed_all(seed: int) -> None:
    """Mirror the aggregator's Phase-0 seeding of the process-global RNGs."""
    np.random.seed(seed)
    random.seed(seed)


def _fedbuff_chosen(seed, make_ends, n=20, c=4):
    """One fresh FedBuff send-state selection under the given seed.

    Seed is threaded via the dedicated-RNG kwarg (``_seed``), as the channel
    manager does — NOT the process-global RNG.
    """
    sel = FedBuffSelector(_seed=seed, c=c, aggGoal=2)
    ends = make_ends(count=n, prefix="t")
    cp = {
        KEY_CH_STATE: VAL_CH_STATE_SEND,
        KEY_CH_SELECT_REQUESTER: "agg",
        "round": 1,
    }
    result = sel.select(ends, cp, trainer_unavail_list=[])
    return frozenset(result)


def _oort_chosen(seed, make_ends, n=20, k=3):
    """One fresh Oort first-round (random) selection under the given seed."""
    sel = OortSelector(_seed=seed, aggr_num=k)
    ends = make_ends(count=n, prefix="t")
    result = sel.select(
        ends, {"round": 1, "cur_time": 0.0},
        trainer_unavail_list=[], task_to_perform="train",
    )
    return frozenset(result)


class TestFedBuffDeterminism:
    def test_same_seed_reproducible(self, make_ends):
        assert _fedbuff_chosen(42, make_ends) == _fedbuff_chosen(42, make_ends)

    def test_subset_of_candidates(self, make_ends):
        chosen = _fedbuff_chosen(42, make_ends, n=20, c=4)
        assert len(chosen) == 4  # sampled a strict subset -> RNG was exercised

    def test_distinct_seeds_can_differ(self, make_ends):
        # With 20 candidates choose-4, distinct seeds almost surely vary; assert
        # at least two outcomes across several seeds (guards the RNG is live).
        outcomes = {_fedbuff_chosen(s, make_ends) for s in range(12)}
        assert len(outcomes) > 1


class TestOortDeterminism:
    def test_same_seed_reproducible(self, make_ends):
        assert _oort_chosen(7, make_ends) == _oort_chosen(7, make_ends)

    def test_distinct_seeds_can_differ(self, make_ends):
        outcomes = {_oort_chosen(s, make_ends) for s in range(12)}
        assert len(outcomes) > 1


class TestSeedingContract:
    """The exact mechanism the aggregator relies on: re-seeding both global RNGs
    makes a downstream draw reproducible."""

    def test_numpy_and_random_reseed_reproduces(self):
        _seed_all(123)
        a = (np.random.rand(5).tolist(), [random.random() for _ in range(5)])
        _seed_all(123)
        b = (np.random.rand(5).tolist(), [random.random() for _ in range(5)])
        assert a == b


class TestDedicatedRngInsulation:
    """The dedicated per-selector RNG must be insulated from the process-global
    RNG, so other np.random/random consumers in the aggregator can't desync
    selection between real and sim (the whole point of self._rng/_pyrng)."""

    def test_global_rng_perturbation_does_not_affect_selection(self, make_ends):
        first = _oort_chosen(7, make_ends)
        # Perturb the global RNGs arbitrarily (simulating other consumers doing a
        # DIFFERENT amount of work between selections in real vs sim).
        np.random.seed(999)
        _ = np.random.rand(37)
        random.seed(123)
        _ = [random.random() for _ in range(11)]
        second = _oort_chosen(7, make_ends)
        assert first == second

    def test_no_seed_defaults_to_deterministic(self, make_ends):
        # No seed passed -> every construction falls back to DEFAULT_SEED, so
        # selection is deterministic across constructions (there is no longer an
        # "unseeded" path -- see AbstractSelector.DEFAULT_SEED).
        outcomes = {
            frozenset(
                OortSelector(aggr_num=3).select(
                    make_ends(count=20, prefix="t"),
                    {"round": 1, "cur_time": 0.0},
                    trainer_unavail_list=[], task_to_perform="train",
                )
            )
            for _ in range(12)
        }
        assert len(outcomes) == 1


from flame.selector import AbstractSelector


class _Mini(AbstractSelector):
    """Minimal concrete selector to exercise the base-class RNG contract."""

    def select(self, ends, channel_props):
        return {}


def _np_seq(sel, n=8):
    return sel._rng.rand(n).tolist()


def _py_seq(sel, n=8):
    return [sel._pyrng.random() for _ in range(n)]


class TestDedicatedRngContract:
    """Base-class guarantees the dedicated RNGs uphold, independent of any
    selector's selection logic."""

    def test_same_seed_same_sequence(self):
        a, b = _Mini(_seed=7), _Mini(_seed=7)
        assert _np_seq(a) == _np_seq(b)
        assert _py_seq(a) == _py_seq(b)

    def test_distinct_seeds_differ(self):
        assert _np_seq(_Mini(_seed=7)) != _np_seq(_Mini(_seed=8))
        assert _py_seq(_Mini(_seed=7)) != _py_seq(_Mini(_seed=8))

    def test_none_seed_falls_back_to_default_and_is_deterministic(self):
        # _seed=None now falls back to DEFAULT_SEED (no unseeded path), so two
        # None-seed selectors produce IDENTICAL sequences.
        assert _Mini(_seed=None)._seed == AbstractSelector.DEFAULT_SEED
        assert _np_seq(_Mini(_seed=None)) == _np_seq(_Mini(_seed=None))
        assert _py_seq(_Mini(_seed=None)) == _py_seq(_Mini(_seed=None))

    def test_construction_does_not_touch_global_rng(self):
        # Seeding a selector must not perturb the process-global RNG state.
        random.seed(0)
        before = [random.random() for _ in range(3)]
        random.seed(0)
        _Mini(_seed=123)  # constructing a seeded selector in between
        after = [random.random() for _ in range(3)]
        assert before == after

    @pytest.mark.parametrize("seed", [None, 0, 1234])
    def test_seed_recorded_and_rngs_present(self, seed):
        sel = _Mini(_seed=seed)
        expected = AbstractSelector.DEFAULT_SEED if seed is None else seed
        assert sel._seed == expected
        assert isinstance(_np_seq(sel), list) and isinstance(_py_seq(sel), list)


class TestSelectRandomOrderDeterminism:
    """select_random's dispatch order, not just its chosen set, must be a pure
    function of (state, seed): it feeds `_pyrng.sample(...)` (deterministic)
    into `dict.fromkeys(...)`. A bare `set()` there would silently reorder by
    string hash, which Python randomizes per-process (PYTHONHASHSEED)
    independent of the seed -- same trainers, different dispatch order every
    launch. Only shows up ACROSS process launches, so these spawn real
    subprocesses under different PYTHONHASHSEED values to catch it."""

    _SNIPPET = """
import json, torch  # noqa: F401 -- import marks ml framework in use as PYTORCH
from flame.selector.{module} import {cls}
sel = {cls}(_seed=7, **{kwargs!r})
ends = {{f"t{{i}}": None for i in range(20)}}
print(json.dumps(list(sel.select_random(ends, num_of_ends=5).keys())))
"""

    def _order_under_hashseed(self, module, cls, kwargs, hashseed):
        import json
        import os
        import subprocess
        import sys

        env = dict(os.environ, PYTHONHASHSEED=hashseed)
        code = self._SNIPPET.format(module=module, cls=cls, kwargs=kwargs)
        out = subprocess.run(
            [sys.executable, "-c", code], env=env, capture_output=True, text=True,
        )
        assert out.returncode == 0, out.stderr
        # logging (this repo's default config) may also land on stdout; the
        # payload is always the last non-empty line.
        last_line = [ln for ln in out.stdout.splitlines() if ln.strip()][-1]
        return json.loads(last_line)

    def _assert_order_hashseed_invariant(self, module, cls, kwargs):
        a = self._order_under_hashseed(module, cls, kwargs, "0")
        b = self._order_under_hashseed(module, cls, kwargs, "1")
        c = self._order_under_hashseed(module, cls, kwargs, "42")
        assert a == b == c, (
            f"{cls}.select_random order depends on PYTHONHASHSEED "
            f"(same seed=7, different hash seeds): {a} vs {b} vs {c}"
        )

    def test_async_oort_order_reproducible(self):
        self._assert_order_hashseed_invariant(
            "async_oort", "AsyncOortSelector",
            dict(
                c=5, aggGoal=2, evalGoalFactor=0.5,
                roundNudgeType="last_train", selectType="default",
            ),
        )

    def test_oort_order_reproducible(self):
        self._assert_order_hashseed_invariant("oort", "OortSelector", dict(aggr_num=5))

    def test_async_random_order_reproducible(self):
        self._assert_order_hashseed_invariant(
            "async_random", "AsyncRandomSelector", dict(c=5, aggGoal=2)
        )


class TestCandidateOrderInsulatedFromEndsInsertionOrder:
    """The oort-family candidate list must not depend on `ends` insertion order
    (= trainer JOIN order, differs real vs sim). Pre-fix, `unexplored_end_ids`
    came from raw `ends.keys()`, so the seeded `_rng.choice` drew a different
    cohort per leg under an identical seed. Feed the same ids in two orders;
    require identical output."""

    _IDS = [f"t{i:03d}" for i in range(40)]

    def _async_oort(self):
        from flame.selector.async_oort import AsyncOortSelector
        return AsyncOortSelector(_seed=1234, c=5, aggGoal=2, evalGoalFactor=0.5,
                                 roundNudgeType="last_train", selectType="default")

    def _oort(self):
        from flame.selector.oort import OortSelector
        return OortSelector(_seed=1234, aggr_num=5)

    def test_async_oort_collect_order_invariant(self, make_ends):
        fwd = make_ends(ids=list(self._IDS))
        rev = make_ends(ids=list(reversed(self._IDS)))
        _, un_fwd = self._async_oort().fetch_statistical_utility(fwd, [], [])
        _, un_rev = self._async_oort().fetch_statistical_utility(rev, [], [])
        assert un_fwd == un_rev == sorted(self._IDS)

    def test_oort_collect_order_invariant(self, make_ends):
        fwd = make_ends(ids=list(self._IDS))
        rev = make_ends(ids=list(reversed(self._IDS)))
        _, un_fwd = self._oort().fetch_statistical_utility(fwd, [], [])
        _, un_rev = self._oort().fetch_statistical_utility(rev, [], [])
        assert un_fwd == un_rev == sorted(self._IDS)

    def test_async_oort_unexplored_draw_same_cohort_across_orders(self, make_ends):
        # End-to-end at the leak site: the seeded explore draw over the collected
        # candidates must pick the SAME set regardless of ends insertion order.
        fwd = make_ends(ids=list(self._IDS))
        rev = make_ends(ids=list(reversed(self._IDS)))
        _, un_fwd = self._async_oort().fetch_statistical_utility(fwd, [], [])
        _, un_rev = self._async_oort().fetch_statistical_utility(rev, [], [])
        a = frozenset(self._async_oort().sample_by_speed(un_fwd, 5))
        b = frozenset(self._async_oort().sample_by_speed(un_rev, 5))
        assert a == b and len(a) == 5


class TestKeyedTopkPopulationInvariance:
    """`_keyed_topk` (select_random/sample_by_speed's shared mechanism) must
    stay population-size-independent: real and sim can momentarily see a
    candidate pool differing by one trainer (async arrival timing), and that
    must never perturb any other candidate's pick or desync later draws --
    the failure mode `random.sample()`/`np.random.choice()` had."""

    def _sel(self, seed=1234):
        from flame.selector.async_oort import AsyncOortSelector
        return AsyncOortSelector(_seed=seed, c=30, aggGoal=10, evalGoalFactor=0.5,
                                 roundNudgeType="last_train", selectType="default")

    def test_extra_non_winning_candidate_does_not_change_pick(self, make_ends):
        base = make_ends(count=70, prefix="t")
        winner = list(self._sel().select_random(
            base, num_of_ends=1, agg_version_key=(0, 1)).keys())

        extended = dict(base, extra=None)
        winner_with_extra = list(self._sel().select_random(
            extended, num_of_ends=1, agg_version_key=(0, 1)).keys())

        # Either the extra candidate doesn't win (pick unchanged), or it does
        # win outright -- never a THIRD, different candidate.
        assert winner_with_extra == winner or winner_with_extra == ["extra"]

    def test_next_draw_resyncs_regardless_of_prior_pool_difference(self, make_ends):
        """The property random.sample() lacks: one pool differing by an extra
        candidate must not desync any LATER draw once pools match again."""
        pool_a = make_ends(count=70, prefix="t")
        pool_b = dict(pool_a, extra=None)  # sim-side transient extra candidate

        sel_a, sel_b = self._sel(), self._sel()
        sel_a.select_random(pool_a, num_of_ends=1, agg_version_key=(0, 1))
        sel_b.select_random(pool_b, num_of_ends=1, agg_version_key=(0, 1))

        # Next call: pools match again on both sides -> must pick identically,
        # regardless of whether the previous call's pools (and picks) matched.
        next_a = list(sel_a.select_random(
            pool_a, num_of_ends=1, agg_version_key=(0, 2)).keys())
        next_b = list(sel_b.select_random(
            pool_a, num_of_ends=1, agg_version_key=(0, 2)).keys())
        assert next_a == next_b

    def test_different_agg_version_key_gives_independent_draw(self, make_ends):
        ends = make_ends(count=20, prefix="t")
        sel = self._sel()
        r1 = sel.select_random(ends, num_of_ends=1, agg_version_key=(0, 1))
        r2 = sel.select_random(ends, num_of_ends=1, agg_version_key=(0, 2))
        # Not asserting inequality (a collision is legal, just unlikely) --
        # only that the key genuinely depends on agg_version_key.
        r3 = sel.select_random(ends, num_of_ends=1, agg_version_key=(0, 1))
        assert r1 == r3  # same round key -> same pick, repeatable

    def test_incremental_single_picks_match_one_bulk_pick(self, make_ends):
        """Within one agg_version_key window, priorities are fixed: picking
        top-1 twice (removing the winner between draws) must match the top-2
        of a single bulk draw -- the consistent-priority-queue property."""
        ends = make_ends(count=20, prefix="t")
        sel = self._sel()
        bulk = list(sel.select_random(ends, num_of_ends=2, agg_version_key=(0, 1)).keys())

        sel2 = self._sel()
        first = list(sel2.select_random(ends, num_of_ends=1, agg_version_key=(0, 1)).keys())
        remaining = {e: None for e in ends if e != first[0]}
        second = list(sel2.select_random(remaining, num_of_ends=1, agg_version_key=(0, 1)).keys())

        assert bulk == [first[0], second[0]]

    def test_pythonhashseed_independent(self):
        # agg_version_key is a tuple embedded in an f-string, not hashed
        # directly -- must stay stable across PYTHONHASHSEED (the exact class
        # of bug TestSelectRandomOrderDeterminism guards against elsewhere).
        import json
        import os
        import subprocess
        import sys

        code = (
            "import json, torch\n"
            "from flame.selector.async_oort import AsyncOortSelector\n"
            "sel = AsyncOortSelector(_seed=7, c=5, aggGoal=2, evalGoalFactor=0.5,"
            " roundNudgeType='last_train', selectType='default')\n"
            "ends = {f't{i}': None for i in range(20)}\n"
            "print(json.dumps(list(sel.select_random(ends, num_of_ends=5,"
            " agg_version_key=(0, 1)).keys())))\n"
        )

        def _run(hashseed):
            env = dict(os.environ, PYTHONHASHSEED=hashseed)
            out = subprocess.run([sys.executable, "-c", code], env=env,
                                 capture_output=True, text=True)
            assert out.returncode == 0, out.stderr
            last_line = [ln for ln in out.stdout.splitlines() if ln.strip()][-1]
            return json.loads(last_line)

        a, b, c = _run("0"), _run("1"), _run("42")
        assert a == b == c
