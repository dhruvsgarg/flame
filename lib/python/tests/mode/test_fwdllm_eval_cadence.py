# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Which commits evaluate must be decided by the commit INDEX, never by whether
the eval thread happens to be free.

`_eval_snapshot_model()` used to return None whenever `_eval_inflight` was set,
so the eval cadence was a wall-clock race between the test-set pass and the
inter-commit wall gap. Sim systematically LOST that race: it compresses the
inter-commit gap (skipping real transport waits) while the eval pass costs the
same or more wall. Measured on the 3600s batch -- real kept 99-100% of its evals
on all nine baselines, sim kept 49-60% on seven of them:

    fedbuff_it_unaware  real  eval 11.6s / gap 28.4s -> 114/115 (99%)
    fedbuff_it_unaware  sim   eval 17.0s / gap 12.7s ->  63/111 (57%)

So the two modes sampled the accuracy trajectory at different, host-speed-
dependent progress points (breaking logical determinism) and `_check_target_stop`
saw a subsampled series in sim only.

Each test below fails against the pre-fix `if self._eval_inflight: return None`.
"""

import threading

from flame.mode.horizontal.syncfl.top_aggregator import TopAggregator


class _StrideHarness:
    """Borrows the real cadence methods; stubs only the weight snapshot, which
    is what `test_fwdllm_eval_background.py` already covers."""

    _eval_stride = TopAggregator._eval_stride
    _eval_snapshot_model = TopAggregator._eval_snapshot_model
    _eval_release = TopAggregator._eval_release
    _EVAL_WAIT_TIMEOUT_S = 2.0

    def __init__(self, stride=None, hp_stride=None):
        self.model = _FakeModel()
        self._eval_inflight = False
        # Non-None so `_eval_snapshot_model` takes the reuse path rather than
        # deepcopy(None); load_state_dict is the only method it calls.
        self._eval_model = _FakeModel()
        if stride is not None:
            self._eval_every_n_commits = stride
        if hp_stride is not None:
            self.config = _FakeConfig(hp_stride)


class _FakeModel:
    def load_state_dict(self, *a, **k):
        pass

    def state_dict(self):
        return {}


class _FakeConfig:
    def __init__(self, stride):
        self.hyperparameters = type("_HP", (), {"eval_every_n_commits": stride})()


def _evaluated_commits(agg, n_commits, release_each=True):
    """Indices (1-based) of the commits that produced an eval."""
    got = []
    for i in range(1, n_commits + 1):
        if agg._eval_snapshot_model() is not None:
            got.append(i)
            if release_each:
                agg._eval_release()
    return got


class TestStrideIsDeterministic:
    def test_stride_one_evaluates_every_commit(self):
        agg = _StrideHarness(stride=1)
        assert _evaluated_commits(agg, 6) == [1, 2, 3, 4, 5, 6]

    def test_stride_two_phases_on_the_first_commit(self):
        """1, 3, 5 -- not 2, 4, 6. The first commit is a real trajectory point
        and stride 1 must reduce to 'every commit' exactly."""
        agg = _StrideHarness(stride=2)
        assert _evaluated_commits(agg, 6) == [1, 3, 5]

    def test_stride_three(self):
        agg = _StrideHarness(stride=3)
        assert _evaluated_commits(agg, 10) == [1, 4, 7, 10]

    def test_both_modes_land_on_identical_commit_indices(self):
        """The parity property itself: two aggregators stepping the same commit
        sequence evaluate the same indices, whatever their wall-clock speed."""
        real, sim = _StrideHarness(stride=2), _StrideHarness(stride=2)
        assert _evaluated_commits(real, 25) == _evaluated_commits(sim, 25)


class TestCadenceIsNotAWallClockRace:
    def test_busy_eval_thread_does_not_shift_the_cadence(self):
        """The regression under test: with the eval thread never released, the
        pre-fix code returned None forever after the first commit. The stride
        must still select 1, 3, 5 -- the busy thread is waited out, not skipped.
        """
        agg = _StrideHarness(stride=2)
        got = []
        for i in range(1, 6):
            # Release only AFTER the next snapshot call would have seen it
            # in-flight, i.e. never proactively.
            if agg._eval_snapshot_model() is not None:
                got.append(i)
            threading.Thread(target=agg._eval_release, daemon=True).start()
        assert got == [1, 3, 5]

    def test_slow_side_and_fast_side_agree(self):
        """A 'sim' whose eval thread is always still busy at the next commit
        must produce the same eval indices as a 'real' whose thread is always
        free -- the exact real/sim asymmetry this fix removes."""
        fast = _StrideHarness(stride=2)
        fast_idx = _evaluated_commits(fast, 12, release_each=True)

        slow = _StrideHarness(stride=2)
        slow_idx = []
        for i in range(1, 13):
            if slow._eval_snapshot_model() is not None:
                slow_idx.append(i)
            threading.Thread(target=slow._eval_release, daemon=True).start()
        assert slow_idx == fast_idx

    def test_wait_timeout_degrades_to_skip_rather_than_stalling(self):
        """A wedged eval thread must not stall training forever (§F-13)."""
        agg = _StrideHarness(stride=1)
        assert agg._eval_snapshot_model() is not None  # commit 1, now in flight
        # Never released -> the wait times out and the eval is skipped.
        assert agg._eval_snapshot_model() is None


class TestStrideConfiguration:
    def test_reads_hyperparameter(self):
        agg = _StrideHarness(hp_stride=4)
        assert agg._eval_stride() == 4
        assert _evaluated_commits(agg, 9) == [1, 5, 9]

    def test_defaults_to_two_when_unset(self):
        """Code-level default, so a yaml that never heard of the knob still gets
        the matched cadence (the fix ships on by default)."""
        agg = _StrideHarness()
        agg.config = type("_C", (), {"hyperparameters": object()})()
        assert agg._eval_stride() == 2

    def test_defaults_to_two_without_any_config(self):
        agg = _StrideHarness()
        assert agg._eval_stride() == 2

    def test_zero_or_negative_is_clamped_to_every_commit(self):
        for bad in (0, -3):
            agg = _StrideHarness(hp_stride=bad)
            assert agg._eval_stride() == 1

    def test_explicit_null_is_clamped_to_every_commit(self):
        """A yaml carrying `eval_every_n_commits:` with no value."""
        agg = _StrideHarness()
        agg.config = _FakeConfig(None)
        assert agg._eval_stride() == 1

    def test_stride_logged_once(self):
        """§F-18: the value is resolved once and cached, so the log line and
        every subsequent decision cannot disagree."""
        agg = _StrideHarness(hp_stride=3)
        assert agg._eval_stride() == 3
        agg.config.hyperparameters.eval_every_n_commits = 99
        assert agg._eval_stride() == 3


class TestReleaseAlwaysClearsInflight:
    def test_release_clears_flag_and_sets_event(self):
        agg = _StrideHarness(stride=1)
        agg._eval_snapshot_model()
        assert agg._eval_inflight is True
        agg._eval_release()
        assert agg._eval_inflight is False
        assert agg._eval_done.is_set()

    def test_release_is_safe_before_any_snapshot(self):
        """It runs in the eval thread's `finally`, which can fire on a snapshot
        that failed before `_eval_done` was ever created."""
        agg = _StrideHarness(stride=1)
        agg._eval_release()
        assert agg._eval_inflight is False
