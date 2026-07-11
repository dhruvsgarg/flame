# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""§R, 2026-07-11: `send_timeout_wait_s` (async_oort's in-flight abandon
timeout) is a workload/comm-latency property that belongs on the aggregator's
hyperparameters, not buried in `selector.kwargs` -- threaded into the
selector's kwargs the same way `_seed` already is."""

from types import SimpleNamespace

from flame.channel_manager import _build_selector_kwargs


def _hp(**kwargs):
    return SimpleNamespace(**kwargs)


class TestBuildSelectorKwargs:
    def test_hyperparameter_send_timeout_is_threaded_in(self):
        merged = _build_selector_kwargs(
            _hp(send_timeout_wait_s=300), {"c": 10, "minInitialTrainers": 10}
        )
        assert merged["send_timeout_wait_s"] == 300
        assert merged["c"] == 10  # untouched selector kwargs survive

    def test_absent_hyperparameter_leaves_selector_kwargs_untouched(self):
        merged = _build_selector_kwargs(_hp(), {"c": 10})
        assert "send_timeout_wait_s" not in merged
        assert merged == {"c": 10}

    def test_selector_kwargs_override_wins_on_conflict(self):
        merged = _build_selector_kwargs(
            _hp(send_timeout_wait_s=300),
            {"c": 10, "send_timeout_wait_s": 42},
        )
        assert merged["send_timeout_wait_s"] == 42

    def test_does_not_mutate_the_input_selector_kwargs_dict(self):
        original = {"c": 10}
        _build_selector_kwargs(_hp(send_timeout_wait_s=300), original)
        assert original == {"c": 10}
