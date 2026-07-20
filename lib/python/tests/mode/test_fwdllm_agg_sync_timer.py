# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""`_agg_sync_timer` (simulate_fwdllm.md §B row 1, 2026-07-20 pm-5): isolates
the wall time of a single CPU/GPU sync point (`.item()`, `.to("cpu")`) from
the rest of its enclosing `@timer_decorator`-wrapped aggregator function, so
`agg_step_timing_breakdown_parity`'s still-open fixed per-call tax can be
localized to the sync call itself vs the rest of the function. Same shape as
the trainer-side `_stage_timer` (already covered by test_tb_prepare_
perturbation.py's family) -- this covers the aggregator-side analog.
"""
import time

import pytest

from examples.fwdllm.aggregator.FedSgdAggregator import _agg_sync_timer
from flame.monitor.runtime import FwdLLMStage


class _FakeOwner:
    def __init__(self, stage):
        self.fwd_llm_stage = stage


@pytest.fixture
def captured_events(monkeypatch):
    # `_agg_sync_timer` does `from flame import telemetry` INSIDE the function
    # body (deferred import, same pattern as `_stage_timer`) -- patch the
    # actual `flame.telemetry` module, not a (nonexistent) module-level name
    # on FedSgdAggregator.
    from flame import telemetry as telemetry_mod

    events: list = []
    monkeypatch.setattr(telemetry_mod, "is_enabled", lambda: True)
    monkeypatch.setattr(
        telemetry_mod, "emit",
        lambda ev, **fields: events.append((ev, fields)),
    )
    return events


class TestAggSyncTimer:
    def test_emits_step_timing_with_func_name_and_duration(self, captured_events):
        stage = FwdLLMStage(round_id=3, data_id=7, iteration=2, trainer_id=None)
        owner = _FakeOwner(stage)

        with _agg_sync_timer(owner, "agg_var_item_sync"):
            time.sleep(0.001)

        assert len(captured_events) == 1
        ev, fields = captured_events[0]
        assert ev == "step_timing"
        assert fields["func"] == "agg_var_item_sync"
        assert fields["duration_s"] >= 0.001
        assert fields["round"] == 3
        assert fields["data_id"] == 7
        assert fields["iteration_per_data_id"] == 2

    def test_noop_when_no_stage(self, captured_events):
        owner = _FakeOwner(None)
        with _agg_sync_timer(owner, "agg_var_item_sync"):
            pass
        assert captured_events == []

    def test_noop_when_telemetry_disabled(self, monkeypatch):
        from flame import telemetry as telemetry_mod

        monkeypatch.setattr(telemetry_mod, "is_enabled", lambda: False)
        calls = []
        monkeypatch.setattr(telemetry_mod, "emit", lambda ev, **f: calls.append(ev))
        stage = FwdLLMStage(round_id=1, data_id=1, iteration=1, trainer_id=None)
        owner = _FakeOwner(stage)

        with _agg_sync_timer(owner, "agg_var_item_sync"):
            pass

        assert calls == []

    def test_propagates_exception_from_body(self, captured_events):
        stage = FwdLLMStage(round_id=1, data_id=1, iteration=1, trainer_id=None)
        owner = _FakeOwner(stage)

        with pytest.raises(ValueError):
            with _agg_sync_timer(owner, "agg_var_item_sync"):
                raise ValueError("boom")
        # still emits on the way out (finally-block), same as _stage_timer.
        assert len(captured_events) == 1
