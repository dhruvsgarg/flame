# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""I-1's instrument: the update actually subtracted from the server weights.

EXPTS_CHARTER root-causes the round-boundary collapse (84% -> 25%) to an
undamped forward-gradient optimizer applying each noisy commit raw. That claim
is about the size of the applied step, and nothing on disk records it — the
`agg_round` event carries the INPUT grad norms, not the OUTPUT delta.

Two things must hold. The audit stays OFF by default, because the norms cost
wall time at the CPU-sync site and a timing perturbation is what separates two
otherwise identical legs (simulate_fwdllm.md §D-45). And with it off, the
applied update must be byte-identical to what the un-instrumented code applied.
"""
import math
from pathlib import Path

import pytest

_AGG = (Path(__file__).resolve().parents[2] / "examples" / "fwdllm"
        / "aggregator" / "FedSgdAggregator.py")


class TestGatedOff:
    def test_defaults_off(self):
        src = _AGG.read_text(encoding="utf-8")
        assert 'getattr(self.args, "server_update_audit", False)' in src

    def test_norms_are_inside_the_gate(self):
        """The reduction must not run when the knob is off — that is the whole
        point of the gate, and an ungated `.pow(2).sum()` per param per commit
        is a wall cost on every run."""
        src = _AGG.read_text(encoding="utf-8")
        body = src.split("def _apply_weighted_update", 1)[1].split("\n    def ", 1)[0]
        for line in body.splitlines():
            if ".pow(2).sum()" in line:
                assert line.startswith(" " * 20), (
                    "norm accumulation must sit under `if _audit:`, not at loop level"
                )

    def test_emit_is_a_separate_never_faulting_helper(self):
        src = _AGG.read_text(encoding="utf-8")
        helper = src.split("def _emit_server_update", 1)[1].split("\n    @", 1)[0]
        assert "except Exception:" in helper
        assert "telemetry.is_enabled()" in helper


class TestApplyIsUnchanged:
    """`_param.sub_(_update)` replaced a chained `.sub_(...)`. The rebind is only
    safe if `.to("cpu")` returns the same object — otherwise the update lands on
    a copy and training silently stops. Pin that with a real tensor."""

    def test_to_cpu_on_a_cpu_tensor_is_identity(self):
        torch = pytest.importorskip("torch")
        detached = torch.zeros(3).detach()   # detach() itself returns a fresh view
        assert detached.to("cpu") is detached

    def test_sub_through_the_rebind_mutates_the_original(self):
        torch = pytest.importorskip("torch")
        p = torch.ones(3)
        rebound = p.detach().to("cpu")
        rebound.sub_(torch.full((3,), 0.25))
        assert torch.allclose(p, torch.full((3,), 0.75))


class TestRecordShape:
    def test_ratio_is_delta_over_weight(self):
        from flame.telemetry.events import EVENT_SERVER_UPDATE, build_server_update
        ev, f = build_server_update(
            round_num=2, data_id=7, iteration=3, model_version=41,
            update_delta_norm=0.5, weight_norm=4.0, learning_rate=1e-3,
        )
        assert ev == EVENT_SERVER_UPDATE
        assert f["update_ratio"] == pytest.approx(0.125)
        assert f["model_version"] == 41

    def test_zero_weight_norm_reports_none_not_inf(self):
        """A blown-up run is exactly when this must not raise."""
        from flame.telemetry.events import build_server_update
        _, f = build_server_update(
            round_num=None, data_id=None, iteration=None, model_version=None,
            update_delta_norm=1.0, weight_norm=0.0, learning_rate=1e-3,
        )
        assert f["update_ratio"] is None

    def test_event_is_registered(self):
        from flame.telemetry.events import EVENT_SERVER_UPDATE, KNOWN_EVENTS
        assert EVENT_SERVER_UPDATE in KNOWN_EVENTS


class TestNormMath:
    def test_per_param_squares_sum_to_the_global_l2(self):
        """The aggregator accumulates squared norms across params and roots once.
        That must equal the norm of the concatenated update."""
        torch = pytest.importorskip("torch")
        parts = [torch.tensor([3.0, 4.0]), torch.tensor([12.0])]
        acc = sum(float(p.pow(2).sum()) for p in parts)
        assert math.sqrt(acc) == pytest.approx(torch.cat(parts).norm().item())
