"""Regression tests for BRIDGE_DESIGN.md §2b reducer-gap fixes (N3/N5/N6) as
wired into compare_baselines.py's per-experiment reducers.

Run:  python -m pytest test_compare_baselines.py -q
  or:  python test_compare_baselines.py
"""

from __future__ import annotations

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import compare_baselines as C  # noqa: E402
from plotlib import reducers as R  # noqa: E402


# --------------------------------------------------------------------------- #
# is_async resolution (was hardcoded `b == "fluxtune"` -- silently mislabeled
# every new async baseline as sync; same bug class run_sequential.sh's
# _BL_INTERNALS lookup already fixed this session)
# --------------------------------------------------------------------------- #
def test_is_async_resolves_from_baselines_yaml_not_a_hardcoded_name():
    assert C._is_async("fluxtune") is True
    assert C._is_async("fwdllm") is False
    assert C._is_async("fwdllm_it_oracular") is False
    # the whole point of the fix: these are NOT "fluxtune" but ARE async
    assert C._is_async("fedbuff_round") is True
    assert C._is_async("fedbuff_it_unaware") is True
    assert C._is_async("fedbuff_it_oracular") is True
    assert C._is_async("felix_round") is True
    assert C._is_async("felix_it") is True


def test_is_async_unknown_baseline_defaults_false_with_warning(capsys):
    assert C._is_async("not_a_real_baseline") is False
    assert "WARN" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# N3 — expt1 prefers converge.json, falls back to reconstruction
# --------------------------------------------------------------------------- #
def _rr(**kw):
    rr = R.RunResult(key="k", run_dir="/tmp/x")
    for k, v in kw.items():
        setattr(rr, k, v)
    return rr


def test_expt1_uses_converge_json_when_target_and_window_match():
    rr = _rr(t0=0.0, evals=[{"ts": 5.0, "round": 0, "data_id": 0, "acc": 0.9, "loss": 0.1}],
             converge_json={"converged": True, "target_accuracy": 0.84, "window": 20,
                             "time_to_converge_wall_s": 42.0, "time_to_converge_vclock_s": 99.0,
                             "rounds_at_converge": 3, "n_bins_completed": 7})
    out = C.expt1_time_to_target(rr, target=0.84, window=20)
    assert out["source"] == "converge.json"
    assert out["wall_s"] == 42.0
    assert out["rounds"] == 3
    assert out["data_bins"] == 7


def test_expt1_falls_back_to_reconstruction_when_target_mismatches():
    """A converge.json exists but for a DIFFERENT --target-acc -- must not be
    trusted for this call (would silently misreport a different threshold)."""
    rr = _rr(t0=0.0,
             evals=[{"ts": 5.0, "round": 0, "data_id": 0, "acc": 0.9, "loss": 0.1}] * 20,
             converge_json={"converged": True, "target_accuracy": 0.5, "window": 20,
                             "time_to_converge_wall_s": 1.0})
    out = C.expt1_time_to_target(rr, target=0.84, window=1)
    assert out["source"] == "reconstructed"


def test_expt1_falls_back_when_no_converge_json():
    rr = _rr(t0=0.0, evals=[{"ts": 5.0, "round": 0, "data_id": 0, "acc": 0.9, "loss": 0.1}],
             converge_json=None)
    out = C.expt1_time_to_target(rr, target=0.84, window=1)
    assert out["source"] == "reconstructed"
    assert out["reached"] is True


# --------------------------------------------------------------------------- #
# N5 — expt2 idle decomposition
# --------------------------------------------------------------------------- #
def test_expt2_decomposes_idle_into_net_wait_and_residual():
    rr = _rr(busy_frac=[0.6], net_wait_frac=[0.3], agg_wall_s=100.0)
    out = C.expt2_utilization(rr)
    assert out["trainer_busy_frac_p50"] == 0.6
    assert out["trainer_net_wait_frac_p50"] == 0.3
    assert out["trainer_idle_frac_p50"] == pytest.approx(0.1)


# --------------------------------------------------------------------------- #
# N6 — expt5 picks round_span_durs (sync) vs session_durs (async)
# --------------------------------------------------------------------------- #
def test_expt5_sync_uses_round_span_durs():
    rr = _rr(session_durs=[1.0, 1.0], round_span_durs=[9.0, 11.0],
             part_rounds=[], part_bins=[], part_iters=[])
    out = C.expt5_sessions(rr, is_async=False)
    assert out["session_def"] == "one_round_span"
    assert out["n_sessions"] == 2
    assert out["session_s_p50"] == 10.0


def test_expt5_async_uses_session_durs_unchanged():
    rr = _rr(session_durs=[1.0, 3.0], round_span_durs=[999.0],
             part_rounds=[], part_bins=[], part_iters=[])
    out = C.expt5_sessions(rr, is_async=True)
    assert out["session_def"] == "dispatch_to_commit"
    assert out["n_sessions"] == 2
    assert out["session_s_p50"] == 2.0


if __name__ == "__main__":
    import pytest as _pytest
    raise SystemExit(_pytest.main([__file__, "-q"]))
