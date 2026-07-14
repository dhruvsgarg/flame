"""Regression test for scripts.parity.cli's run-dir discovery.

Run:  python -m pytest scripts/parity/test_cli.py -q
  or:  python scripts/parity/test_cli.py
"""

from __future__ import annotations

import os
import sys

_SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from parity.cli import _find_run_dirs  # noqa: E402


def test_prefix_sharing_baseline_does_not_collide(tmp_path):
    """Regression, 2026-07-14: `fwdllm` is a strict prefix of `fwdllm_plus`'s run
    dir names. The old glob (`*{tag}*real*`) matched both, and `sorted()[-1]`
    silently picked the fwdllm_plus pair for the `fwdllm` tag whenever both
    baselines' run dirs coexisted -- double-reporting fwdllm_plus's numbers
    under the fwdllm label with no error. Anchoring on `_{tag}_n<digits>_` (the
    exact shape run_sequential.sh emits) must resolve each tag to its own dirs."""
    for name in [
        "run_20260714_003007_fwdllm_n10_smoke_syn_0_real",
        "run_20260714_023125_fwdllm_n10_smoke_syn_0_sim",
        "run_20260714_032159_fwdllm_plus_n10_smoke_syn_0_real",
        "run_20260714_052349_fwdllm_plus_n10_smoke_syn_0_sim",
    ]:
        (tmp_path / name).mkdir()

    fwdllm_real, fwdllm_sim = _find_run_dirs(str(tmp_path), "fwdllm")
    assert "fwdllm_plus" not in fwdllm_real, fwdllm_real
    assert "fwdllm_plus" not in fwdllm_sim, fwdllm_sim
    assert fwdllm_real.endswith("fwdllm_n10_smoke_syn_0_real")
    assert fwdllm_sim.endswith("fwdllm_n10_smoke_syn_0_sim")

    plus_real, plus_sim = _find_run_dirs(str(tmp_path), "fwdllm_plus")
    assert plus_real.endswith("fwdllm_plus_n10_smoke_syn_0_real")
    assert plus_sim.endswith("fwdllm_plus_n10_smoke_syn_0_sim")


if __name__ == "__main__":
    import pytest as _pytest
    raise SystemExit(_pytest.main([__file__, "-q"]))
