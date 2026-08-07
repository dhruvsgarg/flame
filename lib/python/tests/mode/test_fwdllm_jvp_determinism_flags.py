# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""H12 flags: `FWDLLM_JVP_FP32` and `FWDLLM_STRICT_DETERMINISM`.

Both are env-gated (a standalone probe must set them before torch touches CUDA,
where no aggregator config is in scope) and both default OFF, so an unflagged run
is byte-identical to before. These tests pin the gate semantics and the default;
whether the flags actually close the replicate floor is an EXPERIMENT
(`expt_scripts/probe_jvp_determinism.py`), not a unit test.
"""

import importlib
import os

import pytest

torch = pytest.importorskip("torch")

from examples.fwdllm.trainer.forward_training import fwdgrad_utils


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in ("FWDLLM_JVP_FP32", "FWDLLM_STRICT_DETERMINISM"):
        monkeypatch.delenv(k, raising=False)


class TestJvpFp32Gate:
    def test_defaults_off(self):
        assert fwdgrad_utils.jvp_fp32_enabled() is False

    @pytest.mark.parametrize("val", ["1", "true", "TRUE", "yes", " 1 "])
    def test_truthy_spellings(self, monkeypatch, val):
        monkeypatch.setenv("FWDLLM_JVP_FP32", val)
        assert fwdgrad_utils.jvp_fp32_enabled() is True

    @pytest.mark.parametrize("val", ["", "0", "false", "no", "off"])
    def test_falsy_spellings(self, monkeypatch, val):
        monkeypatch.setenv("FWDLLM_JVP_FP32", val)
        assert fwdgrad_utils.jvp_fp32_enabled() is False

    def test_flag_off_still_enters_autocast(self, monkeypatch):
        """Byte-identical when off: the autocast context is still entered."""
        entered: list = []

        class _Spy:
            def __enter__(self): entered.append(True)
            def __exit__(self, *a): return False

        monkeypatch.setattr(fwdgrad_utils, "autocast", lambda: _Spy())
        _run_jvp()
        assert entered == [True]

    def test_flag_on_skips_autocast(self, monkeypatch):
        entered = []

        class _Spy:
            def __enter__(self): entered.append(True)
            def __exit__(self, *a): return False

        monkeypatch.setattr(fwdgrad_utils, "autocast", lambda: _Spy())
        monkeypatch.setenv("FWDLLM_JVP_FP32", "1")
        _run_jvp()
        assert entered == []

    def test_the_central_difference_itself_is_unchanged(self, monkeypatch):
        """The flag changes PRECISION, never the estimator -- same h, same form."""
        base = _run_jvp()
        monkeypatch.setenv("FWDLLM_JVP_FP32", "1")
        flagged = _run_jvp()
        assert torch.allclose(base[1], flagged[1])


def _run_jvp():
    """A 1-param linear loss: jvp is exactly d/de f(p+e*v) = sum(v)."""
    params = [torch.tensor([1.0, 2.0])]
    v = [torch.tensor([1.0, 1.0])]
    return fwdgrad_utils.calculate_jvp(lambda ps: ps[0].sum(), params, v)


class TestStrictDeterminismGate:
    def _initializer(self):
        return importlib.import_module("examples.fwdllm.expts.initializer")

    def test_defaults_off(self):
        assert self._initializer().strict_determinism_enabled() is False

    @pytest.mark.parametrize("val,want", [("1", True), ("yes", True),
                                          ("0", False), ("", False)])
    def test_spellings(self, monkeypatch, val, want):
        monkeypatch.setenv("FWDLLM_STRICT_DETERMINISM", val)
        assert self._initializer().strict_determinism_enabled() is want

    def test_off_does_not_touch_deterministic_algorithms(self, monkeypatch):
        calls = []
        monkeypatch.setattr(torch, "use_deterministic_algorithms",
                            lambda *a, **k: calls.append(a))
        self._initializer().set_seed(1234)
        assert calls == []

    def test_on_pins_algorithms_and_cublas_workspace(self, monkeypatch):
        calls = []
        monkeypatch.setattr(torch, "use_deterministic_algorithms",
                            lambda *a, **k: calls.append(a))
        monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
        monkeypatch.setenv("FWDLLM_STRICT_DETERMINISM", "1")
        self._initializer().set_seed(1234)
        assert calls == [(True,)]
        # cuBLAS reads this at context creation; unset => the first GEMM raises.
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"

    def test_on_disables_tf32(self, monkeypatch):
        monkeypatch.setattr(torch, "use_deterministic_algorithms", lambda *a, **k: None)
        monkeypatch.setenv("FWDLLM_STRICT_DETERMINISM", "1")
        self._initializer().set_seed(1234)
        assert torch.backends.cudnn.allow_tf32 is False
        assert torch.backends.cuda.matmul.allow_tf32 is False

    def test_seeding_still_happens_either_way(self, monkeypatch):
        monkeypatch.setattr(torch, "use_deterministic_algorithms", lambda *a, **k: None)
        init = self._initializer()
        init.set_seed(4321)
        a = torch.randn(3)
        monkeypatch.setenv("FWDLLM_STRICT_DETERMINISM", "1")
        init.set_seed(4321)
        assert torch.allclose(a, torch.randn(3))
