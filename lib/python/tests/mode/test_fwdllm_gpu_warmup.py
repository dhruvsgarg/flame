# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""`_warmup_gpu_kernels` runs one throwaway forward pass before round 1 is
dispatched, to absorb CUDA-context/kernel-compile cost outside the timed
window. Must no-op on CPU, never raise (best-effort), and never touch the
trainer's perturbation RNG state.
"""

import pytest
from unittest.mock import MagicMock, patch

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from examples.fwdllm.trainer.forward_training.tc_transformer_trainer_distribute import (
    ForwardTextClassificationTrainer as FTC,
)


def _stub(device_type, model=None, args=None):
    s = MagicMock()
    s.device = MagicMock(type=device_type)
    s.model = model if model is not None else MagicMock()
    s.args = args if args is not None else MagicMock(max_seq_length=16)
    s.trainer_id = "t1"
    # sentinel RNGs -- must be untouched by warmup
    s.torch_rng = "REAL_RNG_SENTINEL"
    s.torch_cuda_rng = "REAL_CUDA_RNG_SENTINEL"
    return s


class TestGpuWarmup:
    def test_cpu_device_is_noop(self):
        s = _stub("cpu")
        FTC._warmup_gpu_kernels(s)
        s.model.to.assert_not_called()
        s.model.assert_not_called()

    def test_never_raises_even_if_model_forward_fails(self):
        s = _stub("cuda")
        s.model.side_effect = RuntimeError("no CUDA device in this environment")
        FTC._warmup_gpu_kernels(s)  # must not propagate

    def test_never_raises_if_embeddings_lookup_fails(self):
        s = _stub("cuda")
        s.model.get_input_embeddings.side_effect = AttributeError("no embeddings")
        FTC._warmup_gpu_kernels(s)  # must not propagate

    def test_does_not_touch_perturbation_rng_state(self):
        s = _stub("cuda")
        FTC._warmup_gpu_kernels(s)
        assert s.torch_rng == "REAL_RNG_SENTINEL"
        assert s.torch_cuda_rng == "REAL_CUDA_RNG_SENTINEL"

    def test_cuda_path_calls_model_once_with_expected_shape(self):
        s = _stub("cuda", args=MagicMock(max_seq_length=32))
        s.model.get_input_embeddings.return_value = MagicMock(num_embeddings=5000)
        s.model.training = True
        dummy = torch.zeros((1, 32), dtype=torch.long)
        with patch("torch.randint", return_value=dummy) as mock_randint:
            FTC._warmup_gpu_kernels(s)
        s.model.to.assert_called_once_with(s.device)
        mock_randint.assert_called_once()
        args, kwargs = mock_randint.call_args
        assert args[0] == 0 and args[1] == 5000
        assert args[2] == (1, 32)
        s.model.assert_called_once_with(dummy)
        s.model.train.assert_called_once_with(True)

    def test_missing_max_seq_length_falls_back_to_128(self):
        bare_args = MagicMock(spec=[])  # no max_seq_length attribute at all
        s = _stub("cuda", args=bare_args)
        s.model.get_input_embeddings.return_value = MagicMock(num_embeddings=1000)
        with patch("torch.randint", return_value=torch.zeros((1, 128), dtype=torch.long)) as mock_randint:
            FTC._warmup_gpu_kernels(s)
        args, kwargs = mock_randint.call_args
        assert args[2] == (1, 128)
