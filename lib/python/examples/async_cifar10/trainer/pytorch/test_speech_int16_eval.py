# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""FX-N84: the int16 speech test set must feed eval exactly the float waveform the old decode produced."""

import numpy as np
import torch

import fl_data


def test_model_input_matches_float_decode_for_every_int16():
    pcm = np.arange(-32768, 32768, dtype=np.int16)
    old = torch.from_numpy(pcm.astype(np.float32) / 32768.0)
    assert torch.equal(fl_data.model_input(torch.from_numpy(pcm)), old)


def test_model_input_passes_float_through():
    x = torch.randn(2, 3, 32, 32)
    assert fl_data.model_input(x) is x


def test_in_memory_keeps_speech_int16(monkeypatch):
    class _Speech(fl_data.SpeechCommands):
        def __init__(self):
            pass

        def __len__(self):
            return 3

        def pcm16(self, i):
            return torch.full((1, 4), i - 1, dtype=torch.int16), i

    ds = fl_data.in_memory(_Speech())
    assert ds.tensors[0].dtype == torch.int16
    sub = fl_data.in_memory(torch.utils.data.Subset(_Speech(), [2]))
    assert sub.tensors[0].tolist() == [[[1, 1, 1, 1]]] and sub.tensors[1].tolist() == [2]


def test_agg_malloc_trim_runs():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "aggregator" / "pytorch"))
    from agg_common import trim_host_heap

    assert trim_host_heap() >= 0.0
