# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""FX-N10: the dataset switch (fl_data) and the imported google_speech splits."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

EX = Path(__file__).resolve().parents[2] / "examples"
sys.path.insert(0, str(EX / "async_cifar10"))
import fl_data  # noqa: E402


def test_default_is_cifar10_and_names_normalize():
    assert fl_data.spec_for(SimpleNamespace()).name == "cifar10"
    assert fl_data.spec_for(SimpleNamespace(dataset_name="google-speech")).name == "google_speech"
    with pytest.raises(ValueError):
        fl_data.spec_for(SimpleNamespace(dataset_name="mnist"))


@pytest.mark.parametrize("name", sorted(fl_data.SPECS))
def test_models_take_their_stub_and_real_shapes(name):
    spec = fl_data.SPECS[name]
    m = spec.model().eval()
    with torch.no_grad():
        for shape in (spec.stub_shape, spec.shape):
            out = m(torch.randn(2, *shape))
            assert out.shape == (2, spec.num_classes)


def test_speech_splits_are_disjoint_partitions_of_training():
    d = yaml.safe_load(open(EX / "_metadata" / "dataset_splits" / "google_speech_alpha0.1_n100.yaml"))
    splits = d["trainer_data_splits"]
    assert len(splits) == 100 and d["total_samples"] == 84843
    allidx = [i for v in splits.values() for i in v]
    assert len(allidx) == len(set(allidx)) and max(allidx) < 84843


@pytest.mark.skipif(not fl_data.speech_root().exists(), reason="SpeechCommands not on this node")
def test_speech_reader_matches_torchaudio_counts():
    assert len(fl_data.SpeechCommands("training")) == 84843
    assert len(fl_data.SpeechCommands("testing")) == 11005
    x, y = fl_data.SpeechCommands("testing")[0]
    assert x.shape == (1, fl_data.SPEECH_LEN) and 0 <= y < 35


def test_dataset_dir_prefers_env_root_then_falls_back(tmp_path, monkeypatch):
    (tmp_path / "google_speech").mkdir()
    monkeypatch.setenv("FLAME_DATA_ROOT", str(tmp_path))
    assert fl_data.dataset_dir("google_speech") == tmp_path / "google_speech"
    assert fl_data.dataset_dir("cifar10") != tmp_path / "cifar10"  # not there -> next root / legacy dir


def test_incomplete_speech_copy_is_refused(tmp_path):
    for n in ("testing_list.txt", "validation_list.txt"):
        (tmp_path / n).write_text("")
    (tmp_path / "yes").mkdir()
    (tmp_path / "yes" / "a_nohash_0.wav").write_bytes(b"")
    with pytest.raises(RuntimeError, match="incomplete copy"):
        fl_data.SpeechCommands("training", root=tmp_path)
