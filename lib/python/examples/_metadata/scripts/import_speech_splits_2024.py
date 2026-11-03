#!/usr/bin/env python3
# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""FX-N10: turn the 2024 SoCC google_speech per-trainer JSONs into launcher split files.

Reads async_google_speech/trainer/config_dir<a>_num100_traceFail_*_oort/trainer_<i>.json and writes
dataset_splits/google_speech_alpha<a>_n100.yaml (same schema as cifar10_*). Checks every trainer's task id
and delay against trainer_registry.yaml, and every index against SpeechCommands v0.02 training (84843).
Only n=100 is imported: the 2024 n=300 dirs reuse trainer 1's task id past trainer 100, and the alpha=100
set was cut for another registry (task ids differ).
"""

import glob
import json
import os
import sys

import yaml

META = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(META, "..", "async_google_speech", "trainer")
TRAIN_SIZE = 84843  # = fl_data.SPEECH_TRAIN_SIZE
DIRS = {0.1: "config_dir0.1_num100_traceFail_48h_oort", 1.0: "config_dir1_num100_traceFail_48h_oort",
        10.0: "config_dir10_num100_traceFail_48h_oort"}


def main() -> int:
    reg = yaml.safe_load(open(os.path.join(META, "trainer_registry.yaml")))["trainers"]
    for alpha, d in DIRS.items():
        splits, seen = {}, set()
        for i in range(1, 101):
            c = json.load(open(os.path.join(SRC, d, f"trainer_{i}.json")))
            key = f"trainer_{i:03d}"
            t = reg[key]
            h = c["hyperparameters"]
            assert c["taskid"] == t["task_id"], (d, key)
            assert float(h["training_delay_s"]) == float(t["training_delay_s"]), (d, key)
            idx = [int(x) for x in h["trainer_indices_list"]]
            assert all(0 <= x < TRAIN_SIZE for x in idx), (d, key)
            assert not seen & set(idx), (d, key, "overlapping partition")
            seen |= set(idx)
            splits[key] = idx
        out = {"dataset_name": "google_speech", "dirichlet_alpha": alpha, "num_trainers": 100,
               "total_samples": TRAIN_SIZE, "source": f"async_google_speech/trainer/{d} (2024 SoCC)",
               "trainer_data_splits": splits}
        path = os.path.join(META, "dataset_splits", f"google_speech_alpha{alpha}_n100.yaml")
        with open(path, "w") as f:
            yaml.safe_dump(out, f, sort_keys=False, default_flow_style=None)
        print(f"{path}: {len(seen)} samples over 100 trainers")
    assert len(glob.glob(os.path.join(META, "dataset_splits", "google_speech_*"))) >= len(DIRS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
