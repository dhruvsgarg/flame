# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""FX-D126: opt-in client grad-norm clip; default off keeps baselines source-faithful."""

from flame.config import Hyperparameters


def test_clip_grad_norm_defaults_off_and_reads_alias():
    assert Hyperparameters(rounds=1, epochs=1).trainer_clip_grad_norm == 0.0
    assert Hyperparameters(rounds=1, epochs=1, trainerClipGradNorm=1.0).trainer_clip_grad_norm == 1.0
