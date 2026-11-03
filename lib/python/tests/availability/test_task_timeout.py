# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""FX-D46: the abandon timeout is the run's send_timeout_wait_s (90s default)."""

import types

from flame.availability.client_availability import ClientAvailability


def _ca(hp):
    ca = ClientAvailability.__new__(ClientAvailability)
    ca.config = types.SimpleNamespace(hyperparameters=hp)
    return ca


def test_default_90():
    assert _ca(types.SimpleNamespace())._task_timeout_s() == 90.0


def test_scaled_timeout():
    assert _ca(types.SimpleNamespace(send_timeout_wait_s=450.0))._task_timeout_s() == 450.0
