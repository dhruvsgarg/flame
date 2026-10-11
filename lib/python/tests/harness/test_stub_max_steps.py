# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""FX-D110: stub mode may cap real local steps; other modes never do."""

from types import SimpleNamespace

from flame import harness


def test_cap_applies_only_in_stub():
    hp = SimpleNamespace(harness_stub_max_steps=1)
    assert harness.stub_max_steps(hp, "stub") == 1
    assert harness.stub_max_steps(hp, "tiny_cpu") is None and harness.stub_max_steps(hp, "off") is None
    assert harness.stub_max_steps(SimpleNamespace(), "stub") is None
