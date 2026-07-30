# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""sim_charge_registry (FWDLLM_DESIGN.md §P): looks up a profiled vclock
charge for a real-only-artifact category, gated on the registry's own
`charge:` flag -- never on whether the caller merely asked."""

from flame.mode.horizontal.sim_charge_registry import get_profiled_charge_s

_YAML = """
redispatch_turnaround:
  weights:
    charge: true
    mean_s: 0.4365
    n: 3264
  var_bad:
    charge: false
    mean_s: 0.0201
    n: 7456
"""


def _write(tmp_path, text=_YAML):
    p = tmp_path / "registry.yaml"
    p.write_text(text)
    return str(p)


def test_returns_mean_when_charge_true(tmp_path):
    path = _write(tmp_path)
    assert get_profiled_charge_s(path, "redispatch_turnaround", "weights") == 0.4365


def test_returns_none_when_charge_false(tmp_path):
    path = _write(tmp_path)
    assert get_profiled_charge_s(path, "redispatch_turnaround", "var_bad") is None


def test_returns_none_for_unknown_label(tmp_path):
    path = _write(tmp_path)
    assert get_profiled_charge_s(path, "nonexistent", "weights") is None


def test_returns_none_for_unknown_payload_kind(tmp_path):
    path = _write(tmp_path)
    assert get_profiled_charge_s(path, "redispatch_turnaround", "other") is None


def test_none_path_is_a_noop():
    assert get_profiled_charge_s(None, "redispatch_turnaround", "weights") is None


def test_missing_file_is_a_noop_not_a_crash(tmp_path):
    assert get_profiled_charge_s(str(tmp_path / "nope.yaml"), "x", "y") is None


def test_default_payload_kind_key(tmp_path):
    path = _write(tmp_path, "drain_tail:\n  _default:\n    charge: true\n    mean_s: 0.33\n")
    assert get_profiled_charge_s(path, "drain_tail") == 0.33


_SHARED_COMPUTE_YAML = """
drain_tail:
  _default:
    charge: true
    mean_s: 0.2783
    n: 3654
fedavg:
  _default:
    charge: true
    mean_s: 0.0645
    n: 3654
"""


def test_shared_compute_categories_resolve_a_profiled_charge(tmp_path):
    """§D-18: drain_tail/fedavg were charged from sim's OWN live span on the
    theory that "sim runs the identical op, so its span IS the cost". At n=100
    that span is contention-inflated (fluxtune: sim 0.585 vs real 0.330 s/cycle),
    so the live path folds sim-host noise onto the vclock (§F-1/§F-20). They are
    profiled from real now, like any other charged category."""
    path = _write(tmp_path, _SHARED_COMPUTE_YAML)
    assert get_profiled_charge_s(path, "drain_tail") == 0.2783
    assert get_profiled_charge_s(path, "fedavg") == 0.0645


def test_shared_compute_charge_is_inert_without_a_registry(tmp_path):
    """A baseline that never opted in must keep the old live-span behavior --
    the call site passes profiled_s=None and falls back."""
    assert get_profiled_charge_s(None, "drain_tail") is None
    assert get_profiled_charge_s(_write(tmp_path), "drain_tail") is None
