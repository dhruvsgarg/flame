"""Regression test for experiments.yaml's `main_v2` run-set (BRIDGE_DESIGN.md
checklist step 4) -- this file has no other test coverage; run_sequential.sh's
`--run-set` only reads the `condition:` block (verified separately via a live
`--dry-run`), so `baselines:`/`analyses:` structure has nothing else guarding it.

Run:  python -m pytest test_experiments_yaml.py -q
  or:  python test_experiments_yaml.py
"""

from __future__ import annotations

import os

import yaml

_YAML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments.yaml")


def _load():
    with open(_YAML_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def test_no_local_baselines_stanza():
    """Decision #5 (no duplication): baseline knobs live only in
    _metadata/baselines.yaml, never re-declared here."""
    assert "baselines" not in _load()


def test_main_v2_is_the_4_anchor_plus_attribution_baseline():
    d = _load()
    assert d["run_sets"]["main_v2"]["baselines"] == [
        "fwdllm", "fedbuff_round", "felix_round", "fluxtune", "fwdllm_it_oracular",
    ]


def test_main_v2_condition_matches_main_shared_axes():
    """main_v2 is meant to be `main`'s sim-mode successor for E1-E5 -- same
    shared condition axes (N/K/C/partition/trace/delays/target), so a future
    edit to one that forgets the other doesn't silently diverge the comparison."""
    d = _load()
    main_c = d["run_sets"]["main"]["condition"]
    v2_c = d["run_sets"]["main_v2"]["condition"]
    for key in ("N", "K", "C", "partition_method", "avail_trace",
                "delays", "delay_factor", "target_accuracy", "converge_window"):
        assert main_c[key] == v2_c[key], f"main_v2.condition.{key} diverged from main"


def test_e1_through_e5_reference_main_v2():
    d = _load()
    by_id = {a["id"]: a for a in d["analyses"]}
    for eid in ("e1_time_to_accuracy", "e2_resource_utilization",
                "e3_compute_effectiveness", "e4_communication", "e5_session_length"):
        assert by_id[eid]["run_set"] == "main_v2", eid


def test_a0_attribution_scoped_to_fluxtune_vs_fwdllm_it_oracular():
    d = _load()
    by_id = {a["id"]: a for a in d["analyses"]}
    a0 = by_id["a0_attribution"]
    assert a0["run_set"] == "main_v2"
    assert a0["baselines"] == ["fluxtune", "fwdllm_it_oracular"]


def test_old_main_analyses_untouched_decision_3():
    """`main`'s original 3-baseline real-mode analyses are KEPT AS-IS (decision
    #3) -- main_v2 is additive, not a replacement of the old ids."""
    d = _load()
    old_ids = {a["id"] for a in d["analyses"]}
    for eid in ("1_time_to_target", "2_resource_utilization", "3_compute_productivity",
                "4_network_data", "5_client_sessions"):
        assert eid in old_ids


if __name__ == "__main__":
    import pytest as _pytest
    raise SystemExit(_pytest.main([__file__, "-q"]))
