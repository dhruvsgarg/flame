# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""FX-N74: every baseline-defining value comes from _metadata/baseline_reference.yaml, cited, and reaches the config."""

import math

import pytest
import yaml

from flame.launch import baseline_reference as br
from tests.launch.test_debug_run_trace_substitution import _run_generator, generator_source  # noqa: F401

B6 = ("felix", "fedbuff", "oort", "oort_star", "refl", "feddance")
DATASETS = ("cifar10", "google_speech")
TAGS = ("[code]", "[paper]", "[ours]")
REF = br.load()


def _cited(item, where):
    assert isinstance(item, dict) and "src" in item and item["src"].startswith(TAGS), f"{where}: needs v + src [code|paper|ours]"


@pytest.mark.parametrize("bl", B6)
def test_every_cell_is_complete_and_cited(bl):
    e = br.entry(REF, bl)
    _cited(e["participation"], f"{bl}.participation")
    if "selector" in e:
        _cited(e["selector"], f"{bl}.selector")
    for ds in DATASETS:
        d = e["datasets"][ds]
        assert ("local_steps" in d) != ("local_epochs" in d), f"{bl}.{ds}: exactly one of local_steps / local_epochs"
        assert {"source", "ours"} <= set(d.get("model", {})), f"{bl}.{ds}: model parity needs source + ours"
        for k in br.REQUIRED + tuple(k for k in d if k not in br.REQUIRED + ("model",)):
            assert k in d, f"{bl}.{ds}.{k} missing"
            _cited(d[k], f"{bl}.{ds}.{k}")
            if "source_v" in d[k]:  # adapted for our model: the paper's deviation table needs why + evidence
                assert d[k].get("why") and d[k].get("evidence"), f"{bl}.{ds}.{k}: adapted value needs why + evidence"


def _leaves(d, pre=""):
    for k, v in d.items():
        if isinstance(v, dict) and v:
            yield from _leaves(v, f"{pre}{k}.")
        else:
            yield f"{pre}{k}", v


def _get(d, path):
    for k in path.split("."):
        d = d[k]
    return d


@pytest.mark.parametrize("ds", DATASETS)
@pytest.mark.parametrize("bl", B6)
def test_resolved_config_equals_reference(generator_source, tmp_path, monkeypatch, bl, ds):
    for k in ("AGG_GOAL", "CONC", "TRAINER_HP", "AGG_HP"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("DATASET", ds)
    e = _run_generator(generator_source, tmp_path, bl, "", mode="sim")[0]
    ov = br.overlay(REF, bl, ds, e["trainer"]["num_trainers"])
    for path, v in _leaves(ov):
        assert _get(e, path) == v, f"{bl}.{ds}: {path} = {_get(e, path)!r}, reference {v!r}"


@pytest.mark.parametrize("ds,n", [("cifar10", 300), ("google_speech", 100)])
def test_sync_k_has_a_five_percent_floor_and_async_keeps_its_buffer(ds, n):
    for bl in ("oort", "oort_star", "refl", "feddance"):
        frac = br.entry(REF, bl)["participation"]["k_over_n"]
        assert br.agg_goal(REF, bl, n) == math.ceil(max(frac, 0.05) * n) >= 0.05 * n
    assert br.agg_goal(REF, "felix", n) is None and br.agg_goal(REF, "fedbuff", n) is None


def test_no_second_source_for_baseline_values():
    """datasets.yaml carries no per-baseline values any more (they were the old by_baseline layer)."""
    prof = yaml.safe_load(open(br.PATH.replace("baseline_reference.yaml", "datasets.yaml")))["datasets"]
    assert all("by_baseline" not in (p or {}) for p in prof.values())


def test_deviation_table_lists_every_model_mismatch():
    import importlib.util
    from pathlib import Path
    p = Path(br.PATH).parent.parent / "scripts" / "baseline_deviations.py"
    spec = importlib.util.spec_from_file_location("baseline_deviations", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rows = mod.rows(REF)
    assert ("refl", "cifar10", "model", "ResNet18 ([paper] Table 1)", "CifarNet (fl_data.py:56)", "", "") in rows
    assert not any(r[0] == "felix" for r in rows)  # ours: no deviation by definition
