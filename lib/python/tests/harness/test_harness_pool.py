# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""FX-N22 P4: harness_pool.py pure parts (delta map, slot sizing, packing, sharding) + the real bank."""

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "examples" / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


pool = _load("harness_pool")
bank = _load("harness_bank")
B6 = pool.B6


@pytest.mark.parametrize("paths,want,real", [
    (["lib/python/flame/mode/horizontal/asyncfl/top_aggregator.py"], ("felix", "fedbuff"), False),
    (["lib/python/flame/selector/refl_oort.py"], ("refl",), False),
    (["lib/python/flame/selector/oort.py"], ("oort", "oort_star", "refl"), False),
    (["lib/python/flame/channel.py"], B6, True),
    (["lib/python/flame/mode/horizontal/syncfl/trainer.py"], B6, True),
    (["lib/python/flame/mode/horizontal/syncfl/fwdllm_aggregator.py"], (), False),
    (["lib/python/examples/_metadata/FELIX_READINESS.md"], (), False),
    (["lib/python/flame/selector/feddance.py", "lib/python/flame/optimizer/fedbuff.py"],
     ("felix", "fedbuff", "feddance"), False),
])
def test_affected(paths, want, real):
    assert pool.affected(paths) == (want, real)


def test_slot_cpus_is_aggregator_share_plus_trainer_share():
    assert pool.slot_cpus(12) == 14          # 2 aggregator cores + 12
    assert pool.slot_cpus(15, 0.5) == 10
    assert pool.slot_cpus(60) == 64 and pool.slot_cpus(120, 0.25) == 38
    assert pool.slot_cpus(300, 0.4, 8) == 128  # a GPU leg at n=300 = the whole node


def _job(jid, est, cpus=10, deps=(), unit=None):
    return pool.Job(jid, "P", "syn_0", "felix", "sim", [], 10, 100, "stub", cpus, 1.0, 0, tuple(deps), est,
                    unit or jid)


def test_shard_keeps_units_together_and_balances():
    jobs = [_job("a_real", 300, unit="a"), _job("a_sim", 200, unit="a"), _job("a_grade", 60, deps=("a_real", "a_sim"), unit="a"),
            _job("b", 500), _job("c", 100)]
    s1, s2 = pool.shard(jobs, 1, 2), pool.shard(jobs, 2, 2)
    assert {j.jid for j in s1} | {j.jid for j in s2} == {j.jid for j in jobs}
    assert not {j.jid for j in s1} & {j.jid for j in s2}
    unit_a = {"a_real", "a_sim", "a_grade"}
    assert unit_a <= {j.jid for j in s1} or unit_a <= {j.jid for j in s2}


def test_coremap_whole_cores_one_numa_node_and_release():
    groups = {0: [[i, i + 8] for i in range(4)], 1: [[i, i + 8] for i in range(4, 8)]}
    cm = pool.CoreMap(groups, reserve_cores=0)
    a = cm.alloc(6)  # 3 physical cores on one node
    assert len(a) == 6 and len({c % 8 < 4 for c in a}) == 1
    b = cm.alloc(4)
    assert len({c % 8 < 4 for c in b}) == 1 and not set(a) & set(b)
    c = cm.alloc(6)  # no single node has 3 free cores any more -> spans
    assert len(c) == 6 and len({x % 8 < 4 for x in c}) == 2
    assert cm.alloc(2) is None
    cm.release(a, groups)
    assert cm.total() == 6


def test_makespan_packs_in_parallel_and_respects_deps():
    jobs = [_job("r", 100, cpus=50), _job("s", 60, cpus=50), _job("g", 10, cpus=2, deps=("r", "s"))]
    assert pool.simulate_makespan(jobs, cpus=120, max_parallel=8) == pytest.approx(110)
    assert pool.simulate_makespan(jobs, cpus=120, max_parallel=1) == pytest.approx(170)


def test_pair_phase_builds_parallel_legs_plus_grade():
    ph = pool.shaped("P1", ("felix",), "syn_0", "pair", "cifar10")
    jobs = pool.build_jobs([ph], 1.0, 8, {})
    by = {j.mode: j for j in jobs}
    assert set(by) == {"real", "sim", "grade"}
    assert by["grade"].deps == (by["real"].jid, by["sim"].jid)
    args = by["real"].args
    assert args[args.index("--mode") + 1] == "real"
    assert args[args.index("--agg-goal") + 1] == "3" and args[args.index("--concurrency") + 1] == "5"
    assert by["real"].est_s > by["sim"].est_s


def test_t2_sim_legs_share_the_bank_key_config_of_t3_real_legs():
    key = lambda p: (p.trace, p.runtime_s, p.n, p.trace_scale, p.agg_goal, p.c, p.dataset)
    for ds in pool.DATASETS:
        t2 = {key(p) for p in pool.tier_phases("T2", B6, ds) if p.kind == "sim"}
        t3 = {key(p) for p in pool.tier_phases("T3", B6, ds)}
        assert t2 == t3


def test_datasets_get_their_own_phase_ids_and_gpu_cohorts():
    c, g = pool.tier_phases("G1", B6, "cifar10")[0], pool.tier_phases("G1", B6, "google_speech")[0]
    assert (c.pid, c.n, c.agg_goal) == ("G1", 300, None) and (g.pid, g.n) == ("gs_G1", 100)
    assert {p.pid for p in pool.tier_phases("T4", B6, "google_speech")} >= {"gs_P1", "gs_P11a"}
    gs = pool.build_jobs([g], 1.0, 8, {})
    assert all(j.dataset == "google_speech" and "--dataset" in j.args for j in gs)
    assert next(j for j in gs if j.mode == "real").cpus == 48  # n=100 GPU leg leaves room for CPU slots


def test_g0_screen_scales_cohort_keeps_ratios_and_gpu_density():
    # FX-N34: all six, syn_0 + syn_50, 30 min, reference c/n and aggGoal/c, reference trainers per GPU.
    for ds, (n, k, c, g) in {"cifar10": (100, 3, 10, 3), "google_speech": (50, 5, 15, 4)}.items():
        phs = pool.tier_phases("G0", B6, ds)
        assert [p.trace for p in phs] == ["syn_0", "syn_50"]
        for p in phs:
            assert (p.n, p.agg_goal, p.c, p.gpus, p.harness, p.baselines, p.runtime_s) == (n, k, c, g, "none", B6, 1800)
        jobs = pool.build_jobs(phs, 1.0, 8, {})
        legs = [j for j in jobs if j.mode in ("real", "sim")]
        assert len(legs) == 24 and {j.gpus for j in legs} == {g}
        assert {j.cpus for j in legs} == {pool.slot_cpus(n, 0.4, 8)}


def test_gpu_allow_skips_gpus_busy_at_start_and_honours_explicit(monkeypatch):
    # A foreign job on a GPU (jayne 2026-09-27: GPUs 0,2-4 at 23 GB) keeps pool legs off it; ECC GPUs stay out.
    rows = [(0, 0, 23000.0), (1, 3, 0.0), (2, 0, 5.0), (3, 0, 0.0)]
    monkeypatch.setattr(pool, "_gpu_query", lambda: rows)
    pool.set_gpu_allow("")
    assert pool.gpu_ids() == [2, 3] and pool.gpu_ids(healthy_only=False) == [0, 1, 2, 3]
    pool.set_gpu_allow("0,3")
    assert pool.gpu_ids() == [0, 3]
    monkeypatch.setattr(pool, "GPU_ALLOW", None)


def test_bank_lookup_ok_stale_none(tmp_path):
    b = tmp_path / "bank.tsv"
    real = tmp_path / "run_x_real"
    real.mkdir()
    key = {k: "" for k in bank.KEY_FIELDS} | {"baseline": "felix", "trace": "syn_0", "dataset": "cifar10"}
    assert bank.lookup(key, b, hashes={"f": "1"}) == ("", "NONE")
    bank.register(str(real), key, b, hashes={"f": "1"})
    assert bank.lookup(key, b, hashes={"f": "1"}) == (str(real), "OK")
    assert bank.lookup(key, b, hashes={"f": "2"}) == (str(real), "STALE:f")
    assert bank.lookup(key | {"trace": "syn_50"}, b, hashes={"f": "1"}) == ("", "NONE")
