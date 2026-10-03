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


def test_t3s_over_selecting_sync_pair_has_stragglers():
    # FX-N67: sync oort-family selects 1.3 x aggGoal, so aggGoal 10 leaves ~3 stragglers per round to carry over.
    phs = pool.tier_phases("T3S", B6, "cifar10")
    assert [p.trace for p in phs] == ["syn_0", "syn_20"]
    for ph in phs:
        assert (ph.baselines, ph.agg_goal, ph.kind) == (("oort", "oort_star", "refl"), 10, "pair")
        assert int(ph.agg_goal * 1.3) > ph.agg_goal and ph.n >= 2 * int(ph.agg_goal * 1.3)


def test_g2_replicate_tiers_run_one_side_at_the_g2_config():
    # FX-N68: sim-only / real-only copies of a G2 cell, same n and length, for the real<->real / sim<->sim floor.
    for tier, kind in (("G2S", "sim_ev"), ("G2C", "real")):
        (ph,) = pool.tier_phases(tier, ("feddance",), "cifar10")
        g2 = pool.tier_phases("G2", ("feddance",), "cifar10")[0]
        assert (ph.kind, ph.baselines, ph.n, ph.runtime_s) == (kind, ("feddance",), g2.n, g2.runtime_s)
        assert ph.pid == tier


def test_datasets_get_their_own_phase_ids_and_gpu_cohorts():
    c, g = pool.tier_phases("G1", B6, "cifar10")[0], pool.tier_phases("G1", B6, "google_speech")[0]
    assert (c.pid, c.n, c.agg_goal) == ("G1", 300, None) and (g.pid, g.n) == ("gs_G1", 100)
    assert {p.pid for p in pool.tier_phases("T4", B6, "google_speech")} >= {"gs_P1", "gs_P11a"}
    gs = pool.build_jobs([g], 1.0, 8, {})
    assert all(j.dataset == "google_speech" and "--dataset" in j.args for j in gs)
    assert next(j for j in gs if j.mode == "real").cpus == 48  # n=100 GPU leg leaves room for CPU slots


def test_g0_screen_scales_cohort_keeps_ratios_and_gpu_density():
    # FX-N34: all six, syn_0 + syn_20, 30 min, reference c/n and aggGoal/c, reference trainers per GPU.
    for ds, (n, k, c, g) in {"cifar10": (100, 3, 10, 3), "google_speech": (50, 5, 15, 4)}.items():
        phs = pool.tier_phases("G0", B6, ds)
        assert [p.trace for p in phs] == ["syn_0", "syn_20"]
        for p in phs:
            assert (p.n, p.agg_goal, p.c, p.gpus, p.harness, p.baselines, p.runtime_s) == (n, k, c, g, "none", B6, 1800)
        jobs = pool.build_jobs(phs, 1.0, 8, {})
        legs = [j for j in jobs if j.mode in ("real", "sim")]
        assert len(legs) == 24 and {j.gpus for j in legs} == {g}
        assert {j.cpus for j in legs} == {pool.slot_cpus(n, 0.4, 8)}


def test_gpu_allow_and_foreign_busy_gpus(monkeypatch):
    # A foreign job on a GPU (jayne 2026-09-27: GPUs 0,2-4 at 23 GB) is skipped per leg start; ECC GPUs never used.
    rows = [(0, 0, 23000.0), (1, 3, 0.0), (2, 0, 5.0), (3, 0, 0.0)]
    monkeypatch.setattr(pool, "_gpu_query", lambda: rows)
    pool.set_gpu_allow("")
    assert pool.gpu_ids() == [0, 2, 3] and pool.busy_gpus() == {0}
    pool.set_gpu_allow("0,3")
    assert pool.gpu_ids() == [0, 3]
    pool.set_gpu_allow("")


def test_g0c_is_one_real_replicate_per_g0_syn_0_cell():
    for ds in ("cifar10", "google_speech"):
        (c,) = pool.tier_phases("G0C", B6, ds)
        g0 = pool.tier_phases("G0", B6, ds)[0]
        assert (c.kind, c.trace, c.n, c.agg_goal, c.c, c.gpus, c.runtime_s) == \
            ("real", "syn_0", g0.n, g0.agg_goal, g0.c, g0.gpus, g0.runtime_s)
        jobs = pool.build_jobs([c], 1.0, 8, {})
        assert [j.mode for j in jobs] == ["real"] * 6


def test_coremap_skips_cores_another_process_keeps_busy():
    cm = pool.CoreMap({0: [[0, 64], [1, 65], [2, 66], [3, 67]]}, reserve_cores=0)
    assert cm.alloc(4, frozenset({65})) == [0, 2, 64, 66]  # core 1 (CPUs 1, 65) skipped
    assert cm.alloc(4, frozenset({65})) is None and cm.alloc(4) == [1, 3, 65, 67]


@pytest.mark.parametrize("tier", ["T4", "G0"])
def test_unavailability_phases_run_long_enough_for_their_trace(tier):
    # FX-N35: a syn_X leg must see >= 80% of X% unavailable over its own run (ramped traces need long runs).
    from flame.availability.trace import effective_unavailability
    seen = set()
    for ds in ("cifar10", "google_speech"):
        for p in pool.tier_phases(tier, B6, ds):
            if not p.trace.startswith("syn_") or p.trace == "syn_0":
                continue
            key = (p.trace, p.runtime_s, p.trace_scale, p.n)
            if key in seen:
                continue
            seen.add(key)
            got = effective_unavailability(p.trace, p.runtime_s, float(p.trace_scale or 1), p.n)
            assert got >= 0.8 * int(p.trace[4:]) / 100, (p.pid, key, round(got, 3))
    assert seen


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


def test_order_is_unit_major():
    # Run 4 L6: longest-first ran all 18 reals before any sim, so the deadline would skip every sim.
    jobs = [_job("a_real", 30, unit="a"), _job("b_real", 30, unit="b"), _job("a_sim", 10, unit="a"),
            _job("b_sim", 5, unit="b"), _job("a_grade", 1, deps=("a_real", "a_sim"), unit="a")]
    assert [j.jid for j in sorted(jobs, key=pool._order_key(jobs))] == ["a_real", "a_sim", "a_grade", "b_real", "b_sim"]


def test_progress_line_shows_eta_and_deadline_cut(capsys, monkeypatch):
    monkeypatch.setattr(pool, "gpu_ids", lambda *a, **k: [])
    jobs = [_job("a_real", 600, unit="a"), _job("b_real", 600, unit="b")]
    p = pool.Pool(Path("."), jobs, 1, 0, 0, 900, False, label="L6 (rung 2/2)")
    p.t0, p.done = pool.time.time(), {}
    p.progress(jobs, {}, 100)
    out = capsys.readouterr().out
    assert "PROGRESS L6 (rung 2/2) [" in out and "0/2 done" in out and "left ~20m" in out
    assert "later starts SKIPPED" in out


def test_leases_keep_two_pools_off_one_slot(tmp_path):
    # Run 4: two ladders on jayne took port 18830 and cores 9,17-24 two seconds apart.
    a, b = pool.Leases(tmp_path), pool.Leases(tmp_path)
    assert a.take_all(pool.slot_leases([9, 17], [4]))
    assert b.foreign(pool.slot_leases([9, 17, 18], [4, 5])) == {"cpu9", "cpu17", "gpu4"}
    assert not b.take_all(pool.slot_leases([17, 18], [])) and "cpu18" not in b.held  # all or nothing
    pa = pool.free_port(18830, set(), a)
    assert pool.free_port(18830, set(), b) != pa
    a.drop(pool.slot_leases([9, 17], [4], pa))
    assert b.take_all(pool.slot_leases([9, 17], [4]))


def test_oort_mobiperf_legs_run_long_enough_to_commit():
    # Run 4: unaware oort committed 0 updates in 240s on mobiperf_3st (both datasets); nothing graded.
    ph = pool.shaped("P3", ("oort", "felix"), "mobiperf_3st", "pair", "cifar10")
    rt = {(j.baseline, j.mode): j.runtime_s for j in pool.build_jobs([ph], 1.0, 1, {})}
    assert rt[("oort", "real")] == rt[("oort", "sim")] == 960 and rt[("felix", "real")] == 240
    ph.runtime_s = 60  # --smoke keeps its 60s
    assert {j.runtime_s for j in pool.build_jobs([ph], 1.0, 1, {})} == {60}
