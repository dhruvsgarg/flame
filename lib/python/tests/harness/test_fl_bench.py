"""FX-N129 (V5): the bench's in-memory channel, pool discovery and B2 leg stats (CPU only)."""
import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples" / "scripts"))
import bench_pool  # noqa: E402
import bench_validate  # noqa: E402
import fl_bench  # noqa: E402


class _Sel:
    def select(self, **kw):
        return {e: None for e in sorted(kw["ends"])[:2]}


def _channel():
    sent = []
    router = type("R", (), {"deliver": lambda self, e, p: sent.append((e, p))})()
    ch = fl_bench.make_agg_channel(router, _Sel(), "job", "param-channel", "aggregator", "trainer", "default")
    for e in ("a", "b", "c"):
        ch.add_end(e)
    return ch, sent


def _payload(v):
    from flame.channel import encode_message
    return encode_message({"v": v})


def test_send_delivers_and_replies_drain_in_arrival_order():
    ch, sent = _channel()
    assert ch.ends("send") == ["a", "b"]
    ch.send_payload("a", b"x")
    assert sent == [("a", b"x")]
    ch.put_reply("b", _payload(1))
    ch.put_reply("a", _payload(2))
    got = ch.drain_ready(["a", "b", "c"])
    assert [md[0] for _, md in got] == ["a", "b"] and [m["v"] for m, _ in got] == [2, 1]
    assert ch.ends_with_pending_rx() == set() and ch._ends["a"].is_rxq_empty()


def test_recv_fifo_waits_for_a_late_reply_then_times_out():
    ch, _ = _channel()
    threading.Timer(0.1, lambda: ch.put_reply("c", _payload(3))).start()
    t0 = time.time()
    out = list(ch.recv_fifo(["b", "c"], first_k=2, timeout=0.5))
    assert out[0][0]["v"] == 3 and out[-1][0] is None and time.time() - t0 < 2.0


def test_jobs_from_pools_collapses_seed_replicates(tmp_path):
    for leg in ("F42_s0u_ctl_syn_0_felix_sim", "F42_s0u_ctl_r2_syn_0_felix_sim", "gs_F42_s50u_l5p_syn_50_3st_felix_sim"):
        d = tmp_path / "P" / leg.rsplit("_syn", 1)[0] / leg / "runs" / "x"
        d.mkdir(parents=True)
        (d / "debug_run.yaml").write_text("experiments: []\n")
    jobs = bench_pool.jobs_from_pools([str(tmp_path / "P")])
    assert [j[:3] for j in jobs] == [("F42_s0u_ctl", "felix", "syn_0"), ("gs_F42_s50u_l5p", "felix", "syn_50_3st")]


def test_pick_places_by_free_memory_and_counts_ramping_legs():
    from types import SimpleNamespace
    a = SimpleNamespace(per_gpu=3, leg_gpu_gb=4.0, leg_ram_gb=8.0, gpu_reserve_gb=2.0, ramp_s=240)
    pick = bench_pool.pick
    assert pick(["0", "7"], {}, {"0": 20.0, "7": 7.0}, 100.0, a, 0) == "0"  # most free
    assert pick(["7"], {}, {"7": 5.0}, 100.0, a, 0) is None  # 5 − 2 reserve < 4
    young = {i: (None,) * 6 + (0.0, "0") for i in range(4)}  # 4 legs ramping on GPU 0 hold 16 GB not yet visible
    assert pick(["0", "7"], young, {"0": 20.0, "7": 7.0}, 100.0, a, 10) == "7"
    assert pick(["0"], {}, {"0": 20.0}, 7.0, a, 0) is None  # host RAM
    full = {i: (None,) * 6 + (-999.0, "0") for i in range(3)}
    assert pick(["0"], full, {"0": 40.0}, 100.0, a, 0) is None  # per-GPU cap


def test_leg_stats_matched_horizon(tmp_path):
    tel = tmp_path / "telemetry"
    tel.mkdir()
    ev = [dict(event="agg_round", round=r, vclock_now=600.0 * r, task_to_perform="train", staleness=[0]) for r in range(1, 5)]
    ev[3:3] = [dict(event="agg_eval", round=3, **{"test-accuracy": 0.3}), dict(event="wire", bytes=1000)]
    (tel / "aggregator_x.jsonl").write_text("".join(json.dumps(e) + "\n" for e in ev))
    (tel / "trainer_a.jsonl").write_text(json.dumps(dict(event="trainer_round", task_to_perform="train", sim_send_ts=10,
                                                         training_budget_s=12.0)) + "\n")
    end, evals, commits, nbytes, bud, fwd = bench_validate.leg_stats(str(tmp_path), horizon=1800)
    assert end == 1800 and commits == 3 and evals == [(1800.0, 0.3)] and nbytes == 1000 and bud == [12.0] and fwd == {}


def test_leg_stats_fwdllm_cadence(tmp_path):
    """FT-N18: iterations per data_id at commit, variance pass rate, commits / h."""
    tel = tmp_path / "telemetry"
    tel.mkdir()
    ok = [False, False, True, False, True, False]
    ev = [dict(event="agg_round", vclock_now=600.0 * (r + 1), staleness=[0], var=0.5, var_good_enough=g,
               commit_reason="natural" if g else None, cycle_iteration=r) for r, g in enumerate(ok)]
    (tel / "aggregator_x.jsonl").write_text("".join(json.dumps(e) + "\n" for e in ev))
    fwd = bench_validate.leg_stats(str(tmp_path))[5]
    assert fwd == dict(iters=3.0, var_pass=2 / 6, commits_h=2.0)


def test_fwdllm_stack_detection():
    assert fl_bench._is_fwdllm("aggregator/main_fedfwd_agg.py")  # baselines.yaml's relative aggregator_main
    assert fl_bench._is_fwdllm(fl_bench.FWD / "trainer" / "main.py")
    assert not fl_bench._is_fwdllm(fl_bench.EX / "trainer" / "pytorch" / "main.py")


def test_fwdllm_shared_model_swap_restores_trainable_and_rng():
    """FT-N18: trainers sharing one model each get their own trainable slice and global RNG back; frozen weights untouched."""
    import random
    import torch
    host = fl_bench.TrainerHost.__new__(fl_bench.TrainerHost)
    host._model = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.Linear(3, 2))
    host._model[0].requires_grad_(False)
    frozen = host._model[0].weight.clone()
    torch.manual_seed(0)
    random.seed(0)
    a = host._snapshot()
    a_draw = (torch.rand(1).item(), random.random())
    with torch.no_grad():
        host._model[1].weight.add_(1.0)  # trainer b's task moves the trainable slice and the RNG streams
    torch.rand(5)
    host._restore(a)
    assert torch.equal(host._model[1].weight, a[0][0]) and torch.equal(host._model[0].weight, frozen)
    assert (torch.rand(1).item(), random.random()) == a_draw


def _leg_yaml(tmp_path, name, baseline, dataset):
    y = tmp_path / f"{name}.yaml"
    y.write_text(f"experiments:\n- baseline: {baseline}\n  trainer:\n    dataset:\n      name: {dataset}\n")
    return str(y)


def test_parse_jobs_reads_resource_columns():
    """FT-N19: optional key=value columns after the yaml; an unknown key fails loudly."""
    import pytest
    jobs = bench_pool.parse_jobs("# c\nP\tfwdllm\tsyn_0\ta.yaml\tgpu_gb=6\tworkers=4\nQ\tfelix\tsyn_0\tb.yaml\n")
    assert jobs == [("P", "fwdllm", "syn_0", "a.yaml", {"gpu_gb": 6.0, "workers": 4}), ("Q", "felix", "syn_0", "b.yaml", {})]
    with pytest.raises(ValueError):
        bench_pool.parse_jobs("P\tfwdllm\tsyn_0\ta.yaml\tgpus=2\n")


def test_resources_column_over_profile_over_flags_and_project(tmp_path):
    from types import SimpleNamespace
    a = SimpleNamespace(leg_gpu_gb=4.0, leg_ram_gb=8.0, workers=2, runtime_s=None)
    prof = {"agnews": {"gpu_gb": 3.6, "ram_gb": 12.0, "workers": 4}}
    fwd = _leg_yaml(tmp_path, "f", "fwdllm", "agnews")
    r = bench_pool.resources(("P", "fwdllm", "syn_0", fwd, {"ram_gb": 20.0}), prof, a)
    assert r == dict(gpu_gb=3.6, ram_gb=20.0, workers=4, runtime_s=None, project="fluxtune")
    r = bench_pool.resources(("P", "felix", "syn_0", _leg_yaml(tmp_path, "c", "felix", "cifar10"), {}), prof, a)
    assert r == dict(gpu_gb=4.0, ram_gb=8.0, workers=2, runtime_s=None, project="felix")


def test_pick_sizes_each_leg_by_its_own_need():
    from types import SimpleNamespace
    a = SimpleNamespace(per_gpu=4, leg_gpu_gb=4.0, leg_ram_gb=8.0, gpu_reserve_gb=2.0, ramp_s=240)
    assert bench_pool.pick(["0"], {}, {"0": 7.0}, 100.0, a, 0, (6.0, 8.0)) is None  # 7 - 2 reserve < 6
    assert bench_pool.pick(["0"], {}, {"0": 7.0}, 100.0, a, 0, (4.0, 8.0)) == "0"
    young = {0: (None,) * 6 + (0.0, "0", 10.0, 30.0, None)}  # a ramping 10 GB / 30 GB leg counts at its own size
    assert bench_pool.pick(["0"], young, {"0": 15.0}, 100.0, a, 10, (4.0, 8.0)) is None
    assert bench_pool.pick(["0"], young, {"0": 17.0}, 100.0, a, 10, (4.0, 8.0)) == "0"
    assert bench_pool.pick(["0"], young, {"0": 40.0}, 35.0, a, 10, (4.0, 8.0)) is None  # host RAM: 35 - 30 < 8


def test_dead_worker_fails_the_leg(monkeypatch):
    """A worker that died (e.g. CUDA OOM) exits the leg instead of stranding its trainers until the wall timeout."""
    r = fl_bench.PoolRouter.__new__(fl_bench.PoolRouter)
    r._closing, r.procs = False, [type("P", (), {"exitcode": None})(), type("P", (), {"exitcode": 1})()]

    def _exit(rc):
        raise SystemExit(rc)
    monkeypatch.setattr(fl_bench.os, "_exit", _exit)
    try:
        r._watch()
    except SystemExit as e:
        assert e.code == 70
    else:
        raise AssertionError("watchdog did not fail the leg")
