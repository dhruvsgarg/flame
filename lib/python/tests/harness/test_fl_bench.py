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
    end, evals, commits, nbytes, bud = bench_validate.leg_stats(str(tmp_path), horizon=1800)
    assert end == 1800 and commits == 3 and evals == [(1800.0, 0.3)] and nbytes == 1000 and bud == [12.0]
