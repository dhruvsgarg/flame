"""FX-D186 dispatch.py planner: balanced finish across nodes, late binding away from a slow node, multi-resource fit."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples" / "scripts"))
import dispatch as dp  # noqa: E402


def _jobs(n_units, est, gpus=1, mem=80.0):
    js = []
    for i in range(n_units):
        u = f"G3C_syn_0_b{i}"
        js.append({"jid": f"{u}_real", "unit": u, "mode": "real", "est_s": est, "cpus": 10, "mem_gb": mem, "gpus": gpus, "deps": []})
    return {"jobs": js}


def test_units_group_pair_and_grade():
    d = {"jobs": [{"jid": "P_x_real", "unit": "P_x", "est_s": 100, "cpus": 4, "mem_gb": 10, "gpus": 1, "deps": []},
                  {"jid": "P_x_sim", "unit": "P_x", "est_s": 40, "cpus": 8, "mem_gb": 20, "gpus": 1, "deps": []},
                  {"jid": "P_x_grade", "unit": "P_x", "est_s": 30, "cpus": 2, "mem_gb": 1, "gpus": 0, "deps": ["P_x_real", "P_x_sim"]}]}
    (u,) = dp.units_from_jobs("S", 0, {"a": d}, ["a"])
    assert (u.est["a"], u.cpus, u.mem_gb, u.gpus) == (130.0, 12, 30, 2)  # parallel legs + grade after


def test_plan_balances_unequal_nodes():
    # 12 two-hour units; node a has 7 GPUs, node b 4: a static half split would leave b running 3 waves vs a 1
    units = dp.units_from_jobs("R", 0, {"a": _jobs(12, 7200), "b": _jobs(12, 7200)}, ["a", "b"])
    caps = {"a": dp.NodeCap(120, 7, 450), "b": dp.NodeCap(120, 4, 450)}
    assign, _, spans = dp.plan(units, caps, {})
    assert sum(n == "b" for n in assign.values()) >= 4
    assert max(spans.values()) == 2 * 7200  # two waves on both, not three on b


def test_ram_binds_before_gpus():
    units = dp.units_from_jobs("R", 0, {"a": _jobs(6, 3600, mem=150)}, ["a"])
    span, _ = dp.simulate([(u.uid, 3600, u.cpus, u.mem_gb, u.gpus, None) for u in units], dp.NodeCap(120, 7, 450))
    assert span == 2 * 3600  # 3 x 150 GB fit at once, so two waves


def test_slow_node_receives_fewer_remaining_units():
    units = dp.units_from_jobs("R", 0, {"a": _jobs(4, 3600), "b": _jobs(4, 3600)}, ["a", "b"])
    caps = {"a": dp.NodeCap(120, 1, 450), "b": dp.NodeCap(120, 1, 450)}
    # a's running unit is 3 h over its estimate: its remaining time grows, so every free unit goes to b first
    committed = {"a": [("R:old", 3 * 3600, 10, 80, 1, 0.0)], "b": []}
    assign, _, _ = dp.plan(units, caps, committed)
    assert sum(n == "b" for n in assign.values()) == 3
