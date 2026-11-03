"""FX-D186 dispatch.py planner: leg-level list scheduling, balanced finish across nodes, late binding away from a slow node."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples" / "scripts"))
import dispatch as dp  # noqa: E402


def _leg(jid, unit, est, cpus=10, mem=80.0, gpus=1, deps=()):
    return {"jid": jid, "unit": unit, "est_s": est, "cpus": cpus, "mem_gb": mem, "gpus": gpus, "deps": list(deps)}


def _reals(n_units, est, mem=80.0):
    return {"jobs": [_leg(f"G3C_b{i}_real", f"G3C_b{i}", est, mem=mem) for i in range(n_units)]}


def test_unit_span_is_parallel_legs_then_grade():
    d = {"jobs": [_leg("P_x_real", "P_x", 100), _leg("P_x_sim", "P_x", 40),
                  _leg("P_x_grade", "P_x", 30, 2, 1, 0, deps=("P_x_real", "P_x_sim"))]}
    (u,) = dp.units_from_jobs("S", 0, {"a": d}, ["a"])
    assert u.span("a") == 130.0 and len(u.legs) == 3


def test_short_sim_frees_its_ram_for_the_next_unit():
    # two pairs (real 2 h 90 GB + sim 10 min 90 GB) on 200 GB: the second real starts when the first sim ends, not after 2 h
    d = {"jobs": [j for i in range(2) for j in (_leg(f"P{i}_real", f"P{i}", 7200, mem=90), _leg(f"P{i}_sim", f"P{i}", 600, mem=90))]}
    units = dp.units_from_jobs("S", 0, {"a": d}, ["a"])
    span, starts = dp.simulate([it for u in units for it in dp.unit_items(u, "a")], dp.NodeCap(120, 7, 200))
    assert starts["S:P1_real"] == 600 and span == 7800


def test_plan_balances_unequal_nodes():
    # 12 two-hour reals; node a has 7 GPUs, node b 4: two waves on each, never three on b
    units = dp.units_from_jobs("R", 0, {"a": _reals(12, 7200), "b": _reals(12, 7200)}, ["a", "b"])
    caps = {"a": dp.NodeCap(120, 7, 450), "b": dp.NodeCap(120, 4, 450)}
    assign, _, spans = dp.plan(units, caps, {})
    assert sum(n == "b" for n in assign.values()) >= 4 and max(spans.values()) == 2 * 7200


def test_ram_binds_before_gpus():
    units = dp.units_from_jobs("R", 0, {"a": _reals(6, 3600, mem=150)}, ["a"])
    span, _ = dp.simulate([it for u in units for it in dp.unit_items(u, "a")], dp.NodeCap(120, 7, 450))
    assert span == 2 * 3600  # 3 x 150 GB at once


def test_slow_node_receives_fewer_remaining_units():
    units = dp.units_from_jobs("R", 0, {"a": _reals(4, 3600), "b": _reals(4, 3600)}, ["a", "b"])
    caps = {"a": dp.NodeCap(120, 1, 450), "b": dp.NodeCap(120, 1, 450)}
    committed = {"a": [("R:old_real", 3 * 3600, 10, 80, 1, 0.0, [])], "b": []}  # a's running leg is 3 h over its estimate
    assign, _, _ = dp.plan(units, caps, committed)
    assert sum(n == "b" for n in assign.values()) == 3


def test_finished_units_leave_the_status_count():
    # PR38 printed "unfinished/unreleased 40" after every unit had finished.
    (u,) = dp.units_from_jobs("R", 0, {"a": _reals(1, 60)}, ["a"])
    assert not dp.unit_done(u, {})
    u.node = "a"
    assert not dp.unit_done(u, {("a", "R"): {"done": {}}})
    assert dp.unit_done(u, {("a", "R"): {"done": {"G3C_b0_real": "x"}}})
