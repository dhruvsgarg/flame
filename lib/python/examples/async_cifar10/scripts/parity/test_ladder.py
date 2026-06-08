"""Fault-injection tests for the parity causal ladder.

Each test builds a synthetic real/sim pair, injects a *single* mechanism fault,
and asserts the verdict names exactly that rung as ROOT-CAUSE with the rest
demoted to downstream.  This is the regression net that keeps the ladder
self-reinforcing (PARITY.md §3d).

Run:  python -m pytest scripts/parity/test_ladder.py -q
  or:  python scripts/parity/test_ladder.py
"""

from __future__ import annotations

import os
import sys

_SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from parity.checks import run_all_parity, overall_verdict  # noqa: E402

AGG_GOAL = 4
TRAINERS = ["0001", "0002", "0003", "0004"]
SPEEDS = [1.0, 2.0, 3.0, 4.0]
STALENESS = [0, 1, 0, 1]
UTILS = [10.0, 20.0, 30.0, 40.0]
PHASES = {
    "pre_train_s": 0.01, "weights_to_gpu_s": 0.5, "gpu_compute_s": 8.0,
    "mqtt_fetch_s": 1.0, "weights_to_ram_s": 0.3, "post_train_s": 0.01,
}


def _build_mode(n_rounds: int, advance: float, *, with_vclock: bool) -> tuple:
    """Build (agg, trainers) for one mode.

    real mode (with_vclock=False): progress carried by ts = (r-1)*advance.
    sim  mode (with_vclock=True):  progress carried by vclock_now=(r-1)*advance;
                                   ts is a separate fast wall clock.
    """
    agg_rounds, selection_train, agg_evals = [], [], []
    for r in range(1, n_rounds + 1):
        prog = (r - 1) * advance
        e = {
            "event": "agg_round", "round": r, "agg_goal_count": AGG_GOAL,
            "trainer_speed_s": list(SPEEDS),
            "contributing_trainers": list(TRAINERS),
            "staleness": list(STALENESS),
            "stat_utility": list(UTILS),
        }
        if with_vclock:
            e["vclock_now"] = prog
            e["ts"] = (r - 1) * 1.0  # fast independent wall clock
        else:
            e["ts"] = prog
        agg_rounds.append(e)
        selection_train.append({
            "event": "selection", "task": "train", "round": r,
            "ts": e["ts"], "chosen": list(TRAINERS),
            "avail_composition": {"TRAIN": 10}, "num_eligible": 10,
            "num_candidates": 10, "num_chosen": AGG_GOAL, "in_flight": AGG_GOAL,
            "effective_c": AGG_GOAL, "selector": "OortSelector",
        })
    for r in range(1, n_rounds + 1, max(1, n_rounds // 5)):
        agg_evals.append({"event": "agg_eval", "round": r,
                          "test-accuracy": 0.5 + r * 0.001,
                          "test-loss": 2.0 - r * 0.001})

    trainers: dict = {}
    for tid in TRAINERS:
        task_recv, trainer_round = [], []
        for r in range(1, n_rounds + 1):
            tr = {"event": "trainer_round", "round": r,
                  "real_gpu_time_s": PHASES["gpu_compute_s"],
                  "training_budget_s": PHASES["gpu_compute_s"] + 1.0}
            tr.update(PHASES)
            trainer_round.append(tr)
            task_recv.append({
                "event": "task_recv", "round": r,
                "sim_send_ts": ((r - 1) * advance if with_vclock else None),
            })
        trainers[tid] = {"task_recv": task_recv, "trainer_round": trainer_round}

    agg = {"agg_rounds": agg_rounds, "selection_train": selection_train,
           "agg_evals": agg_evals}
    return agg, trainers


def _verdict(real_agg, sim_agg, real_tr, sim_tr):
    results = run_all_parity(real_agg, sim_agg, real_tr, sim_tr,
                             agg_goal=AGG_GOAL)
    return results, overall_verdict(results)


# ───────────────────────────── tests ──────────────────────────────────────

def test_clean_pair_passes():
    """Identical content + matched time-base ⇒ no enforced failures."""
    real_agg, real_tr = _build_mode(20, advance=10.0, with_vclock=False)
    sim_agg, sim_tr = _build_mode(20, advance=10.0, with_vclock=True)
    results, (passed, roots, downstream, warnings) = _verdict(
        real_agg, sim_agg, real_tr, sim_tr)
    assert passed, (
        f"clean pair should pass; roots={roots} downstream={downstream}")
    assert not roots and not downstream


def test_overhead_residual_is_root():
    """Sim under-charges the clock (no per-commit overhead) ⇒ Stage-1
    overhead_residual is the root; throughput/per_round_advance demote to
    downstream; the speed control (P3) still passes."""
    real_agg, real_tr = _build_mode(20, advance=10.0, with_vclock=False)
    sim_agg, sim_tr = _build_mode(20, advance=3.0, with_vclock=True)
    results, (passed, roots, downstream, warnings) = _verdict(
        real_agg, sim_agg, real_tr, sim_tr)
    assert not passed
    assert "overhead_residual" in roots, f"roots={roots}"
    assert results["trainer_speed"]["ok"], "P3 control must stay green"
    # emergent clock checks are consequences, not independent roots
    assert "per_round_advance" in downstream
    assert "throughput" in downstream
    assert "per_round_advance" not in roots and "throughput" not in roots


def test_missing_vclock_is_stage0_root():
    """Sim path never stamps vclock_now ⇒ Stage-0 coverage is the root and the
    Stage-1 clock checks SKIP (not FAIL)."""
    real_agg, real_tr = _build_mode(20, advance=10.0, with_vclock=False)
    sim_agg, sim_tr = _build_mode(20, advance=10.0, with_vclock=False)  # no vclock
    results, (passed, roots, downstream, warnings) = _verdict(
        real_agg, sim_agg, real_tr, sim_tr)
    assert not passed
    assert roots and roots[0] in ("field_coverage", "vclock_telemetry"), \
        f"expected stage-0 root, got {roots}"
    # downstream clock checks must SKIP, never silently FAIL
    assert results["overhead_residual"].get("status") == "SKIP"
    assert results["throughput"].get("note", "").startswith("K10:") or \
        results["throughput"].get("status") == "SKIP"


def test_phase_split_localizes_single_phase():
    """A divergence confined to mqtt_fetch flags only that phase, not the rest."""
    real_agg, real_tr = _build_mode(20, advance=10.0, with_vclock=False)
    sim_agg, sim_tr = _build_mode(20, advance=10.0, with_vclock=True)
    # blow up only the mqtt_fetch phase in sim
    for d in sim_tr.values():
        for e in d["trainer_round"]:
            e["mqtt_fetch_s"] = 50.0
    results, _ = _verdict(real_agg, sim_agg, real_tr, sim_tr)
    assert not results["phase_mqtt_fetch"]["ok"], "mqtt phase should fail"
    for other in ("phase_pre_train", "phase_gpu_compute", "phase_weights_to_gpu"):
        assert results[other]["ok"], f"{other} should stay green"


def test_avail_timebase_detects_trajectory_shift():
    """A systematic eligible-count shift in sim trips A3."""
    real_agg, real_tr = _build_mode(20, advance=10.0, with_vclock=False)
    sim_agg, sim_tr = _build_mode(20, advance=10.0, with_vclock=True)
    for i, e in enumerate(sim_agg["selection_train"]):
        e["num_eligible"] = 10 if i < 10 else 2  # second half collapses
    results, _ = _verdict(real_agg, sim_agg, real_tr, sim_tr)
    assert not results["avail_timebase"]["ok"], "A3 should catch the shift"


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL  {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
