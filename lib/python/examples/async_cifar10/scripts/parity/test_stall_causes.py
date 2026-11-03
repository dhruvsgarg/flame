# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""FX-N62: a sync round is a stall by cause (abandon, or a fresh gated update it took), not only by length."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from parity import checks as c  # noqa: E402


def _agg(rounds):
    return {"agg_rounds": [{"event": "agg_round", "round": r, "ts": float(t), "vclock_now": float(t),
                            "contributing_trainers": tr} for r, t, tr in rounds],
            "withheld_deliveries": [], "abandon_timeouts": []}


def _cfg(tmp_path, sort="fedavg"):
    (tmp_path / "aggregator_config.json").write_text(json.dumps({"optimizer": {"sort": sort}}))
    return str(tmp_path)


def test_fresh_gated_update_marks_its_round_stale_one_does_not(tmp_path):
    agg = _agg([(1, 3, ["a", "b"]), (2, 55, ["c", "d"]), (3, 58, ["e", "x"])])
    agg["withheld_deliveries"] = [{"end_id": "c", "round": 2, "staleness": 0, "accepted": True},
                                  {"end_id": "x", "round": 3, "staleness": 1, "accepted": True}]
    real_sends = {"t": {"task_send": [{"end_id": "a", "round": 1, "send_gate_wait_s": 0.2}]}}
    c._mark_stall_causes(agg, real_sends, _cfg(tmp_path))
    assert [e.get("stall_cause") for e in agg["agg_rounds"]] == [None, "gated", None]


def test_real_gated_send_and_abandon(tmp_path):
    agg = _agg([(1, 3, ["a"]), (2, 50, ["b"]), (3, 150, ["c"])])
    agg["abandon_timeouts"] = [{"round": 3}]
    sends = {"t": {"task_send": [{"end_id": "b", "round": 2, "send_gate_wait_s": 40.0}]}}
    c._mark_stall_causes(agg, sends, _cfg(tmp_path))
    assert [e.get("stall_cause") for e in agg["agg_rounds"]] == [None, "gated", "abandon"]


def test_async_rounds_stay_unstamped(tmp_path):
    agg = _agg([(1, 3, ["a"]), (2, 50, ["b"])])
    agg["abandon_timeouts"] = [{"round": 2}]
    c._mark_stall_causes(agg, {}, _cfg(tmp_path, "fedbuff"))
    assert all("stall_cause" not in e for e in agg["agg_rounds"])


def test_cause_round_is_an_episode_and_excess_uses_stall_free_median():
    adv = [c._RoundAdv(x, cause) for x, cause in ((3, None), (50, "gated"), (3, None), (4, None), (5, None))]
    assert c._stall_episodes(adv, 72.0) == [[1]]
    assert c._stall_excess_s(adv, 72.0) == 50 - 3.5


def test_few_stall_free_rounds_skip_only_when_stalls_dropped_rounds():
    r = {"ok": False}
    c._skip_if_few_free(r, [1.0] * 4, [1.0] * 5, n_rounds=12)
    assert r["status"] == "SKIP" and r["ok"] is True
    r = {"ok": False}
    c._skip_if_few_free(r, [1.0] * 6, [1.0] * 6, n_rounds=6)
    assert "status" not in r and r["ok"] is False


def test_tail_round_cut_by_one_stop_is_not_graded():
    # PR36 B1 fedbuff G0U mobiperf_3st: sim == real replicate to 0.1 s, but real's stop cut round 6 after 1 of 2 commits.
    t = [(1, 192.1), (1, 277.1), (2, 454.1), (2, 544.1), (3, 641.5), (3, 655.2), (4, 666.3), (4, 728.2), (5, 767.3),
         (5, 822.2), (6, 862.4)]
    real = {"agg_rounds": [{"event": "agg_round", "round": r, "ts": 1e9 + x} for r, x in t]}
    sim = _agg([(r, x, ["a"]) for r, x in t + [(6, 915.2)]])
    assert c._tail_cut(real, sim) == 1 and c._tail_cut(sim, sim) == 0
    for chk in (c.throughput_parity, c.overhead_residual, c.per_round_advance_parity):
        assert chk(real, sim)["matched_window_n"] == 4, chk.__name__
