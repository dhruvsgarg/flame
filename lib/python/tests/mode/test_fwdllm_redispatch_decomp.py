# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""New telemetry for the fedbuff_round/felix_round throughput investigation
(simulate_fwdllm.md §B, 2026-07-27): fwdllm never emitted the shared
asyncfl/top_aggregator.py's `[LAG_DECOMP]` per-message round-trip breakdown
(§D-3 -- it's a separate subclass that didn't inherit that logging), and had
no way to tell whether a committed trainer's wait for its next dispatch was
genuine peer-wait (waiting on its round/cohort-mates, already modeled via the
vclock's max()) or unmodeled server-side redispatch overhead.

Covers two additions to `TopAggregator`:
  - `_process_single_trainer_message` now logs `[LAG_DECOMP]` (same field
    order as `_LAG_DECOMP_RE` in scripts/analysis/analyze_run.py, so the
    existing parser picks it up) and stamps `_last_commit_wall_ts[end]`.
  - `_distribute_weights_async` now emits a `redispatch_decomp` telemetry
    event per redispatch -- both a genuine WEIGHTS send and a VAR=bad
    keep-training ping (widened 07-28: VAR=bad retries are the majority of
    cycles and were previously uninstrumented, simulate_fwdllm.md §D-11) --
    splitting the commit->dispatch wall gap into `peer_wait_wall_s` (bounded
    by the round's own close) and `post_close_overhead_wall_s` (the residual
    -- what would actually calibrate `sim_redispatch_gap_s`, if non-trivial).
    `payload_kind` distinguishes the two.
"""

import time
import types

from flame import telemetry
from flame.channel import VAL_CH_STATE_RECV
from flame.mode.horizontal.syncfl.fwdllm_aggregator import (
    TopAggregator,
    _OrderedContributorList,
    _PendingCommitUnion,
)
from flame.mode.message import MessageType
from flame.selector.properties import PROP_ROUND_START_TIME


# ---------------------------------------------------------------------------
# [LAG_DECOMP] + _last_commit_wall_ts (_process_single_trainer_message)
# ---------------------------------------------------------------------------

class _FakeSelector:
    def __init__(self):
        self.ordered_updates_recv_ends = []


class _FakeChannel:
    def __init__(self, round_start_time=None):
        self._selector = _FakeSelector()
        self._props = {}
        if round_start_time is not None:
            self._props[("t1", PROP_ROUND_START_TIME)] = (1, round_start_time)

    def set_end_property(self, end, key, value):
        self._props[(end, key)] = value

    def get_end_property(self, end, key):
        return self._props.get((end, key))

    def cleanup_recvd_end(self, end):
        pass

    def cleanup_provided_ends(self, end):
        pass


class _FakeAggregator:
    process = TopAggregator._process_single_trainer_message
    _round_cache_clock_now = TopAggregator._round_cache_clock_now
    _release_end_on_return = TopAggregator._release_end_on_return

    def __init__(self, simulated: bool):
        self.simulated = simulated
        self.is_async = True
        self._per_agg_trainer_list = _OrderedContributorList()
        self._trainer_inflight_dispatch_version = {}
        self._real_pending_commit = _PendingCommitUnion(
            self._trainer_inflight_dispatch_version, self._per_agg_trainer_list
        )
        self._round = 1
        self.data_id = 0
        self.iteration_per_data_id = 0
        self._model_version = 0
        self._agg_goal_cnt = 0
        self._round_cache_activity_ts = {}
        self._updates_in_queue = 0
        self._trainer_state_dict = {}
        self._updates_received = {}
        self.grad_pool = []
        self._trainer_last_model_version = {}
        self._inflight_residence = False
        self._sim_pending_commit = set()
        self._sim_contrib_intervals = {}
        self._last_commit_wall_ts = {}


def _grad_msg(wall_send_ts=None, wall_recv_ts=None):
    msg = {
        MessageType.MODEL_VERSION: 0,
        MessageType.GRADIENTS: {},
        MessageType.GRADIENTS_FOR_VAR_CHECK: None,
        MessageType.STAT_UTILITY: 1.0,
    }
    if wall_send_ts is not None:
        msg[MessageType.WALL_SEND_TS] = wall_send_ts
    if wall_recv_ts is not None:
        msg[MessageType.WALL_RECV_TS] = wall_recv_ts
    return msg


class TestLastCommitWallTsStamped:
    def test_commit_stamps_last_commit_wall_ts(self):
        agg = _FakeAggregator(simulated=False)
        channel = _FakeChannel()

        before = time.time()
        agg.process(channel, _grad_msg(), "t1", timestamp=0)
        after = time.time()

        assert "t1" in agg._last_commit_wall_ts
        assert before <= agg._last_commit_wall_ts["t1"] <= after

    def test_stamped_in_sim_mode_too(self):
        agg = _FakeAggregator(simulated=True)
        channel = _FakeChannel()

        agg.process(channel, _grad_msg(), "t1", timestamp=0)

        assert "t1" in agg._last_commit_wall_ts


class TestLagDecompLog:
    def test_logs_numeric_components_when_stamps_present(self, caplog):
        import datetime as _dt
        sent = _dt.datetime(2026, 1, 1, 0, 0, 0)
        recv = _dt.datetime(2026, 1, 1, 0, 0, 2)  # 2s round trip
        agg = _FakeAggregator(simulated=False)
        channel = _FakeChannel(round_start_time=sent)

        wst = sent.timestamp() + 1.5   # trainer sent update back at +1.5s
        wrt = sent.timestamp() + 0.1   # trainer received weights at +0.1s
        with caplog.at_level("INFO"):
            agg.process(channel, _grad_msg(wall_send_ts=wst, wall_recv_ts=wrt),
                       "t1", timestamp=recv)

        lag_lines = [r.message for r in caplog.records if "[LAG_DECOMP]" in r.message]
        assert len(lag_lines) == 1
        line = lag_lines[0]
        assert "end=t1" in line
        assert "wall_lag_s=2.000" in line          # recv - sent
        assert "agg_to_trainer_s=0.100" in line     # wrt - sent
        assert "compute_s=1.400" in line            # wst - wrt
        assert "mqtt_lag_s=0.500" in line            # recv - wst
        assert "post_wait_s=-" in line
        assert "queue_wait_s=-" in line
        assert "process_s=-" in line

    def test_placeholders_when_stamps_absent(self, caplog):
        import datetime as _dt
        sent = _dt.datetime(2026, 1, 1, 0, 0, 0)
        agg = _FakeAggregator(simulated=False)
        channel = _FakeChannel(round_start_time=sent)

        with caplog.at_level("INFO"):
            agg.process(channel, _grad_msg(), "t1", timestamp=sent)

        lag_lines = [r.message for r in caplog.records if "[LAG_DECOMP]" in r.message]
        assert len(lag_lines) == 1
        assert "agg_to_trainer_s=-" in lag_lines[0]
        assert "compute_s=-" in lag_lines[0]
        assert "mqtt_lag_s=-" in lag_lines[0]

    def test_no_log_without_round_start_time(self, caplog):
        """No dispatch record for this end -> the whole PROP_CLIENT_TASK_TRAIN_
        DURATION branch (and LAG_DECOMP with it) is skipped, matching existing
        behavior for a trainer the aggregator never dispatched to."""
        agg = _FakeAggregator(simulated=False)
        channel = _FakeChannel()  # no round_start_time

        with caplog.at_level("INFO"):
            agg.process(channel, _grad_msg(), "t1", timestamp=0)

        assert not [r for r in caplog.records if "[LAG_DECOMP]" in r.message]


# ---------------------------------------------------------------------------
# redispatch_decomp telemetry (_distribute_weights_async)
# ---------------------------------------------------------------------------

class _DChan:
    def __init__(self):
        self.properties = {}
        self.props = {}
        self.sent = []

    def await_join(self):
        return True

    def set_curr_unavailable_trainers(self, **_kw):
        pass

    def ends(self, *args, **kwargs):
        if args and args[0] == VAL_CH_STATE_RECV:
            return []
        return ["A"]

    def set_end_property(self, end, key, value):
        self.props[(end, key)] = value

    def get_end_property(self, end, key):
        return self.props.get((end, key))

    def send(self, end, payload):
        self.sent.append(end)

    def get_c(self):
        return 30


class _Buf:
    def has(self, _e):
        return False


class _DAgg:
    _distribute_weights_async = TopAggregator._distribute_weights_async
    _select_ends_for_async_respecting_reselect_gate = (
        TopAggregator._select_ends_for_async_respecting_reselect_gate
    )
    _should_send_full_weights = TopAggregator._should_send_full_weights
    _warn_if_redundant_weights_resend = TopAggregator._warn_if_redundant_weights_resend
    _already_served_current_instruction = TopAggregator._already_served_current_instruction
    _mark_instruction_served = TopAggregator._mark_instruction_served

    def __init__(self, simulated=False):
        self.simulated = simulated
        self.trainer_event_dict = None
        self._chan = _DChan()
        self.cm = types.SimpleNamespace(get_by_tag=lambda _t: self._chan)
        self._round = 1
        self.data_id = 0
        self.iteration_per_data_id = 0
        self._model_version = 0
        self.var_good_enough = True
        self._weights_sent_this_cycle = set()
        self._redundant_weights_suppressed_total = 0
        self._trainer_last_model_version = {}
        self._trainer_state_dict = {}
        self._end_served_version_key = {}
        self._sim_inflight_expected = {}
        self._sim_known_delay_s = {"A": 5.0}
        self._sim_pending_commit = set()
        self._sim_dispatch_wall = {}
        self._sim_buffer = _Buf()
        self._sim_staggered_redispatch = False
        self._reselect_each_iteration = True
        self.weights = None
        self._last_commit_wall_ts = {}
        self._last_round_close_wall_ts = None
        self.config = types.SimpleNamespace(hyperparameters=types.SimpleNamespace(
            sim_model_dispatch_queue=False, sim_overhead_warn_s=5.0))

    @property
    def version_key(self):
        return (1, 0)

    @property
    def vclock_now(self):
        return 100.0

    def get_global_model_params(self):
        return {"w": 0}

    def _prepare_distribution_payload(self, _task, force_weights=False):
        return {"w": 0} if force_weights else {"v": 0}

    def _update_state_after_payload_prepared(self):
        pass


def _events(tmp_path, event_name):
    import json
    path = tmp_path / "aggregator.jsonl"
    if not path.exists():
        return []
    lines = path.read_text().splitlines()
    return [e for e in (json.loads(l) for l in lines) if e["event"] == event_name]


class TestRedispatchDecompTelemetry:
    def test_first_ever_dispatch_emits_nothing(self, tmp_path):
        """No prior commit for "A" -> no meaningful gap to report."""
        telemetry.configure(role="aggregator", run_dir=str(tmp_path))
        try:
            agg = _DAgg()
            agg._distribute_weights_async("t")
            assert _events(tmp_path, "redispatch_decomp") == []
        finally:
            telemetry.shutdown()

    def test_redispatch_splits_peer_wait_and_post_close_overhead(self, tmp_path):
        """"A" committed at t=0; its round closed at t=1 (1s peer-wait, since
        the round waited on other cohort-mates past A's own commit); the
        actual redispatch fires "now" -- whatever's left is post-close
        overhead."""
        telemetry.configure(role="aggregator", run_dir=str(tmp_path))
        try:
            agg = _DAgg()
            t0 = time.time() - 5.0
            agg._last_commit_wall_ts["A"] = t0
            agg._last_round_close_wall_ts = t0 + 1.0

            agg._distribute_weights_async("t")

            evs = _events(tmp_path, "redispatch_decomp")
            assert len(evs) == 1
            ev = evs[0]
            assert ev["end_id"] == "A"
            assert ev["time_mode"] == "real"
            assert ev["payload_kind"] == "weights"
            assert abs(ev["peer_wait_wall_s"] - 1.0) < 0.3
            assert abs(ev["redispatch_gap_wall_s"] - 5.0) < 0.3
            # gap == peer_wait + post_close, by construction.
            assert abs(ev["redispatch_gap_wall_s"]
                       - (ev["peer_wait_wall_s"] + ev["post_close_overhead_wall_s"])) < 1e-6
        finally:
            telemetry.shutdown()

    def test_var_bad_redispatch_also_splits_peer_wait_and_post_close(self, tmp_path):
        """A VAR=bad ping shares the WEIGHTS path's channel.send() call, so it
        gets the same peer_wait/post_close split -- previously uninstrumented
        despite being most of the cycles (simulate_fwdllm.md §D-11)."""
        telemetry.configure(role="aggregator", run_dir=str(tmp_path))
        try:
            agg = _DAgg()
            agg.var_good_enough = False
            agg._trainer_last_model_version["A"] = agg._model_version  # not stale -> VAR=bad
            t0 = time.time() - 5.0
            agg._last_commit_wall_ts["A"] = t0
            agg._last_round_close_wall_ts = t0 + 1.0

            agg._distribute_weights_async("t")

            evs = _events(tmp_path, "redispatch_decomp")
            assert len(evs) == 1
            ev = evs[0]
            assert ev["payload_kind"] == "var_bad"
            assert abs(ev["peer_wait_wall_s"] - 1.0) < 0.3
            assert abs(ev["redispatch_gap_wall_s"] - 5.0) < 0.3
        finally:
            telemetry.shutdown()

    def test_own_commit_closed_the_round_zero_peer_wait(self, tmp_path):
        """"A" was itself the round-closing commit (or no round has closed
        since) -- attribute the whole gap to post-close overhead, not
        peer-wait."""
        telemetry.configure(role="aggregator", run_dir=str(tmp_path))
        try:
            agg = _DAgg()
            t0 = time.time() - 3.0
            agg._last_commit_wall_ts["A"] = t0
            agg._last_round_close_wall_ts = t0 - 10.0  # stale, predates this commit

            agg._distribute_weights_async("t")

            evs = _events(tmp_path, "redispatch_decomp")
            assert len(evs) == 1
            assert evs[0]["peer_wait_wall_s"] == 0.0
            assert abs(evs[0]["post_close_overhead_wall_s"] - 3.0) < 0.3
        finally:
            telemetry.shutdown()

    def test_redispatch_also_emits_measurement_only_vclock_charge(self, tmp_path):
        """The post-close span also flows through `charge_sim_vclock_overhead`
        as a `charge=False` candidate category (§D-11) -- must appear in
        `vclock_charge` with charged_s==0.0, never altering behavior."""
        telemetry.configure(role="aggregator", run_dir=str(tmp_path))
        try:
            agg = _DAgg()
            t0 = time.time() - 5.0
            agg._last_commit_wall_ts["A"] = t0
            agg._last_round_close_wall_ts = t0 + 1.0

            agg._distribute_weights_async("t")

            decomp = _events(tmp_path, "redispatch_decomp")[0]
            charges = _events(tmp_path, "vclock_charge")
            assert len(charges) == 1
            ch = charges[0]
            assert ch["label"] == "redispatch_turnaround"
            assert ch["payload_kind"] == "weights"
            assert ch["charged_s"] == 0.0
            assert abs(ch["span_s"] - decomp["post_close_overhead_wall_s"]) < 1e-6
        finally:
            telemetry.shutdown()

    def test_sim_mode_tagged_correctly(self, tmp_path):
        telemetry.configure(role="aggregator", run_dir=str(tmp_path))
        try:
            agg = _DAgg(simulated=True)
            t0 = time.time() - 2.0
            agg._last_commit_wall_ts["A"] = t0
            agg._last_round_close_wall_ts = t0

            agg._distribute_weights_async("t")

            evs = _events(tmp_path, "redispatch_decomp")
            assert len(evs) == 1
            assert evs[0]["time_mode"] == "sim"
        finally:
            telemetry.shutdown()

    def test_redispatch_charges_from_registry_when_configured(self, tmp_path):
        """§P wiring: with `sim_charge_profile_path` set and a `weights` entry
        marked `charge: true`, the call site charges the PROFILED mean, not
        the live (near-zero) sim span."""
        from flame.sim.virtual_clock import VirtualClock
        profile = tmp_path / "profile.yaml"
        profile.write_text(
            "redispatch_turnaround:\n"
            "  weights:\n"
            "    charge: true\n"
            "    mean_s: 0.4365\n"
        )
        telemetry.configure(role="aggregator", run_dir=str(tmp_path))
        try:
            agg = _DAgg(simulated=True)
            agg._vclock = VirtualClock()
            agg.config.hyperparameters.sim_charge_profile_path = str(profile)
            t0 = time.time() - 5.0
            agg._last_commit_wall_ts["A"] = t0
            agg._last_round_close_wall_ts = t0 + 1.0

            agg._distribute_weights_async("t")

            ch = _events(tmp_path, "vclock_charge")[0]
            assert ch["charge_source"] == "profiled"
            assert ch["charged_s"] == 0.4365
            assert abs(agg._vclock.now - 0.4365) < 1e-9
        finally:
            telemetry.shutdown()

    def test_redispatch_var_bad_not_charged_when_only_weights_enabled(self, tmp_path):
        """The registry entry is looked up per `payload_kind` -- `var_bad`
        stays uncharged even with the same profile configured, matching the
        yaml's own `charge: false` for that kind (avoids the overshoot found
        when both kinds are charged, simulate_fwdllm.md §B)."""
        from flame.sim.virtual_clock import VirtualClock
        profile = tmp_path / "profile.yaml"
        profile.write_text(
            "redispatch_turnaround:\n"
            "  weights:\n"
            "    charge: true\n"
            "    mean_s: 0.4365\n"
            "  var_bad:\n"
            "    charge: false\n"
            "    mean_s: 0.0201\n"
        )
        telemetry.configure(role="aggregator", run_dir=str(tmp_path))
        try:
            agg = _DAgg(simulated=True)
            agg._vclock = VirtualClock()
            agg.config.hyperparameters.sim_charge_profile_path = str(profile)
            agg.var_good_enough = False
            agg._trainer_last_model_version["A"] = agg._model_version
            t0 = time.time() - 5.0
            agg._last_commit_wall_ts["A"] = t0
            agg._last_round_close_wall_ts = t0 + 1.0

            agg._distribute_weights_async("t")

            ch = _events(tmp_path, "vclock_charge")[0]
            assert ch["charge_source"] == "none"
            assert ch["charged_s"] == 0.0
            assert agg._vclock.now == 0.0
        finally:
            telemetry.shutdown()
