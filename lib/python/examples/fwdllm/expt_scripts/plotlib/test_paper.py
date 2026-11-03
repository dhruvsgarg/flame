"""Tests for the paper-figure series (reducers §6 fields) and plotlib/paper.py.

Run:  python -m pytest plotlib/test_paper.py -q
"""

from __future__ import annotations

import json
import os
import sys

_EXPT_SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _EXPT_SCRIPTS not in sys.path:
    sys.path.insert(0, _EXPT_SCRIPTS)

import pytest  # noqa: E402

from plotlib import paper as P  # noqa: E402
from plotlib import reducers as R  # noqa: E402
from plotlib import style as S  # noqa: E402
from plotlib import synthetic as SY  # noqa: E402


def _write_jsonl(path, events):
    with open(path, "w", encoding="utf-8") as fh:
        for e in events:
            fh.write(json.dumps(e) + "\n")


@pytest.fixture
def run_dir(tmp_path):
    """Sim run: 3 cycles, 2 commits, vclock = 10 x wall offset."""
    d = tmp_path / "run_sim"
    t = d / "telemetry"
    t.mkdir(parents=True)
    (d / "aggregator_config.json").write_text(json.dumps({"hyperparameters": {"time_mode": "simulated"}}))
    ag = lambda ts, reason, ends: {  # noqa: E731
        "event": "agg_round", "ts": ts, "round": 0, "data_id": 0, "is_async": True,
        "vclock_now": (ts - 100) * 10, "commit_reason": reason, "contributing_trainers": ends,
        "contributor_intervals": [{"end": e, "dispatch_ts": 0.0, "commit_ts": 30.0} for e in ends]}
    _write_jsonl(t / "aggregator_1.jsonl", [
        {"event": "agg_eval", "ts": 100.0, "round": 0, "data_id": 0, "test-accuracy": 0.5, "test-loss": 2.0},
        ag(101.0, None, ["t1", "t2"]),
        ag(102.0, "natural", ["t1"]),
        {"event": "comm", "ts": 102.5, "direction": "agg_to_trainer", "size_bytes": 1000},
        {"event": "server_update", "ts": 102.6, "rho": 0.05, "phi": 1.1},
        {"event": "agg_eval", "ts": 103.0, "round": 0, "data_id": 1, "test-accuracy": 0.9, "test-loss": 1.0},
        ag(104.0, "cap", ["t3"]),
        {"event": "comm", "ts": 105.0, "direction": "agg_to_trainer", "size_bytes": 5000},
    ])
    _write_jsonl(t / "trainer_t1.jsonl", [
        {"event": "comm", "ts": 101.5, "direction": "trainer_to_agg", "size_bytes": 200},
        {"event": "comm", "ts": 104.5, "direction": "trainer_to_agg", "size_bytes": 300},
    ])
    return str(d)


def test_reducer_commit_comm_and_contrib_series(run_dir):
    rr = R.load_run(run_dir, post_peak_grace_s=None)
    assert rr.is_sim and rr.is_async
    assert rr.commit_ts == [102.0, 104.0]
    assert rr.commit_uploads == [3, 1]            # pending uploads fold into the next commit
    assert [e for _, e in rr.contribs] == ["t1", "t2", "t1", "t3"]
    assert rr.server_updates[0]["rho"] == 0.05
    assert rr.bytes_until(103.0) == (1000, 200)
    assert rr.bytes_until(None) == (6000, 500)
    assert rr.commits_until(103.0) == 1
    assert rr.native_s(103.0) == pytest.approx(20.0)   # vclock at last agg_round <= 103
    assert rr.session_durs == [30.0] * 4


def test_seed_metrics_to_target_and_dnf(run_dir):
    rr = R.load_run(run_dir, post_peak_grace_s=None)
    m = P.seed_metrics(rr, 0.85)
    assert m["reached"] and m["commits"] == 1
    assert m["down_gb"] == pytest.approx(1e-6) and m["up_gb"] == pytest.approx(2e-7)
    dnf = P.seed_metrics(rr, 0.95)
    assert not dnf["reached"] and dnf["tta_h"] is None and dnf["commits"] == 2


def test_first_crossing_consecutive():
    rr = SY.fake_run("fluxtune")
    one, three = rr.first_crossing(0.5, 1), rr.first_crossing(0.5, 3)
    assert one is not None and three["ts"] >= one["ts"]


def test_representative_is_median_tta_seed():
    runs = [SY.fake_run("fluxtune", speed=s, seed=i) for i, s in enumerate((1.0, 3.0, 2.0))]
    arm = P.Arm("fluxtune", "FluxTune", P.arm_style("fluxtune"), runs)
    assert P.representative(arm, 0.8) is runs[2]


def test_gini_bounds():
    assert P._gini([5, 5, 5, 5]) == pytest.approx(0.0)
    assert P._gini([0, 0, 0, 10]) == pytest.approx(0.75)


@pytest.mark.parametrize("manifest", ["figs_e2e.yaml", "figs_availability.yaml", "figs_ablation.yaml"])
def test_manifests_render_synthetic_and_empty(manifest, tmp_path, monkeypatch):
    import make_paper_figs as M
    path = os.path.join(_EXPT_SCRIPTS, manifest)
    monkeypatch.setattr(sys, "argv", ["x", "--manifest", path, "--synthetic",
                                      "--out-root", str(tmp_path / "syn")])
    assert M.main() == 0
    written = json.load(open(tmp_path / "syn" / "manifest.json"))["figures"]
    assert written and all((tmp_path / "syn" / f"{n}.pdf").exists() for n in written)
    assert (tmp_path / "syn" / "summary.csv").exists()
    monkeypatch.setattr(sys, "argv", ["x", "--manifest", path, "--out-root", str(tmp_path / "empty")])
    assert M.main() == 0                          # no runs yet: renders nothing, no crash


def test_step_size_skips_arms_without_audit():
    S.use_paper_style()
    arm = P.Arm("fwdllm", "FwdLLM", P.arm_style("fwdllm"), [SY.fake_run("fwdllm", with_audit=False)])
    assert P.fig_c_step_size([arm], {"target_acc": 0.8}, {}) is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
