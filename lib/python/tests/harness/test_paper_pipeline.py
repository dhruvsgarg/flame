"""FX-N153: leg digest (series + task fates on a synthetic leg), manifest (Table 1 / 2 legs), launch script."""
import gzip
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples" / "scripts"))
import leg_digest  # noqa: E402
import paper_runs as pr  # noqa: E402


def _w(path, rows):
    path.write_text("".join(json.dumps(dict(ts=0, role="x", **r)) + "\n" for r in rows))


def _leg(tmp_path):
    """Async leg, c 2, two clients: a commits twice, b is evicted once then commits; target hit at the 2nd eval."""
    run = tmp_path / "FB2E2ES_smu_fx" / "runs" / "syn_0_felix_s1" / "run_x"
    tel = run / "telemetry"
    tel.mkdir(parents=True)
    (run / "aggregator_config.json").write_text(json.dumps(dict(
        hyperparameters=dict(targetAccuracy=0.5, targetWindow=1, availability_trace="syn_0"), selector=dict(sort="async_oort", kwargs=dict(c=2)),
        optimizer={})))
    (run / "debug_run.yaml").write_text("experiments: [{}]\n")
    _w(tel / "aggregator_j.jsonl", [
        dict(event="selection", task="train", vclock_now=0.0, effective_c=2, num_eligible=2, num_chosen=2),
        dict(event="wire", dir="send", bytes=100), dict(event="wire", dir="send", bytes=100),
        dict(event="abandon_timeout", end_id="b", sim_send_ts=0.0, vclock_now=5.0, reason="aware_boundary_eviction"),
        dict(event="wire", dir="recv", bytes=100),
        dict(event="agg_round", round=1, vclock_now=10.0, staleness=[0], contributing_trainers=["a"]),
        dict(event="agg_timing", round=1, vclock_now=10.0, cycle_s=0.1, commit_s=0.01, ingest_s=0.01),
        dict(event="agg_eval", round=1, **{"test-accuracy": 0.3}),
        dict(event="withheld_delivery", staleness=2, accepted=True),
        dict(event="agg_round", round=2, vclock_now=30.0, staleness=[1], contributing_trainers=["b"]),
        dict(event="agg_round", round=2, vclock_now=40.0, staleness=[2], contributing_trainers=["a"]),
        dict(event="agg_eval", round=2, **{"test-accuracy": 0.6}),
        dict(event="run_end", vclock_now=50.0)])
    _w(tel / "trainer_a.jsonl", [dict(event="trainer_round", sim_send_ts=0.0, sim_completion_ts=10.0, sim_round_duration_s=10.0,
                                      total_samples=50, model_samples=32, round=0),
                                 dict(event="trainer_round", sim_send_ts=10.0, sim_completion_ts=40.0, sim_round_duration_s=30.0,
                                      total_samples=50, model_samples=32, round=1)])
    _w(tel / "trainer_b.jsonl", [dict(event="trainer_round", sim_send_ts=0.0, sim_completion_ts=20.0, sim_round_duration_s=20.0,
                                      total_samples=10, model_samples=10, round=0)])
    return run


def test_digest_series_and_fates(tmp_path):
    run = _leg(tmp_path)
    d = leg_digest.digest(str(run))
    assert leg_digest.check(d) == [] and set(leg_digest.REQUIRED) <= set(leg_digest.SOURCES)
    assert d["acc"]["acc"] == [0.3, 0.6] and d["acc"]["t"] == [10.0, 40.0] and d["acc"]["updates"] == [1, 3]
    assert d["hit"] == dict(target=0.5, h=40 / 3600, gb=300 / 1e9, acc=0.6, updates=3)
    assert d["updates"]["total"] == 3 and d["updates"]["versions"] == 1
    assert d["staleness"]["all"] == {"0": 1, "1": 1, "2": 1} and d["staleness"]["withheld"] == {"2": 1}
    assert d["fates"]["counts"] == {"aggregated": 2, "late": 1}  # b: evicted at 5, its update still committed at 30
    s = d["slots"]
    assert s["busy_s"] == 10 + 30 + 5 and s["capacity_s"] == 2 * 50.0 and s["idle_s"] == 100 - 45
    assert d["picks"]["n"] == 3 and d["picks"]["coverage"] == 1.0 and d["sync"] is False


def test_withheld_slot_frees_at_completion_only_for_notify(tmp_path, monkeypatch):
    """FX-N167: x leaves mid-task (5 s), finishes (10 s), returns (25 s), commits (30 s): Felix's send gate frees the slot at
    completion, a check-in stack holds it until delivery."""
    run = _leg(tmp_path)
    tel = run / "telemetry"
    for f in tel.glob("trainer_*.jsonl"):
        f.unlink()
    _w(tel / "aggregator_j.jsonl", [
        dict(event="agg_round", round=1, vclock_now=30.0, staleness=[0], contributing_trainers=["x"]),
        dict(event="agg_eval", round=1, **{"test-accuracy": 0.6}), dict(event="run_end", vclock_now=50.0)])
    _w(tel / "trainer_x.jsonl", [dict(event="trainer_round", sim_send_ts=0.0, sim_completion_ts=10.0, sim_round_duration_s=10.0,
                                      total_samples=10, model_samples=10, round=0)])
    monkeypatch.setattr(leg_digest, "trace_events", lambda *_: {"x": [(0.0, "AVL_TRAIN"), (5.0, "UN_AVL"), (25.0, "AVL_TRAIN")]})
    cfg = json.loads((run / "aggregator_config.json").read_text())
    held = {}
    for k in ("notify", "checkin"):
        cfg["hyperparameters"]["avail_knowledge"] = k
        (run / "aggregator_config.json").write_text(json.dumps(cfg))
        held[k] = leg_digest.digest(str(run))["slots"]["held_departed_s"]
    assert held == {"notify": 10 - 5, "checkin": 30 - 5}
    assert leg_digest.digest(str(run))["slots"]["by_fate"] == {"aggregated": [5.0, 0.0, 25.0]}  # checkin: compute to the departure


def test_digest_main_writes_then_gzips_raw(tmp_path):
    run = _leg(tmp_path)
    assert leg_digest.main([str(run)]) == 0
    assert (run / "digest.json.gz").exists() and not list((run / "telemetry").glob("*.jsonl"))
    assert leg_digest.check(json.load(gzip.open(run / "digest.json.gz", "rt"))) == []
    assert leg_digest.digest(str(run))["updates"]["total"] == 3  # re-digest from the gzipped raw logs
    assert leg_digest.main([str(run), "--check"]) == 0


def test_manifest_matches_table_1_and_2():
    e2e = {ds: pr.legs("E2E", ds) for ds in pr.DS}
    assert all(len(v) == 44 for v in e2e.values())  # 4 cells x 11 stacks
    assert {lg[2] for lg in e2e["cifar10"]} >= {"fx", "ctl_srv", "fbstar_srv", "favstar_srv", "sync_srv", "rstar_srv", "fdstar_srv"}
    assert len(pr.required("headline")) == 88 and len(pr.required("teaser")) == 11
    assert {lg[1] for _, _, lg in pr.required("ladder")} == set(pr.hp.FB2_ABL)
    smoke = {(rs, ds): pr.legs(rs, ds, smoke=True) for rs in pr.hp.FB2_SETS for ds in pr.DS}
    assert sum(map(len, smoke.values())) == 60 and all(lg[1] == pr.SMOKE_CELL.get(rs, "smu") for (rs, _), v in smoke.items() for lg in v)


def test_status_and_launch_on_an_empty_root(tmp_path):
    states = pr.status(str(tmp_path), [1], smoke=True, key="k")
    assert {st for st, _, _ in states.values()} == {"missing"}
    sh = pr.launch_script(str(tmp_path), states, 4, [1], True, "0")
    assert sh.count("scripts/bench_pool.py") == 2 and "--digest" in sh and "FB2E2ES" in sh
    f = tmp_path / "l.sh"
    f.write_text(sh)
    assert subprocess.run(["bash", "-n", str(f)]).returncode == 0
    assert pr.launch_script(str(tmp_path), {k: ("done", "", None) for k in states}, 4, [1], True, "0") is None
    sh = pr.launch_script(str(tmp_path), states, 4, [1], True, "0", ("cifar10",))  # FX-N159
    assert sh.count("scripts/bench_pool.py") == 1 and "google_speech" not in sh


def test_every_table_2_row_has_a_plot():
    import paper_plots  # asserts FIGS == paper_runs.FIGURES at import
    assert set(paper_plots.FIGS) == set(pr.FIGURES)


def test_paper_json_metrics_and_ratios():
    import paper_json as pj
    d = dict(config=dict(target=0.8, window=2, K=10), end=dict(h=3.0, gb=9.0, acc_last=0.82, hit=True),
             acc=dict(t=[3600, 7200, 10800], acc=[0.5, 0.7, 0.9], gb=[1.0, 2.0, 3.0], updates=[10, 20, 30]), updates=dict(total=30),
             fates=dict(gb=dict(aggregated=9.0, late=1.0)), staleness=dict(all={"0": 1, "2": 1}), slots={}, departures={}, overhead={})
    assert pj.first_cross(d, 0.75) == (3.0, 3.0, 30)  # 2-eval mean 0.6 at h 2, 0.8 at h 3
    assert pj.first_cross(d, 0.75) is not None and pj.first_cross(d, 0.95) is None
    m = pj.leg_metrics(d, "cifar10", "s0u")
    assert (m["status"], m["time_h"], m["gb"], m["mean_staleness"], m["paid_not_used_pct"]) == ("hit", 3.0, 3.0, 1.0, 10.0)
    fx, o = dict(time_h=3.0, gb=3.0, target=0.8), dict(time_h=6.0, gb=6.0, acc_at_felix_hit_h=70.0)
    assert pj.vs(fx, o) == dict(speedup=2.0, gb_ratio=0.5, gb_saving_pct=50.0, acc_gain_pp_at_felix_hit_h=10.0)


def test_report_target_reads_dataset_name():
    cfg = dict(job=dict(name="cifar-10"), hyperparameters=dict(dataset_name="google_speech", targetAccuracy=0.88, availability_trace="mobiperf_3st"))
    assert leg_digest.report_target(cfg) == 0.84
