"""FX-N130: per-arm holdout rows and the P3 / P4 trained-count check (config_audit.holdout_check)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples" / "scripts"))
import config_audit as ca  # noqa: E402


def _leg(tmp_path, rows):
    (tmp_path / "telemetry").mkdir()
    with open(tmp_path / "telemetry" / "trainer_a.jsonl", "w") as f:
        for v, n, task in rows:
            f.write(json.dumps(dict(event="trainer_round", task_to_perform=task, visible_samples=v, dataset_size=n)) + "\n")
    return str(tmp_path)


def test_holdout_zero_trains_on_every_visible_sample(tmp_path):
    run = _leg(tmp_path, [(150, 150, "train"), (37, 37, "train"), (150, 0, "eval")])
    assert ca.holdout_check(run, "-") == "ok"
    assert ca.holdout_check(run, "0.1") == "2/2 bad"


def test_holdout_tenth_drops_every_tenth_arrival(tmp_path):
    run = _leg(tmp_path, [(150, 135, "train"), (37, 34, "train"), (9, 9, "train")])
    assert ca.holdout_check(run, "0.1") == "ok"
    assert ca.holdout_check(run, "-") == "2/3 bad"


def test_only_validation_arms_expect_a_holdout():
    assert not any(v.get("holdout") == "0.1" or v.get("vprobe") == "true" for v in ca.EXPECT.values())  # no arm reads held-out data


def test_bench_pool_layout_is_audited(tmp_path):
    # FX-D213: bench_pool writes POOL/<phase>/runs/<trace>_<baseline>_s<seed>/run_*; its dry-run inputs (_yaml/) are not legs
    run = tmp_path / "F42_s0u_v2p" / "runs" / "syn_0_felix_s2" / "run_x"
    (run / "telemetry").mkdir(parents=True)
    (run / "telemetry" / "trainer_a.jsonl").write_text(json.dumps(dict(event="trainer_round", task_to_perform="train",
                                                                        visible_samples=20, dataset_size=18)) + "\n")
    (run / "aggregator_config.json").write_text(json.dumps(dict(hyperparameters={}, selector={}, optimizer={})))
    (run / "debug_run.yaml").write_text("experiments:\n- trainer: {config_overrides: {hyperparameters: {validation_holdout: 0.1}}}\n")
    (tmp_path / "_yaml" / "j_sim" / "runs" / "x").mkdir(parents=True)
    (tmp_path / "_yaml" / "j_sim" / "runs" / "x" / "debug_run.yaml").write_text("experiments: []\n")
    [(leg, b, arm, k)] = list(ca.legs(str(tmp_path)))
    assert (leg, b, arm, k["holdout"], k["p34"]) == ("F42_s0u_v2p/syn_0_felix_s2", "felix", "v2p", "0.1", "ok")
