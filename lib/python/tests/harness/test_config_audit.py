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
    held = {k for k, v in ca.EXPECT.items() if v.get("holdout") == "0.1"}
    assert held == {("felix", "l5p"), ("felix", "e0p"), ("felix", "fck")}
    assert all((v.get("vprobe") == "true") == (k in held) for k, v in ca.EXPECT.items())
