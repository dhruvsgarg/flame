# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""H13: the probe's reporting decisions, which twice invalidated a run silently.

Runs 1-2 read bit-exact on every arm because `_build_real` forced `.eval()`, and
their `--hetero` sweeps printed a cross-process "spread" that was different WORK,
not nondeterminism. Both are reporting bugs, not measurement bugs, so they are
unit-testable even though the probe itself is an experiment
(`expt_scripts/probe_jvp_determinism.py`, simulate_fwdllm.md §D-47).
"""

import importlib.util
import logging
import pathlib
import sys
from pathlib import Path

import pytest
import yaml

_FWDLLM = Path(__file__).resolve().parents[2] / "examples" / "fwdllm"
_PROBE = _FWDLLM / "expt_scripts" / "probe_jvp_determinism.py"


@pytest.fixture(scope="module")
def probe():
    spec = importlib.util.spec_from_file_location("_probe", _PROBE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_probe"] = mod
    spec.loader.exec_module(mod)
    return mod


def _rep(seed, loss, jvp, repeats=2):
    """One replica record, shaped like `_run_arm`'s return."""
    return {"seed": seed, "repeats": repeats, "device": "cuda", "model": "real",
            "eval_mode": False, "live_dropout": 13,
            "loss_exact_frac": 1.0 if len(set(loss)) == 1 else 0.5,
            "loss_rel_spread": 0.0 if len(set(loss)) == 1 else 1e-3,
            "jvp_rel_spread": 0.0 if len(set(jvp)) == 1 else 1e-1,
            "jvp_fp32": False, "strict_determinism": False,
            "losses": loss, "jvps": jvp}


class TestHeteroDetection:
    def test_same_seed_is_not_hetero(self, probe):
        reps = [_rep(7, [1.0], [2.0]), _rep(7, [1.0], [2.0])]
        assert probe._merge_replicas("base", reps)["hetero"] is False

    def test_differing_seeds_are_hetero(self, probe):
        reps = [_rep(7, [1.0], [2.0]), _rep(8, [1.5], [9.0])]
        assert probe._merge_replicas("base", reps)["hetero"] is True

    def test_hetero_suppresses_the_cross_process_table(self, probe, capsys):
        reps = [_rep(7, [1.0], [2.0]), _rep(8, [1.5], [9.0])]
        probe._report([probe._merge_replicas("base", reps)])
        out = capsys.readouterr().out
        assert "ACROSS concurrent processes: SUPPRESSED" in out
        assert "replicate-floor analogue" not in out


class TestVerdictColumn:
    def test_within_process_spread_counts_as_reproduction(self, probe, capsys):
        """Dropout moves repeats inside one process while same-seed processes stay
        aligned: reading only the cross-process column calls that INCONCLUSIVE."""
        reps = [_rep(7, [1.0, 1.001], [2.0, 2.2]), _rep(7, [1.0, 1.001], [2.0, 2.2])]
        base = probe._merge_replicas("base", reps)
        assert base["jvp_rel_spread"] == 0.0 and base["within_jvp_rel_spread"] > 0
        probe._report([base])
        out = capsys.readouterr().out
        assert "INCONCLUSIVE" not in out
        assert "Read the within column" in out

    def test_bit_exact_everywhere_is_still_inconclusive(self, probe, capsys):
        reps = [_rep(7, [1.0, 1.0], [2.0, 2.0]), _rep(7, [1.0, 1.0], [2.0, 2.0])]
        probe._report([probe._merge_replicas("base", reps)])
        assert "INCONCLUSIVE" in capsys.readouterr().out


class TestEvalModeArm:
    def test_evalmode_is_an_arm_and_base_leaves_dropout_live(self, probe):
        assert probe._ARMS["evalmode"] == {"FWDLLM_PROBE_EVAL": "1"}
        assert probe._ARMS["base"] == {}

    def test_eval_gate_is_env_driven_and_defaults_off(self, probe, monkeypatch):
        monkeypatch.delenv("FWDLLM_PROBE_EVAL", raising=False)
        assert probe._eval_mode_requested() is False
        monkeypatch.setenv("FWDLLM_PROBE_EVAL", "1")
        assert probe._eval_mode_requested() is True


class TestEvalModePromoted:
    """`jvp_eval_mode` is a promoted correctness fix: ON unless explicitly
    disabled, at BOTH read sites, and declared in every baseline yaml.

    A yaml-only knob is invisible to the launch fingerprint (§F-18), so "the
    config says true" is not evidence the model saw it — these pin the layers
    the config cannot reach.
    """

    @staticmethod
    def _default_at(path, needle):
        """The literal default in a `getattr(..., "jvp_eval_mode", X)` call."""
        src = pathlib.Path(path).read_text(encoding="utf-8")
        i = src.index(needle)
        return src[i:i + 400]

    def test_trainer_entrypoint_defaults_on(self):
        chunk = self._default_at(
            _FWDLLM / "trainer" / "main.py", '"jvp_eval_mode": getattr(')
        assert 'jvp_eval_mode", True)' in chunk

    def test_trainer_class_defaults_on(self):
        chunk = self._default_at(
            _FWDLLM / "trainer" / "forward_training"
            / "tc_transformer_trainer_distribute.py",
            'self.jvp_eval_mode = bool(')
        assert '"jvp_eval_mode", True)' in chunk

    @pytest.mark.parametrize("yml", sorted(
        p for p in (_FWDLLM / "expt_scripts").glob("*_smoke*.yaml")
        if not p.name.startswith("figs")))
    def test_every_baseline_yaml_declares_it(self, yml):
        """The §F-18 contract, enforced instead of remembered: a baseline that
        silently loses the knob trains a different experiment."""
        cfg = yaml.safe_load(yml.read_text(encoding="utf-8"))
        hp = (cfg["experiments"][0]["trainer"]
              ["config_overrides"]["hyperparameters"])
        assert hp.get("jvp_eval_mode") is True, f"{yml.name} does not declare it ON"


class TestDropoutCensus:
    """The run must PROVE dropout was off, not assert the flag was set (§D-48):
    `model.training` reads False while its dropout leaves train on."""

    @pytest.fixture
    def trainer(self):
        torch = pytest.importorskip("torch")
        mod = pytest.importorskip(
            "examples.fwdllm.trainer.forward_training."
            "tc_transformer_trainer_distribute")
        m = torch.nn.Sequential(torch.nn.Dropout(0.1), torch.nn.Dropout(0.1),
                                torch.nn.Linear(4, 4))
        m.training = False                      # the shape train_adapter leaves
        for child in m.children():
            child.training = True
        cls = mod.ForwardTextClassificationTrainer
        t = cls.__new__(cls)
        t.model, t._dropout_census_logged, t.jvp_eval_mode = m, False, True
        return mod, t

    def test_census_reads_zero_inside_the_block(self, trainer, caplog):
        mod, t = trainer
        with caplog.at_level(logging.INFO):
            with mod._eval_mode(t.model):
                t._log_dropout_census()
        assert "live_dropout=0/2" in caplog.text

    def test_census_counts_live_leaves_when_off(self, trainer, caplog):
        """Without the block the leaves stay training — the defect's signature."""
        mod, t = trainer
        t.jvp_eval_mode = False
        with caplog.at_level(logging.INFO):
            t._log_dropout_census()
        assert "live_dropout=2/2" in caplog.text

    def test_logged_once_per_process(self, trainer, caplog):
        mod, t = trainer
        with caplog.at_level(logging.INFO):
            for _ in range(3):
                t._log_dropout_census()
        assert caplog.text.count("live_dropout=") == 1


class TestEvalModeContext:
    """`_eval_mode` is the trainer-side H13 fix, now promoted to default ON."""

    @pytest.fixture(scope="class")
    def trainer_mod(self):
        pytest.importorskip("torch")
        return pytest.importorskip(
            "examples.fwdllm.trainer.forward_training."
            "tc_transformer_trainer_distribute")

    @pytest.fixture
    def model(self):
        torch = pytest.importorskip("torch")
        m = torch.nn.Sequential(torch.nn.Dropout(0.1), torch.nn.Linear(4, 4))
        # The shape `train_adapter` leaves behind: root reads False, children True.
        m.training = False
        for child in m.children():
            child.training = True
        return m

    def test_dropout_is_off_inside(self, trainer_mod, model):
        with trainer_mod._eval_mode(model):
            assert not any(m.training for m in model.modules())

    def test_every_module_flag_is_restored(self, trainer_mod, model):
        before = [m.training for m in model.modules()]
        with trainer_mod._eval_mode(model):
            pass
        assert [m.training for m in model.modules()] == before

    def test_restores_on_exception(self, trainer_mod, model):
        before = [m.training for m in model.modules()]
        with pytest.raises(RuntimeError):
            with trainer_mod._eval_mode(model):
                raise RuntimeError("boom")
        assert [m.training for m in model.modules()] == before


class TestReplicateFloorConfigSplit:
    """A floor pooled across `jvp_eval_mode` values measures the flag, not the
    floor (§D-45). The knob is in no config file in the run dir, so the grouping
    reads it back from the trainer log."""

    @pytest.fixture(scope="class")
    def rf(self):
        import importlib.util
        path = (Path(__file__).resolve().parents[2] / "examples" / "fwdllm"
                / "expt_scripts" / "replicate_floor.py")
        spec = importlib.util.spec_from_file_location("_rf", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["_rf"] = mod
        spec.loader.exec_module(mod)
        return mod

    def _run_dir(self, tmp_path, name, line):
        d = tmp_path / name
        d.mkdir()
        (d / "x_trainers.log").write_text(
            "some preamble\n" + (line + "\n" if line else "") + "more\n")
        return d

    def test_on_leg_detected(self, rf, tmp_path):
        d = self._run_dir(tmp_path, "on", "INFO | [JVP_EVAL_MODE] jvp_eval_mode=True (…)")
        assert rf._jvp_eval_mode(str(d)) is True

    def test_off_leg_detected(self, rf, tmp_path):
        d = self._run_dir(tmp_path, "off", "INFO | [JVP_EVAL_MODE] jvp_eval_mode=False (…)")
        assert rf._jvp_eval_mode(str(d)) is False

    def test_pre_flag_run_reads_as_off(self, rf, tmp_path):
        """Runs older than the flag never log it; the code default was dropout live."""
        d = self._run_dir(tmp_path, "legacy", None)
        assert rf._jvp_eval_mode(str(d)) is False

    def test_missing_log_reads_as_off(self, rf, tmp_path):
        d = tmp_path / "empty"
        d.mkdir()
        assert rf._jvp_eval_mode(str(d)) is False
