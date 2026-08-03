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
import sys
from pathlib import Path

import pytest

_PROBE = (Path(__file__).resolve().parents[2] / "examples" / "fwdllm"
          / "expt_scripts" / "probe_jvp_determinism.py")


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


class TestEvalModeContext:
    """`_eval_mode` is the trainer-side H13 fix (`jvp_eval_mode`, default OFF)."""

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
