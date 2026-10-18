"""FX-N108 C1 (operator 10-18): a below-target speech baseline is round-bound when learning, or degenerate with timely abandons."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples" / "scripts"))
import felix_claims as fc  # noqa: E402


def _leg(tmp_path, events):
    d = tmp_path / "run_x_real" / "telemetry"
    d.mkdir(parents=True)
    (d / "aggregator_x.jsonl").write_text("\n".join(json.dumps(e) for e in events))
    return str(d.parent)


def _evals(losses, accs):
    return [{"event": "agg_eval", "round": 10 * i, "test-loss": l, "test-accuracy": a, "ts": i}
            for i, (l, a) in enumerate(zip(losses, accs))]


def test_learning_needs_falling_loss_and_rising_accuracy(tmp_path):
    rising = _evals([3.6, 3.5, 3.4, 3.2, 3.0, 2.9], [0.04, 0.05, 0.07, 0.09, 0.12, 0.15])
    assert fc.round_bound(_leg(tmp_path / "a", rising), "syn_0") == "learning"
    flat = _evals([3.56, 3.55, 3.55, 3.55, 3.56, 3.59], [0.037] * 5 + [0.044])  # PR38 speech refl syn_50_3st
    assert fc.round_bound(_leg(tmp_path / "b", flat), "syn_0") is None


def test_a_loss_spike_does_not_flip_a_learning_leg(tmp_path):
    spike = _evals([3.6, 3.6, 3.5, 3.4, 3.3, 3.2, 3.1, 2.9, 5.2], [0.04, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.16, 0.07])  # PR38 speech refl lin real r100
    assert fc.round_bound(_leg(tmp_path, spike), "lin") == "learning"


def test_degenerate_counts_only_with_timely_abandons_and_redispatch(tmp_path):
    ab = [{"event": "abandon_timeout", "ts": 500.0 * i, "age_s": 450.01} for i in range(1, 4)]
    ok = ab + [{"event": "dispatch", "ts": 500.0 * i + 0.1} for i in range(1, 4)]
    assert fc.round_bound(_leg(tmp_path / "a", _evals([3.6, 3.5], [0.02, 0.04]) + ok), "syn_50_3st") == "degenerate"
    late = ab + [{"event": "dispatch", "ts": 500.0 * i + 300} for i in range(1, 4)]
    assert fc.round_bound(_leg(tmp_path / "b", _evals([3.6, 3.5], [0.02, 0.04]) + late), "syn_50_3st") is None
    assert fc.round_bound(_leg(tmp_path / "c", _evals([3.6, 3.5], [0.02, 0.04]) + ok), "syn_0") is None
