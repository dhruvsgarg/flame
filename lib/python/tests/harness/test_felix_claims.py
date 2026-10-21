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


def test_f41_cells_split_by_stream_and_grade_one_felix_arm(tmp_path):
    assert fc.scenario("F41_s0u_c3 run_x", "syn_0") == "syn_0_usage"
    assert fc.scenario("F41_s0_ctl run_x", "syn_0") == "syn_0"
    assert fc.scenario("gs_F41_s50u_ctl run_x", "syn_50_3st") == "syn_50_3st_usage"
    assert fc.base_scenario("syn_50_3st_usage") == "syn_0_usage" and fc.base_scenario("syn_50_3st") == "syn_0"
    pool = tmp_path / "pool"
    for phase, b in (("F41_s0u_ctl", "felix"), ("F41_s0u_ctl", "fedbuff"), ("F41_s0u_c3", "felix")):
        d = tmp_path / f"run_{phase}_dbg_{b}_n75_alpha0.1_syn_0_stream_sim"
        (d / "telemetry").mkdir(parents=True)
        (pool / phase).mkdir(parents=True, exist_ok=True)
        f = pool / phase / "summary.tsv"
        rows = f.read_text() if f.exists() else "baseline\ttrace\treal_dir\tsim_dir\n"
        f.write_text(rows + f"{b}\tsyn_0\t\t{d}\n")
    cells = fc.collect([str(pool)], [])
    assert set(cells) == {("cifar10", "syn_0_usage", "felix"), ("cifar10", "syn_0_usage", "fedbuff")}
    assert "_s0u_ctl_" in cells[("cifar10", "syn_0_usage", "felix")]
    assert "_s0u_c3_" in fc.collect([str(pool)], [], felix_arm="c3")[("cifar10", "syn_0_usage", "felix")]
