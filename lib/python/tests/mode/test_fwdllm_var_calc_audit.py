# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""H14's instrument: per-contribution grad norms into the variance gate.

Every summary statistic already matches between real and sim — pool size,
staleness, aggregation rate, raw grad norm to 0.09% — while the variance over
those gradients runs ~7% low in sim. Variance is dispersion, so the pool's
individual entries are the only place left to look, and `var_calc` is the record
that carries them. It must stay OFF by default (a GPU->CPU sync per entry) and
must be readable by the differ without a live run.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_DIFF = (Path(__file__).resolve().parents[2] / "examples" / "fwdllm"
         / "expt_scripts" / "diff_var_pool.py")


@pytest.fixture(scope="module")
def differ():
    spec = importlib.util.spec_from_file_location("_diff_var_pool", _DIFF)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_diff_var_pool"] = mod
    spec.loader.exec_module(mod)
    return mod


def _leg(tmp_path, name, per_cycle):
    """A run dir carrying var_calc records: {(bin, iter): [norms]}."""
    d = tmp_path / name / "telemetry"
    d.mkdir(parents=True)
    with open(d / "aggregator_x.jsonl", "w") as fh:
        for (b, it), norms in sorted(per_cycle.items()):
            fh.write(json.dumps({
                "event": "var_calc", "round": 0, "data_id": b,
                "iteration_per_data_id": it, "input_grad_norms": norms,
                "output_var": sum((x - sum(norms) / len(norms)) ** 2
                                  for x in norms) / len(norms),
            }) + "\n")
    return str(tmp_path / name)


class TestVarCalcAuditGate:
    def test_defaults_off(self):
        """The audit costs a sync per pool entry — it must never be the default."""
        src = (Path(__file__).resolve().parents[2] / "examples" / "fwdllm"
               / "aggregator" / "FedSgdAggregator.py").read_text(encoding="utf-8")
        assert 'getattr(self.args, "var_calc_audit", False)' in src

    def test_emission_is_gated_on_the_knob_or_debug(self):
        src = (Path(__file__).resolve().parents[2] / "examples" / "fwdllm"
               / "aggregator" / "FedSgdAggregator.py").read_text(encoding="utf-8")
        assert ('if getattr(self, "_var_calc_audit", False) '
                'or logger.isEnabledFor(logging.DEBUG):') in src


class TestDiffVarPool:
    def test_reports_nothing_to_do_without_records(self, differ, tmp_path, capsys):
        """The failure mode to avoid is a confident report over zero records."""
        a = _leg(tmp_path, "real", {})
        b = _leg(tmp_path, "sim", {})
        assert differ.main([a, b]) == 1
        assert "var_calc_audit" in capsys.readouterr().out

    def test_pairs_cycles_and_names_dispersion(self, differ, tmp_path, capsys):
        real = {(0, 0): [1.0, 2.0, 3.0], (1, 0): [1.0, 2.0, 3.0]}
        sim = {(0, 0): [2.0, 2.0, 2.0], (1, 0): [2.0, 2.0, 2.0]}  # same mean, no spread
        assert differ.main([_leg(tmp_path, "real", real),
                            _leg(tmp_path, "sim", sim)]) == 0
        out = capsys.readouterr().out
        assert "paired cycles at identical (bin, iteration): 2" in out
        assert "CoV of pool" in out
        assert "-100.00%" in out          # sim dispersion collapses to zero

    def test_flags_a_reduction_divergence_when_inputs_match(self, differ, tmp_path,
                                                            capsys):
        """Identical inputs with a differing output would indict the reduction
        itself rather than pool assembly — the two must be distinguishable."""
        same = {(0, 0): [1.0, 2.0, 3.0]}
        assert differ.main([_leg(tmp_path, "real", same),
                            _leg(tmp_path, "sim", same)]) == 0
        out = capsys.readouterr().out
        assert "INPUT norms match bit-close: 1/1" in out
        assert "+0.000%" in out           # same reduction on same inputs
