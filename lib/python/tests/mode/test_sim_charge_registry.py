# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""sim_charge_registry (FWDLLM_DESIGN.md §P): looks up a profiled vclock
charge for a real-only-artifact category, gated on the registry's own
`charge:` flag -- never on whether the caller merely asked."""

import json
import pathlib
import subprocess
import sys

import yaml

from flame.mode.horizontal.sim_charge_registry import get_profiled_charge_s

_YAML = """
redispatch_turnaround:
  weights:
    charge: true
    mean_s: 0.4365
    n: 3264
  var_bad:
    charge: false
    mean_s: 0.0201
    n: 7456
"""


def _write(tmp_path, text=_YAML):
    p = tmp_path / "registry.yaml"
    p.write_text(text)
    return str(p)


def test_returns_mean_when_charge_true(tmp_path):
    path = _write(tmp_path)
    assert get_profiled_charge_s(path, "redispatch_turnaround", "weights") == 0.4365


def test_returns_none_when_charge_false(tmp_path):
    path = _write(tmp_path)
    assert get_profiled_charge_s(path, "redispatch_turnaround", "var_bad") is None


def test_returns_none_for_unknown_label(tmp_path):
    path = _write(tmp_path)
    assert get_profiled_charge_s(path, "nonexistent", "weights") is None


def test_returns_none_for_unknown_payload_kind(tmp_path):
    path = _write(tmp_path)
    assert get_profiled_charge_s(path, "redispatch_turnaround", "other") is None


def test_none_path_is_a_noop():
    assert get_profiled_charge_s(None, "redispatch_turnaround", "weights") is None


def test_missing_file_is_a_noop_not_a_crash(tmp_path):
    assert get_profiled_charge_s(str(tmp_path / "nope.yaml"), "x", "y") is None


def test_default_payload_kind_key(tmp_path):
    path = _write(tmp_path, "drain_tail:\n  _default:\n    charge: true\n    mean_s: 0.33\n")
    assert get_profiled_charge_s(path, "drain_tail") == 0.33


_SHARED_COMPUTE_YAML = """
drain_tail:
  _default:
    charge: true
    mean_s: 0.2783
    n: 3654
fedavg:
  _default:
    charge: true
    mean_s: 0.0645
    n: 3654
"""


def test_shared_compute_categories_resolve_a_profiled_charge(tmp_path):
    """§D-18: drain_tail/fedavg were charged from sim's OWN live span on the
    theory that "sim runs the identical op, so its span IS the cost". At n=100
    that span is contention-inflated (fluxtune: sim 0.585 vs real 0.330 s/cycle),
    so the live path folds sim-host noise onto the vclock (§F-1/§F-20). They are
    profiled from real now, like any other charged category."""
    path = _write(tmp_path, _SHARED_COMPUTE_YAML)
    assert get_profiled_charge_s(path, "drain_tail") == 0.2783
    assert get_profiled_charge_s(path, "fedavg") == 0.0645


def test_shared_compute_charge_is_inert_without_a_registry(tmp_path):
    """A baseline that never opted in must keep the old live-span behavior --
    the call site passes profiled_s=None and falls back."""
    assert get_profiled_charge_s(None, "drain_tail") is None
    assert get_profiled_charge_s(_write(tmp_path), "drain_tail") is None


class TestPerBaselineProfileGeneration:
    """`profile_sim_charges.py --only-observed`: a PER-BASELINE profile must not
    carry a number for an op that baseline never performs. Without the flag the
    seed's entry survives the refresh and is re-stamped with the new run's
    provenance -- sync fwdllm inherited `redispatch_turnaround` (an async-only
    dispatch cost) that way, claiming a value it never measured."""

    @staticmethod
    def _run_dir(tmp_path, name, labels):
        d = tmp_path / name / "telemetry"
        d.mkdir(parents=True)
        rows = [json.dumps({"event": "vclock_charge", "time_mode": "real",
                            "label": lbl, "span_s": span})
                for lbl, span in labels for _ in range(8)]
        (d / "aggregator_x.jsonl").write_text("\n".join(rows))
        return str(tmp_path / name)

    def _profile(self, tmp_path, run, seed, extra_args=()):
        out = tmp_path / "profile.yaml"
        out.write_text(seed)
        script = (pathlib.Path(__file__).resolve().parents[2]
                  / "examples/fwdllm/expt_scripts/profile_sim_charges.py")
        subprocess.run([sys.executable, str(script), "--real-run", run,
                        "--out", str(out), *extra_args], check=True,
                       capture_output=True)
        return yaml.safe_load(out.read_text())

    _SEED = ("drain_tail:\n  _default:\n    charge: true\n    rationale: keep me\n"
             "redispatch_turnaround:\n  weights:\n    charge: true\n    rationale: async only\n")

    def test_only_observed_drops_an_op_this_baseline_never_ran(self, tmp_path):
        run = self._run_dir(tmp_path, "run_x_fwdllm_n100_real", [("drain_tail", 0.10)])
        got = self._profile(tmp_path, run, self._SEED, ("--only-observed",))
        assert "redispatch_turnaround" not in got
        assert got["drain_tail"]["_default"]["mean_s"] == 0.10

    def test_without_the_flag_the_unobserved_entry_survives(self, tmp_path):
        run = self._run_dir(tmp_path, "run_x_fwdllm_n100_real", [("drain_tail", 0.10)])
        got = self._profile(tmp_path, run, self._SEED)
        assert "redispatch_turnaround" in got

    def test_charge_flag_and_rationale_survive_a_refresh(self, tmp_path):
        """The numbers are the script's to write; `charge:` is a human decision
        it must never flip on its own."""
        run = self._run_dir(tmp_path, "run_x_fwdllm_n100_real", [("drain_tail", 0.10)])
        got = self._profile(tmp_path, run, self._SEED, ("--only-observed",))
        assert got["drain_tail"]["_default"]["charge"] is True
        assert got["drain_tail"]["_default"]["rationale"] == "keep me"

    def test_provenance_records_the_source_run(self, tmp_path):
        run = self._run_dir(tmp_path, "run_x_fwdllm_n100_real", [("drain_tail", 0.10)])
        got = self._profile(tmp_path, run, self._SEED, ("--only-observed",))
        e = got["drain_tail"]["_default"]
        assert e["source_runs"] == ["run_x_fwdllm_n100_real"]
        assert e["profiled_at"]
