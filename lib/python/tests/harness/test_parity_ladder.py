# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""FX-N42: the parity ladder gates a pool rung by rung; known items never block, lower stages gate first."""

import importlib.util
import json
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "examples" / "scripts"
spec = importlib.util.spec_from_file_location("parity_ladder", SCRIPTS / "parity_ladder.py")
lad = importlib.util.module_from_spec(spec)
sys.modules["parity_ladder"] = lad
spec.loader.exec_module(lad)

HDR = "trace\tbaseline\tev_real\tev_sim\tparity\n"


def _pool(tmp_path, phase, rows, parity=None):
    d = tmp_path / phase
    d.mkdir(parents=True, exist_ok=True)
    (d / "summary.tsv").write_text(HDR + "".join("\t".join(r) + "\n" for r in rows))
    for (tr, b), checks in (parity or {}).items():
        p = d / f"{phase}_{b}_grade" / "parity"
        p.mkdir(parents=True)
        (p / f"{tr}_{b}.json").write_text(json.dumps({**checks, "summary": {}}))
    return tmp_path


def _status(cells):
    return {(c.phase, c.baseline): (c.status, c.why) for c in cells}


def test_ev_gate_known_and_red(tmp_path):
    root = _pool(tmp_path, "P2", [("syn_50", "felix", "PASS", "FAIL:EV10"), ("syn_50", "oort", "FAIL:EV2", "PASS"),
                                  ("syn_50", "refl", "PASS", "PASS"), ("syn_50", "fedbuff", "FAIL:EV14", "PASS")])
    s = _status(lad.grade_pool(root, -1))
    assert s[("P2", "felix")] == ("known", "sim EV10 (FX-N38)")
    assert s[("P2", "oort")][0] == "red" and s[("P2", "refl")][0] == "green"
    assert s[("P2", "fedbuff")] == ("known", "real EV14 (FX-N15)")


def test_injected_bug_must_fail_its_check(tmp_path):
    root = _pool(tmp_path, "P11b", [("syn_50", "felix", "PASS", "FAIL:EV10,EV16")])
    assert _status(lad.grade_pool(root, -1))[("P11b", "felix")][0] == "green"
    root2 = _pool(tmp_path / "x", "P11b", [("syn_50", "felix", "PASS", "PASS")])
    assert _status(lad.grade_pool(root2, -1))[("P11b", "felix")] == ("red", "sim did not FAIL EV16")


def test_parity_gates_only_inv_exact_up_to_the_rung_stage(tmp_path):
    checks = {"overhead_residual": {"ok": False, "tier": "EXACT"},           # stage 1
              "trainer_speed_identity": {"ok": False, "tier": "DIST"},       # stage 1, DIST: reported, not gated
              "participation": {"ok": False, "tier": "EXACT"}}               # stage 3
    root = _pool(tmp_path, "T3_syn_0", [("syn_0", "felix", "PASS", "PASS")], {("syn_0", "felix"): checks})
    assert _status(lad.grade_pool(root, -1))[("T3_syn_0", "felix")][0] == "green"
    assert _status(lad.grade_pool(root, 1))[("T3_syn_0", "felix")] == ("red", "S1 overhead_residual [EXACT]")
    assert "S3 participation" in _status(lad.grade_pool(root, 3))[("T3_syn_0", "felix")][1]


def test_rung_ranges():
    assert lad._rung_ids("L1-L3") == ["L1", "L2", "L3"]
    assert lad._rung_ids("L5") == ["L5"] and lad._rung_ids("L2,L4") == ["L2", "L4"]
    assert all(r.reuse in lad.BY_ID for r in lad.LADDER if r.reuse)


def test_real_only_control_is_not_sim_missing(tmp_path):
    # Run 4 L6: every G0C cell read red "sim MISSING"; G0C has no sim leg by design.
    root = _pool(tmp_path, "gs_G0C_syn_0", [("syn_0", "felix", "PASS", "MISSING"), ("syn_0", "oort", "FAIL:EV0", "MISSING")])
    s = _status(lad.grade_pool(root, 99))
    assert s[("gs_G0C_syn_0", "felix")] == ("green", "") and s[("gs_G0C_syn_0", "oort")] == ("red", "real EV0")
    root = _pool(tmp_path / "g0", "G0_syn_0", [("syn_0", "felix", "PASS", "MISSING")])
    assert _status(lad.grade_pool(root, 99))[("G0_syn_0", "felix")] == ("red", "sim MISSING")


def test_interrupt_waits_for_pool_teardown(tmp_path):
    # R20, run 4 L6: subprocess.run SIGKILLed the pool 0.25s after Ctrl+C, orphaning its GPU legs.
    import os, signal, subprocess, time
    mark = tmp_path / "torn_down"
    pool = tmp_path / "pool.py"
    pool.write_text("import signal, sys, time\n"
                    "signal.signal(signal.SIGINT, lambda *_: (time.sleep(1.5), open(sys.argv[1], 'w').close(), sys.exit(130)))\n"
                    "print('up', flush=True)\ntime.sleep(60)\n")
    drv = f"import sys; sys.path.insert(0, {str(SCRIPTS)!r}); import parity_ladder as l; " \
          f"sys.exit(l._run_pool([sys.executable, {str(pool)!r}, {str(mark)!r}]))"
    p = subprocess.Popen([sys.executable, "-c", drv], start_new_session=True, stdout=subprocess.PIPE, text=True)
    assert p.stdout.readline().strip() == "up"
    os.killpg(p.pid, signal.SIGINT)  # the terminal's Ctrl+C: the whole foreground group
    assert p.wait(timeout=30) == 130 and mark.exists()
