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
    root = _pool(tmp_path, "P2", [("syn_50_3st", "felix", "PASS", "FAIL:EV10"), ("syn_50_3st", "oort", "FAIL:EV2", "PASS"),
                                  ("syn_50_3st", "refl", "PASS", "PASS"), ("syn_50_3st", "fedbuff", "FAIL:EV14", "PASS")])
    s = _status(lad.grade_pool(root, -1))
    assert s[("P2", "felix")][0] == "red"  # FX-N38 closed: syn_50_3st EV10 is a regression again
    assert s[("P2", "oort")][0] == "red" and s[("P2", "refl")][0] == "green"
    assert s[("P2", "fedbuff")][0] == "red"  # FX-N15 closed
    known = _status(lad.grade_pool(_pool(tmp_path / "k", "gs_P7o", [("syn_0", "felix", "PASS", "FAIL:EV12")]), -1))
    assert known[("gs_P7o", "felix")] == ("known", "sim EV12 (FX-N30)")


def test_injected_bug_must_fail_its_check(tmp_path):
    root = _pool(tmp_path, "P11b", [("syn_50_3st", "felix", "PASS", "FAIL:EV10,EV16")])
    assert _status(lad.grade_pool(root, -1))[("P11b", "felix")][0] == "green"
    root2 = _pool(tmp_path / "x", "P11b", [("syn_50_3st", "felix", "PASS", "PASS")])
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
    # Run 17: T3C read "sim MISSING"; G1AS (sim-only) grades its sim leg alone.
    for ph in ("T3C_syn_50_3st", "G0UC_syn_50_3st", "gs_G2C_syn_0"):
        root = _pool(tmp_path / ph, ph, [("syn_50_3st", "refl", "PASS", "MISSING")])
        assert _status(lad.grade_pool(root, 99))[(ph, "refl")] == ("green", "")
    root = _pool(tmp_path / "s", "G1AS", [("syn_0", "refl", "MISSING", "PASS"), ("syn_0", "fedbuff", "-", "FAIL:EV14")])
    s = _status(lad.grade_pool(root, 99))
    assert s[("G1AS", "refl")] == ("green", "") and s[("G1AS", "fedbuff")] == ("red", "sim EV14")


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


def test_known_parity_fail_is_known(tmp_path):
    """FX-N62: a parity fail matching a KNOWN row is known, not red."""
    root = _pool(tmp_path, "T3_syn_50_3st", [("syn_50_3st", "oort", "PASS", "PASS")],
                 parity={("syn_50_3st", "oort"): {"overhead_residual": {"ok": False, "tier": "EXACT"}}})
    assert _status(lad.grade_pool(root, 1))[("T3_syn_50_3st", "oort")][0] == "known"


def test_floors_take_every_same_code_replicate(tmp_path, monkeypatch):
    # FX-D159: three reals give three pairs; the floor is their max spread, not the last pool's.
    exp = tmp_path / "experiments"
    spreads = {("r0", "r1"): 0.02, ("r0", "r2"): 0.0, ("r1", "r2"): 0.07}
    for pool, rep in (("pool_A", "r1"), ("pool_B", "r2")):
        (exp / pool).mkdir(parents=True)
        (exp / pool / "code_key.txt").write_text("k1")
        _pool(exp / pool, "gs_G0UC_syn_50_3st", [("syn_50_3st", "feddance", "PASS", "-")])
        with open(exp / pool / "gs_G0UC_syn_50_3st" / "summary.tsv", "w") as fh:
            fh.write("trace\tbaseline\treal_dir\nsyn_50\tfeddance\t" + rep + "\n")
    g = exp / "pool_A" / "gs_G0U_syn_50_3st" / "gs_G0U_syn_50_3st_feddance_grade"
    g.mkdir(parents=True)
    (g / "cmd.txt").write_text("x --real-dir r0 --sim-dir s0 --agg-goal 5 --runtime-s 1800")

    def run(cmd, **_):
        a, b, out = cmd[cmd.index("--real") + 1], cmd[cmd.index("--sim") + 1], cmd[cmd.index("--json-out") + 1]
        Path(out).write_text(json.dumps({"terminal_state": {"trainers_rel_diff": spreads[(a, b)]}}))
    monkeypatch.setattr(lad.subprocess, "run", run)
    floors = lad._control_floors(exp / "pool_A", Path("checker"), 2)
    assert floors[("gs_G0U_syn_50_3st", "feddance")] == {"trainers_at_n": 0.07}


def test_g3c_replicate_floors_its_g3_cell_and_syn_50(tmp_path, monkeypatch):
    # FX-D181: a G3C real (another same-code pool) floors its G3 cell; syn_50_3st takes the syn_0 band (operator 10-15).
    exp = tmp_path / "experiments"
    for pool in ("pool_A", "pool_B"):
        (exp / pool).mkdir(parents=True)
        (exp / pool / "code_key.txt").write_text("k1")
    for ph in ("G3C_syn_0", "G3C_lin_syn_0"):
        (exp / "pool_B" / ph).mkdir()
        (exp / "pool_B" / ph / "summary.tsv").write_text(f"trace\tbaseline\treal_dir\nsyn_0\toort\tr_{ph}\n")
    for ph in ("G3_syn_0", "G3_lin_syn_0"):
        g = exp / "pool_A" / ph / f"{ph}_oort_grade"
        g.mkdir(parents=True)
        (g / "cmd.txt").write_text(f"x --real-dir own_{ph} --sim-dir s --agg-goal 10 --runtime-s 7200")

    def run(cmd, **_):
        Path(cmd[cmd.index("--json-out") + 1]).write_text(json.dumps({"terminal_state": {"trainers_rel_diff": 0.04}}))
    monkeypatch.setattr(lad.subprocess, "run", run)
    floors = lad._control_floors(exp / "pool_A", Path("checker"), 2)
    assert {k for k in floors} == {("G3_syn_0", "oort"), ("G3_lin_syn_0", "oort"), ("G3_syn_50_3st", "oort")}
    assert lad._replicate_cells(exp / "pool_A")[("G3_lin_syn_0", "oort")][2][0][0] == Path("r_G3C_lin_syn_0")


def test_syn_50_band_from_a_sibling_pools_syn_0_cell(tmp_path, monkeypatch):
    # PR38: the syn_0 cell + G3C ran on shepherd, syn_50_3st on jayne; jayne's syn_50_3st still takes the syn_0 band.
    exp = tmp_path / "experiments"
    for pool in ("pool_J", "pool_S", "pool_SR"):
        (exp / pool).mkdir(parents=True)
        (exp / pool / "code_key.txt").write_text("k1")
    (exp / "pool_SR" / "G3C_syn_0").mkdir()
    (exp / "pool_SR" / "G3C_syn_0" / "summary.tsv").write_text("trace\tbaseline\treal_dir\nsyn_0\toort\tr2\n")
    for pool, ph in (("pool_S", "G3_syn_0"), ("pool_J", "G3_syn_50_3st")):
        g = exp / pool / ph / f"{ph}_oort_grade"
        g.mkdir(parents=True)
        (g / "cmd.txt").write_text(f"x --real-dir own_{ph} --sim-dir s --agg-goal 10 --runtime-s 7200")

    def run(cmd, **_):
        Path(cmd[cmd.index("--json-out") + 1]).write_text(json.dumps({"terminal_state": {"trainers_rel_diff": 0.04}}))
    monkeypatch.setattr(lad.subprocess, "run", run)
    assert ("G3_syn_50_3st", "oort") in lad._control_floors(exp / "pool_J", Path("checker"), 2)


def test_minority_of_reals_is_the_draw(tmp_path):
    # L13, PR34 speech feddance eve: sim red vs its own real only (1 of 3 reals) is the draw; vs a majority, the code.
    red = {"terminal_state": {"ok": False, "tier": "EXACT"}}
    root = _pool(tmp_path, "gs_G0T_eve_syn_50_3st", [("syn_50_3st", "feddance", "PASS", "PASS")])
    p = root / "gs_G0T_eve_syn_50_3st" / "gs_G0T_eve_syn_50_3st_feddance_grade" / "parity_regrade"
    p.mkdir(parents=True)
    for name, checks in (("syn_50_3st_feddance.json", red), ("syn_50_3st_feddance.rep1.json", {}), ("syn_50_3st_feddance.rep2.json", {})):
        (p / name).write_text(json.dumps({**checks, "summary": {}}))
    assert _status(lad.grade_pool(root, 9))[("gs_G0T_eve_syn_50_3st", "feddance")] == ("green", "draw S8 terminal_state 1/3")
    (p / "syn_50_3st_feddance.rep2.json").write_text(json.dumps({**red, "summary": {}}))
    assert _status(lad.grade_pool(root, 9))[("gs_G0T_eve_syn_50_3st", "feddance")][0] == "red"
    (p / "syn_50_3st_feddance.rep2.json").unlink()  # n=2: no vote, any fail is red (L12)
    assert _status(lad.grade_pool(root, 9))[("gs_G0T_eve_syn_50_3st", "feddance")][0] == "red"


def test_reals_pair_across_a_real_path_identical_key(tmp_path, monkeypatch):
    # PR37 -> PR38 changed only sim paths for oort: PR37's oort real votes on PR38's cell; Felix's (lr changed) does not.
    exp = tmp_path / "experiments"
    for pool, key in (("pool_37", "1e6004b6c305"), ("pool_38", "44d44bd2d67b")):
        (exp / pool).mkdir(parents=True)
        (exp / pool / "code_key.txt").write_text(key)
    real37 = tmp_path / "r37"
    real37.mkdir()
    (exp / "pool_37" / "G3_syn_50_3st").mkdir()
    (exp / "pool_37" / "G3_syn_50_3st" / "summary.tsv").write_text(
        f"trace\tbaseline\treal_dir\nsyn_50_3st\toort\t{real37}\nsyn_50_3st\tfelix\t{real37}\n")
    for b in ("oort", "felix"):
        g = exp / "pool_38" / "G3_syn_50_3st" / f"G3_syn_50_3st_{b}_grade"
        g.mkdir(parents=True)
        (g / "cmd.txt").write_text(f"x --real-dir own --sim-dir s --baselines {b} --agg-goal 10 --runtime-s 7200")
    cells = lad._replicate_cells(exp / "pool_38")
    assert [r for r, _ in cells[("G3_syn_50_3st", "oort")][2]] == [real37]
    assert ("G3_syn_50_3st", "felix") not in cells
