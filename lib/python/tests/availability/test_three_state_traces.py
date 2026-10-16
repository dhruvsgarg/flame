# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""FX-N96: syn_X_3st keeps syn_X's AVL_TRAIN timeline and relabels non-train windows with MobiPerf's 3-state structure."""

import importlib.util
import random
from collections import Counter
from pathlib import Path

import pytest

from flame.availability.trace import load_trace

_GEN = Path(__file__).resolve().parents[2] / "examples/_metadata/scripts/gen_synthetic_trace.py"
_spec = importlib.util.spec_from_file_location("gen_synthetic_trace", _GEN)
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)

LIKE = {"p_train_to_eval": 0.55, "p_eval_to_train": 0.60, "p_un_to_eval": 0.05, "eval_share_of_nontrain": 0.403}
HORIZON = 600 * 2000


def _source(n=60, level=0.5):
    rng = random.Random(1)
    return {f"t{i}": gen.trainer_events(rng, level, 1200.0, 600, HORIZON) for i in range(n)}


def test_generator_keeps_train_and_verifies():
    src = _source()
    new = gen.three_state_events(src, LIKE, "x")
    res = gen.verify_three_state(src, new, LIKE, share_tol=0.02)
    assert res["full"][gen.EVAL] == pytest.approx(0.403 * (1 - res["full"][gen.TRAIN]), abs=0.02)
    assert gen.three_state_events(src, LIKE, "x") == new  # deterministic for a seed
    assert gen.three_state_events(src, LIKE, "y") != new


def test_verify_catches_a_changed_train_interval():
    src = _source(n=10)
    new = gen.three_state_events(src, LIKE, "x")
    new["t0"][0] = [new["t0"][0][0], gen.TRAIN] if new["t0"][0][1] != gen.TRAIN else [new["t0"][0][0], gen.UN]
    with pytest.raises(AssertionError):
        gen.verify_three_state(src, new, LIKE, share_tol=0.5, exit_tol=1.0)


@pytest.mark.parametrize("level", [10, 20, 50])
def test_stored_trace_matches_mobiperf_structure(level):
    per = {f"trainer_{i:03d}": [[t, s] for t, s in load_trace(f"syn_{level}_3st", f"trainer_{i:03d}").items()]
           for i in range(1, 301)}
    like = {"p_train_to_eval": 0.55, "p_eval_to_train": 0.60}
    for t_end in (536400.0, 10800.0):
        sh = gen._shares(per, t_end)
        assert sh[gen.TRAIN] == pytest.approx(1 - level / 100, abs=0.03)
        assert sh[gen.EVAL] / (sh[gen.EVAL] + sh[gen.UN]) == pytest.approx(0.403, abs=0.03)
    ex = Counter((a, b) for ev in per.values() for (_, a), (_, b) in zip(ev, ev[1:]))
    assert ex[gen.TRAIN, gen.EVAL] / (ex[gen.TRAIN, gen.EVAL] + ex[gen.TRAIN, gen.UN]) == pytest.approx(like["p_train_to_eval"], abs=0.05)
    assert ex[gen.EVAL, gen.TRAIN] / (ex[gen.EVAL, gen.TRAIN] + ex[gen.EVAL, gen.UN]) == pytest.approx(like["p_eval_to_train"], abs=0.05)
    for ev in per.values():
        assert all(a[1] != b[1] for a, b in zip(ev, ev[1:])) and ev[-1][1] == gen.UN
