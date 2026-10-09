# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""FX-D108: sim charges each recipient's fan-out delivery lag on its completion time."""

from flame.config import Hyperparameters
from flame.mode.message import MessageType


def test_delivery_lag_charge_is_on_with_a_revert_knob():
    # C9: A/B `pool_20261009_D108_T3` closed the speech K=10 throughput gap (18.7% -> 3.8%).
    assert Hyperparameters(rounds=1, epochs=1).sim_charge_delivery_lag is True
    assert Hyperparameters(rounds=1, epochs=1, simChargeDeliveryLag=False).sim_charge_delivery_lag is False


def test_wall_send_stamp_has_its_own_id():
    ids = [m.value for m in MessageType]
    assert MessageType.SIM_WALL_SEND_TS.value == 46 and ids.count(46) == 1


def test_download_leg_is_profiled_apart():
    # FX-D116: completion_leg held agg_to_trainer, which the measured lag charged again (+0.4 s per speech sync round).
    import importlib.util
    import os
    import tempfile

    path = os.path.join(os.path.dirname(__file__), "..", "..", "examples", "scripts", "profile_felix_charges.py")
    spec = importlib.util.spec_from_file_location("profile_felix_charges", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    line = ("2026-10-08 05:33:37,641 | x | INFO | M | f | [LAG_DECOMP] end=e version=1 wall_lag_s=15.389 "
            "agg_to_trainer_s=0.212 compute_s=15.000 post_wait_s=0.108 mqtt_lag_s=0.069 queue_wait_s=0.016\n")
    with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as f:
        f.write(line)
    legs, _, down = mod.leg_samples(f.name, "oort_sync")
    os.unlink(f.name)
    assert abs(legs[0] - 0.085) < 1e-9 and down == [0.212]
    hp = Hyperparameters(rounds=1, epochs=1)
    assert hp.sim_download_leg_s == 0.0 and hp.real_recv_until_awaited is True
