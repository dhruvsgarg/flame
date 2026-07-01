from timing_model import compute_modeled_delay_s
import numpy as np

def test_disabled_returns_zero():
    delay, rtt, timestep = compute_modeled_delay_s(
        computation_time_ms=3000.0,
        satellite_latencies=None,   # never touched when enabled=False
        satellite_index=0,
        elapsed_s=100.0,
        enabled=False,
    )
    assert delay == 0.0
    assert rtt == 0.0
    assert timestep == 0

def test_known_values_at_timestep_zero():
    fake_latencies = np.array([
        [10.0, 20.0],   # row 0 (timestep 0): satellite 0 -> 10ms, satellite 1 -> 20ms
        [11.0, 21.0],   # row 1 (timestep 1)
    ])

    delay, rtt, timestep = compute_modeled_delay_s(
        computation_time_ms=3000.0,
        satellite_latencies=fake_latencies,
        satellite_index=1,     # picking the "20.0" column
        elapsed_s=0.0,          # elapsed=0 -> should land on timestep 0
        enabled=True,
    )

    assert timestep == 0
    assert rtt == 40.0          # fake_latencies[0,1] = 20.0, doubled = 40.0
    assert delay == 3.04        # (3000 + 40) / 1000