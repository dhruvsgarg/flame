def compute_modeled_delay_s(computation_time_ms, satellite_latencies, satellite_index, elapsed_s, enabled):
    if not enabled:
        return 0.0, 0.0, 0
    timestep = min(int(elapsed_s), satellite_latencies.shape[0] - 1)
    current_rtt_ms = float(satellite_latencies[timestep, satellite_index]) * 2
    modeled_delay_s = (computation_time_ms + current_rtt_ms) / 1000
    return modeled_delay_s, current_rtt_ms, timestep