# LEO Satellite / Ground-Station RF Link-Budget Gate — Design & Call Flow

Trainers are reframed as LEO satellites (positions from the existing
`_metadata/leo/ecef.npz` constellation) and the aggregator as a fixed ground
station. The ground station computes free-space + atmospheric + ionospheric
path loss to each satellite, derives a throughput-loss fraction, and drops
any update (and withholds the next round's weights) from a satellite whose
modeled loss exceeds a configurable threshold.

Status: **implemented**. Off by default — every existing baseline/parity run
is unaffected unless a config explicitly sets `hyperparameters.link_budget.enabled`.

## Requirements this satisfies

1. Trainers simulated as satellites orbiting per constellation ephemeris.
2. Aggregator = ground station.
3. Roles/functionality of Trainer and Aggregator otherwise unchanged.
4. All satellites share one downlink channel/bandwidth (SpaceX FCC-filing figure).
5. Ground station computes FSPL + atmospheric loss + ionospheric loss.
6. Both ground station and satellite independently compute throughput loss.
7. Aggregator drops an update whose throughput loss exceeds threshold, excludes
   it from the aggregate, and withholds the next round's weights from it.

## What already existed (reused, not rebuilt)

- `lib/python/examples/_metadata/leo/ecef.npz` — 300-satellite constellation,
  ECEF position in km, `[10800 timesteps, 300 sats, 3]`, 1 s resolution.
- `trainer/pytorch/main.py` already loaded this constellation's `geodetic.npz`
  sibling for a location-logging line (`self.coords`/`self.satellite_index`) —
  the "trainer = satellite" framing was already half-wired.
- `flame/launch/spawner.py` already assigns `hyperparameters.satellite_index =
  trainer_id - 1` per trainer, and `trainer_registry.yaml` maps `trainer_id ->
  task_id` (the hex string a channel `end` id actually equals — an `end` is
  **not** the integer trainer_id, it's the MQTT task_id, so a registry lookup
  is required to go from `end` back to a satellite row).

## SpaceX bandwidth (FCC filing)

Starlink's Ku-band user downlink is 8 channels of 240 MHz each (~1.92–2 GHz
aggregate across the constellation; one filing rounds this to 8×250 MHz),
uplink 8×62.5 MHz (500 MHz aggregate, a declared 4:1 downlink:uplink split).
Downlink band: 10.7–12.7 GHz. Used here: **`channel_bandwidth_hz = 240 MHz`**
per link, **`downlink_freq_mhz = 12000`** (center of the Ku downlink band).

Sources: [FCC DA 24-222](https://docs.fcc.gov/public/attachments/DA-24-222A1.pdf),
[Modeling Starlink capacity — Mike Puchol](https://mikepuchol.com/modeling-starlink-capacity-843b2387f501),
[Starlink Ku-Band Downlink signal structure — UT Austin RadioNavLab](https://radionavlab.ae.utexas.edu/wp-content/uploads/starlink_structure.pdf).

## Physics model

`flame/availability/rf_link_budget.py` (pure functions, no I/O):

```
fspl_db(distance_km, freq_mhz)        = 20*log10(d_km) + 20*log10(f_mhz) + 32.44
atmospheric_loss_db(elevation_deg)    = zenith_db / sin(max(elevation_deg, 5deg))   # ITU-R P.676 cosecant approx
ionospheric_loss_db(elevation_deg, freq_mhz, tec_tecu)
                                       = 40.3*TEC/(f_mhz*1e6)^2, dB-scaled, / sin(elevation)  # negligible at Ku-band, kept for lower-band reuse
total_path_loss_db                    = fspl + atmospheric + ionospheric
shannon_throughput_mbps(snr_db, bw_hz) = bw_hz * log2(1 + 10^(snr_db/10)) / 1e6
throughput_loss_fraction(extra_loss_db, ref_snr_db)
                                       = 1 - log2(1+SNR)/log2(1+SNR_ref), SNR = SNR_ref / 10^(extra_loss_db/10)
```

`extra_loss_db` is the candidate link's `total_path_loss_db` minus the loss at
a configured reference condition (`ref_distance_km`/`ref_elevation_deg`) —
this is what makes the loss *fraction* relative to a realistic operating
point rather than to zero-loss free space.

### `ref_snr_db` derivation

Default **20.0 dB**. Independent RF analysis of live Starlink Ku-band
downlinks measured the assigned beam's SNR at **≈21 dB clear-sky**
([UT Austin RadioNavLab](https://radionavlab.ae.utexas.edu/wp-content/uploads/starlink_structure.pdf)).
20 dB is a clean round number matching that measurement, paired with a
reference geometry of 550 km range / 40° elevation (Starlink's nominal shell
altitude, a middling elevation) — so "extra loss" is measured against a
realistic well-pointed link, not an idealized one. Fully configurable via
`hyperparameters.link_budget.ref_snr_db` if you have a better-calibrated
figure (e.g. from a real EIRP/G-T budget).

### Known consequence — visibility scarcity

Verified against the repo's own `ecef.npz`: with **one** fixed ground station,
only **~2–3%** of the 300 satellites are above a 5° elevation mask at any
instant (matches the pre-existing `latency.npz`'s `direct_visible ≈ 1.96%`
average almost exactly — an independent cross-check that the geometry here is
correct). At the default `throughput_loss_threshold=0.5` this leaves roughly
2–4 trainers eligible per round — a severe bottleneck for async FL
throughput. Physically correct for a single gateway; raise
`throughput_loss_threshold`, lower `min_elevation_deg`, or add multiple
ground stations before running real experiments.

## Config reference

Set under `hyperparameters` (trainer `configs/trainer_base.yaml`, mirrored in
aggregator `_metadata/aggregator_base.json` — keep both in sync):

```yaml
ground_station:
  lat_deg: 47.6062
  lon_deg: -122.3321
  alt_m: 50.0
link_budget:
  enabled: "False"                 # master gate; off = zero behavior change
  downlink_freq_mhz: 12000.0
  channel_bandwidth_hz: 240000000  # SpaceX FCC filing, 8x240 MHz Ku channels
  ref_snr_db: 20.0
  ref_distance_km: 550.0
  ref_elevation_deg: 40.0
  atmo_zenith_db: 0.2
  tec_tecu: 50.0
  throughput_loss_threshold: 0.5   # drop if modeled loss exceeds this fraction
  min_elevation_deg: 5.0           # below this, satellite isn't visible at all
  ecef_path: null                  # override; default <geodetic dir>/ecef.npz
  registry_path: null              # aggregator only; default _metadata/trainer_registry.yaml
```

## New files

| File | Purpose |
|---|---|
| `flame/availability/rf_link_budget.py` | Library-level, example-agnostic physics + `LinkBudgetConfig` |
| `examples/async_cifar10/leo/ground_station.py` | `GroundStation` (lat/lon/alt -> ECEF), `SatelliteLinkGate` (evaluates a satellite's link at any timeline second), `end`->satellite_index registry lookup |

## Modified files — two small library hooks, default no-op

| File | Change |
|---|---|
| `flame/mode/horizontal/asyncfl/top_aggregator.py` | `_extra_ineligible_trainers()` (distribute-time gate), `_extra_drop_update()` (aggregate-time gate); both default to "nothing excluded" |
| `flame/mode/horizontal/syncfl/trainer.py` | `_extra_send_fields()`, merged into the upload message; default `{}` |
| `flame/mode/message.py` | `MessageType.LINK_THROUGHPUT_LOSS_SELF` |
| `examples/async_cifar10/aggregator/pytorch/main_asyncfl_agg.py` | `_init_link_gate()` + hook overrides |
| `examples/async_cifar10/trainer/pytorch/main.py` | `_init_link_gate()`, `_compute_own_throughput_loss()`, `_extra_send_fields()` override |
| `examples/async_cifar10/configs/trainer_base.yaml`, `examples/_metadata/aggregator_base.json` | new `ground_station`/`link_budget` blocks, disabled by default |

## Call flow

Every existing baseline's tasklet chain is unchanged (`distribute ->
aggregate` inside the asyncfl loop — see `TopAggregator.compose()` in
`flame/mode/horizontal/asyncfl/top_aggregator.py`). The two new hooks slot
into the existing `_distribute_weights`/`_aggregate_weights` methods at the
points marked `[NEW]` below.

```mermaid
sequenceDiagram
    participant Agg as Ground Station<br/>(TopAggregator)
    participant LBA as SatelliteLinkGate<br/>[NEW]
    participant RF as rf_link_budget<br/>[NEW]
    participant Chan as Channel (MQTT)
    participant Sat as Satellite<br/>(Trainer)
    participant Opt as FedBuff Optimizer

    loop every round
        rect rgb(240, 248, 255)
        note over Agg,Chan: Distribute
        Agg->>LBA: _extra_ineligible_trainers(channel, task) [NEW]
        LBA->>RF: total_path_loss_db(range_km, elevation_deg, freq_mhz)
        RF-->>LBA: fspl_db + atmo_db + iono_db
        LBA->>RF: throughput_loss_fraction(extra_loss_db, ref_snr_db)
        RF-->>LBA: loss_frac
        note right of LBA: loss_frac > threshold => ineligible<br/>(recomputed LIVE from orbital position,<br/>no static trace file)
        LBA-->>Agg: ineligible_end_ids
        Agg->>Chan: ends(VAL_CH_STATE_SEND, task) [selector picks from eligible only]
        Chan-->>Agg: selected_ends
        Agg->>Sat: send({WEIGHTS, MODEL_VERSION, TASK_TO_PERFORM})
        note left of Agg: satellites below threshold never<br/>receive this round's global weights (req 7)
        end

        rect rgb(255, 250, 240)
        note over Sat: Local training + self-check
        Sat->>Sat: _fetch_weights() -> load_state_dict(weights)
        Sat->>RF: _compute_own_throughput_loss() [NEW, req 6]
        RF-->>Sat: loss_frac_self
        Sat->>Sat: train() [local SGD]
        Sat->>Chan: send({WEIGHTS_BYTES, MODEL_VERSION, LINK_THROUGHPUT_LOSS_SELF})
        end

        rect rgb(245, 255, 245)
        note over Agg,Opt: Aggregate
        Agg->>Chan: recv_fifo(recv_ends, 1)
        Chan-->>Agg: msg, end
        Agg->>RF: _extra_drop_update(msg, end) [NEW, authoritative recheck @ arrival time]
        alt loss_frac_authoritative > threshold
            Agg->>Agg: DROP (cleanup_provided_ends(end); return)
            note right of Agg: optimizer.do() NOT called;<br/>agg_goal_cnt NOT incremented;<br/>telemetry: link_loss_dropped
        else within threshold
            Agg->>Opt: do(agg_goal_weights, cache, version, ...)
            Opt-->>Agg: rate-weighted delta buffered
            Agg->>Agg: agg_goal_cnt += 1
        end
        opt agg_goal_cnt == agg_goal
            Agg->>Opt: scale_add_agg_weights(weights, agg_goal_weights, agg_goal)
            Opt-->>Agg: updated global weights
            Agg->>Agg: _update_model(); round += 1
        end
        end
    end

    note over Agg,Sat: Next round's distribute re-queries the gate live -<br/>a dropped satellite is automatically re-excluded while<br/>still below threshold, and automatically re-eligible once<br/>geometry improves. No separate withhold-ledger needed.
```

## Verification performed

- All modified/new files byte-compile clean.
- `SatelliteLinkGate` run against the real `ecef.npz` + `trainer_registry.yaml`:
  geometry (range/elevation) cross-checked against the pre-existing
  `latency.npz`'s `direct_visible` statistic (independent data, same
  constellation) — matched within ~0.3 percentage points.
  - Visible satellite at t=0: elevation 29.5°, range 918 km, path loss 174.8 dB,
    throughput-loss fraction 0.236, effective throughput 1221 Mbps (of a
    1598 Mbps reference cap at 20 dB/240 MHz) — eligible.
  - Below-horizon satellite: elevation -55.7°, correctly excluded
    (`reason=below_horizon`).
- Aggregator/trainer hooks instantiated directly (bypassing the full gRPC/MQTT
  stack) and confirmed:
  - `enabled: true` -> eligibility/drop decisions match the standalone gate.
  - `enabled: false` (default) -> `_extra_ineligible_trainers` always `[]`,
    `_extra_drop_update` always `False`, `_extra_send_fields` always `{}` —
    zero behavior change for every existing baseline/parity run.
