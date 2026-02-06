# REFL Integration Plan for Flame/Felix System

**Date:** February 5, 2026  
**Purpose:** Integrate REFL (Resource-Efficient Federated Learning) into Flame to enable head-to-head comparison with Felix on identical workloads.

---

## Executive Summary

This document outlines the strategy to implement REFL's availability tracking, client selection, and aggregation algorithms within Flame's abstraction framework. REFL uses **synchronous FL** with sophisticated handling of stragglers and unavailability. The integration will be modular, allowing individual components to be toggled independently.

---

## 1. Understanding REFL's Core Components

### 1.1 Availability Tracking
**REFL Implementation (`client_manager.py`):**
- Loads device availability traces from pickle files (`device_avail_file`)
- Each client has availability periods: `[(start_time, end_time), ...]`
- Key methods:
  - `isClientActive(clientId, cur_time, time_window)`: checks if client is available at a future time
  - `isAvailable(clientId, cur_time, time_window, time_slots)`: checks availability across time slots
  - `getPriority(clientId, cur_time, time_window)`: returns priority based on near-term availability (0-2)
  - `getPeriodCount(clientId, cur_time, deadline)`: counts availability periods within deadline

**Flame Current State:**
- Uses `TrainerAvailState` enum: `AVL_TRAIN`, `AVL_EVAL`, `UN_AVL`
- Trainers self-manage state transitions based on traces
- Already supports trace-based unavailability in `async_cifar10`

### 1.2 Client Selection (Oort Enhancement)
**REFL Implementation (`oort.py` + `aggregator.py`):**
- UCB-based selection with exploration/exploitation
- Statistical utility: normalized loss + temporal uncertainty
- System utility: penalizes slow clients based on `round_prefer_duration`
- Pacer mechanism: adaptively adjusts `round_threshold` to control client speed filtering
- Blacklisting: excludes clients selected too frequently
- Priority-based selection (`args.avail_priority`):
  - 0: No priority
  - 1: Fill remaining slots from non-priority clients
  - 2: Only select high-priority clients first

**Key Selection Parameters:**
- `exploration_factor`: Initially 0.9, decays by `exploration_decay` (0.98)
- `exploration_min`: Floor at 0.2
- `round_threshold`: Controls speed filtering (default 30%, adaptive)
- `alpha`: Weight for staleness in utility (default 2)
- `clip_bound`: Caps utility at 95th percentile
- `cut_off_util`: Prunes low-utility clients (95% of cutoff)

**Flame Current State:**
- Has basic Oort selector (`flame/selector/oort.py`)
- Synchronized FL support in `syncfl/top_aggregator.py`
- Needs enhancements for:
  - Priority-based selection using availability
  - Pacer mechanism
  - Blacklisting per REFL

### 1.3 Aggregation with Staleness Handling
**REFL Implementation (`aggregator.py`):**
- Tracks stale updates in `self.staleWeights[clientId]`
- Applies **deadline filtering** (`exp_type=0` or `exp_type=2`):
  - Fixed deadline: `args.deadline`
  - Moving average deadline: `mov_avg_deadline`
  - Clients exceeding deadline become stragglers, updates cached as "stale"
- Stale update lifecycle:
  1. Client times out → update stored in `staleWeights[clientId]`
  2. Each round: `staleRemainDuration[clientId]` decrements by `round_duration`
  3. When `staleRemainDuration <= 0` and `stale_rounds <= args.stale_update`: apply update
  4. If `stale_rounds > args.stale_update`: discard (too stale)

**Stale Weighting Strategies (`args.stale_factor`):**
- `> 1`: Divide by constant factor
- `1`: Equal weight (baseline FedAvg)
- `-1`: Divide by average staleness across all stale updates
- `-2`: AdaSGD - divide by `(staleness + 1)`
- `-3`: DynSGD - multiply by `exp(-(staleness + 1))`
- `-4`: **REFL method** - hybrid formula:
  ```python
  weight *= (1 - beta) / (staleness + 1) + beta * (1 - exp(-client_ratio / max_ratio) / scale_coff)
  ```
  - `beta` (`args.stale_beta`): balance between staleness and utility
  - `client_ratio`: importance based on dataset size or loss
  - `scale_coff`: scaling coefficient (default 10.0)

**Aggregation Formula:**
```python
global_model += client_weight * client_importance * update
```
Where:
- `client_weight`: normalized by dataset size
- `client_importance`: adjusted by staleness strategy
- Normalize after aggregation to maintain model scale

**Flame Current State:**
- `FedAvg` (`flame/optimizer/fedavg.py`): Simple weighted averaging
- `FedBuff` (`flame/optimizer/fedbuff.py`): Asynchronous aggregation with staleness
- FedBuff already has staleness handling with `alpha_polynomial`, `alpha_exponential`, etc.
- Needs: REFL-specific staleness strategy and deadline-based filtering

### 1.4 Experimental Configurations (`exp_type`)
REFL uses `exp_type` to control aggregation behavior:
- **0**: Deadline + target ratio (SAFA baseline)
- **1**: No deadline, wait for all selected clients
- **2**: Overcommitment with deadline
- **3**: Overcommitment without deadline

For Flame integration, we'll focus on **exp_type=0** and **exp_type=2** (deadline-based) since these align with REFL's core contribution.

---

## 2. Integration Strategy

### 2.1 Modular Design Principles
1. **Component Independence**: Each REFL feature toggleable via config
2. **Backward Compatibility**: Existing Flame experiments unaffected
3. **Minimal Breaking Changes**: Leverage Flame's abstractions
4. **Approximation Where Necessary**: Document deviations from REFL's exact behavior

### 2.2 Three-Tier Implementation

#### Tier 1: Availability Tracking (REFL-Compatible)
**Goal:** Enable REFL-style availability priority and deadline filtering.

**Implementation:**
1. **Extend Availability Traces:**
   - Current: Trainers use trace files with state transitions
   - Add: Compute availability periods from traces at aggregator side
   - Location: `lib/python/flame/mode/horizontal/syncfl/top_aggregator.py`

2. **Aggregator-Side Availability Manager:**
   ```python
   class REFLAvailabilityTracker:
       def __init__(self, trace_file_path):
           self.traces = self.load_traces(trace_file_path)
           self.availability_periods = self.compute_availability_periods()
       
       def is_available(self, trainer_id, cur_time, time_window):
           # Check if trainer available in [cur_time, cur_time + time_window]
       
       def get_priority(self, trainer_id, cur_time, time_window):
           # Return 0-2 based on near-term availability
       
       def compute_availability_periods(self):
           # Convert traces to [(start, end), ...] per trainer
   ```

3. **Config Extension:**
   ```yaml
   refl:
     enabled: true
     availability_trace_file: "path/to/traces.pkl"
     use_priority_selection: true  # Enable priority-based selection
     avail_priority: 2  # 0=none, 1=fill, 2=strict
     avail_probability: 1.0  # Accuracy of predictions (0-1)
   ```

**Flame Approximation:**
- REFL's traces are centralized at aggregator, but Flame trainers self-manage state
- **Solution:** Aggregator loads same trace file and mirrors trainer state predictions
- **Trade-off:** Slight desync possible, but acceptable for fair comparison

#### Tier 2: REFL-Enhanced Oort Selector
**Goal:** Implement REFL's priority-based selection, pacer, and blacklisting.

**Implementation:**
1. **Create `REFLOortSelector` (extends `OortSelector`):**
   - Location: `lib/python/flame/selector/refl_oort.py`
   - Inherits from `flame/selector/oort.py`
   - Adds:
     - **Priority selection logic** (uses `REFLAvailabilityTracker`)
     - **Pacer mechanism** for adaptive `round_threshold`
     - **Blacklisting** based on selection frequency

2. **Key Methods:**
   ```python
   class REFLOortSelector(OortSelector):
       def __init__(self, **kwargs):
           super().__init__(**kwargs)
           self.avail_tracker = REFLAvailabilityTracker(kwargs['trace_file'])
           self.blacklist_rounds = kwargs.get('blacklist_rounds', -1)
           self.blacklist_max_len = kwargs.get('blacklist_max_len', 0.3)
           self.pacer_step = kwargs.get('pacer_step', 20)
           self.pacer_delta = kwargs.get('pacer_delta', 5)
       
       def select(self, ends, channel_props, trainer_unavail_list, task, **kwargs):
           # 1. Build priority lists using avail_tracker
           priority_ends, remaining_ends = self.build_priority_lists(ends)
           
           # 2. Apply blacklist filter
           available_ends = self.filter_blacklist(priority_ends + remaining_ends)
           
           # 3. Run Oort UCB selection on available_ends
           selected = super()._select_with_ucb(available_ends, ...)
           
           # 4. Fill from priority first if avail_priority >= 1
           if self.avail_priority >= 1:
               selected = self.fill_priority_first(priority_ends, selected)
           
           # 5. Run pacer to adjust round_threshold
           self.pacer()
           
           return selected
       
       def pacer(self):
           # Adaptive adjustment of round_threshold based on utility trends
           if self.round % self.pacer_step == 0:
               util_change = self.compute_utility_change()
               if abs(util_change) < 0.1:
                   self.round_threshold = min(100, self.round_threshold + self.pacer_delta)
               elif abs(util_change) > 5.0:
                   self.round_threshold = max(self.pacer_delta, self.round_threshold - self.pacer_delta)
       
       def get_blacklist(self, ends):
           # Return set of end_ids selected > blacklist_rounds times
           blacklist = set()
           for end_id, end in ends.items():
               if end.get_property(PROP_SELECTED_COUNT) > self.blacklist_rounds:
                   blacklist.add(end_id)
           # Cap at blacklist_max_len * total_clients
           return blacklist[:int(self.blacklist_max_len * len(ends))]
   ```

3. **Config Extension:**
   ```yaml
   selector:
     sort: refl_oort
     kwargs:
       aggr_num: 10
       blacklist_rounds: -1  # -1 disables, else max selections before blacklist
       blacklist_max_len: 0.3  # Max 30% of clients blacklisted
       avail_priority: 2  # 0=none, 1=fill, 2=strict
       avail_probability: 1.0  # Availability prediction accuracy
       pacer_step: 20  # Evaluate pacer every N rounds
       pacer_delta: 5  # % adjustment to round_threshold
   ```

**Flame Approximation:**
- REFL's `ucbSampler` is tightly coupled with `clientManager`
- **Solution:** Extend Flame's `OortSelector` to call `REFLAvailabilityTracker` for priorities
- **Trade-off:** Cleaner separation, minor implementation differences

#### Tier 3: REFL-Aware Aggregator (Staleness Handling)
**Goal:** Implement deadline filtering and REFL's stale update weighting.

**Implementation:**
1. **Create `REFLFedAvg` Optimizer:**
   - Location: `lib/python/flame/optimizer/reflfedavg.py`
   - Implements:
     - **Deadline-based filtering** of trainer updates
     - **Stale update caching** for stragglers
     - **REFL staleness weighting** (`stale_factor=-4`)
     - Support for other weighting strategies (AdaSGD, DynSGD, etc.)

2. **Key Data Structures:**
   ```python
   class REFLFedAvg(AbstractOptimizer):
       def __init__(self, **kwargs):
           super().__init__(**kwargs)
           self.stale_weights = {}  # {trainer_id: [stale_params]}
           self.stale_remain_duration = {}  # {trainer_id: remaining_time}
           self.stale_rounds = {}  # {trainer_id: num_rounds_stale}
           
           # Config
           self.deadline = kwargs.get('deadline', 0)  # 0 = moving avg
           self.mov_avg_deadline = 0
           self.stale_update_max = kwargs.get('stale_update', -1)  # -1 = no limit
           self.stale_factor = kwargs.get('stale_factor', 1)
           self.stale_beta = kwargs.get('stale_beta', 0.9)
           self.scale_coff = kwargs.get('scale_coff', 10.0)
   ```

3. **Aggregation Flow:**
   ```python
   def do(self, base_weights, cache, total, version, **kwargs):
       round_duration = kwargs.get('round_duration', 0)
       
       # 1. Separate fast and slow trainers based on deadline
       fast_trainers, slow_trainers = self.filter_by_deadline(cache, round_duration)
       
       # 2. Cache stale updates from slow trainers
       for tres in slow_trainers:
           self.stale_weights[tres.end_id] = tres.weights
           self.stale_remain_duration[tres.end_id] = tres.duration - self.deadline
           self.stale_rounds[tres.end_id] = 0
       
       # 3. Retrieve applicable stale updates from previous rounds
       applicable_stale = self.get_applicable_stale_updates(round_duration)
       
       # 4. Compute importance weights for fast + applicable stale
       all_trainers = fast_trainers + applicable_stale
       importance_weights = self.compute_importance_weights(all_trainers)
       
       # 5. Aggregate with weighted averaging
       for tres in all_trainers:
           rate = (tres.count / total) * importance_weights[tres.end_id]
           self.aggregate_fn(tres, rate)
       
       # 6. Update moving average deadline
       if self.deadline == 0:
           self.mov_avg_deadline = self.compute_moving_avg_deadline(fast_trainers)
       
       return self.agg_weights
   
   def compute_importance_weights(self, trainers):
       # Implements REFL's stale weighting strategies
       weights = {}
       for tres in trainers:
           staleness = self.stale_rounds.get(tres.end_id, 0)
           
           if self.stale_factor == 1:  # Equal weight
               weights[tres.end_id] = 1.0
           elif self.stale_factor == -2:  # AdaSGD
               weights[tres.end_id] = 1.0 / (staleness + 1)
           elif self.stale_factor == -3:  # DynSGD
               weights[tres.end_id] = math.exp(-(staleness + 1))
           elif self.stale_factor == -4:  # REFL
               client_ratio = tres.stat_utility / max_stat_utility
               weights[tres.end_id] = (
                   (1 - self.stale_beta) / (staleness + 1) + 
                   self.stale_beta * (1 - math.exp(-client_ratio / max_ratio) / self.scale_coff)
               )
       
       # Normalize weights
       total_weight = sum(weights.values())
       return {k: v / total_weight for k, v in weights.items()}
   
   def get_applicable_stale_updates(self, round_duration):
       # Returns stale updates that are now ready to apply
       applicable = []
       for trainer_id in list(self.stale_weights.keys()):
           self.stale_remain_duration[trainer_id] -= round_duration
           self.stale_rounds[trainer_id] += 1
           
           if (self.stale_remain_duration[trainer_id] <= 0 and
               (self.stale_update_max < 0 or self.stale_rounds[trainer_id] <= self.stale_update_max)):
               # Apply this stale update
               applicable.append(self.create_tres_from_stale(trainer_id))
               del self.stale_weights[trainer_id]
               del self.stale_remain_duration[trainer_id]
               del self.stale_rounds[trainer_id]
       
       return applicable
   ```

4. **Top Aggregator Integration:**
   - Modify `syncfl/top_aggregator.py` to:
     - Track round durations (start/end timestamps)
     - Pass `round_duration` to optimizer
     - Apply deadline filtering before aggregation

5. **Config Extension:**
   ```yaml
   optimizer:
     sort: reflfedavg
     kwargs:
       deadline: 100  # seconds, 0 = use moving average
       stale_update: -1  # max staleness rounds, -1 = no limit
       stale_factor: -4  # -4=REFL, -3=DynSGD, -2=AdaSGD, 1=equal
       stale_beta: 0.9  # REFL beta parameter
       scale_coff: 10.0  # REFL scaling coefficient
       target_ratio: 0.8  # Target fraction of selected clients to wait for
   ```

**Flame Approximation:**
- REFL's aggregator handles raw client updates directly
- **Solution:** Adapt to Flame's `TrainResult` objects and diskcache
- **Trade-off:** Slight memory overhead, but maintains Flame's architecture

---

## 3. Config-Driven Feature Toggles

To enable modular experimentation, all REFL components are independently toggleable:

```yaml
# Baseline FedAvg (no REFL)
selector:
  sort: random
optimizer:
  sort: fedavg

# REFL Availability Only
selector:
  sort: refl_oort
  kwargs:
    avail_priority: 2
    blacklist_rounds: -1  # Disable blacklisting
    pacer_step: -1  # Disable pacer
optimizer:
  sort: fedavg  # No staleness handling

# REFL Selection + Staleness
selector:
  sort: refl_oort
  kwargs:
    avail_priority: 2
    blacklist_rounds: 50
    pacer_step: 20
optimizer:
  sort: reflfedavg
  kwargs:
    stale_factor: -4  # REFL weighting

# REFL Full (All Features)
selector:
  sort: refl_oort
  kwargs:
    avail_priority: 2
    blacklist_rounds: 50
    pacer_step: 20
optimizer:
  sort: reflfedavg
  kwargs:
    deadline: 100
    stale_factor: -4
```

---

## 4. Integration with async_cifar10

### 4.1 Trainer Config Updates
Trainers in `async_cifar10` already support:
- Trace-based availability (`TrainerAvailState`)
- Self-managed state transitions
- Statistical utility reporting (loss, dataset size)

**Additions Needed:**
- Report additional metrics for REFL:
  - Training duration (already captured)
  - Completion timestamp
  - Staleness (if applicable)

**No breaking changes required** - trainers continue operating as before.

### 4.2 Aggregator Config Updates
Create REFL-specific configs in `/aggregator/`:

```json
{
  "realm": "...",
  "selector": {
    "sort": "refl_oort",
    "kwargs": {
      "aggr_num": 10,
      "avail_priority": 2,
      "blacklist_rounds": 50,
      "pacer_step": 20,
      "pacer_delta": 5
    }
  },
  "optimizer": {
    "sort": "reflfedavg",
    "kwargs": {
      "deadline": 100,
      "stale_update": 5,
      "stale_factor": -4,
      "stale_beta": 0.9,
      "scale_coff": 10.0
    }
  },
  "refl": {
    "availability_trace_file": "metadata/availability_traces/mobiperf_traces.yaml",
    "use_priority_selection": true
  }
}
```

### 4.3 Metadata Integration
REFL experiments require availability traces. Extend `experiments/metadata/availability_traces/`:

```yaml
# synthetic_traces.yaml (matching REFL's exp_type configs)
syn_0:
  description: "Always available (0% dropout)"
  traces:
    1: {periods: [[0, 1e12]], duration: 1e12}
    2: {periods: [[0, 1e12]], duration: 1e12}
    # ... for all trainers

syn_20:
  description: "20% synthetic dropout"
  # Generated from REFL's trace generation logic

mobiperf_2st:
  description: "Real-world MobiPerf traces (2-state)"
  # Converted from REFL's pickle format
```

---

## 5. Implementation Phases

### Phase 1: Foundation (Week 1-2)
1. **Create `REFLAvailabilityTracker`**
   - Load/parse REFL trace format
   - Implement `is_available`, `get_priority`
   - Unit tests against REFL's test cases

2. **Extend TrainResult**
   - Add `completion_time`, `round_duration` fields
   - Modify `syncfl/top_aggregator.py` to track round timing

3. **Config Schema Updates**
   - Add `refl` section to aggregator config
   - Document all REFL-specific parameters

### Phase 2: Selector (Week 3-4)
1. **Create `REFLOortSelector`**
   - Implement priority-based selection
   - Add pacer mechanism
   - Add blacklisting logic

2. **Integration Testing**
   - Compare selections against REFL on synthetic workloads
   - Validate pacer behavior matches REFL

3. **Config Templates**
   - Create test configs for each feature toggle combination

### Phase 3: Aggregator (Week 5-6)
1. **Create `REFLFedAvg` Optimizer**
   - Implement deadline filtering
   - Add stale update caching
   - Implement all staleness weighting strategies

2. **Top Aggregator Modifications**
   - Integrate round duration tracking
   - Pass metadata to optimizer

3. **End-to-End Testing**
   - Run full REFL experiments on async_cifar10
   - Compare metrics against REFL's reported results

### Phase 4: Validation & Tuning (Week 7-8)
1. **Reproduce REFL Experiments**
   - CIFAR-10 with various configurations
   - Google Speech (if time permits)

2. **Head-to-Head Comparison**
   - REFL vs. Felix on identical workloads
   - Document performance differences

3. **Documentation & Examples**
   - Update README with REFL integration
   - Provide example configs and launch commands

---

## 6. Testing Strategy

### 6.1 Unit Tests
- **Availability Tracker:**
  - Test `is_available` logic with synthetic traces
  - Validate priority computation
- **REFL Selector:**
  - Test UCB selection matches expected distributions
  - Verify blacklist enforcement
  - Confirm pacer adjustments
- **REFL Optimizer:**
  - Test staleness weighting formulas
  - Validate deadline filtering
  - Ensure stale cache lifecycle correct

### 6.2 Integration Tests
- **Small-Scale Experiments (5 trainers):**
  - Use `test_phase3_mini.yaml` as baseline
  - Enable REFL features one at a time
  - Compare against non-REFL baseline

### 6.3 Validation Against REFL
- **Reproduce Key Results:**
  - Use REFL's published hyperparameters
  - Run on CIFAR-10 with `n=300`, `alpha=0.1`
  - Compare:
    - Convergence speed (rounds to target accuracy)
    - Resource efficiency (compute + communication)
    - Fairness metrics (Gini coefficient, KL divergence)

---

## 7. Known Deviations & Approximations

### 7.1 Centralized vs. Distributed Availability
**REFL:** Aggregator has centralized view of all client availability traces.  
**Flame:** Trainers self-manage state, aggregator observes.  
**Approximation:** Aggregator loads same trace file and predicts trainer states.  
**Impact:** Minimal - acceptable for controlled experiments.

### 7.2 Event-Driven vs. Round-Based Timing
**REFL:** Uses event queue with virtual clock for simulation.  
**Flame:** Real-time system with actual network delays.  
**Approximation:** Track actual round durations, apply deadline filtering post-facto.  
**Impact:** Moderate - may affect deadline tuning.

### 7.3 Model Update Format
**REFL:** Direct NumPy arrays in memory.  
**Flame:** PyTorch tensors via `TrainResult` + diskcache.  
**Approximation:** Convert between formats as needed.  
**Impact:** Negligible - performance overhead only.

### 7.4 Executor Model
**REFL:** Explicit executor processes managed by aggregator.  
**Flame:** Trainers spawn independently, communicate via channels.  
**Approximation:** Maintain REFL's aggregator-centric selection logic.  
**Impact:** None - selection algorithm unchanged.

---

## 8. Success Criteria

### 8.1 Functional Completeness
- [ ] All three REFL components implemented (availability, selection, aggregation)
- [ ] All staleness weighting strategies available (`-4`, `-3`, `-2`, `1`)
- [ ] Priority-based selection working with configurable `avail_priority`
- [ ] Deadline filtering operational with both fixed and moving average
- [ ] Stale update caching and lifecycle management correct

### 8.2 Experimental Validation
- [ ] Reproduce REFL's reported accuracy on CIFAR-10 (within ±2%)
- [ ] Confirm resource efficiency gains (compute + communication)
- [ ] Validate fairness metrics match REFL's trends

### 8.3 Fair Comparison Setup
- [ ] Run REFL and Felix on identical:
  - Datasets (same splits, same Dirichlet alpha)
  - Availability traces (same trainer-to-trace mappings)
  - Hyperparameters (batch size, learning rate, etc.)
  - Evaluation protocol (same test sets, same metrics)

### 8.4 Documentation & Usability
- [ ] README with REFL integration guide
- [ ] Example configs for all feature toggle combinations
- [ ] Troubleshooting guide for common issues
- [ ] Performance tuning recommendations

---

## 9. Config Examples

### Example 1: REFL Full Configuration (300 trainers, alpha=0.1, syn_0)
```yaml
# experiments/configs/refl_n300_alpha0.1_syn0.yaml
experiments:
  - name: refl_n300_alpha0.1_syn0
    description: "REFL with 300 trainers, alpha=0.1, always available"
    
    trainer:
      num_trainers: 300
      dataset:
        dirichlet_alpha: 0.1
      availability:
        mode: syn_0
    
    aggregator:
      selector:
        sort: refl_oort
        kwargs:
          aggr_num: 10
          avail_priority: 2
          blacklist_rounds: 50
          pacer_step: 20
          pacer_delta: 5
          exploration_factor: 0.9
          exploration_decay: 0.98
          exploration_min: 0.2
          round_threshold: 30
          alpha: 2
          clip_bound: 0.95
          cut_off_util: 0.95
      
      optimizer:
        sort: reflfedavg
        kwargs:
          deadline: 100
          stale_update: 5
          stale_factor: -4
          stale_beta: 0.9
          scale_coff: 10.0
          target_ratio: 0.8
      
      refl:
        availability_trace_file: "metadata/availability_traces/synthetic_traces.yaml"
        use_priority_selection: true
    
    execution:
      num_gpus: 8
      sleep_between_spawns: 5.0
      aggregator_warmup_time: 600
```

### Example 2: Comparison - Felix vs. REFL
```yaml
# experiments/configs/comparison_felix_vs_refl.yaml
experiments:
  - name: felix_n300_alpha0.1_syn0
    # ... Felix config ...
    aggregator:
      selector:
        sort: async_oort  # Felix selector
      optimizer:
        sort: fedbuff  # Felix optimizer
  
  - name: refl_n300_alpha0.1_syn0
    # ... REFL config (as above) ...
```

### Example 3: Ablation - Availability Only
```yaml
# experiments/configs/ablation_avail_only.yaml
experiments:
  - name: refl_avail_only
    aggregator:
      selector:
        sort: refl_oort
        kwargs:
          avail_priority: 2
          blacklist_rounds: -1  # Disable
          pacer_step: -1  # Disable
      optimizer:
        sort: fedavg  # Standard FedAvg, no staleness
```

---

## 10. Implementation Roadmap

### Week 1-2: Foundation
- **Deliverables:**
  - `lib/python/flame/availability/refl_tracker.py`
  - Updated `TrainResult` with timing fields
  - Config schema extensions
  - Unit tests for availability tracker

### Week 3-4: Selector
- **Deliverables:**
  - `lib/python/flame/selector/refl_oort.py`
  - Integration with `syncfl/top_aggregator.py`
  - Selector unit tests
  - Test configs for priority selection

### Week 5-6: Aggregator
- **Deliverables:**
  - `lib/python/flame/optimizer/reflfedavg.py`
  - Deadline filtering in top aggregator
  - Stale update lifecycle tests
  - End-to-end integration tests

### Week 7-8: Validation
- **Deliverables:**
  - Reproduced REFL experiments on CIFAR-10
  - Comparison report: REFL vs. Felix
  - Documentation updates
  - Example configs and launch scripts

---

## 11. Open Questions & Future Work

### 11.1 Open Questions
1. **Moving Average Deadline:** REFL uses `target_ratio` to compute deadline - should we replicate exact formula or use Flame's existing approach?
2. **Overcommitment vs. Aggr Goal:** REFL selects `overcommitment × aggr_goal` clients - should this be selector or aggregator responsibility?
3. **Stale Update Expiry:** REFL uses `args.stale_skip_round` flag - do we need this for Flame?

### 11.2 Future Enhancements
1. **SAFA Integration:** REFL compares against SAFA (exp_type=0) - consider implementing SAFA as another baseline.
2. **Additional Datasets:** Extend beyond CIFAR-10 to Google Speech, OpenImage.
3. **Adaptive Hyperparameters:** Auto-tune `deadline`, `stale_beta` based on workload characteristics.
4. **Visualization Tools:** Dashboards showing availability patterns, staleness distributions, etc.

---

## 12. References

### REFL Papers
- **REFL arXiv:** https://arxiv.org/abs/2111.01108
- **REFL EuroSys'23:** ACM EuroSys 2023 proceedings

### REFL Codebase
- **Location:** `/home/dgarg39/flame/third_party/REFL`
- **Key Files:**
  - `core/aggregator.py`: Main aggregation logic
  - `core/client_manager.py`: Availability tracking and Oort integration
  - `thirdparty/oort/oort.py`: UCB-based selector

### Flame Codebase
- **Selectors:** `lib/python/flame/selector/`
- **Optimizers:** `lib/python/flame/optimizer/`
- **SyncFL Aggregator:** `lib/python/flame/mode/horizontal/syncfl/top_aggregator.py`
- **async_cifar10 Example:** `lib/python/examples/async_cifar10/`

---

## Conclusion

This integration plan provides a **comprehensive roadmap** to implement REFL within Flame's abstraction framework. By leveraging Flame's modular design, we can implement REFL's availability tracking, client selection, and aggregation algorithms as **composable components** that can be toggled independently.

The plan prioritizes:
1. **Correctness:** Faithfully implementing REFL's algorithms
2. **Modularity:** Each component toggleable for ablation studies
3. **Fair Comparison:** Ensuring identical experimental conditions for REFL vs. Felix
4. **Minimal Disruption:** Maintaining backward compatibility with existing Flame experiments

**Next Steps:** Review this plan, identify any concerns or needed clarifications, then proceed with Phase 1 implementation.
