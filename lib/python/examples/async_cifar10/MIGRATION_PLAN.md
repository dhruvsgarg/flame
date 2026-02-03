# Async CIFAR-10 Configuration Refactoring Plan

## Executive Summary

**Goal**: Migrate from static config files to programmatic configuration for scalable, maintainable federated learning experiments.

**Current Problem**: 5,924 trainer config files across 20+ directories with massive redundancy, making it nearly impossible to:
- Compare configurations across experiments
- Update shared parameters (e.g., availability traces)
- Maintain consistency across dataset splits
- Scale to new datasets/examples

**Solution**: Centralized metadata with programmatic trainer spawning.

---

## Current Architecture Analysis

### File Structure Issues
```
trainer/
├── config_dir0.1_num300_traceFail_6d_3state_oort/
│   ├── trainer_1.json (312 lines)
│   ├── trainer_2.json (312 lines)
│   └── ... (300 files × 312 lines = 93,600 lines!)
├── config_dir100_num300_traceFail_6d_3state_oort/
│   └── ... (another 300 files)
└── ... (20+ directories)
```

### Redundancy Analysis

**Identical Across All Trainers (70% of config):**
- `backend`, `brokers`, `groupAssociation`
- `channels` definition (47 lines)
- `dataset`, `dependencies`
- `hyperparameters`: `batchSize`, `learningRate`, `rounds`, `epochs`
- `use_oort_loss_fn`, `heartbeats`, `client_notify`
- `baseModel`, `job`, `registry`, `selector`, `optimizer`
- `maxRunTime`, `realm`, `role`

**Trainer-Specific Properties (Intrinsic, 15%):**
- `taskid` (trainer ID - deterministic hash)
- `training_delay_s` (4.0, 8.0, 12.0, 16.0s - speed classes)
- Availability traces (mobiperf: per-trainer real-world patterns)

**Dataset/Experiment-Specific (15%):**
- `trainer_indices_list` (local dataset indices - varies by alpha)
- Synthetic availability traces (syn_0, syn_20, syn_50 - uniform patterns)

---

## Proposed Architecture

### Directory Structure
```
examples/
├── metadata/                           # NEW: Centralized metadata
│   ├── trainer_registry.yaml          # Static trainer properties
│   ├── dataset_splits/                 # Dataset partitions
│   │   ├── cifar10_alpha0.1_n300.yaml
│   │   ├── cifar10_alpha1.0_n300.yaml
│   │   └── cifar10_alpha100_n300.yaml
│   └── availability_traces/            # Trace definitions
│       ├── mobiperf_traces.yaml        # Real-world per-trainer
│       └── synthetic_traces.yaml       # Uniform synthetic
│
├── async_cifar10/
│   ├── configs/                        # NEW: Template configs only
│   │   ├── aggregator_base.yaml       # Base agg config
│   │   └── trainer_base.yaml          # Base trainer template
│   ├── launch/                         # NEW: Experiment launchers
│   │   ├── spawner.py                  # Programmatic trainer spawning
│   │   ├── run_experiment.py           # High-level experiment runner
│   │   └── experiment_configs.yaml     # Experiment definitions
│   ├── aggregator/
│   │   └── pytorch/
│   │       └── main_oort_agg.py        # (minimal changes)
│   └── trainer/
│       └── pytorch/
│           └── main.py                 # (refactored to use metadata)
│
└── README.md                            # Updated documentation
```

### Key Files Design

#### 1. `metadata/trainer_registry.yaml`
```yaml
# Global trainer registry - 300 trainers with intrinsic properties
trainers:
  trainer_001:
    task_id: "505f9fc483cf4df68a2409257b5fad7d3c580370"
    training_delay_s: 4.0
    speed_class: "fast"
    mobiperf_trace_id: "device_001"
    
  trainer_002:
    task_id: "505f9fc483cf4df68a2409257b5fad7d3c580371"
    training_delay_s: 16.0
    speed_class: "slow"
    mobiperf_trace_id: "device_002"
    
  # ... 298 more trainers
  
# Speed distribution (example)
speed_distribution:
  fast: [1-50]     # 4s delay
  medium: [51-150] # 8s delay
  slow: [151-300]  # 12-16s delay
```

#### 2. `metadata/dataset_splits/cifar10_alpha0.1_n300.yaml`
```yaml
# CIFAR-10 split with Dirichlet α=0.1, 300 clients
dataset_name: "cifar10"
num_trainers: 300
dirichlet_alpha: 0.1
total_samples: 50000

trainer_data_splits:
  trainer_001: [8630, 6636, 35954, ...]  # 217 indices
  trainer_002: [38308, 4308, 7268, ...]  # 13 indices
  trainer_003: [...]
  # ... 300 entries
```

#### 3. `metadata/availability_traces/mobiperf_traces.yaml`
```yaml
# Real-world MobiPerf availability traces per device
traces:
  device_001:
    name: "mobiperf_device_001"
    states_2st: [(0, 'AVL_TRAIN'), (250, 'UN_AVL'), ...]
    states_3st_50: [(0, 'AVL_TRAIN'), (250, 'UN_AVL'), (1053, 'AVL_TRAIN'), (7162, 'AVL_EVAL'), ...]
    states_3st_75: [(0, 'AVL_TRAIN'), (300, 'UN_AVL'), ...]
    
  device_002:
    name: "mobiperf_device_002"
    states_2st: [(0, 'AVL_TRAIN'), (300, 'UN_AVL'), ...]
    states_3st_50: [(0, 'AVL_TRAIN'), (300, 'UN_AVL'), ...]
    states_3st_75: [(0, 'AVL_TRAIN'), (300, 'UN_AVL'), ...]
```

#### 4. `metadata/availability_traces/synthetic_traces.yaml`
```yaml
# Synthetic availability traces (uniform across all trainers)
traces:
  syn_0:
    description: "Always available"
    pattern: [(0, 'AVL_TRAIN')]
    
  syn_20:
    description: "20% unavailability"
    pattern: [(0, 'AVL_TRAIN'), (600, 'UN_AVL'), (1200, 'AVL_TRAIN'), ...]
    
  syn_50:
    description: "50% unavailability"  
    pattern: [(0, 'AVL_TRAIN'), (4800, 'UN_AVL'), (5400, 'AVL_TRAIN'), ...]
```

#### 5. `async_cifar10/launch/spawner.py`
```python
"""Programmatic trainer spawner - replaces exec_300_trainers_2state.sh"""
import yaml
import subprocess
from pathlib import Path
from typing import Dict, List

class TrainerSpawner:
    def __init__(self, metadata_dir: Path):
        self.metadata_dir = metadata_dir
        self.trainer_registry = self._load_yaml('trainer_registry.yaml')
        self.mobiperf_traces = self._load_yaml('availability_traces/mobiperf_traces.yaml')
        self.synthetic_traces = self._load_yaml('availability_traces/synthetic_traces.yaml')
        
    def load_dataset_split(self, dataset: str, alpha: float, num_trainers: int):
        """Load dataset split configuration."""
        split_file = f"dataset_splits/{dataset}_alpha{alpha}_n{num_trainers}.yaml"
        return self._load_yaml(split_file)
    
    def generate_trainer_config(self, 
                               trainer_id: int,
                               dataset_split: Dict,
                               trace_type: str,
                               trace_name: str,
                               base_config: Dict) -> Dict:
        """Generate runtime trainer config from metadata."""
        trainer_key = f"trainer_{trainer_id:03d}"
        trainer_meta = self.trainer_registry['trainers'][trainer_key]
        
        # Build config dict (no file needed)
        config = base_config.copy()
        config['taskid'] = trainer_meta['task_id']
        config['hyperparameters']['training_delay_s'] = trainer_meta['training_delay_s']
        config['hyperparameters']['trainer_indices_list'] = dataset_split['trainer_data_splits'][trainer_key]
        
        # Set availability trace
        if trace_type == 'synthetic':
            config['hyperparameters'][f'avl_events_{trace_name}'] = \
                self.synthetic_traces['traces'][trace_name]['pattern']
        elif trace_type == 'mobiperf':
            trace_id = trainer_meta['mobiperf_trace_id']
            config['hyperparameters']['avl_events_mobiperf_2st'] = \
                self.mobiperf_traces['traces'][trace_id]['states_2st']
            # ... handle 3st variants
        
        return config
    
    def spawn_trainers(self, 
                      num_trainers: int,
                      dataset_split: Dict,
                      trace_type: str,
                      trace_name: str,
                      base_config: Dict,
                      num_gpus: int = 8,
                      delay_between_trainers: float = 1.0):
        """Spawn trainer processes with generated configs."""
        processes = []
        
        for trainer_id in range(1, num_trainers + 1):
            # Generate config in-memory
            config = self.generate_trainer_config(
                trainer_id, dataset_split, trace_type, trace_name, base_config
            )
            
            # Assign GPU
            gpu_id = trainer_id % num_gpus
            
            # Spawn trainer with config passed as JSON string or temp file
            cmd = [
                'python', '../pytorch/main.py',
                '--config-dict', json.dumps(config)  # OR write temp file
            ]
            
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            
            proc = subprocess.Popen(cmd, env=env)
            processes.append((trainer_id, proc))
            
            time.sleep(delay_between_trainers)
        
        return processes
```

#### 6. `async_cifar10/launch/run_experiment.py`
```python
"""High-level experiment runner."""
import argparse
from spawner import TrainerSpawner
from pathlib import Path

def run_experiment(exp_config: Dict):
    """Run a complete FL experiment."""
    # Load metadata
    metadata_dir = Path(__file__).parent.parent.parent / 'metadata'
    spawner = TrainerSpawner(metadata_dir)
    
    # Load dataset split
    dataset_split = spawner.load_dataset_split(
        dataset=exp_config['dataset'],
        alpha=exp_config['dirichlet_alpha'],
        num_trainers=exp_config['num_trainers']
    )
    
    # Start aggregator
    agg_proc = start_aggregator(exp_config['aggregator_config'])
    time.sleep(15)  # Wait for agg ready
    
    # Spawn trainers
    trainer_procs = spawner.spawn_trainers(
        num_trainers=exp_config['num_trainers'],
        dataset_split=dataset_split,
        trace_type=exp_config['trace_type'],
        trace_name=exp_config['trace_name'],
        base_config=load_base_config(),
        num_gpus=exp_config['num_gpus']
    )
    
    # Monitor and manage experiment
    monitor_experiment(agg_proc, trainer_procs, exp_config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', required=True, 
                       help='Experiment name from experiment_configs.yaml')
    args = parser.parse_args()
    
    # Load experiment definition
    exp_config = load_experiment_config(args.experiment)
    run_experiment(exp_config)
```

#### 7. `async_cifar10/launch/experiment_configs.yaml`
```yaml
# High-level experiment definitions (replaces multiple script files)
experiments:
  oort_n300_oracular_syn0:
    description: "Oort selector, 300 trainers, oracular tracking, always available"
    dataset: "cifar10"
    dirichlet_alpha: 0.1
    num_trainers: 300
    trace_type: "synthetic"
    trace_name: "syn_0"
    selector: "oort"
    aggregator_concurrency: 13
    aggregation_goal: 10
    tracking_mode: "oracular"
    num_gpus: 8
    aggregator_config: "configs/aggregator_base.yaml"
    
  oort_n300_oracular_syn20:
    description: "Oort selector, 300 trainers, oracular tracking, 20% unavailable"
    # Similar structure...
    trace_name: "syn_20"
    
  oort_n300_oracular_mobiperf:
    description: "Oort selector, 300 trainers, oracular tracking, real-world traces"
    trace_type: "mobiperf"
    trace_name: "mobiperf_2st"
```

---

## Migration Strategy

### Phase 1: Automated Metadata Extraction & Verification (Week 1)
**Goal**: Programmatically extract static properties into centralized metadata with full validation

**Critical Principle**: 100% automated extraction with verification - NO MANUAL DATA ENTRY

---

#### Step 1.1: Setup Extraction Infrastructure (Day 1)

**Create**: `scripts/extract_metadata.py`

```python
"""
Automated metadata extraction from existing trainer configs.
Handles JSON parsing, deduplication, and YAML generation.
"""
import json
import yaml
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set
import hashlib

class MetadataExtractor:
    def __init__(self, trainer_config_dirs: List[Path], output_dir: Path):
        self.config_dirs = trainer_config_dirs
        self.output_dir = output_dir
        self.trainer_registry = {}
        self.dataset_splits = defaultdict(dict)
        self.mobiperf_traces = {}
        self.synthetic_traces = {}
        
    def extract_all(self):
        """Run full extraction pipeline."""
        print("Phase 1.1: Scanning trainer configs...")
        self._scan_all_configs()
        
        print("Phase 1.2: Building trainer registry...")
        self._build_trainer_registry()
        
        print("Phase 1.3: Extracting dataset splits...")
        self._extract_dataset_splits()
        
        print("Phase 1.4: Extracting availability traces...")
        self._extract_traces()
        
        print("Phase 1.5: Writing YAML files...")
        self._write_yaml_files()
        
        print("✓ Extraction complete!")
        
    def _scan_all_configs(self):
        """Scan all config directories and load JSON files."""
        self.all_configs = {}
        for config_dir in self.config_dirs:
            dir_name = config_dir.name
            self.all_configs[dir_name] = {}
            
            # Load all trainer_N.json files
            for config_file in sorted(config_dir.glob("trainer_*.json")):
                if config_file.name == "trainer_0_test.json":
                    continue  # Skip test files
                    
                trainer_num = int(config_file.stem.split('_')[1])
                with open(config_file) as f:
                    self.all_configs[dir_name][trainer_num] = json.load(f)
        
        print(f"  Loaded configs from {len(self.all_configs)} directories")
        for dir_name, configs in self.all_configs.items():
            print(f"    {dir_name}: {len(configs)} trainers")
    
    def _build_trainer_registry(self):
        """Extract trainer-specific intrinsic properties."""
        # Use first config directory as reference (properties should be consistent)
        reference_dir = list(self.all_configs.keys())[0]
        reference_configs = self.all_configs[reference_dir]
        
        for trainer_num, config in sorted(reference_configs.items()):
            trainer_key = f"trainer_{trainer_num:03d}"
            
            self.trainer_registry[trainer_key] = {
                'trainer_id': trainer_num,
                'task_id': config['taskid'],
                'training_delay_s': float(config['hyperparameters']['training_delay_s']),
            }
            
            # Infer speed class from delay
            delay = float(config['hyperparameters']['training_delay_s'])
            if delay <= 4.0:
                speed_class = 'fast'
            elif delay <= 8.0:
                speed_class = 'medium'
            elif delay <= 12.0:
                speed_class = 'slow'
            else:
                speed_class = 'very_slow'
            
            self.trainer_registry[trainer_key]['speed_class'] = speed_class
            
        print(f"  Built registry for {len(self.trainer_registry)} trainers")
        
    def _extract_dataset_splits(self):
        """Extract dataset index assignments per trainer per alpha."""
        for dir_name, configs in self.all_configs.items():
            # Parse directory name: config_dir<alpha>_num<N>_<rest>
            # e.g., config_dir0.1_num300_traceFail_6d_3state_oort
            if not dir_name.startswith('config_dir'):
                continue
                
            parts = dir_name.split('_')
            alpha_str = parts[0].replace('config_dir', '')
            
            try:
                alpha = float(alpha_str)
            except ValueError:
                print(f"  Warning: Could not parse alpha from {dir_name}")
                continue
            
            split_key = f"cifar10_alpha{alpha}_n300"
            
            for trainer_num, config in configs.items():
                trainer_key = f"trainer_{trainer_num:03d}"
                indices = config['hyperparameters']['trainer_indices_list']
                
                if split_key not in self.dataset_splits:
                    self.dataset_splits[split_key] = {
                        'dataset_name': 'cifar10',
                        'num_trainers': 300,
                        'dirichlet_alpha': alpha,
                        'total_samples': 50000,
                        'trainer_data_splits': {}
                    }
                
                self.dataset_splits[split_key]['trainer_data_splits'][trainer_key] = indices
        
        print(f"  Extracted {len(self.dataset_splits)} dataset splits")
        for split_name, split_data in self.dataset_splits.items():
            num_trainers = len(split_data['trainer_data_splits'])
            print(f"    {split_name}: {num_trainers} trainers")
    
    def _extract_traces(self):
        """Extract availability traces (both mobiperf and synthetic)."""
        # Use first config directory as reference
        reference_dir = list(self.all_configs.keys())[0]
        reference_configs = self.all_configs[reference_dir]
        
        # Extract synthetic traces (uniform across all trainers)
        first_config = reference_configs[1]
        hp = first_config['hyperparameters']
        
        self.synthetic_traces = {
            'description': 'Synthetic availability traces - uniform across all trainers',
            'traces': {
                'syn_0': {
                    'description': 'Always available (0% unavailability)',
                    'pattern': eval(hp['avl_events_syn_0'])
                },
                'syn_20': {
                    'description': '20% unavailability',
                    'pattern': eval(hp['avl_events_syn_20'])
                },
                'syn_50': {
                    'description': '50% unavailability',
                    'pattern': eval(hp['avl_events_syn_50'])
                }
            }
        }
        
        # Extract mobiperf traces (per-trainer)
        self.mobiperf_traces = {
            'description': 'Real-world MobiPerf availability traces - per device',
            'traces': {}
        }
        
        for trainer_num, config in reference_configs.items():
            trainer_key = f"trainer_{trainer_num:03d}"
            device_id = f"device_{trainer_num:03d}"
            hp = config['hyperparameters']
            
            self.mobiperf_traces['traces'][device_id] = {
                'trainer': trainer_key,
                'states_2st': eval(hp['avl_events_mobiperf_2st']),
                'states_3st_50': eval(hp['avl_events_mobiperf_3st_50']),
                'states_3st_75': eval(hp['avl_events_mobiperf_3st_75']),
            }
            
            # Link device to trainer in registry
            self.trainer_registry[trainer_key]['mobiperf_device_id'] = device_id
        
        print(f"  Extracted {len(self.synthetic_traces['traces'])} synthetic traces")
        print(f"  Extracted {len(self.mobiperf_traces['traces'])} mobiperf device traces")
    
    def _write_yaml_files(self):
        """Write all metadata to YAML files."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Write trainer registry
        registry_file = self.output_dir / 'trainer_registry.yaml'
        with open(registry_file, 'w') as f:
            yaml.dump({
                'description': 'Global trainer registry with intrinsic properties',
                'num_trainers': len(self.trainer_registry),
                'trainers': self.trainer_registry
            }, f, default_flow_style=False, sort_keys=False)
        print(f"  ✓ Wrote {registry_file}")
        
        # Write dataset splits
        splits_dir = self.output_dir / 'dataset_splits'
        splits_dir.mkdir(exist_ok=True)
        for split_name, split_data in self.dataset_splits.items():
            split_file = splits_dir / f"{split_name}.yaml"
            with open(split_file, 'w') as f:
                yaml.dump(split_data, f, default_flow_style=False)
            print(f"  ✓ Wrote {split_file}")
        
        # Write availability traces
        traces_dir = self.output_dir / 'availability_traces'
        traces_dir.mkdir(exist_ok=True)
        
        synthetic_file = traces_dir / 'synthetic_traces.yaml'
        with open(synthetic_file, 'w') as f:
            yaml.dump(self.synthetic_traces, f, default_flow_style=False)
        print(f"  ✓ Wrote {synthetic_file}")
        
        mobiperf_file = traces_dir / 'mobiperf_traces.yaml'
        with open(mobiperf_file, 'w') as f:
            yaml.dump(self.mobiperf_traces, f, default_flow_style=False)
        print(f"  ✓ Wrote {mobiperf_file}")

# Main execution
if __name__ == '__main__':
    import sys
    
    # Define config directories to process
    base_dir = Path(__file__).parent.parent / 'trainer'
    config_dirs = [
        base_dir / 'config_dir0.1_num300_traceFail_6d_3state_oort',
        base_dir / 'config_dir1_num300_traceFail_6d_3state_oort',
        base_dir / 'config_dir10_num300_traceFail_6d_3state_oort',
        base_dir / 'config_dir100_num300_traceFail_6d_3state_oort',
    ]
    
    output_dir = Path(__file__).parent.parent.parent / 'metadata'
    
    extractor = MetadataExtractor(config_dirs, output_dir)
    extractor.extract_all()
    
    print("\n" + "="*60)
    print("✓ Phase 1.1 Complete: Metadata extraction successful")
    print("="*60)
```

**Run**: 
```bash
cd /home/dgarg39/flame/lib/python/examples/async_cifar10
python scripts/extract_metadata.py
```

**Expected Output**:
- `metadata/trainer_registry.yaml` (300 trainers)
- `metadata/dataset_splits/cifar10_alpha*.yaml` (4 files)
- `metadata/availability_traces/synthetic_traces.yaml`
- `metadata/availability_traces/mobiperf_traces.yaml`

---

#### Step 1.2: Comprehensive Validation (Day 2-3)

**Create**: `scripts/validate_metadata.py`

```python
"""
Comprehensive validation: verify YAML metadata matches original JSON configs.
This is the GATE before proceeding to Phase 2.
"""
import json
import yaml
from pathlib import Path
from typing import Dict, List, Tuple
import sys

class MetadataValidator:
    def __init__(self, metadata_dir: Path, trainer_config_dirs: List[Path]):
        self.metadata_dir = metadata_dir
        self.config_dirs = trainer_config_dirs
        self.errors = []
        self.warnings = []
        
    def validate_all(self) -> bool:
        """Run all validation checks. Returns True if all pass."""
        print("="*70)
        print("METADATA VALIDATION SUITE")
        print("="*70)
        
        all_passed = True
        
        checks = [
            ("Trainer Registry", self._validate_trainer_registry),
            ("Dataset Splits", self._validate_dataset_splits),
            ("Synthetic Traces", self._validate_synthetic_traces),
            ("MobiPerf Traces", self._validate_mobiperf_traces),
            ("Cross-Reference Integrity", self._validate_cross_references),
            ("Completeness Check", self._validate_completeness),
        ]
        
        for check_name, check_func in checks:
            print(f"\n{'─'*70}")
            print(f"Running: {check_name}")
            print('─'*70)
            passed = check_func()
            
            if passed:
                print(f"✓ {check_name}: PASSED")
            else:
                print(f"✗ {check_name}: FAILED")
                all_passed = False
        
        print("\n" + "="*70)
        print("VALIDATION SUMMARY")
        print("="*70)
        
        if all_passed:
            print("✓ ALL VALIDATIONS PASSED")
            print("\n🎉 Metadata is verified correct. Safe to proceed to Phase 2.")
        else:
            print(f"✗ VALIDATION FAILED with {len(self.errors)} errors")
            print("\nErrors:")
            for error in self.errors[:10]:  # Show first 10
                print(f"  • {error}")
            if len(self.errors) > 10:
                print(f"  ... and {len(self.errors) - 10} more errors")
            print("\n⚠ DO NOT PROCEED TO PHASE 2 until all errors are fixed.")
        
        if self.warnings:
            print(f"\nWarnings: {len(self.warnings)}")
            for warning in self.warnings[:5]:
                print(f"  ⚠ {warning}")
        
        return all_passed
    
    def _load_metadata(self):
        """Load all YAML metadata files."""
        with open(self.metadata_dir / 'trainer_registry.yaml') as f:
            self.trainer_registry = yaml.safe_load(f)
        
        self.dataset_splits = {}
        splits_dir = self.metadata_dir / 'dataset_splits'
        for split_file in splits_dir.glob('*.yaml'):
            with open(split_file) as f:
                self.dataset_splits[split_file.stem] = yaml.safe_load(f)
        
        traces_dir = self.metadata_dir / 'availability_traces'
        with open(traces_dir / 'synthetic_traces.yaml') as f:
            self.synthetic_traces = yaml.safe_load(f)
        with open(traces_dir / 'mobiperf_traces.yaml') as f:
            self.mobiperf_traces = yaml.safe_load(f)
    
    def _load_original_config(self, config_dir: Path, trainer_num: int) -> Dict:
        """Load original JSON config for comparison."""
        config_file = config_dir / f"trainer_{trainer_num}.json"
        with open(config_file) as f:
            return json.load(f)
    
    def _validate_trainer_registry(self) -> bool:
        """Validate trainer registry against original configs."""
        self._load_metadata()
        passed = True
        
        # Check count
        expected_count = 300
        actual_count = self.trainer_registry['num_trainers']
        if actual_count != expected_count:
            self.errors.append(f"Registry has {actual_count} trainers, expected {expected_count}")
            passed = False
        
        # Validate each trainer against first config directory
        reference_dir = self.config_dirs[0]
        print(f"  Validating against: {reference_dir.name}")
        
        for trainer_key, trainer_meta in self.trainer_registry['trainers'].items():
            trainer_num = trainer_meta['trainer_id']
            
            try:
                orig_config = self._load_original_config(reference_dir, trainer_num)
            except FileNotFoundError:
                self.errors.append(f"{trainer_key}: Original config not found")
                passed = False
                continue
            
            # Check task_id
            if trainer_meta['task_id'] != orig_config['taskid']:
                self.errors.append(
                    f"{trainer_key}: task_id mismatch - "
                    f"YAML={trainer_meta['task_id']}, JSON={orig_config['taskid']}"
                )
                passed = False
            
            # Check training_delay_s
            orig_delay = float(orig_config['hyperparameters']['training_delay_s'])
            if trainer_meta['training_delay_s'] != orig_delay:
                self.errors.append(
                    f"{trainer_key}: training_delay_s mismatch - "
                    f"YAML={trainer_meta['training_delay_s']}, JSON={orig_delay}"
                )
                passed = False
        
        if passed:
            print(f"  ✓ All {actual_count} trainers validated")
        
        return passed
    
    def _validate_dataset_splits(self) -> bool:
        """Validate dataset splits against original configs."""
        passed = True
        
        for split_name, split_data in self.dataset_splits.items():
            alpha = split_data['dirichlet_alpha']
            print(f"  Validating split: {split_name} (alpha={alpha})")
            
            # Find corresponding config directory
            config_dir = None
            for cd in self.config_dirs:
                if f"config_dir{alpha}_" in cd.name:
                    config_dir = cd
                    break
            
            if not config_dir:
                self.errors.append(f"{split_name}: No matching config directory found")
                passed = False
                continue
            
            # Validate each trainer's indices
            for trainer_key, indices in split_data['trainer_data_splits'].items():
                trainer_num = int(trainer_key.split('_')[1])
                
                try:
                    orig_config = self._load_original_config(config_dir, trainer_num)
                    orig_indices = orig_config['hyperparameters']['trainer_indices_list']
                    
                    if indices != orig_indices:
                        self.errors.append(
                            f"{split_name}/{trainer_key}: Indices mismatch - "
                            f"lengths YAML={len(indices)}, JSON={len(orig_indices)}"
                        )
                        passed = False
                except FileNotFoundError:
                    self.errors.append(f"{split_name}/{trainer_key}: Original config not found")
                    passed = False
            
            if passed:
                num_trainers = len(split_data['trainer_data_splits'])
                print(f"    ✓ {num_trainers} trainers validated")
        
        return passed
    
    def _validate_synthetic_traces(self) -> bool:
        """Validate synthetic traces."""
        passed = True
        
        # Check against first trainer config
        reference_dir = self.config_dirs[0]
        orig_config = self._load_original_config(reference_dir, 1)
        hp = orig_config['hyperparameters']
        
        for trace_name in ['syn_0', 'syn_20', 'syn_50']:
            yaml_pattern = self.synthetic_traces['traces'][trace_name]['pattern']
            json_pattern = eval(hp[f'avl_events_{trace_name}'])
            
            if yaml_pattern != json_pattern:
                self.errors.append(f"Synthetic trace {trace_name} mismatch")
                passed = False
            else:
                print(f"  ✓ {trace_name}: {len(yaml_pattern)} events")
        
        return passed
    
    def _validate_mobiperf_traces(self) -> bool:
        """Validate mobiperf traces for all trainers."""
        passed = True
        
        reference_dir = self.config_dirs[0]
        
        for device_id, trace_data in self.mobiperf_traces['traces'].items():
            trainer_key = trace_data['trainer']
            trainer_num = int(trainer_key.split('_')[1])
            
            try:
                orig_config = self._load_original_config(reference_dir, trainer_num)
                hp = orig_config['hyperparameters']
                
                # Check all three trace variants
                for variant in ['2st', '3st_50', '3st_75']:
                    yaml_trace = trace_data[f'states_{variant}']
                    json_trace = eval(hp[f'avl_events_mobiperf_{variant}'])
                    
                    if yaml_trace != json_trace:
                        self.errors.append(
                            f"{device_id} ({trainer_key}): mobiperf_{variant} mismatch"
                        )
                        passed = False
            except Exception as e:
                self.errors.append(f"{device_id}: Validation error - {e}")
                passed = False
        
        if passed:
            num_devices = len(self.mobiperf_traces['traces'])
            print(f"  ✓ {num_devices} device traces validated (3 variants each)")
        
        return passed
    
    def _validate_cross_references(self) -> bool:
        """Validate that all cross-references are valid."""
        passed = True
        
        # Check that mobiperf device IDs in registry exist in traces
        for trainer_key, trainer_meta in self.trainer_registry['trainers'].items():
            device_id = trainer_meta.get('mobiperf_device_id')
            if device_id not in self.mobiperf_traces['traces']:
                self.errors.append(f"{trainer_key}: References non-existent device {device_id}")
                passed = False
        
        # Check that dataset splits reference valid trainers
        for split_name, split_data in self.dataset_splits.items():
            for trainer_key in split_data['trainer_data_splits'].keys():
                if trainer_key not in self.trainer_registry['trainers']:
                    self.errors.append(f"{split_name}: References non-existent {trainer_key}")
                    passed = False
        
        if passed:
            print("  ✓ All cross-references valid")
        
        return passed
    
    def _validate_completeness(self) -> bool:
        """Check that no data is missing."""
        passed = True
        
        # Check all 300 trainers present
        trainer_ids = set(range(1, 301))
        actual_ids = {meta['trainer_id'] for meta in self.trainer_registry['trainers'].values()}
        missing = trainer_ids - actual_ids
        
        if missing:
            self.errors.append(f"Missing trainers: {sorted(missing)}")
            passed = False
        else:
            print("  ✓ All 300 trainers present")
        
        # Check all dataset splits have 300 trainers
        for split_name, split_data in self.dataset_splits.items():
            if len(split_data['trainer_data_splits']) != 300:
                self.errors.append(
                    f"{split_name}: Has {len(split_data['trainer_data_splits'])}/300 trainers"
                )
                passed = False
        
        if passed:
            print(f"  ✓ All splits complete (4 splits × 300 trainers)")
        
        return passed

# Main execution
if __name__ == '__main__':
    metadata_dir = Path(__file__).parent.parent.parent / 'metadata'
    base_dir = Path(__file__).parent.parent / 'trainer'
    
    config_dirs = [
        base_dir / 'config_dir0.1_num300_traceFail_6d_3state_oort',
        base_dir / 'config_dir1_num300_traceFail_6d_3state_oort',
        base_dir / 'config_dir10_num300_traceFail_6d_3state_oort',
        base_dir / 'config_dir100_num300_traceFail_6d_3state_oort',
    ]
    
    validator = MetadataValidator(metadata_dir, config_dirs)
    passed = validator.validate_all()
    
    sys.exit(0 if passed else 1)
```

**Run**:
```bash
cd /home/dgarg39/flame/lib/python/examples/async_cifar10
python scripts/validate_metadata.py
```

**Gate Criteria**: Script must exit with code 0 (all checks pass) before proceeding to Phase 2.

---

#### Step 1.3: Phase 1 Deliverables Checklist

**Before proceeding to Phase 2, verify:**

- [ ] `extract_metadata.py` runs without errors
- [ ] All YAML files generated:
  - [ ] `metadata/trainer_registry.yaml` (300 trainers)
  - [ ] `metadata/dataset_splits/cifar10_alpha0.1_n300.yaml`
  - [ ] `metadata/dataset_splits/cifar10_alpha1_n300.yaml`
  - [ ] `metadata/dataset_splits/cifar10_alpha10_n300.yaml`
  - [ ] `metadata/dataset_splits/cifar10_alpha100_n300.yaml`
  - [ ] `metadata/availability_traces/synthetic_traces.yaml`
  - [ ] `metadata/availability_traces/mobiperf_traces.yaml`
- [ ] `validate_metadata.py` exits with code 0
- [ ] All 6 validation checks pass:
  - [ ] Trainer Registry validation
  - [ ] Dataset Splits validation
  - [ ] Synthetic Traces validation
  - [ ] MobiPerf Traces validation
  - [ ] Cross-reference integrity
  - [ ] Completeness check
- [ ] Manual spot-check: Compare 5 random trainer configs (JSON vs YAML) by hand
- [ ] Git commit with message: "Phase 1 complete: Extracted and validated metadata"

**Time Estimate**: 3 days
- Day 1: Write and test extraction script
- Day 2: Write and test validation script
- Day 3: Run full validation, fix any issues, final verification

**Sign-off Required**: All checkboxes above must be ticked before Phase 2 begins.

### Phase 2: Programmatic Spawning (Week 2)
**Goal**: Implement trainer spawner without changing experiment behavior

**Prerequisites**: Phase 1 complete with all validation checks passing

**Tasks** (detailed implementation in separate section after Phase 1 approval):
1. **Implement `launch/spawner.py`**
   - Metadata loader
   - Config generator (runtime assembly)
   - Process spawner with GPU affinity

2. **Refactor `trainer/pytorch/main.py`**
   - Accept config as dict or JSON string (not just file)
   - Keep backward compatibility with file-based configs

3. **Create minimal base configs**
   - `configs/trainer_base.yaml` (common properties only)
   - `configs/aggregator_base.yaml`

**Validation**: Run parallel experiments - old script vs new spawner
- Same trainer IDs spawned
- Same dataset splits loaded
- Same availability traces used

*Note: Detailed Phase 2 implementation will be provided after Phase 1 sign-off.*
Prerequisites**: Phase 2 complete with parallel validation passing

**Tasks** (detailed implementation in separate section after Phase 2 approval):
1. **Implement `launch/run_experiment.py`**
   - Aggregator lifecycle management
   - Trainer spawning orchestration
   - Process monitoring and cleanup (replaces Ctrl+C cleanup)

2. **Create `launch/experiment_configs.yaml`**
   - Define current experiments (syn0, syn20, syn50, mobiperf)
   - Add metadata for logging, wandb integration

3. **Update `oort_n300_oracular_1feb_all4unavail.sh`**
   - Replace trainer spawning with Python call
   - Keep monitoring/termination logic temporarily

**Validation**: Run full experiment end-to-end with new system

*Note: Detailed Phase 3 implementation will be provided after Phase 2 sign-off.*
   - Replace trainer spawning with Python call
   - Keep monitoring/termination logic temporarily

**Validation**: Run full experiment end-to-end with new system

### Phase 4: Cleanup & Documentation (Week 4)
**Goal**: Remove legacy files and update documentation

**Tasks**:
1. **Archive old configs**
   - Move `config_dir*` to `trainer/legacy_configs/` (Git history preserved)
   - Keep one example config for reference

2. **Update shell script**
   - Simplify to call Python experiment runner
   - Remove bash trainer spawning logic

3. **Documentation**
   - Update `README.md` with new workflow
   - Add metadata file format documentation
   - Create developer guide for adding new experiments

4. **Validation script**
   - `scripts/validate_metadata.py` - check consistency
   - Unit tests for spawner

---

## Code Changes Required

### 1. `trainer/pytorch/main.py`
**Current**:
```python
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)  # File path only
    args = parser.parse_args()
    config = Config(args.config)  # Loads from file
```

**New**:
```python
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Config file path (legacy)')
    parser.add_argument('--config-dict', help='Config as JSON string')
    parser.add_argument('--trainer-id', type=int, help='Trainer ID for metadata lookup')
    parser.add_argument('--metadata-dir', help='Path to metadata directory')
    args = parser.parse_args()
    
    if args.config_dict:
        config = Config(config_dict=json.loads(args.config_dict))
    elif args.trainer_id and args.metadata_dir:
        # Load from metadata + generate config
        config = load_from_metadata(args.trainer_id, args.metadata_dir)
    else:
        config = Config(args.config)  # Legacy file-based
```

### 2. `aggregator/pytorch/main_oort_agg.py`
**Minimal changes**: Already uses single config file - no spawning needed

### 3. `flame/config.py`
**Add support for dict-based config**:
```python
class Config:
    def __init__(self, config_path=None, config_dict=None):
        if config_dict:
            self._config = config_dict
        elif config_path:
            self._config = self._load_from_file(config_path)
        else:
            raise ValueError("Must provide config_path or config_dict")
```

---

## Benefits

### Immediate
- **99% reduction** in config file size (5,924 files → ~10 metadata files)
- **Easy comparison**: `diff` two YAML files instead of 300 pairs
- **Single source of truth**: Update trainer delay once, affects all experiments
- **Git-friendly**: Small YAML diffs instead of 300-file changes

### Long-term
- **Scalability**: Add new dataset (MNIST, Speech) with one split file
- **Reproducibility**: Experiment definition in one YAML file
- **Debugging**: Clear separation of static metadata vs experiment params
- **Testing**: Mock metadata for unit tests
- **Extensibility**: Easy to add new trainer properties (battery level, network speed)

---

## Migration Risks & Mitigation

### Risk 1: Data Loss During Extraction
**Mitigation**: 
- Keep all original configs in Git history
- Validation script to compare old vs new configs
- Parallel run tests (Phase 2)

### Risk 2: Breaking Existing Experiments
**Mitigation**:
- Maintain backward compatibility (file-based configs)
- Phased rollout with validation at each step
- Feature flag to switch between old/new system

### Risk 3: Performance Overhead
**Mitigation**:
- Config generation is O(1) per trainer (fast)
- Spawn delay already 1s per trainer (no change)
- Benchmark before/after

---

## Success Criteria

1. **Functionally equivalent**: New system produces identical experiment results
2. **Reduced complexity**: <1% of original config file count
3. **Maintainability**: Can add ne Gate |
|-------|----------|-------------|------|
| **1. Metadata Extraction** | **3 days** | YAML metadata files + validation | ✓ Validation script passes |
| 2. Programmatic Spawning | 5 days | `spawner.py` + refactored `main.py` | ✓ Parallel validation passes |
| 3. Experiment Runner | 5 days | `run_experiment.py` + experiment configs | ✓ Full experiment succeeds |
| 4. Cleanup & Docs | 5 days | Updated docs, archived legacy, tests | ✓ Review complete |
| **Total** | **18 days** | Production-ready refactored system | |

---

## Execution Plan

### Immediate Next Steps (START HERE)

1. **Create extraction script** (`scripts/extract_metadata.py`)
   - Copy full implementation from Phase 1 Step 1.1 above
   - Run and verify output files generated

2. **Create validation script** (`scripts/validate_metadata.py`)
   - Copy full implementation from Phase 1 Step 1.2 above
   - Run and verify all checks pass

3. **Phase 1 Gate Review**
   - Complete Step 1.3 checklist
   - Manual spot-check 5 random trainers
   - Git commit with Phase 1 sign-off

4. **Phase 1 Approval Required**
   - User review of generated YAML files
   - Confirmation all validation checks pass
   - Sign-off before Phase 2 detailed plan

### After Phase 1 Approval

- Detailed Phase 2 implementation plan will be provided
- Phase 3 implementation plan follows Phase 2 approval
- Iterative approach ensures no wasted effort on incorrect architecture

## Next Steps

1. **Review this plan** - Get feedback on architecture
2. **Proof of concept** - Implement Phase 1 for 10 trainers
3. **Validate extraction** - Ensure no data loss
4. **Iterate** - Refine based on learnings
5. **Full rollout** - Complete all phases

---

## Questions for Discussion

1. **YAML vs JSON**: Prefer YAML for readability, but JSON for Python compatibility?
2. **Metadata location**: `examples/metadata/` vs `lib/python/flame/metadata/`?
3. **Backward compatibility**: Keep file-based configs indefinitely or deprecate after migration?
4. **Testing strategy**: Unit tests, integration tests, or manual validation sufficient?
5. **Config assembly**: Runtime generation vs pre-generated temp files?
6. **Extension to other examples**: Apply same pattern to `async_mnist`, `async_google_speech`?

---

**Document Version**: 1.0  
**Last Updated**: February 2, 2026  
**Author**: Based on codebase analysis and user requirements
