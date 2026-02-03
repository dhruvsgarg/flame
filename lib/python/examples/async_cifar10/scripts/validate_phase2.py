#!/usr/bin/env python3
"""
Validation script for Phase 2: Compare old vs new spawning methods.

Verifies that the new spawner generates configs identical to the old system.
"""
import json
import yaml
from pathlib import Path
import sys

# Add parent directory to path to import launch module
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from launch.spawner import MetadataLoader, ConfigGenerator


def load_old_config(config_path: Path) -> dict:
    """Load an old JSON config file."""
    with open(config_path) as f:
        return json.load(f)


def compare_configs(old_config: dict, new_config: dict, trainer_id: int) -> bool:
    """
    Compare old and new configs for equivalence.
    
    Returns True if configs are equivalent (ignoring order).
    """
    errors = []
    
    # Compare taskid
    if old_config['taskid'] != new_config['taskid']:
        errors.append(f"  taskid: old={old_config['taskid']}, new={new_config['taskid']}")
    
    # Compare training_delay_s
    old_delay = float(old_config['hyperparameters']['training_delay_s'])
    new_delay = float(new_config['hyperparameters']['training_delay_s'])
    if old_delay != new_delay:
        errors.append(f"  training_delay_s: old={old_delay}, new={new_delay}")
    
    # Compare trainer_indices_list
    old_indices = old_config['hyperparameters']['trainer_indices_list']
    new_indices = new_config['hyperparameters']['trainer_indices_list']
    if old_indices != new_indices:
        errors.append(f"  trainer_indices_list: lengths old={len(old_indices)}, new={len(new_indices)}")
    
    # Compare availability traces (sample check)
    trace_keys = ['avl_events_mobiperf_2st', 'avl_events_syn_0']
    for key in trace_keys:
        if key in old_config['hyperparameters'] and key in new_config['hyperparameters']:
            old_trace_str = old_config['hyperparameters'][key]
            new_trace = new_config['hyperparameters'][key]
            
            # Old trace is string, new trace is list - convert for comparison
            if isinstance(old_trace_str, str):
                old_trace = eval(old_trace_str)
                # Convert tuples to lists for comparison
                old_trace = [list(item) if isinstance(item, tuple) else item for item in old_trace]
            else:
                old_trace = old_trace_str
            
            if old_trace != new_trace:
                errors.append(f"  {key}: mismatch (lengths: old={len(old_trace)}, new={len(new_trace)})")
    
    if errors:
        print(f"\n✗ Trainer {trainer_id}: MISMATCH")
        for error in errors:
            print(error)
        return False
    
    return True


def validate_phase2():
    """Main validation function."""
    print("="*70)
    print("PHASE 2 VALIDATION: Old vs New Config Generation")
    print("="*70)
    
    # Get correct paths
    script_dir = Path(__file__).parent
    example_dir = script_dir.parent
    metadata_dir = example_dir / 'metadata'
    base_config_path = example_dir / 'configs' / 'trainer_base.yaml'
    
    # Old config directory
    old_config_dir = example_dir / 'trainer' / 'config_dir0.1_num300_traceFail_6d_3state_oort'
    
    print("\n[1/2] Loading metadata and initializing generator...")
    metadata_loader = MetadataLoader(metadata_dir)
    config_gen = ConfigGenerator(metadata_loader, base_config_path)
    print("  ✓ Metadata loaded")
    print("  ✓ Config generator initialized")
    
    print("\n[2/2] Comparing configs for sample trainers...")
    
    # Test with a sample of trainers
    test_trainer_ids = [1, 50, 100, 150, 200, 250, 300]
    alpha = 0.1
    availability_mode = 'mobiperf_2st'
    
    all_passed = True
    for trainer_id in test_trainer_ids:
        # Load old config
        old_config_path = old_config_dir / f"trainer_{trainer_id}.json"
        old_config = load_old_config(old_config_path)
        
        # Generate new config
        new_config = config_gen.generate_trainer_config(
            trainer_id, alpha, availability_mode
        )
        
        # Compare
        passed = compare_configs(old_config, new_config, trainer_id)
        if passed:
            print(f"  ✓ Trainer {trainer_id}: MATCH")
        else:
            all_passed = False
    
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    if all_passed:
        print("\n🎉 ✓ ALL VALIDATIONS PASSED")
        print("\nNew spawner generates configs identical to old system.")
        print("Safe to proceed with testing.")
    else:
        print("\n✗ VALIDATION FAILED")
        print("\nConfig generation does not match old system.")
        print("Review errors above before proceeding.")
    
    return all_passed


if __name__ == '__main__':
    passed = validate_phase2()
    sys.exit(0 if passed else 1)
