"""
Aggregator spawner for Phase 3.

Spawns aggregator process with log capture.
"""
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional


class AggregatorSpawner:
    """Spawns aggregator process with log capture."""
    
    def __init__(self, log_file: Optional[Path] = None):
        self.log_file = log_file
        self.process = None
        self._log_handle = None
    
    def spawn(
        self,
        aggregator_main_path: Path,
        config_path: Path,
        log_to_wandb: bool = False,
        wandb_run_name: Optional[str] = None,
    ) -> subprocess.Popen:
        """
        Spawn aggregator process.
        
        Args:
            aggregator_main_path: Path to aggregator main.py
            config_path: Path to aggregator config JSON
            log_to_wandb: Enable wandb logging
            wandb_run_name: Custom wandb run name
        
        Returns:
            subprocess.Popen object
        """
        # Open log file if specified
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            self._log_handle = open(self.log_file, 'w', buffering=1)  # Line buffered
            stdout_target = self._log_handle
            stderr_target = subprocess.STDOUT
        else:
            stdout_target = subprocess.PIPE
            stderr_target = subprocess.PIPE
        
        # Build command
        cmd = [
            sys.executable,
            str(aggregator_main_path),
            str(config_path)  # Positional argument, not --config
        ]
        
        # Add optional wandb flags
        if log_to_wandb:
            cmd.append('--log_to_wandb')
            if wandb_run_name:
                cmd.extend(['--wandb_run_name', wandb_run_name])
        
        # Spawn process
        self.process = subprocess.Popen(
            cmd,
            stdout=stdout_target,
            stderr=stderr_target,
            text=True
        )
        
        print(f"  ✓ Aggregator started (PID: {self.process.pid})")
        if self.log_file:
            print(f"    Logs: {self.log_file}")
        
        return self.process
    
    def is_running(self) -> bool:
        """Check if aggregator is still running."""
        if not self.process:
            return False
        return self.process.poll() is None
    
    def wait_until_ready(self, timeout: int = 30) -> bool:
        """
        Wait for aggregator to be ready.
        
        Simple implementation: just wait fixed time and check process is alive.
        Could be enhanced with log monitoring for "ready" message.
        
        Args:
            timeout: Maximum time to wait in seconds
        
        Returns:
            True if aggregator appears ready, False otherwise
        """
        print(f"  Waiting for aggregator to be ready (up to {timeout}s)...")
        
        start_time = time.time()
        check_interval = 1.0
        
        while time.time() - start_time < timeout:
            if not self.is_running():
                print(f"  ✗ Aggregator process died")
                return False
            
            time.sleep(check_interval)
            elapsed = time.time() - start_time
            
            # Simple heuristic: if process is alive for 5 seconds, assume ready
            if elapsed >= 5:
                print(f"  ✓ Aggregator ready (process alive for {elapsed:.1f}s)")
                return True
        
        print(f"  ⚠ Timeout waiting for aggregator")
        return self.is_running()
    
    def terminate(self):
        """Terminate aggregator process."""
        if self.process:
            try:
                self.process.terminate()
                time.sleep(2)
                if self.process.poll() is None:
                    self.process.kill()
            except:
                pass
        
        if self._log_handle:
            self._log_handle.close()
            self._log_handle = None
