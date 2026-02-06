# Copyright 2023 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
"""REFL Availability Tracker for client availability management."""

import logging
import pickle
from typing import Dict, List, Tuple, Optional
import yaml

logger = logging.getLogger(__name__)


class REFLAvailabilityTracker:
    """
    Tracks client availability based on REFL's availability trace format.
    
    Supports both pickle (REFL native) and YAML formats for traces.
    Each trace contains availability periods for each client.
    """

    def __init__(self, trace_file_path: Optional[str] = None):
        """
        Initialize the availability tracker.
        
        Args:
            trace_file_path: Path to trace file (pickle or YAML format)
        """
        self.traces = {}
        self.availability_periods = {}
        
        if trace_file_path:
            self.load_traces(trace_file_path)
            self.compute_availability_periods()
    
    def load_traces(self, trace_file_path: str) -> None:
        """
        Load availability traces from file.
        
        Args:
            trace_file_path: Path to pickle or YAML file containing traces
        """
        try:
            # Try loading as pickle first (REFL native format)
            if trace_file_path.endswith('.pkl') or trace_file_path.endswith('.pickle'):
                with open(trace_file_path, 'rb') as fin:
                    self.traces = pickle.load(fin)
                logger.info(f"Loaded {len(self.traces)} client traces from pickle: {trace_file_path}")
            
            # Try loading as YAML
            elif trace_file_path.endswith('.yaml') or trace_file_path.endswith('.yml'):
                with open(trace_file_path, 'r') as fin:
                    data = yaml.safe_load(fin)
                    
                # Handle both direct trace format and metadata format
                if 'traces' in data:
                    self.traces = data['traces']
                else:
                    self.traces = data
                    
                logger.info(f"Loaded {len(self.traces)} client traces from YAML: {trace_file_path}")
            
            else:
                logger.warning(f"Unknown trace file format: {trace_file_path}")
                self.traces = {}
                
        except Exception as e:
            logger.error(f"Failed to load traces from {trace_file_path}: {e}")
            self.traces = {}
    
    def compute_availability_periods(self) -> None:
        """
        Compute availability periods from raw traces.
        
        Converts trace data into [(start_time, end_time), ...] format
        for efficient availability queries.
        """
        self.availability_periods = {}
        
        for client_id, trace_data in self.traces.items():
            if isinstance(trace_data, dict):
                # YAML format: {periods: [[start, end], ...], duration: X}
                if 'periods' in trace_data:
                    self.availability_periods[client_id] = [
                        tuple(period) for period in trace_data['periods']
                    ]
                # Pickle format: {duration: X, ...} with complex structure
                elif 'duration' in trace_data:
                    # For now, assume always available if no periods specified
                    duration = trace_data['duration']
                    self.availability_periods[client_id] = [(0, duration)]
            else:
                # Simple format: just a list of periods
                self.availability_periods[client_id] = trace_data
        
        logger.info(f"Computed availability periods for {len(self.availability_periods)} clients")
    
    def is_available(
        self, 
        client_id: str, 
        cur_time: float, 
        time_window: float,
        time_slots: int = 1
    ) -> bool:
        """
        Check if client is available during a specific time window.
        
        Args:
            client_id: Client identifier
            cur_time: Current virtual time
            time_window: Duration of time window to check
            time_slots: Number of time slots to check (default 1)
            
        Returns:
            True if client is available for entire duration, False otherwise
        """
        client_id = str(client_id)
        
        if client_id not in self.availability_periods:
            # If no trace data, assume always available
            return True
        
        periods = self.availability_periods[client_id]
        if not periods:
            return True
        
        # Get finish time for wrapping
        finish_time = max(period[1] for period in periods)
        
        # Normalize time to wrap around trace duration
        norm_time = cur_time % finish_time
        start_time = norm_time + (time_slots - 1) * time_window
        end_time = norm_time + time_slots * time_window
        
        # Check if any availability period covers [start_time, end_time]
        for period_start, period_end in periods:
            if period_start <= start_time and period_end >= end_time:
                return True
        
        return False
    
    def is_client_active(
        self, 
        client_id: str, 
        cur_time: float, 
        time_window: float
    ) -> bool:
        """
        Check if client will be active at the end of a time window.
        
        This is used to filter out clients who will disconnect before
        training completes.
        
        Args:
            client_id: Client identifier
            cur_time: Current virtual time
            time_window: Duration until check point (e.g., training duration)
            
        Returns:
            True if client will be active at cur_time + time_window
        """
        return self.is_available(client_id, cur_time, time_window, time_slots=1)
    
    def get_priority(
        self, 
        client_id: str, 
        cur_time: float, 
        time_window: float,
        lookup_timeslots: int = 2
    ) -> int:
        """
        Get priority of client based on near-term availability.
        
        Priority is based on how soon the client will become unavailable:
        - Priority 2: Available now
        - Priority 1: Available in next timeslot
        - Priority 0: Not available in near term
        
        Args:
            client_id: Client identifier
            cur_time: Current virtual time
            time_window: Duration of each time slot
            lookup_timeslots: Number of future timeslots to check
            
        Returns:
            Priority value (0 to lookup_timeslots)
        """
        priority = 0
        
        for i in range(lookup_timeslots, 0, -1):
            if self.is_available(client_id, cur_time, time_window, i):
                priority = lookup_timeslots - i
                break
        
        return priority
    
    def get_period_count(
        self, 
        client_id: str, 
        cur_time: float, 
        deadline: float
    ) -> int:
        """
        Count number of availability periods divided into deadline-sized slots.
        
        Args:
            client_id: Client identifier
            cur_time: Current virtual time
            deadline: Duration of each slot
            
        Returns:
            Number of deadline-sized availability periods
        """
        client_id = str(client_id)
        
        if client_id not in self.availability_periods:
            return 0
        
        periods = self.availability_periods[client_id]
        if not periods:
            return 0
        
        finish_time = max(period[1] for period in periods)
        norm_time = cur_time % finish_time
        
        # Find current period index
        index = 0
        for period_start, period_end in periods:
            if norm_time < period_start:
                break
            index += 1
        
        count = 0
        
        # Count remaining time in current period
        if index > 0:
            v1, v2 = periods[index - 1]
            remaining_in_current = v2 - norm_time
            count += int(remaining_in_current / deadline)
        
        # Count future periods
        for i in range(index, len(periods)):
            start, end = periods[i]
            duration_normed = int((end - start) / deadline)
            if duration_normed > 0:
                count += duration_normed
        
        return count
    
    def get_online_clients(
        self, 
        client_ids: List[str], 
        cur_time: float
    ) -> List[str]:
        """
        Get list of clients that are currently online.
        
        Args:
            client_ids: List of all client IDs to check
            cur_time: Current virtual time
            
        Returns:
            List of client IDs that are currently online
        """
        online = []
        
        for client_id in client_ids:
            if self.is_available(client_id, cur_time, 0, time_slots=1):
                online.append(client_id)
        
        return online
    
    def split_by_priority(
        self,
        client_ids: List[str],
        cur_time: float,
        time_window: float,
        lookup_timeslots: int = 2,
        accuracy: float = 1.0
    ) -> Tuple[List[str], List[str]]:
        """
        Split clients into high-priority and remaining based on availability.
        
        Args:
            client_ids: List of all client IDs
            cur_time: Current virtual time
            time_window: Duration of time window (e.g., round duration)
            lookup_timeslots: Number of timeslots for priority calculation
            accuracy: Probability of correct prediction (0-1), used for simulation
            
        Returns:
            Tuple of (priority_clients, remaining_clients)
        """
        import random
        
        priority_vals = {}
        for client_id in client_ids:
            priority_vals[client_id] = self.get_priority(
                client_id, cur_time, time_window, lookup_timeslots
            )
        
        # High priority: clients with priority == 1 (available in next timeslot)
        priority_clients = [cid for cid, p in priority_vals.items() if p == 1]
        remaining_clients = [cid for cid, p in priority_vals.items() if p == 0]
        
        # Apply accuracy factor (simulate prediction errors)
        if accuracy < 1.0:
            acc_num_priority = int(len(priority_clients) * accuracy)
            priority_clients = random.sample(priority_clients, acc_num_priority) if priority_clients else []
            
            acc_num_remaining = int(len(remaining_clients) * accuracy)
            remaining_clients = random.sample(remaining_clients, acc_num_remaining) if remaining_clients else []
        
        return priority_clients, remaining_clients
    
    def get_trace_duration(self, client_id: str) -> float:
        """
        Get total duration of trace for a client.
        
        Args:
            client_id: Client identifier
            
        Returns:
            Duration of trace in seconds (or time units)
        """
        client_id = str(client_id)
        
        if client_id not in self.availability_periods:
            return float('inf')
        
        periods = self.availability_periods[client_id]
        if not periods:
            return float('inf')
        
        return max(period[1] for period in periods)
