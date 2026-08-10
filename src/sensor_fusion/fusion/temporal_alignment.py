"""Temporal alignment and synchronization for multi-modal sensor streams."""
import numpy as np
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class SensorTimestamp:
    """A timestamped sensor data sample."""
    sensor_id: str
    timestamp: float
    data: Any

class TemporalAligner:
    """Ring buffer per sensor, interpolates/extrapolates data to common timestamp."""
    
    def __init__(self, max_buffer_size: int = 50, max_drift_ms: float = 100.0):
        self.max_buffer_size = max_buffer_size
        self.max_drift_s = max_drift_ms / 1000.0
        self.buffers: Dict[str, deque] = {}
    
    def add_sample(self, sensor_id: str, timestamp: float, data: Any):
        """Add a sensor sample to its ring buffer."""
        if sensor_id not in self.buffers:
            self.buffers[sensor_id] = deque(maxlen=self.max_buffer_size)
        self.buffers[sensor_id].append(SensorTimestamp(sensor_id, timestamp, data))
        
    def get_aligned_batch(self, target_timestamp: float) -> Dict[str, Any]:
        """Retrieve a synchronized batch of sensor data for the target timestamp."""
        aligned_data = {}
        for sensor_id, buffer in self.buffers.items():
            if not buffer:
                continue
                
            # Find the closest samples before and after the target timestamp
            idx_after = 0
            while idx_after < len(buffer) and buffer[idx_after].timestamp < target_timestamp:
                idx_after += 1
                
            if idx_after == 0:
                # Target is before all samples
                sample = buffer[0]
                if abs(sample.timestamp - target_timestamp) <= self.max_drift_s:
                    aligned_data[sensor_id] = sample.data
            elif idx_after == len(buffer):
                # Target is after all samples
                sample = buffer[-1]
                if abs(sample.timestamp - target_timestamp) <= self.max_drift_s:
                    aligned_data[sensor_id] = sample.data
            else:
                # Target is between two samples
                sample_before = buffer[idx_after - 1]
                sample_after = buffer[idx_after]
                
                dt_before = target_timestamp - sample_before.timestamp
                dt_after = sample_after.timestamp - target_timestamp
                
                # Check drift
                if dt_before <= self.max_drift_s or dt_after <= self.max_drift_s:
                    # Depending on sensor type, interpolate or use nearest
                    # For simplicity, using nearest neighbor for all for now, 
                    # but camera frames could be interpolated (flow/morph) or IMU could be linearly interpolated
                    if dt_before < dt_after:
                        aligned_data[sensor_id] = sample_before.data
                    else:
                        aligned_data[sensor_id] = sample_after.data
                        
        return aligned_data
        
    def get_sync_quality(self) -> float:
        """Calculate the overall synchronization quality (0.0 to 1.0)."""
        # A simple metric: proportion of sensors that have recent data
        if not self.buffers:
            return 0.0
        
        current_time = time.time()
        active_sensors = 0
        
        for buffer in self.buffers.values():
            if buffer and (current_time - buffer[-1].timestamp) < self.max_drift_s:
                active_sensors += 1
                
        return float(active_sensors) / len(self.buffers)
