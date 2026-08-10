"""
Common Data Types for OMNIDRIVE Project
"""
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import List, Optional

import numpy as np
import torch


class SafetyFlag(IntEnum):
    NOMINAL = 0
    WARN = 1
    DEGRADED = 2
    FAILSAFE = 3
    EMERGENCY_STOP = 4


class VehicleMode(str, Enum):
    ROBOTAXI = 'robotaxi'
    TRUCK = 'truck'
    MILITARY = 'military'


class HazardLevel(IntEnum):
    CLEAR = 0
    LOW = 1
    WARN = 2
    CRITICAL = 3


# Error codes
class OmniDriveError(Exception):
    """Base exception for OMNIDRIVE errors."""
    pass


class SensorFailureError(OmniDriveError):
    """Raised when a sensor fails."""
    pass


class JEPAInferenceError(OmniDriveError):
    """Raised during JEPA inference failure."""
    pass


class SafetyVetoError(OmniDriveError):
    """Raised when a safety check vetoes an action."""
    pass


class VehicleInterfaceError(OmniDriveError):
    """Raised on vehicle communication errors."""
    pass


@dataclass
class EgoPose:
    x: float  # metres, East
    y: float  # metres, North
    z: float  # metres, Up
    roll: float   # radians
    pitch: float  # radians
    yaw: float    # radians
    vx: float     # m/s body frame
    vy: float
    vz: float
    timestamp: float  # Unix epoch seconds


@dataclass
class SensorHealthStatus:
    camera_ok: bool
    lidar_ok: bool
    radar_ok: bool
    gps_ok: bool
    imu_ok: bool
    thermal_ok: bool  # military only
    degradation_level: int  # 0=full, 1=minor, 2=moderate, 3=major, 4=failsafe


@dataclass
class TrackedObject3D:
    object_id: int
    class_label: str   # 'vehicle', 'pedestrian', 'cyclist', 'obstacle', 'soldier'
    x: float  # metres, world frame
    y: float
    z: float
    length: float
    width: float
    height: float
    yaw: float
    vx: float  # velocity m/s
    vy: float
    confidence: float  # 0.0-1.0
    timestamp: float


@dataclass
class BEVFeatureMap:
    data: np.ndarray  # shape (H, W, C) = (1000, 1000, 8)
    resolution: float  # metres per pixel, default 0.1
    origin_x: float   # world coords of bottom-left corner
    origin_y: float
    timestamp: float


@dataclass
class UnifiedWorldState:
    ego_pose: EgoPose
    tracked_objects: List[TrackedObject3D]
    bev_feature_map: BEVFeatureMap
    sensor_health: SensorHealthStatus
    timestamp: float
    frame_id: int


@dataclass
class LatentState:
    context_tokens: torch.Tensor  # shape (N, D) = (256, 512)
    bev_tokens: torch.Tensor      # shape (H//8, W//8, D)
    hazard_energy_map: torch.Tensor  # shape (16, 16) spatial hazard grid
    timestamp: float


@dataclass
class DrivingAction:
    steering_angle: float  # degrees, [-45, +45]
    throttle: float        # [0.0, 1.0]
    brake: float           # [0.0, 1.0]
    gear: int              # -1=reverse, 0=neutral, 1-8=forward
    # Military extensions
    mission_halt: bool = False
    # Truck extensions
    trailer_brake: float = 0.0
    retarder: float = 0.0

__all__ = [
    'SafetyFlag',
    'VehicleMode',
    'HazardLevel',
    'OmniDriveError',
    'SensorFailureError',
    'JEPAInferenceError',
    'SafetyVetoError',
    'VehicleInterfaceError',
    'EgoPose',
    'SensorHealthStatus',
    'TrackedObject3D',
    'BEVFeatureMap',
    'UnifiedWorldState',
    'LatentState',
    'DrivingAction',
]
