"""Unified World State — the canonical output of the Sensor Fusion layer."""

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class BoundingBox3D:
    """3D bounding box for tracked objects."""

    x: float
    y: float
    z: float
    length: float
    width: float
    height: float
    yaw: float

    def to_dict(self) -> dict[str, float]:
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "length": self.length,
            "width": self.width,
            "height": self.height,
            "yaw": self.yaw,
        }


@dataclass
class EgoPose:
    """Ego vehicle pose in world coordinates."""

    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float
    velocity_x: float
    velocity_y: float
    velocity_z: float
    acceleration_x: float
    acceleration_y: float
    acceleration_z: float

    def to_dict(self) -> dict[str, float]:
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "roll": self.roll,
            "pitch": self.pitch,
            "yaw": self.yaw,
            "velocity_x": self.velocity_x,
            "velocity_y": self.velocity_y,
            "velocity_z": self.velocity_z,
            "acceleration_x": self.acceleration_x,
            "acceleration_y": self.acceleration_y,
            "acceleration_z": self.acceleration_z,
        }


@dataclass
class TrackedObject3D:
    """A tracked object in the 3D environment."""

    object_id: int
    object_class: str  # e.g., 'car', 'pedestrian', 'cyclist'
    confidence: float
    box: BoundingBox3D
    velocity_x: float
    velocity_y: float
    velocity_z: float
    history: list[BoundingBox3D] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "object_class": self.object_class,
            "confidence": self.confidence,
            "box": self.box.to_dict(),
            "velocity_x": self.velocity_x,
            "velocity_y": self.velocity_y,
            "velocity_z": self.velocity_z,
        }


@dataclass
class SensorHealthStatus:
    """Health status of the sensor suite."""

    camera_ok: bool = True
    lidar_ok: bool = True
    radar_ok: bool = True
    gps_ok: bool = True
    imu_ok: bool = True
    last_update_times: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_ok": self.camera_ok,
            "lidar_ok": self.lidar_ok,
            "radar_ok": self.radar_ok,
            "gps_ok": self.gps_ok,
            "imu_ok": self.imu_ok,
            "last_update_times": self.last_update_times,
        }


@dataclass
class UnifiedWorldState:
    """The canonical world state output of the Sensor Fusion layer."""

    timestamp: float
    ego_pose: EgoPose
    tracked_objects: list[TrackedObject3D]
    sensor_health: SensorHealthStatus
    lane_lines: list[np.ndarray] = field(default_factory=list)
    drivable_area: np.ndarray | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the world state to a dictionary."""
        return {
            "timestamp": self.timestamp,
            "ego_pose": self.ego_pose.to_dict(),
            "tracked_objects": [obj.to_dict() for obj in self.tracked_objects],
            "sensor_health": self.sensor_health.to_dict(),
        }

    def is_stale(self, max_age_s: float) -> bool:
        """Check if the world state is older than the max age."""
        return (time.time() - self.timestamp) > max_age_s

    def get_nearby_objects(self, radius_m: float) -> list[TrackedObject3D]:
        """Return objects within a specified radius from the ego vehicle."""
        nearby = []
        for obj in self.tracked_objects:
            # Calculate distance from ego (assuming ego is at 0,0 locally, or compare world coords)
            # Assuming ego_pose and object box are in the same coordinate frame
            dx = obj.box.x - self.ego_pose.x
            dy = obj.box.y - self.ego_pose.y
            dist = np.sqrt(dx**2 + dy**2)
            if dist <= radius_m:
                nearby.append(obj)
        return nearby

    def get_critical_objects(self, ttc_threshold: float, ego_speed: float) -> list[TrackedObject3D]:
        """Return objects with Time-To-Collision (TTC) below the threshold."""
        critical = []
        for obj in self.tracked_objects:
            # Simplified TTC calculation
            dx = obj.box.x - self.ego_pose.x
            dy = obj.box.y - self.ego_pose.y
            dist = np.sqrt(dx**2 + dy**2)

            # Relative velocity towards the ego vehicle
            rel_vx = obj.velocity_x - self.ego_pose.velocity_x
            rel_vy = obj.velocity_y - self.ego_pose.velocity_y
            rel_speed = np.sqrt(rel_vx**2 + rel_vy**2)

            # Assuming heading towards ego
            if rel_speed > 0:
                ttc = dist / rel_speed
                if ttc < ttc_threshold:
                    critical.append(obj)
            else:
                # If moving away or stationary relative, check if absolute distance is very small
                if dist < 2.0:
                    critical.append(obj)

        return critical
