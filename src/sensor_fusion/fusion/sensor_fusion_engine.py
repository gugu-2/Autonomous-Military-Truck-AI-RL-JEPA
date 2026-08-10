"""Master sensor fusion engine combining all sensor modalities."""

import threading
import time

import numpy as np

from src.sensor_fusion.fusion.temporal_alignment import TemporalAligner
from src.sensor_fusion.fusion.unified_world_state import (
    SensorHealthStatus,
    UnifiedWorldState,
)
from src.sensor_fusion.gps_imu.ekf_localizer import EKFLocalizer
from src.sensor_fusion.lidar.bev_projection import BEVProjector


class SensorFusionEngine:
    """Master sensor fusion engine combining all sensor modalities."""

    def __init__(self, config: dict):
        self.config = config
        self.lock = threading.Lock()

        self.temporal_aligner = TemporalAligner(max_drift_ms=100.0)
        self.ekf = EKFLocalizer(config.get("ekf_config", {}))
        self.bev_projector = BEVProjector(config.get("lidar_config", {}))

        self.sensor_health = SensorHealthStatus()
        self.latest_bev: np.ndarray | None = None
        self.latest_radar_tracks: list[dict] = []

    def process_camera_frames(self, frames: list[np.ndarray], timestamp: float):
        """Process multi-camera frames."""
        with self.lock:
            self.temporal_aligner.add_sample("camera", timestamp, frames)
            self.sensor_health.camera_ok = True
            self.sensor_health.last_update_times["camera"] = time.time()

    def process_lidar(self, points: np.ndarray, timestamp: float):
        """Process LiDAR point cloud into BEV."""
        # Convert to BEV (computationally heavy, might run outside lock if optimized)
        bev_map = self.bev_projector.project(points)
        with self.lock:
            self.latest_bev = bev_map
            self.temporal_aligner.add_sample("lidar", timestamp, bev_map)
            self.sensor_health.lidar_ok = True
            self.sensor_health.last_update_times["lidar"] = time.time()

    def process_radar(self, radar_data: list[dict], timestamp: float):
        """Process radar track data."""
        with self.lock:
            self.latest_radar_tracks = radar_data
            self.temporal_aligner.add_sample("radar", timestamp, radar_data)
            self.sensor_health.radar_ok = True
            self.sensor_health.last_update_times["radar"] = time.time()

    def process_gps_imu(self, gps: dict, imu: dict, timestamp: float):
        """Process GPS and IMU data through EKF."""
        with self.lock:
            # Prediction step (IMU)
            accel = np.array([imu["ax"], imu["ay"], imu["az"]])
            gyro = np.array([imu["gx"], imu["gy"], imu["gz"]])
            dt = 0.01  # Should be calculated from timestamps
            self.ekf.predict(accel, gyro, dt)

            # Update step (GPS)
            gps_pos = np.array([gps["x"], gps["y"], gps["z"]])
            gps_cov = np.eye(3) * gps.get("accuracy", 2.0)
            self.ekf.update_gps(gps_pos, gps_cov)

            pose = self.ekf.get_pose()
            self.temporal_aligner.add_sample("pose", timestamp, pose)

            self.sensor_health.gps_ok = True
            self.sensor_health.imu_ok = True
            self.sensor_health.last_update_times["gps_imu"] = time.time()

    def fuse(self, timestamp: float) -> UnifiedWorldState:
        """Main fusion call, merges all modalities into unified state."""
        with self.lock:
            # 1. Align temporal data
            aligned_data = self.temporal_aligner.get_aligned_batch(timestamp)

            # 2. Get Pose
            pose = aligned_data.get("pose", self.ekf.get_pose())

            # 3. Object Level Fusion (Simplified placeholder)
            # In a real system, this would associate tracks from Camera, LiDAR (BEV), and Radar
            tracked_objects = []

            # If we had object detector output, we would fuse it here
            # using something like Hungarian Algorithm and Kalman Filter per object

            # 4. Check Health Status
            current_time = time.time()
            for sensor, last_update in self.sensor_health.last_update_times.items():
                if current_time - last_update > 1.0:  # 1 second timeout
                    if sensor == "camera":
                        self.sensor_health.camera_ok = False
                    if sensor == "lidar":
                        self.sensor_health.lidar_ok = False
                    if sensor == "radar":
                        self.sensor_health.radar_ok = False
                    if sensor == "gps_imu":
                        self.sensor_health.gps_ok = False
                        self.sensor_health.imu_ok = False

            return UnifiedWorldState(
                timestamp=timestamp,
                ego_pose=pose,
                tracked_objects=tracked_objects,
                sensor_health=self.sensor_health,
                drivable_area=None,  # Would be derived from camera/lidar segmentation
            )

    def get_health_status(self) -> SensorHealthStatus:
        """Returns current health status."""
        with self.lock:
            return self.sensor_health
