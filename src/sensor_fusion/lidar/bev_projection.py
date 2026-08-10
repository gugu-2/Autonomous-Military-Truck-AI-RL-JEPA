"""Projects LiDAR point clouds to Bird's Eye View (BEV) feature maps."""

import numpy as np


class BEVProjector:
    """Projects LiDAR point clouds into 2D BEV feature maps."""

    def __init__(self, config: dict):
        self.range_m = config.get("range_m", 100.0)
        self.resolution = config.get("resolution", 0.1)
        self.height_bins = config.get("height_bins", 32)
        self.min_z = config.get("min_z", -3.0)
        self.max_z = config.get("max_z", 5.0)

        self.grid_size = int((self.range_m * 2) / self.resolution)

    def _remove_ground_plane(
        self, points: np.ndarray, distance_threshold: float = 0.2
    ) -> np.ndarray:
        """Simple RANSAC-like or height-based ground removal.
        Using height threshold for speed in this implementation."""
        # A full RANSAC would sample points, fit plane, find inliers.
        # Here we do a fast pseudo-ground removal based on z-height
        # relative to sensor (assuming sensor is at ~1.8m height)
        # Points below a certain threshold are considered ground
        non_ground_mask = points[:, 2] > (self.min_z + 0.5)
        return points[non_ground_mask]

    def _voxelize(self, points: np.ndarray):
        """Group points into 2D voxels in the BEV plane."""
        # Points are (N, 4) -> [x, y, z, intensity]
        x_bins = np.floor((points[:, 0] + self.range_m) / self.resolution).astype(np.int32)
        y_bins = np.floor((points[:, 1] + self.range_m) / self.resolution).astype(np.int32)

        # Filter points outside range
        valid_mask = (
            (x_bins >= 0) & (x_bins < self.grid_size) & (y_bins >= 0) & (y_bins < self.grid_size)
        )

        x_bins = x_bins[valid_mask]
        y_bins = y_bins[valid_mask]
        valid_points = points[valid_mask]

        # Unique voxel indices (y * grid_size + x)
        voxel_indices = y_bins * self.grid_size + x_bins

        # Sort points by voxel index for efficient grouping
        sort_idx = np.argsort(voxel_indices)
        voxel_indices = voxel_indices[sort_idx]
        valid_points = valid_points[sort_idx]

        # Find boundaries of unique voxels
        _, unique_idx, counts = np.unique(voxel_indices, return_index=True, return_counts=True)

        return voxel_indices[unique_idx], valid_points, unique_idx, counts

    def _compute_bev_channels(self, voxels) -> np.ndarray:
        """Compute the 8 BEV channels from voxelized points."""
        voxel_indices, points, unique_idx, counts = voxels

        # Output tensor: (H, W, 8)
        # channels: [max_height, mean_height, min_height, intensity, density, ground_flag, obstacle_height, std_height]
        bev_map = np.zeros((self.grid_size, self.grid_size, 8), dtype=np.float32)

        for i in range(len(unique_idx)):
            start_idx = unique_idx[i]
            count = counts[i]
            end_idx = start_idx + count

            voxel_points = points[start_idx:end_idx]
            z_vals = voxel_points[:, 2]
            intensities = voxel_points[:, 3] if voxel_points.shape[1] > 3 else np.zeros_like(z_vals)

            flat_idx = voxel_indices[i]
            y = flat_idx // self.grid_size
            x = flat_idx % self.grid_size

            max_z = np.max(z_vals)
            min_z = np.min(z_vals)

            bev_map[y, x, 0] = max_z
            bev_map[y, x, 1] = np.mean(z_vals)
            bev_map[y, x, 2] = min_z
            bev_map[y, x, 3] = np.mean(intensities)
            bev_map[y, x, 4] = min(1.0, np.log(count + 1) / np.log(64))  # Normalized density
            bev_map[y, x, 5] = 1.0 if min_z < (self.min_z + 0.5) else 0.0  # Ground flag
            bev_map[y, x, 6] = max_z - min_z  # Obstacle height
            bev_map[y, x, 7] = np.std(z_vals) if count > 1 else 0.0

        return bev_map

    def project(self, points: np.ndarray) -> np.ndarray:
        """Main projection function."""
        if len(points) == 0:
            return np.zeros((self.grid_size, self.grid_size, 8), dtype=np.float32)

        # Optional: points = self._remove_ground_plane(points)
        voxels = self._voxelize(points)
        bev_map = self._compute_bev_channels(voxels)
        return bev_map
