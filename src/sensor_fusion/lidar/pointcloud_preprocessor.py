import numpy as np

class PointCloudPreprocessor:
    """Filters out noise and ground planes from raw point clouds."""
    def filter_ground(self, pointcloud: np.ndarray) -> np.ndarray:
        # Simple z-threshold filter for mockup
        return pointcloud[pointcloud[:, 2] > 0.1]
