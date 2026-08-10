import numpy as np


class LidarDriver:
    """Hardware interface for LiDAR sensors."""

    def get_pointcloud(self) -> np.ndarray:
        # Mock returning a (N, 4) pointcloud [x, y, z, intensity]
        return np.random.randn(10000, 4)
