import numpy as np


class LidarClustering:
    """Clusters points into distinct obstacles using DBSCAN or Euclidean clustering."""

    def cluster(self, pointcloud: np.ndarray) -> list:
        # Mock clustering
        return [{"id": 1, "center": [10.0, 0.0, 0.0], "points": pointcloud[:100]}]
