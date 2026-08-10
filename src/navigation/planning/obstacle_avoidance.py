import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class ObstacleAvoidancePlanner:
    """
    Modifies the local path dynamically to steer around obstacles in the BEV map.
    Acts as a safety layer below global routing but above the RL controller's
    free-space navigation.
    """

    def __init__(self, config: dict[str, Any]):
        self.lateral_avoidance_margin = config.get("lateral_avoidance_margin", 1.5)  # meters
        self.max_steering_correction = config.get("max_steering_correction", 0.5)  # rad

    def adjust_trajectory(
        self, base_trajectory: np.ndarray, obstacle_bev: np.ndarray
    ) -> np.ndarray:
        """
        Adjusts the target trajectory to avoid obstacles detected in the BEV grid.

        Args:
            base_trajectory: Output from LocalPathPlanner (N, 4)
            obstacle_bev: Binary BEV grid of obstacles (H, W)

        Returns:
            Adjusted trajectory (N, 4)
        """
        if base_trajectory.shape[0] == 0:
            return base_trajectory

        adjusted_trajectory = np.copy(base_trajectory)

        # In a full implementation, this uses Potential Fields, A*, or Elastic Bands
        # to bend the trajectory away from occupied cells in the BEV map.

        # For this template, we return the base trajectory. The actual RL controller
        # (trained in JEPA latent space) handles the immediate dynamic obstacle avoidance.
        # This module is meant for static, map-level rerouting around large blockages.

        return adjusted_trajectory
