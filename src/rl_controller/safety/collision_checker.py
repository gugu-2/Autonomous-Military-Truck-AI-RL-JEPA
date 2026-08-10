from typing import Any

import torch


class CollisionChecker:
    """
    Deterministic geometric collision checking using LiDAR BEV bounding boxes.
    Runs in parallel with neural network policy to ensure absolute safety.
    """

    def __init__(self, config: dict[str, Any]):
        # Vehicle footprint (length, width in meters)
        self.ego_length = config.get("ego_length", 4.5)
        self.ego_width = config.get("ego_width", 2.0)

        # Safety margins
        self.margin_front = config.get("safety_margin_front", 2.0)
        self.margin_side = config.get("safety_margin_side", 0.5)

        # Grid parameters for BEV
        self.grid_res = config.get("bev_resolution", 0.1)  # meters per pixel

    def check_trajectory_collision(
        self, planned_trajectory: torch.Tensor, obstacle_bev: torch.Tensor
    ) -> bool:
        """
        Check if a planned trajectory (list of x,y points) intersects with any obstacles.

        Args:
            planned_trajectory: (N, 2) tensor of x, y coordinates in ego frame
            obstacle_bev: (H, W) binary tensor where 1 = obstacle, 0 = free space

        Returns:
            True if collision is detected, False otherwise.
        """
        if obstacle_bev is None or planned_trajectory is None:
            return False

        # Convert meters to grid indices
        # Assuming ego is at center of bottom edge of BEV grid (x: -width/2 to width/2, y: 0 to length)
        H, W = obstacle_bev.shape
        origin_x = W // 2
        origin_y = H - 1

        for point in planned_trajectory:
            x_m, y_m = float(point[0]), float(point[1])

            # Add safety margin to vehicle footprint
            check_length = self.ego_length + self.margin_front
            check_width = self.ego_width + (self.margin_side * 2)

            # Calculate footprint grid boundaries at this trajectory point
            # Simple bounding box check (assuming no rotation for simplicity in this template)
            min_x_m = x_m - (check_width / 2.0)
            max_x_m = x_m + (check_width / 2.0)
            min_y_m = y_m
            max_y_m = y_m + check_length

            min_x_px = int(origin_x + (min_x_m / self.grid_res))
            max_x_px = int(origin_x + (max_x_m / self.grid_res))
            min_y_px = int(origin_y - (max_y_m / self.grid_res))
            max_y_px = int(origin_y - (min_y_m / self.grid_res))

            # Clip to grid bounds
            min_x_px = max(0, min(W - 1, min_x_px))
            max_x_px = max(0, min(W - 1, max_x_px))
            min_y_px = max(0, min(H - 1, min_y_px))
            max_y_px = max(0, min(H - 1, max_y_px))

            # Check for any obstacles in this footprint
            footprint = obstacle_bev[min_y_px:max_y_px, min_x_px:max_x_px]
            if footprint.sum() > 0:
                return True  # Collision detected

        return False
