import torch
import numpy as np
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class LocalPathPlanner:
    """
    Generates a smooth, drivable trajectory along the global route.
    Operates at a higher frequency than the global planner.
    The RL controller uses this as a reference or "progress" reward signal.
    """
    def __init__(self, config: Dict[str, Any]):
        self.lookahead_distance = config.get('lookahead_distance', 50.0) # meters
        self.trajectory_resolution = config.get('trajectory_resolution', 0.5) # meters
        
    def generate_trajectory(self, current_pose: Dict[str, float], global_route: List[Dict[str, Any]]) -> np.ndarray:
        """
        Generates a local trajectory (x,y,yaw,velocity) from the current position.
        
        Args:
            current_pose: {'x': float, 'y': float, 'yaw': float, 'velocity': float}
            global_route: List of global waypoints
            
        Returns:
            np.ndarray of shape (N, 4) where N is number of trajectory points
        """
        # In a full implementation, this uses spline interpolation or clothoid generation
        # to ensure kinematic feasibility.
        
        if not global_route:
            return np.zeros((0, 4))
            
        # Mock straight line trajectory for template
        num_points = int(self.lookahead_distance / self.trajectory_resolution)
        trajectory = np.zeros((num_points, 4))
        
        start_x, start_y = current_pose['x'], current_pose['y']
        yaw = current_pose.get('yaw', 0.0)
        vel = current_pose.get('velocity', 5.0)
        
        for i in range(num_points):
            dist = i * self.trajectory_resolution
            trajectory[i, 0] = start_x + (dist * np.cos(yaw)) # X
            trajectory[i, 1] = start_y + (dist * np.sin(yaw)) # Y
            trajectory[i, 2] = yaw                            # Yaw
            trajectory[i, 3] = vel                            # Target velocity
            
        return trajectory
