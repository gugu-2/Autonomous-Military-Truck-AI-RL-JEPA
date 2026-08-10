"""Hard safety interlock — overrides RL when physical safety is at risk."""
import torch
import numpy as np
from typing import Tuple, Dict, Any
from enum import Enum, auto
from rl_controller.policy.action_space import DrivingAction
from rl_controller.policy.reward_function import UnifiedWorldState

class SafetyFlag(Enum):
    SAFE = auto()
    EMERGENCY_STOP = auto()
    FAILSAFE = auto()
    CLAMPED = auto()
    SPEED_LIMITED = auto()

class SafetyInterlock:
    def __init__(self, config: Dict[str, Any]):
        self.emergency_threshold = config.get('emergency_threshold', 0.9)
        self.min_ttc = config.get('min_ttc', 1.2)
        self.max_steering_rate = config.get('max_steering_rate', 0.5)
        self.school_zone_speed_limit = 25.0 / 3.6 # m/s

    def compute_ttc(self, world_state: UnifiedWorldState, ego_speed: float) -> float:
        """Computes minimum Time-To-Collision to any front obstacle."""
        # Mock logic assuming world_state provides closest obstacle dist/speed
        dist = getattr(world_state, 'closest_obstacle_dist', 100.0)
        rel_speed = ego_speed - getattr(world_state, 'closest_obstacle_speed', 0.0)
        
        if rel_speed <= 0:
            return float('inf')
        return dist / rel_speed

    def compute_stopping_distance(self, speed_ms: float, decel: float = 6.0) -> float:
        """Computes theoretical stopping distance."""
        reaction_time = 0.2
        return (speed_ms * reaction_time) + ((speed_ms ** 2) / (2 * decel))

    def check_and_override(self, proposed_action: DrivingAction, hazard_energy: float, 
                           world_state: UnifiedWorldState, prev_action: DrivingAction) -> Tuple[DrivingAction, SafetyFlag, str]:
        
        safe_action = DrivingAction(**proposed_action.__dict__)
        
        # Check 1: JEPA Hazard Energy Emergency
        if hazard_energy >= self.emergency_threshold:
            safe_action.throttle = 0.0
            safe_action.brake = 1.0
            safe_action.steering = prev_action.steering if prev_action else 0.0
            return safe_action, SafetyFlag.EMERGENCY_STOP, f"Hazard energy {hazard_energy:.2f} >= {self.emergency_threshold}"

        # Check 2: Minimum TTC Failsafe
        ttc = self.compute_ttc(world_state, world_state.speed)
        if ttc < self.min_ttc:
            safe_action.throttle = 0.0
            safe_action.brake = 1.0
            return safe_action, SafetyFlag.FAILSAFE, f"TTC {ttc:.2f}s < {self.min_ttc}s"

        # Check 4: School Zone Speed Limit
        in_school_zone = getattr(world_state, 'in_school_zone', False)
        if in_school_zone and world_state.speed >= self.school_zone_speed_limit:
            safe_action.throttle = 0.0
            safe_action.brake = max(0.2, safe_action.brake)
            return safe_action, SafetyFlag.SPEED_LIMITED, f"School zone limit enforced (Speed: {world_state.speed * 3.6:.1f} km/h)"

        # Check 5: Stopping Distance Constraint
        stop_dist = self.compute_stopping_distance(world_state.speed)
        clear_dist = getattr(world_state, 'clear_path_distance', 100.0)
        if stop_dist > clear_dist * 0.8:
            safe_action.throttle = 0.0
            safe_action.brake = max(0.5, safe_action.brake)
            return safe_action, SafetyFlag.FAILSAFE, f"Stopping distance {stop_dist:.1f}m > 80% clear path {clear_dist:.1f}m"

        # Check 3: Steering Rate Clamping
        if prev_action is not None:
            dt = 0.1 # assuming 10Hz control
            steer_delta = safe_action.steering - prev_action.steering
            max_delta = self.max_steering_rate * dt
            if abs(steer_delta) > max_delta:
                safe_action.steering = prev_action.steering + np.sign(steer_delta) * max_delta
                return safe_action, SafetyFlag.CLAMPED, f"Steering rate clamped to {self.max_steering_rate} rad/s"

        return safe_action, SafetyFlag.SAFE, "No safety violations detected."
