"""Action space definitions and constraints for OMNIDRIVE driving policy."""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

import numpy as np
import torch


class VehicleMode(Enum):
    ROBOTAXI = auto()
    TRUCK = auto()
    MILITARY = auto()


@dataclass
class DrivingAction:
    steering: float
    throttle: float
    brake: float
    trailer_brake: float = 0.0
    retarder: float = 0.0
    mission_halt: bool = False


@dataclass
class ActionBounds:
    steering_min: float = -1.0
    steering_max: float = 1.0
    throttle_min: float = 0.0
    throttle_max: float = 1.0
    brake_min: float = 0.0
    brake_max: float = 1.0
    trailer_brake_min: float = 0.0
    trailer_brake_max: float = 1.0
    retarder_min: float = 0.0
    retarder_max: float = 1.0
    max_steering_rate: float = 0.8
    max_jerk: float = 3.0


class ActionSpace:
    def __init__(self, config: dict[str, Any], vehicle_mode: VehicleMode):
        self.config = config
        self.vehicle_mode = vehicle_mode
        self.bounds = ActionBounds(**config.get("bounds", {}))

        # Base dimensions (steering, throttle, brake)
        self.dim = 3
        if self.vehicle_mode == VehicleMode.TRUCK:
            self.dim += 2  # trailer_brake, retarder
        elif self.vehicle_mode == VehicleMode.MILITARY:
            self.dim += 1  # mission_halt (continuous relaxed to discrete)

    def sample(self) -> DrivingAction:
        """Sample a random valid action within bounds."""
        action = DrivingAction(
            steering=np.random.uniform(self.bounds.steering_min, self.bounds.steering_max),
            throttle=np.random.uniform(self.bounds.throttle_min, self.bounds.throttle_max),
            brake=np.random.uniform(self.bounds.brake_min, self.bounds.brake_max),
        )
        if self.vehicle_mode == VehicleMode.TRUCK:
            action.trailer_brake = np.random.uniform(
                self.bounds.trailer_brake_min, self.bounds.trailer_brake_max
            )
            action.retarder = np.random.uniform(self.bounds.retarder_min, self.bounds.retarder_max)
        elif self.vehicle_mode == VehicleMode.MILITARY:
            action.mission_halt = np.random.choice([True, False])
        return action

    def clip(self, action: DrivingAction) -> DrivingAction:
        """Clips action to valid min/max bounds."""
        clipped = DrivingAction(
            steering=np.clip(action.steering, self.bounds.steering_min, self.bounds.steering_max),
            throttle=np.clip(action.throttle, self.bounds.throttle_min, self.bounds.throttle_max),
            brake=np.clip(action.brake, self.bounds.brake_min, self.bounds.brake_max),
            trailer_brake=np.clip(
                action.trailer_brake, self.bounds.trailer_brake_min, self.bounds.trailer_brake_max
            ),
            retarder=np.clip(action.retarder, self.bounds.retarder_min, self.bounds.retarder_max),
            mission_halt=action.mission_halt,
        )
        return clipped

    def apply_rate_limits(
        self, action: DrivingAction, prev_action: DrivingAction, dt: float
    ) -> DrivingAction:
        """Enforces max steering rate and max jerk."""
        if prev_action is None:
            return self.clip(action)

        max_steer_delta = self.bounds.max_steering_rate * dt
        max_accel_delta = self.bounds.max_jerk * dt

        steer = np.clip(
            action.steering,
            prev_action.steering - max_steer_delta,
            prev_action.steering + max_steer_delta,
        )
        throttle = np.clip(
            action.throttle,
            prev_action.throttle - max_accel_delta,
            prev_action.throttle + max_accel_delta,
        )
        brake = np.clip(
            action.brake, prev_action.brake - max_accel_delta, prev_action.brake + max_accel_delta
        )

        rate_limited_action = DrivingAction(
            steering=float(steer),
            throttle=float(throttle),
            brake=float(brake),
            trailer_brake=action.trailer_brake,
            retarder=action.retarder,
            mission_halt=action.mission_halt,
        )
        return self.clip(rate_limited_action)

    def to_tensor(self, action: DrivingAction) -> torch.Tensor:
        """Converts DrivingAction to an RL tensor."""
        vec = [action.steering, action.throttle, action.brake]
        if self.vehicle_mode == VehicleMode.TRUCK:
            vec.extend([action.trailer_brake, action.retarder])
        elif self.vehicle_mode == VehicleMode.MILITARY:
            vec.append(1.0 if action.mission_halt else -1.0)
        return torch.tensor(vec, dtype=torch.float32)

    def from_tensor(self, tensor: torch.Tensor) -> DrivingAction:
        """Converts RL tensor back to DrivingAction."""
        t = tensor.detach().cpu().numpy()
        action = DrivingAction(steering=float(t[0]), throttle=float(t[1]), brake=float(t[2]))
        if self.vehicle_mode == VehicleMode.TRUCK and len(t) >= 5:
            action.trailer_brake = float(t[3])
            action.retarder = float(t[4])
        elif self.vehicle_mode == VehicleMode.MILITARY and len(t) >= 4:
            action.mission_halt = bool(t[3] > 0.0)

        return self.clip(action)

    def is_valid(self, action: DrivingAction) -> bool:
        """Checks if an action is strictly within bounds."""
        a = self.clip(action)
        return (
            np.isclose(a.steering, action.steering)
            and np.isclose(a.throttle, action.throttle)
            and np.isclose(a.brake, action.brake)
            and np.isclose(a.trailer_brake, action.trailer_brake)
            and np.isclose(a.retarder, action.retarder)
        )
