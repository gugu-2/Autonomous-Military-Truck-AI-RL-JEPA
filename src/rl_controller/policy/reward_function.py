"""Multi-objective reward function for OMNIDRIVE DreamerV3 RL agent."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from rl_controller.policy.action_space import DrivingAction, VehicleMode


@dataclass
class UnifiedWorldState:
    speed: float
    lateral_accel: float
    traffic_light_state: int  # 0=red, 1=yellow, 2=green
    speed_limit: float
    # Other state attributes assumed


@dataclass
class RewardBreakdown:
    r_total: float
    r_safety: float
    r_progress: float
    r_comfort: float
    r_traffic: float
    r_efficiency: float
    r_mission: float = 0.0
    r_stealth: float = 0.0


def symlog(x: float) -> float:
    return np.sign(x) * np.log(np.abs(x) + 1.0)


class RewardFunction:
    def __init__(self, config: dict[str, Any], vehicle_mode: VehicleMode):
        self.config = config
        self.vehicle_mode = vehicle_mode

        # Load mode-specific weights
        self.weights = config.get("weights", {}).get(
            vehicle_mode.name.lower(),
            {
                "safety": 1.0,
                "progress": 0.5,
                "comfort": 0.1,
                "traffic": 0.5,
                "efficiency": 0.05,
                "mission": 0.0,
                "stealth": 0.0,
            },
        )

    def compute(
        self,
        state: UnifiedWorldState,
        action: DrivingAction,
        next_state: UnifiedWorldState,
        hazard_energy: float,
        route_progress: float,
        is_collision: bool,
    ) -> RewardBreakdown:

        # Safety Reward
        if is_collision:
            r_safety = -100.0
        else:
            r_safety = -np.exp(5.0 * hazard_energy)

        # Progress Reward
        speed_factor = min(next_state.speed / max(next_state.speed_limit, 1.0), 1.2)
        r_progress = route_progress * speed_factor

        # Comfort Reward
        jerk_magnitude = np.abs(action.throttle - action.brake)  # Simplified proxy
        lateral_accel_penalty = np.abs(next_state.lateral_accel)
        r_comfort = -(jerk_magnitude + lateral_accel_penalty * 0.5)

        # Traffic Reward
        traffic_compliance = 0.0
        if next_state.traffic_light_state == 0 and next_state.speed > 0.5:
            traffic_compliance = -10.0  # Running red light
        speeding_penalty = max(0.0, next_state.speed - next_state.speed_limit)
        r_traffic = traffic_compliance - (speeding_penalty * 2.0)

        # Efficiency Reward
        fuel_rate = action.throttle**2
        r_efficiency = -fuel_rate

        # Mode-specific extensions
        r_mission = 0.0
        r_stealth = 0.0
        if self.vehicle_mode == VehicleMode.MILITARY:
            if action.mission_halt:
                r_mission = 10.0  # Reward for correctly halting if required by mission (simplified)
            # Stealth: penalize high RPM/throttle
            r_stealth = -(action.throttle * 5.0)

        # Weighted Sum
        r_total_raw = (
            self.weights["safety"] * r_safety
            + self.weights["progress"] * r_progress
            + self.weights["comfort"] * r_comfort
            + self.weights["traffic"] * r_traffic
            + self.weights["efficiency"] * r_efficiency
            + self.weights.get("mission", 0.0) * r_mission
            + self.weights.get("stealth", 0.0) * r_stealth
        )

        # DreamerV3 Symlog Transform
        r_total = symlog(r_total_raw)

        return RewardBreakdown(
            r_total=float(r_total),
            r_safety=float(r_safety),
            r_progress=float(r_progress),
            r_comfort=float(r_comfort),
            r_traffic=float(r_traffic),
            r_efficiency=float(r_efficiency),
            r_mission=float(r_mission),
            r_stealth=float(r_stealth),
        )
