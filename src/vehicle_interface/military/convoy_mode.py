"""Military convoy leader-follower mode for OMNIDRIVE."""
import time
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# Forward declare for type hinting purposes
@dataclass
class DrivingAction:
    steering_angle: float
    throttle: float
    brake: float
    gear: int

@dataclass
class EgoPose:
    x: float
    y: float
    heading: float
    speed_ms: float

@dataclass
class VehicleConvoyState:
    vehicle_id: str
    x: float
    y: float
    speed_ms: float
    heading: float
    timestamp: float

class ConvoyMode:
    """Handles military leader-follower logistics logic."""
    
    def __init__(self, target_gap_m: float = 15.0, gap_tolerance_m: float = 2.0, max_speed_kmh: float = 80.0):
        self.target_gap_m = target_gap_m
        self.gap_tolerance_m = gap_tolerance_m
        self.max_speed_ms = max_speed_kmh / 3.6
        
        self.leader_state: Optional[VehicleConvoyState] = None
        self._is_leader_flag = False
        
        # PID Controller state for gap maintenance
        self.kp = 0.5
        self.ki = 0.05
        self.kd = 0.1
        self.integral_error = 0.0
        self.last_error = 0.0
        self.last_time = time.time()

    def is_leader(self) -> bool:
        return self._is_leader_flag

    def set_leader_state(self, state: VehicleConvoyState):
        """Receive telemetry from the convoy leader."""
        self.leader_state = state

    def compute_follower_action(self, ego_state: EgoPose, current_action: DrivingAction) -> DrivingAction:
        """PID-based gap control adjusting the base driving action."""
        if not self.leader_state:
            logger.warning("No leader state available. Halting.")
            self.emergency_convoy_halt()
            return DrivingAction(steering_angle=0, throttle=0, brake=1.0, gear=current_action.gear)

        current_time = time.time()
        dt = current_time - self.last_time
        if dt <= 0:
            dt = 0.01
            
        # Compute distance to leader
        dx = self.leader_state.x - ego_state.x
        dy = self.leader_state.y - ego_state.y
        distance = (dx**2 + dy**2)**0.5
        
        # Error term
        error = distance - self.target_gap_m
        
        # Anti-windup
        if abs(error) < self.gap_tolerance_m:
            self.integral_error = 0.0
        else:
            self.integral_error += error * dt
            # Clamp integral
            self.integral_error = max(-10.0, min(10.0, self.integral_error))
            
        derivative = (error - self.last_error) / dt
        
        # PID Output (target speed adjustment)
        speed_adjustment = (self.kp * error) + (self.ki * self.integral_error) + (self.kd * derivative)
        target_speed = self.leader_state.speed_ms + speed_adjustment
        
        # Clamp target speed
        target_speed = max(0.0, min(self.max_speed_ms, target_speed))
        
        self.last_error = error
        self.last_time = current_time
        
        # Translate speed into throttle/brake adjustments
        if target_speed > ego_state.speed_ms + 0.5:
            current_action.throttle = min(1.0, current_action.throttle + 0.1)
            current_action.brake = 0.0
        elif target_speed < ego_state.speed_ms - 0.5:
            current_action.throttle = 0.0
            current_action.brake = min(1.0, current_action.brake + 0.2)
            
        return current_action

    def broadcast_state(self, ego_state: EgoPose, vehicle_id: str = "VEH_1") -> VehicleConvoyState:
        """Publishes telemetry for other vehicles."""
        return VehicleConvoyState(
            vehicle_id=vehicle_id,
            x=ego_state.x,
            y=ego_state.y,
            speed_ms=ego_state.speed_ms,
            heading=ego_state.heading,
            timestamp=time.time()
        )

    def emergency_convoy_halt(self):
        """Propagates halt command immediately."""
        logger.critical("EMERGENCY CONVOY HALT TRIGGERED.")
        # Network broadcast hook would go here
