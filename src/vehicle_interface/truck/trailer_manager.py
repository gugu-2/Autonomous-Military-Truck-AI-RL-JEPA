import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class TrailerManager:
    """
    Manages kinematics, braking, and state estimation for articulated truck trailers.
    Interfaces with the J1939 CAN bus to control trailer brakes independently to prevent jackknifing.
    """
    def __init__(self, config: Dict[str, Any]):
        self.num_trailers = config.get('num_trailers', 1)
        self.trailer_length = config.get('trailer_length', 13.6) # Standard 53ft / 13.6m
        self.jackknife_warning_angle = config.get('jackknife_warning_angle', 15.0) # degrees
        self.jackknife_critical_angle = config.get('jackknife_critical_angle', 45.0) # degrees
        
        self.current_articulation_angle = 0.0
        
    def update_state(self, articulation_angle: float):
        """
        Updates the current state of the trailer from articulation sensors.
        """
        self.current_articulation_angle = articulation_angle
        
    def calculate_trailer_brake_bias(self, target_brake: float, current_speed: float) -> float:
        """
        Dynamically adjusts trailer brake pressure vs tractor brake pressure.
        Applying more trailer brake than tractor brake keeps the combination straight.
        """
        if current_speed < 2.0:
            return target_brake # No special handling at very low speeds
            
        # If we are approaching a jackknife angle, increase trailer braking bias
        # to pull the tractor straight.
        angle_mag = abs(self.current_articulation_angle)
        
        if angle_mag > self.jackknife_critical_angle:
            logger.critical(f"CRITICAL JACKKNIFE ANGLE DETECTED: {angle_mag:.1f}°")
            # Apply maximum trailer brakes to pull straight
            return 1.0 
            
        elif angle_mag > self.jackknife_warning_angle:
            logger.warning(f"High articulation angle: {angle_mag:.1f}°")
            # Add bias proportional to how close we are to critical
            bias_factor = (angle_mag - self.jackknife_warning_angle) / (self.jackknife_critical_angle - self.jackknife_warning_angle)
            return min(1.0, target_brake + (bias_factor * 0.5))
            
        return target_brake
