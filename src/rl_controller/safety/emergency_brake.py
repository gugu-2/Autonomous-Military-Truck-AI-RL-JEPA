import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


class EmergencyBrakeController:
    """
    Hardware-level logical controller for Emergency Braking (AEB).
    This overrides all neural network outputs.
    """

    def __init__(self, config: dict[str, Any]):
        self.deceleration_rate = config.get("max_deceleration_g", 1.0) * 9.81
        self.jerk_limit = config.get("jerk_limit", 5.0)
        self.abs_enabled = config.get("abs_enabled", True)

        self.is_active = False
        self.trigger_time = 0.0
        self.reason = ""

    def trigger(self, reason: str = "Unknown hazard") -> dict[str, float]:
        """
        Engage the emergency brake.
        """
        if not self.is_active:
            self.is_active = True
            self.trigger_time = time.time()
            self.reason = reason
            logger.critical(f"🚨 EMERGENCY BRAKE ENGAGED 🚨 Reason: {reason}")

        # Return maximum braking force command, zero throttle
        return {
            "steering": 0.0,  # Hold wheel straight (or maintain current, depending on ESC)
            "throttle": 0.0,
            "brake": 1.0,  # 100% braking force
            "handbrake": (
                1.0 if (time.time() - self.trigger_time > 2.0) else 0.0
            ),  # Engage handbrake after full stop
            "emergency_flag": 1.0,
        }

    def release(self):
        """
        Release the emergency brake (requires manual or high-level safety system clearance).
        """
        if self.is_active:
            logger.info("Emergency brake released.")
            self.is_active = False
            self.reason = ""
