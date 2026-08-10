"""Failsafe controller — graceful controlled stop on system failure."""

import logging
import threading
import time
from enum import Enum, auto


class FailsafeState(Enum):
    NOMINAL = auto()
    DECELERATING = auto()
    STOPPED = auto()


class SafetyFlag(Enum):
    NOMINAL = 0
    WARN = 1
    DEGRADED = 2
    FAILSAFE = 3


class FailsafeController:
    """Manages graceful degraded operations and emergency stops."""

    def __init__(self, vehicle_interface, config):
        self.vehicle_interface = vehicle_interface
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.state = FailsafeState.NOMINAL
        self._lock = threading.Lock()
        self._decel_thread = None

    def trigger_failsafe(self, reason: str, flag: SafetyFlag):
        with self._lock:
            if self.state != FailsafeState.NOMINAL:
                return
            self.logger.critical(f"Triggering FAILSAFE due to {reason} with flag {flag.name}")
            self.state = FailsafeState.DECELERATING
            self.activate_hazard_lights()
            self.broadcast_sos(reason)

            if flag == SafetyFlag.FAILSAFE:
                self.trigger_emergency_stop(reason)
            else:
                self.controlled_deceleration_stop()

    def trigger_emergency_stop(self, reason: str):
        self.logger.critical(f"EMERGENCY STOP Triggered: {reason}")
        try:
            self.vehicle_interface.apply_max_brake()
            self.vehicle_interface.disable_drive_power()
            self.state = FailsafeState.STOPPED
        except Exception as e:
            self.logger.error(f"Failed to apply emergency stop: {e}")

    def controlled_deceleration_stop(self, target_decel: float = -2.5):
        self.logger.info(f"Initiating controlled deceleration at {target_decel} m/s^2")
        self._decel_thread = threading.Thread(
            target=self._decel_loop, args=(target_decel,), daemon=True
        )
        self._decel_thread.start()

    def _decel_loop(self, target_decel: float):
        try:
            while True:
                current_speed = self.vehicle_interface.get_current_speed()
                if self.is_stopped(current_speed):
                    self.logger.info("Vehicle has fully stopped.")
                    with self._lock:
                        self.state = FailsafeState.STOPPED
                    self.vehicle_interface.apply_parking_brake()
                    break

                # Apply brake corresponding to target deceleration
                self.vehicle_interface.apply_brake_for_deceleration(target_decel)
                time.sleep(0.05)
        except Exception as e:
            self.logger.error(f"Error in controlled deceleration loop: {e}")
            self.trigger_emergency_stop("Deceleration loop failed")

    def activate_hazard_lights(self):
        try:
            self.vehicle_interface.send_can_message(0x350, [0x01])
            self.logger.info("Hazard lights activated.")
        except Exception as e:
            self.logger.error(f"Failed to activate hazard lights: {e}")

    def broadcast_sos(self, reason: str):
        self.logger.info(f"Broadcasting SOS to fleet API: {reason}")
        # In a real implementation, this would make an HTTP/gRPC request to fleet management
        pass

    def is_stopped(self, current_speed_ms: float) -> bool:
        return abs(current_speed_ms) < 0.1
