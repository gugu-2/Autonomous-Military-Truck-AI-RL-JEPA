"""Failsafe controller — graceful controlled stop on system failure."""

import logging
import struct
import threading
import time
from enum import Enum, auto

from vehicle_interface.can_bus.can_encoder import CANCommandEncoder, DrivingAction


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

    def __init__(self, can_driver, can_decoder, config):
        self.can_driver = can_driver
        self.can_decoder = can_decoder
        self.can_encoder = CANCommandEncoder()
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
            action1 = DrivingAction(steering_angle=0.0, throttle=0.0, brake=1.0, gear=0)
            for msg in self.can_encoder.encode_action(action1):
                self.can_driver.send(msg.arbitration_id, msg.data, msg.is_extended_id)
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
                current_speed = self.can_decoder.get_vehicle_status().speed_ms
                if self.is_stopped(current_speed):
                    self.logger.info("Vehicle has fully stopped.")
                    with self._lock:
                        self.state = FailsafeState.STOPPED
                    action = DrivingAction(steering_angle=0.0, throttle=0.0, brake=1.0, gear=0)
                    for msg in self.can_encoder.encode_action(action):
                        self.can_driver.send(msg.arbitration_id, msg.data, msg.is_extended_id)
                    break

                # Apply brake corresponding to target deceleration
                brake_val = min(1.0, max(0.0, abs(target_decel) / 10.0))
                action = DrivingAction(steering_angle=0.0, throttle=0.0, brake=brake_val, gear=3)
                for msg in self.can_encoder.encode_action(action):
                    if msg.arbitration_id == 0x102:
                        self.can_driver.send(msg.arbitration_id, msg.data, msg.is_extended_id)
                time.sleep(0.05)
        except Exception as e:
            self.logger.error(f"Error in controlled deceleration loop: {e}")
            self.trigger_emergency_stop("Deceleration loop failed")

    def activate_hazard_lights(self):
        try:
            self.can_driver.send(0x350, bytes([0x01]), False)
            self.logger.info("Hazard lights activated.")
        except Exception as e:
            self.logger.error(f"Failed to activate hazard lights: {e}")

    def broadcast_sos(self, reason: str):
        self.logger.info(f"Broadcasting SOS to fleet API: {reason}")
        try:
            current_speed = self.can_decoder.get_vehicle_status().speed_ms
            payload = b"SOS\x00" + struct.pack("<f", current_speed)
            self.can_driver.send(0x7DF, payload, False)
        except Exception as e:
            self.logger.error(f"Failed to broadcast SOS: {e}")

    def is_stopped(self, current_speed_ms: float) -> bool:
        return abs(current_speed_ms) < 0.1
