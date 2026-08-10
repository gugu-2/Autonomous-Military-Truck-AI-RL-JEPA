"""Master safety monitor — watches all 7 AI layers for anomalies."""

import logging
import threading
from dataclasses import dataclass
from enum import Enum

# Assuming these are imported from their respective modules
from .failsafe_controller import FailsafeController, SafetyFlag
from .watchdog import Watchdog


class SystemSafetyState(Enum):
    NOMINAL = 0
    DEGRADED = 1
    FAILSAFE_ACTIVE = 2


@dataclass
class DrivingAction:
    steering: float
    throttle: float
    brake: float


@dataclass
class SensorHealthStatus:
    lidar_ok: bool
    camera_ok: bool
    radar_ok: bool
    gnss_ok: bool


class SafetyMonitor:
    def __init__(self, config, watchdog: Watchdog, failsafe: FailsafeController):
        self.config = config
        self.watchdog = watchdog
        self.failsafe = failsafe
        self.logger = logging.getLogger(__name__)
        self.consecutive_warns = 0
        self._state = SystemSafetyState.NOMINAL
        self._lock = threading.Lock()

    def check_jepa_health(self, latency_ms: float, hazard_energy: float) -> SafetyFlag:
        if latency_ms > 100 or hazard_energy > self.config.get("jepa_max_energy", 10.0):
            return SafetyFlag.FAILSAFE
        if latency_ms > 50 or hazard_energy > self.config.get("jepa_warn_energy", 5.0):
            return SafetyFlag.WARN
        return SafetyFlag.NOMINAL

    def check_rl_output(
        self, action: DrivingAction, prev_action: DrivingAction, dt: float
    ) -> SafetyFlag:
        if (
            action.steering > 1.0
            or action.steering < -1.0
            or action.throttle > 1.0
            or action.brake > 1.0
        ):
            return SafetyFlag.FAILSAFE
        # Simple rate limit check
        if dt > 0 and abs(action.steering - prev_action.steering) / dt > 5.0:
            return SafetyFlag.WARN
        return SafetyFlag.NOMINAL

    def check_sensor_health(self, health: SensorHealthStatus) -> SafetyFlag:
        if not health.lidar_ok or not health.camera_ok:
            return SafetyFlag.FAILSAFE
        if not health.radar_ok or not health.gnss_ok:
            return SafetyFlag.WARN
        return SafetyFlag.NOMINAL

    def check_can_heartbeat(self, last_can_ms: float) -> SafetyFlag:
        if last_can_ms > 500:
            return SafetyFlag.FAILSAFE
        if last_can_ms > 200:
            return SafetyFlag.WARN
        return SafetyFlag.NOMINAL

    def full_check(
        self,
        jepa_latency: float,
        hazard_energy: float,
        action: DrivingAction,
        prev_action: DrivingAction,
        sensor_health: SensorHealthStatus,
    ) -> SafetyFlag:
        flags = [
            self.check_jepa_health(jepa_latency, hazard_energy),
            self.check_rl_output(action, prev_action, 0.05),  # assuming 20Hz dt
            self.check_sensor_health(sensor_health),
        ]

        worst_flag = max(flags, key=lambda f: f.value)

        with self._lock:
            if worst_flag == SafetyFlag.WARN:
                self.consecutive_warns += 1
                if self.consecutive_warns >= 3:
                    worst_flag = SafetyFlag.DEGRADED
                    self._state = SystemSafetyState.DEGRADED
            else:
                self.consecutive_warns = 0

            if worst_flag == SafetyFlag.FAILSAFE or worst_flag == SafetyFlag.DEGRADED:
                if self._state != SystemSafetyState.FAILSAFE_ACTIVE:
                    self._state = SystemSafetyState.FAILSAFE_ACTIVE
                    self.failsafe.trigger_failsafe("Safety checks failed", worst_flag)

        return worst_flag

    def get_system_state(self) -> SystemSafetyState:
        with self._lock:
            return self._state
