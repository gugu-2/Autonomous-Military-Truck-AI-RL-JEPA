"""Unit tests for safety monitor."""


class SafetyFlag:
    NOMINAL = 0
    WARN = 1
    DEGRADED = 2
    FAILSAFE = 3


class SafetyMonitor:
    def __init__(self):
        self.consecutive_warns = 0

    def check(self, jepa_latency_ms=10, hazard_energy=0.1, steering_rate=0.0, sensors_ok=True):
        if not sensors_ok:
            return SafetyFlag.FAILSAFE

        if hazard_energy >= 0.85:
            return SafetyFlag.FAILSAFE

        flag = SafetyFlag.NOMINAL

        if jepa_latency_ms > 20:
            flag = SafetyFlag.WARN

        if abs(steering_rate) > 300:
            flag = SafetyFlag.WARN

        if flag == SafetyFlag.WARN:
            self.consecutive_warns += 1
            if self.consecutive_warns >= 3:
                return SafetyFlag.DEGRADED
            return SafetyFlag.WARN
        else:
            self.consecutive_warns = 0

        return flag


def test_nominal_state_returns_nominal():
    monitor = SafetyMonitor()
    assert monitor.check() == SafetyFlag.NOMINAL


def test_high_jepa_latency_triggers_warn():
    monitor = SafetyMonitor()
    assert monitor.check(jepa_latency_ms=25) == SafetyFlag.WARN


def test_critical_hazard_energy_triggers_failsafe():
    monitor = SafetyMonitor()
    assert monitor.check(hazard_energy=0.90) == SafetyFlag.FAILSAFE


def test_excessive_steering_rate_triggers_warn():
    monitor = SafetyMonitor()
    assert monitor.check(steering_rate=350) == SafetyFlag.WARN


def test_sensor_all_failed_triggers_failsafe():
    monitor = SafetyMonitor()
    assert monitor.check(sensors_ok=False) == SafetyFlag.FAILSAFE


def test_three_consecutive_warns_escalate():
    monitor = SafetyMonitor()
    assert monitor.check(jepa_latency_ms=25) == SafetyFlag.WARN
    assert monitor.check(jepa_latency_ms=25) == SafetyFlag.WARN
    assert monitor.check(jepa_latency_ms=25) == SafetyFlag.DEGRADED
