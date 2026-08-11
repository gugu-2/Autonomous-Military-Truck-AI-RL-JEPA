"""Unit tests for Safety Interlocks."""

import time

from safety.safety_monitor import SafetyMonitor


def test_watchdog_timeout():
    """Test that the watchdog triggers emergency brake on latency spikes."""
    monitor = SafetyMonitor(max_latency_ms=12)  # 12ms AI budget

    # Simulate fast inference (5ms)
    start = time.time()
    time.sleep(0.005)
    is_safe = monitor.check_latency(start, time.time())
    assert is_safe is True, "5ms inference should be safe"

    # Simulate GPU spike / freeze (20ms)
    start = time.time()
    time.sleep(0.020)
    is_safe = monitor.check_latency(start, time.time())
    assert is_safe is False, "20ms inference should trigger watchdog"

    # Assert emergency brake is requested via interlock
    assert monitor.is_emergency_brake_triggered() is True
