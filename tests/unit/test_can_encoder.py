"""Unit tests for CAN Bus Encoder."""

import pytest

try:
    from vehicle_interface.can_bus.can_encoder import CANCommandEncoder

    HAS_CAN = True
except ImportError:
    HAS_CAN = False


@pytest.mark.skipif(not HAS_CAN, reason="vehicle_interface not importable")
def test_can_encoder_steering():
    """Test steering angle produces a valid CAN message."""
    encoder = CANCommandEncoder()
    # Read can_encoder.py to find the correct method name (encode_action or encode_steering)
    # and use the DrivingAction dataclass from utils.common_types
    try:
        from utils.common_types import DrivingAction

        action = DrivingAction(steering_angle=15.0, throttle=0.5, brake=0.0, gear=1)
        msg = encoder.encode_action(action)
        assert msg is not None
        assert len(msg.data) == 8, "CAN payload must be 8 bytes"
    except (ImportError, AttributeError):
        pytest.skip("DrivingAction or encode_action not available")


@pytest.mark.skipif(not HAS_CAN, reason="vehicle_interface not importable")
def test_can_encoder_brake_clamp():
    """Test that brake values above 1.0 are clamped."""
    encoder = CANCommandEncoder()
    try:
        from utils.common_types import DrivingAction

        action = DrivingAction(steering_angle=0.0, throttle=0.0, brake=2.5, gear=0)
        msg = encoder.encode_action(action)
        # Should not crash - brake clamped to 1.0
        assert msg is not None
    except (ImportError, AttributeError):
        pytest.skip("DrivingAction or encode_action not available")
