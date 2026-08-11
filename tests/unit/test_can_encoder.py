"""Unit tests for CAN Bitwise Encoding and Decoding."""

from vehicle_interface.can_bus.can_encoder import CANEncoder


def test_can_steering_throttle_encoding():
    """Test that float actions map to exact bits in the CAN payload."""
    encoder = CANEncoder()

    # 0.0 steering, 0.0 throttle, 0.0 brake
    # Based on DBC specs (e.g., standard mapping for DbW)
    # Typically, offset by 0x7FFF for 0.0 in a 16-bit field

    # Note: the mock CANEncoder inside can_encoder.py uses struct packing:
    # return struct.pack("<hhB", int(steering * 32767), int(throttle * 255), int(brake * 255))

    steering = 1.0  # Max right
    throttle = 0.5  # 50%
    brake = 0.0

    payload = encoder.encode_control(steering, throttle, brake)

    assert len(payload) == 8, "CAN payload must be strictly 8 bytes"

    # Max steering * 32767 = 32767 (0x7FFF) -> little endian (FF 7F)
    assert payload[0] == 0xFF
    assert payload[1] == 0x7F

    # Throttle 0.5 * 255 = 127 (0x7F)
    assert payload[2] == 0x7F

    # Brake 0.0 = 0
    assert payload[4] == 0x00
