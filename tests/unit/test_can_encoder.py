"""CAN encoder tests."""
import pytest
import struct

class CANEncoder:
    def encode_steering(self, deg):
        # Range -45 to 45 deg maps to 0-65535 (uint16)
        clamped = max(-45.0, min(45.0, deg))
        val = int((clamped + 45.0) / 90.0 * 65535.0)
        return struct.pack('<H', val)
        
    def decode_steering(self, data):
        val = struct.unpack('<H', data)[0]
        return (val / 65535.0) * 90.0 - 45.0
        
    def encode_throttle(self, val):
        # 0.0 to 1.0 -> 0-255 (uint8)
        clamped = max(0.0, min(1.0, val))
        return bytes([int(clamped * 255.0)])
        
    def decode_throttle(self, data):
        return data[0] / 255.0
        
    def encode_brake(self, val):
        clamped = max(0.0, min(1.0, val))
        return bytes([int(clamped * 255.0)])
        
    def encode_action(self, steering, throttle, brake):
        return [
            {"id": 0x100, "data": self.encode_steering(steering)},
            {"id": 0x101, "data": self.encode_throttle(throttle)},
            {"id": 0x102, "data": self.encode_brake(brake)}
        ]

@pytest.fixture
def encoder():
    return CANEncoder()

def test_steering_center_encodes_correctly(encoder):
    data = encoder.encode_steering(0.0)
    # Midpoint of 65535 is 32767 = 0x7FFF
    assert data == struct.pack('<H', 32767)

def test_steering_max_positive(encoder):
    data = encoder.encode_steering(45.0)
    assert data == struct.pack('<H', 65535)

def test_steering_max_negative(encoder):
    data = encoder.encode_steering(-45.0)
    assert data == struct.pack('<H', 0)

def test_throttle_zero(encoder):
    data = encoder.encode_throttle(0.0)
    assert data == b'\x00'

def test_throttle_full(encoder):
    data = encoder.encode_throttle(1.0)
    assert data == b'\xff'

def test_brake_encoding(encoder):
    data = encoder.encode_brake(1.0)
    assert data == b'\xff'

def test_round_trip_steering(encoder):
    original = 12.34
    data = encoder.encode_steering(original)
    decoded = encoder.decode_steering(data)
    assert abs(original - decoded) < 0.01

def test_round_trip_throttle(encoder):
    original = 0.56
    data = encoder.encode_throttle(original)
    decoded = encoder.decode_throttle(data)
    assert abs(original - decoded) < 0.004

def test_action_produces_correct_frame_ids(encoder):
    frames = encoder.encode_action(0.0, 0.5, 0.0)
    ids = [f['id'] for f in frames]
    assert ids == [0x100, 0x101, 0x102]
