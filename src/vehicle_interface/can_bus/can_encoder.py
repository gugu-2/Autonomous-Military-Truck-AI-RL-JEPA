"""Encodes AI driving commands into CAN bus frames."""
import struct
import math
import can
from dataclasses import dataclass
from typing import List

@dataclass
class DrivingAction:
    steering_angle: float
    throttle: float
    brake: float
    gear: int

class CANCommandEncoder:
    """Encodes vehicle control signals into specific CAN frames."""
    
    def encode_steering(self, angle_deg: float) -> can.Message:
        """
        CAN ID 0x100
        Encodes -45 to +45 deg into uint16
        Scale: 0.01 deg/bit
        Offset: -45 deg
        Data layout: Little Endian
        """
        # Clamp value
        angle_deg = max(-45.0, min(45.0, angle_deg))
        
        # Apply offset and scale
        raw_val = int((angle_deg + 45.0) / 0.01)
        
        # Pack into 2 bytes (uint16), little endian
        data = struct.pack('<H', raw_val)
        # Pad to 8 bytes DLC
        data += b'\x00' * 6
        
        return can.Message(arbitration_id=0x100, data=data, is_extended_id=False)

    def encode_throttle(self, value: float) -> can.Message:
        """
        CAN ID 0x101
        Encodes 0.0 to 1.0 into uint8 (0-255)
        """
        value = max(0.0, min(1.0, value))
        raw_val = int(value * 255.0)
        
        data = struct.pack('<B', raw_val)
        data += b'\x00' * 7
        
        return can.Message(arbitration_id=0x101, data=data, is_extended_id=False)

    def encode_brake(self, value: float) -> can.Message:
        """
        CAN ID 0x102
        Encodes 0.0 to 1.0 into uint8 (0-255)
        """
        value = max(0.0, min(1.0, value))
        raw_val = int(value * 255.0)
        
        data = struct.pack('<B', raw_val)
        data += b'\x00' * 7
        
        return can.Message(arbitration_id=0x102, data=data, is_extended_id=False)

    def encode_gear(self, gear: int) -> can.Message:
        """
        CAN ID 0x103
        Gear encoding: 0=Park, 1=Reverse, 2=Neutral, 3=Drive
        """
        gear = max(0, min(3, gear))
        data = struct.pack('<B', gear)
        data += b'\x00' * 7
        
        return can.Message(arbitration_id=0x103, data=data, is_extended_id=False)

    def encode_heartbeat(self, counter: int) -> can.Message:
        """
        CAN ID 0x0FF
        Watchdog heartbeat message to keep drive-by-wire active.
        """
        raw_val = counter % 256
        data = struct.pack('<B', raw_val)
        data += b'\x00' * 7
        
        return can.Message(arbitration_id=0x0FF, data=data, is_extended_id=False)

    def encode_action(self, action: DrivingAction) -> List[can.Message]:
        """Encodes a full driving action into a list of CAN messages."""
        return [
            self.encode_steering(action.steering_angle),
            self.encode_throttle(action.throttle),
            self.encode_brake(action.brake),
            self.encode_gear(action.gear)
        ]
