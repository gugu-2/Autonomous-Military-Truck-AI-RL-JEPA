"""Decodes vehicle feedback from CAN bus into Python structures."""
import struct
import can
from dataclasses import dataclass
from typing import Optional, Dict, Any
import time

@dataclass
class VehicleStatus:
    speed_ms: float = 0.0
    steering_angle_deg: float = 0.0
    engine_rpm: float = 0.0
    brake_pressure: float = 0.0
    gear: int = 0
    battery_voltage: float = 12.0
    fault_code: int = 0
    timestamp: float = 0.0

class CANDecoder:
    """Decodes raw CAN frames into high-level vehicle telemetry."""
    
    def __init__(self):
        self._status = VehicleStatus()

    def decode_speed(self, msg: can.Message) -> float:
        """
        CAN ID 0x200
        Data layout: uint16 for front-left, front-right, rear-left, rear-right
        Scale: 0.01 m/s per bit
        """
        if len(msg.data) >= 8:
            fl, fr, rl, rr = struct.unpack('<HHHH', msg.data[:8])
            # Average wheel speed
            speed = ((fl + fr + rl + rr) / 4.0) * 0.01
            self._status.speed_ms = speed
            return speed
        return self._status.speed_ms

    def decode_steering_feedback(self, msg: can.Message) -> float:
        """
        CAN ID 0x201
        Data layout: int16 (signed)
        Scale: 0.1 deg per bit
        """
        if len(msg.data) >= 2:
            raw_angle = struct.unpack('<h', msg.data[:2])[0]
            angle = raw_angle * 0.1
            self._status.steering_angle_deg = angle
            return angle
        return self._status.steering_angle_deg

    def decode_engine_status(self, msg: can.Message) -> dict:
        """
        CAN ID 0x300
        Data layout: uint16 rpm, uint8 gear, uint8 faults
        Scale RPM: 1.0 rpm per bit
        """
        if len(msg.data) >= 4:
            rpm, gear, faults = struct.unpack('<HBB', msg.data[:4])
            self._status.engine_rpm = float(rpm)
            self._status.gear = gear
            self._status.fault_code = faults
            return {"rpm": rpm, "gear": gear, "faults": faults}
        return {}

    def decode_brake_status(self, msg: can.Message) -> dict:
        """
        CAN ID 0x301
        Data layout: uint16 pressure in bars
        Scale: 0.1 bar per bit
        """
        if len(msg.data) >= 2:
            pressure_raw = struct.unpack('<H', msg.data[:2])[0]
            pressure = pressure_raw * 0.1
            self._status.brake_pressure = pressure
            return {"brake_pressure": pressure}
        return {}

    def decode_message(self, msg: can.Message) -> Optional[dict]:
        """Routes a CAN message to the correct decoder."""
        self._status.timestamp = time.time()
        
        if msg.arbitration_id == 0x200:
            return {"speed_ms": self.decode_speed(msg)}
        elif msg.arbitration_id == 0x201:
            return {"steering_angle_deg": self.decode_steering_feedback(msg)}
        elif msg.arbitration_id == 0x300:
            return self.decode_engine_status(msg)
        elif msg.arbitration_id == 0x301:
            return self.decode_brake_status(msg)
        return None

    def get_vehicle_status(self) -> VehicleStatus:
        """Returns the latest fused vehicle status."""
        return self._status
