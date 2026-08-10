"""SAE J1939 heavy-duty truck protocol interface for OMNIDRIVE."""
import can
import struct
import time
from enum import IntEnum
from dataclasses import dataclass
from typing import Optional

class J1939PGN(IntEnum):
    PGN_EEC1 = 61444  # Electronic Engine Controller 1
    PGN_TC1 = 256     # Transmission Control 1
    PGN_CCVS = 65265  # Cruise Control/Vehicle Speed
    PGN_EBC1 = 61441  # Electronic Brake Controller 1
    PGN_ERC1 = 61440  # Electronic Retarder Controller 1
    PGN_TSC1 = 0      # Torque/Speed Control 1

@dataclass
class J1939Message:
    pgn: int
    source_address: int
    priority: int
    data: bytes

class J1939Interface:
    """Implements SAE J1939 29-bit CAN protocol for heavy duty vehicles."""
    
    def __init__(self, driver, source_address: int = 0xF9):
        # driver is assumed to be an instance of CANDriver
        self.driver = driver
        self.source_address = source_address
        self.last_speed = 0.0

    def _encode_j1939_frame(self, pgn: int, data: bytes, priority: int = 6, destination_address: int = 0xFF) -> can.Message:
        """Builds a 29-bit J1939 CAN frame."""
        # J1939 29-bit ID format:
        # 3 bits Priority, 1 bit Reserved, 1 bit Data Page, 8 bits PDU Format, 8 bits PDU Specific, 8 bits Source Address
        dp = 0
        pf = (pgn >> 8) & 0xFF
        ps = pgn & 0xFF
        
        # PDU1 format (peer to peer) if PF < 240, otherwise PDU2 (broadcast)
        if pf < 240:
            ps = destination_address

        can_id = (priority & 0x07) << 26
        can_id |= (dp & 0x01) << 24
        can_id |= (pf & 0xFF) << 16
        can_id |= (ps & 0xFF) << 8
        can_id |= (self.source_address & 0xFF)
        
        # Pad data to 8 bytes if needed
        if len(data) < 8:
            data = data + b'\xFF' * (8 - len(data))
            
        return can.Message(arbitration_id=can_id, data=data, is_extended_id=True)

    def _decode_j1939_frame(self, msg: can.Message) -> Optional[J1939Message]:
        """Extracts PGN and Source Address from a 29-bit CAN ID."""
        if not msg.is_extended_id:
            return None
            
        can_id = msg.arbitration_id
        priority = (can_id >> 26) & 0x07
        dp = (can_id >> 24) & 0x01
        pf = (can_id >> 16) & 0xFF
        ps = (can_id >> 8) & 0xFF
        sa = can_id & 0xFF
        
        if pf < 240:
            # PDU1 format
            pgn = (dp << 16) | (pf << 8)
        else:
            # PDU2 format
            pgn = (dp << 16) | (pf << 8) | ps
            
        return J1939Message(pgn=pgn, source_address=sa, priority=priority, data=msg.data)

    def send_tsc1_engine_speed_request(self, target_speed_rpm: float, destination_address: int = 0x00):
        """Sends Torque/Speed Control 1 message to engine ECU."""
        # Eng Override Control Mode: 01 (Speed control)
        # Eng Requested Speed Control Conditions: 01 (Transient)
        # Byte 1: 0x05 (Mode 1, Condition 1)
        byte1 = 0x05
        
        # Speed resolution is 0.125 rpm/bit
        rpm_raw = int(target_speed_rpm / 0.125)
        
        # Pack into little endian uint16 for bytes 2-3
        speed_bytes = struct.pack('<H', rpm_raw)
        
        data = bytearray([byte1, speed_bytes[0], speed_bytes[1], 0xFF, 0xFF, 0xFF, 0xFF, 0xFF])
        msg = self._encode_j1939_frame(J1939PGN.PGN_TSC1, bytes(data), priority=3, destination_address=destination_address)
        self.driver.send(msg.arbitration_id, msg.data, True)

    def send_cruise_control_command(self, enable: bool, speed_kmh: float):
        """Sets cruise speed via CCVS."""
        # Custom implementation for CCVS speed command
        speed_raw = int(speed_kmh * 256.0) # 1/256 km/h per bit
        speed_bytes = struct.pack('<H', speed_raw)
        
        # Byte 4 contains states (e.g. CC Active)
        states = 0x01 if enable else 0x00
        
        data = bytearray([0xFF, speed_bytes[0], speed_bytes[1], states, 0xFF, 0xFF, 0xFF, 0xFF])
        msg = self._encode_j1939_frame(J1939PGN.PGN_CCVS, bytes(data), priority=6)
        self.driver.send(msg.arbitration_id, msg.data, True)

    def send_retarder_command(self, retarder_level: int):
        """Engine retarder 0-15."""
        level = max(0, min(15, retarder_level))
        data = bytearray([level, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF])
        msg = self._encode_j1939_frame(J1939PGN.PGN_ERC1, bytes(data), priority=6)
        self.driver.send(msg.arbitration_id, msg.data, True)

    def request_vehicle_speed(self) -> Optional[float]:
        """Reads latest speed based on J1939 callbacks or recent data."""
        # Normally this is passively decoded in a receive loop
        return self.last_speed
