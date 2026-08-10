"""JAUS (Joint Architecture for Unmanned Systems) interface for military vehicles."""

import logging
import socket
import struct
from dataclasses import dataclass
from enum import IntEnum

logger = logging.getLogger(__name__)


@dataclass
class JausAddress:
    subsystem_id: int  # uint16
    node_id: int  # uint8
    component_id: int  # uint8

    def to_bytes(self) -> bytes:
        return struct.pack("<HBB", self.subsystem_id, self.node_id, self.component_id)


class JausMessageType(IntEnum):
    SET_WRENCH_EFFORT = 0x040F
    SET_DESIRED_TRAVEL_SPEED = 0x040A
    SET_ELEMENT = 0x041A
    REPORT_VELOCITY_STATE = 0x2404
    EMERGENCY_HALT = 0x000E


class JausInterface:
    """Production JAUS interface over UDP."""

    def __init__(
        self, local_address: JausAddress, remote_address: JausAddress, udp_port: int = 3794
    ):
        self.local_address = local_address
        self.remote_address = remote_address
        self.udp_port = udp_port
        self.sock: socket.socket | None = None
        self.seq_num = 0

    def connect(self):
        """Initializes the UDP socket for JAUS comms."""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Bind locally if needed, but not strictly required for client sending
        self.sock.settimeout(0.1)
        logger.info(f"JAUS Interface ready on UDP port {self.udp_port}")

    def _build_jaus_header(self, msg_type: int, data_len: int) -> bytes:
        """Builds standard JAUS transport header."""
        # 1 byte properties, 4 bytes dest, 4 bytes src, 4 bytes data len, 2 bytes seq
        properties = 0x00  # Default QoS
        dest = self.remote_address.to_bytes()
        src = self.local_address.to_bytes()

        self.seq_num = (self.seq_num + 1) % 65536

        # Transport header packing
        header = (
            struct.pack("<B", properties)
            + dest
            + src
            + struct.pack("<I", data_len)
            + struct.pack("<H", self.seq_num)
        )

        # Append Message Code
        header += struct.pack("<H", msg_type)
        return header

    def _send_udp(self, data: bytes):
        if not self.sock:
            logger.error("JAUS socket not connected.")
            return
        # Assume broadcast or configured IP for remote. Using localhost for generic demonstration
        self.sock.sendto(data, ("127.0.0.1", self.udp_port))

    def send_set_wrench_effort(self, propulsive_linear_x: float, steering: float):
        """Sends normalized drive command [-100, 100]."""
        x_effort = max(-100.0, min(100.0, propulsive_linear_x))
        yaw_effort = max(-100.0, min(100.0, steering))

        # Map to double
        payload = struct.pack("<dd", x_effort, yaw_effort)

        msg = self._build_jaus_header(JausMessageType.SET_WRENCH_EFFORT, len(payload)) + payload
        self._send_udp(msg)

    def send_set_desired_travel_speed(self, speed_ms: float):
        """Sets target speed."""
        payload = struct.pack("<d", speed_ms)
        msg = (
            self._build_jaus_header(JausMessageType.SET_DESIRED_TRAVEL_SPEED, len(payload))
            + payload
        )
        self._send_udp(msg)

    def send_emergency_halt(self):
        """Immediate halt command."""
        # No payload for halt
        msg = self._build_jaus_header(JausMessageType.EMERGENCY_HALT, 0)
        self._send_udp(msg)

    def send_element_waypoint(self, lat: float, lon: float, heading: float):
        """Mission waypoint."""
        # Simplified lat/lon encoding to doubles
        payload = struct.pack("<ddd", lat, lon, heading)
        msg = self._build_jaus_header(JausMessageType.SET_ELEMENT, len(payload)) + payload
        self._send_udp(msg)

    def receive(self, timeout: float = 0.1) -> dict | None:
        """Reads incoming JAUS messages."""
        if not self.sock:
            return None

        try:
            self.sock.settimeout(timeout)
            data, addr = self.sock.recvfrom(4096)
            if len(data) < 16:
                return None

            # Parse header and msg type
            msg_type = struct.unpack("<H", data[14:16])[0]

            if msg_type == JausMessageType.REPORT_VELOCITY_STATE:
                if len(data) >= 16 + 8:
                    speed = struct.unpack("<d", data[16:24])[0]
                    return {"msg_type": "REPORT_VELOCITY_STATE", "speed": speed}

            return {"msg_type": hex(msg_type)}
        except TimeoutError:
            return None
        except Exception as e:
            logger.error(f"JAUS receive error: {e}")
            return None
