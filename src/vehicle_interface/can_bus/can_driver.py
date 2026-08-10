"""Base CAN bus driver for OMNIDRIVE vehicle interface."""

import logging
import queue
import threading
import time
from collections.abc import Callable

import can

logger = logging.getLogger(__name__)


class CANDriver:
    """Production CAN driver with background reception and bus-off recovery."""

    def __init__(
        self,
        channel: str = "can0",
        bustype: str = "socketcan",
        bitrate: int = 500000,
        timeout: float = 0.5,
    ):
        self.channel = channel
        self.bustype = bustype
        self.bitrate = bitrate
        self.timeout = timeout

        self.bus: can.BusABC | None = None
        self.rx_queue: queue.Queue = queue.Queue(maxsize=10000)
        self.callbacks: dict[int, Callable[[can.Message], None]] = {}

        self._running = False
        self._rx_thread: threading.Thread | None = None

        # Statistics
        self.tx_count = 0
        self.rx_count = 0
        self.error_count = 0

    def _init_bus(self):
        try:
            self.bus = can.Bus(channel=self.channel, interface=self.bustype, bitrate=self.bitrate)
            logger.info(
                f"Initialized CAN bus {self.channel} ({self.bustype}) at {self.bitrate} bps"
            )
        except Exception as e:
            logger.error(f"Failed to initialize CAN bus: {e}")
            raise

    def start(self):
        """Starts the CAN bus and the receiver thread."""
        if self._running:
            return

        self._init_bus()
        self._running = True
        self._rx_thread = threading.Thread(
            target=self._receive_loop, name="CAN_Rx_Thread", daemon=True
        )
        self._rx_thread.start()
        logger.info("CANDriver started.")

    def stop(self):
        """Cleanly stops the CAN driver and closes the bus."""
        self._running = False
        if self._rx_thread and self._rx_thread.is_alive():
            self._rx_thread.join(timeout=2.0)

        if self.bus:
            self.bus.shutdown()
            self.bus = None
        logger.info(
            f"CANDriver stopped. TX: {self.tx_count}, RX: {self.rx_count}, ERR: {self.error_count}"
        )

    def send(self, msg_id: int, data: bytes, extended: bool = False):
        """Sends a CAN frame with error handling."""
        if not self.bus:
            logger.warning("Attempted to send CAN message but bus is not initialized.")
            return

        msg = can.Message(
            arbitration_id=msg_id, data=data, is_extended_id=extended, is_error_frame=False
        )

        try:
            self.bus.send(msg, timeout=self.timeout)
            self.tx_count += 1
        except can.CanError as e:
            self.error_count += 1
            logger.error(f"CAN send error (ID: {hex(msg_id)}): {e}")
            self._handle_bus_error()

    def receive(self, timeout: float = 0.1) -> can.Message | None:
        """Non-blocking receive from the internal queue."""
        try:
            return self.rx_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def subscribe(self, msg_id: int, callback: Callable[[can.Message], None]):
        """Register a callback for a specific CAN ID."""
        self.callbacks[msg_id] = callback
        logger.debug(f"Subscribed callback for CAN ID: {hex(msg_id)}")

    def _receive_loop(self):
        """Background thread loop for receiving messages."""
        while self._running and self.bus:
            try:
                msg = self.bus.recv(timeout=0.1)
                if msg:
                    self.rx_count += 1

                    # Dispatch to callbacks
                    if msg.arbitration_id in self.callbacks:
                        try:
                            self.callbacks[msg.arbitration_id](msg)
                        except Exception as e:
                            logger.error(f"Callback error for ID {hex(msg.arbitration_id)}: {e}")

                    # Put in general queue (drop oldest if full)
                    if self.rx_queue.full():
                        try:
                            self.rx_queue.get_nowait()
                        except queue.Empty:
                            pass
                    self.rx_queue.put_nowait(msg)

            except can.CanError as e:
                self.error_count += 1
                logger.error(f"CAN receive error: {e}")
                self._handle_bus_error()
            except Exception as e:
                logger.error(f"Unexpected error in CAN receive loop: {e}")

    def _handle_bus_error(self):
        """Attempt bus-off recovery with exponential backoff."""
        logger.warning("Attempting bus recovery...")
        if self.bus:
            self.bus.shutdown()

        backoff = 1.0
        max_backoff = 16.0

        while self._running:
            try:
                time.sleep(backoff)
                self._init_bus()
                logger.info("Bus successfully recovered.")
                break
            except Exception as e:
                logger.error(f"Bus recovery failed: {e}. Retrying in {backoff}s...")
                backoff = min(backoff * 2, max_backoff)
