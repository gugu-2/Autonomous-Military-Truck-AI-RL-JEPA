"""Software and hardware watchdog for OMNIDRIVE safety system."""

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class ModuleStatus:
    last_heartbeat: float = 0.0
    miss_count: int = 0
    is_healthy: bool = True


class Watchdog:
    """Watchdog for monitoring module health via heartbeats."""

    def __init__(self, heartbeat_interval_ms: float = 100.0, miss_limit: int = 3):
        self.heartbeat_interval = heartbeat_interval_ms / 1000.0
        self.miss_limit = miss_limit
        self._modules: dict[str, dict] = {}
        self._status: dict[str, ModuleStatus] = {}
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self.logger = logging.getLogger(__name__)

    def register_module(self, name: str, callback_on_failure: Callable):
        with self._lock:
            self._modules[name] = {"callback": callback_on_failure}
            self._status[name] = ModuleStatus(last_heartbeat=time.time())
            self.logger.info(f"Registered module {name} with watchdog.")

    def heartbeat(self, module_name: str):
        with self._lock:
            if module_name in self._status:
                self._status[module_name].last_heartbeat = time.time()
                self._status[module_name].miss_count = 0
                self._status[module_name].is_healthy = True

    def start(self):
        with self._lock:
            if self._running:
                return
            self._running = True
            self._thread = threading.Thread(
                target=self._monitor_loop, daemon=True, name="WatchdogMonitor"
            )
            self._thread.start()
            self.logger.info("Watchdog started.")

    def stop(self):
        with self._lock:
            self._running = False
        if self._thread:
            self._thread.join()
            self.logger.info("Watchdog stopped.")

    def _monitor_loop(self):
        while self._running:
            time.sleep(0.05)  # Check every 50ms
            now = time.time()
            with self._lock:
                for name, status in self._status.items():
                    if not status.is_healthy:
                        continue

                    time_since_last = now - status.last_heartbeat
                    if time_since_last > self.heartbeat_interval:
                        status.miss_count += 1
                        self.logger.warning(
                            f"Module {name} missed heartbeat. Miss count: {status.miss_count}"
                        )

                        if status.miss_count >= self.miss_limit:
                            status.is_healthy = False
                            self.logger.critical(f"CRITICAL: Module {name} failed watchdog check!")
                            callback = self._modules[name]["callback"]
                            try:
                                callback(name)
                            except Exception as e:
                                self.logger.error(f"Error in watchdog callback for {name}: {e}")

    def get_status(self) -> dict[str, ModuleStatus]:
        with self._lock:
            # Return a copy to ensure thread safety
            return {
                name: ModuleStatus(
                    last_heartbeat=s.last_heartbeat,
                    miss_count=s.miss_count,
                    is_healthy=s.is_healthy,
                )
                for name, s in self._status.items()
            }
