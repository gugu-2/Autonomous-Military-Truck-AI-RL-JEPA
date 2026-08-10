"""Encrypted black box logger — last 60 seconds of all system data."""
import collections
import threading
import json
import time
import gzip
import logging
import datetime
from pathlib import Path
from typing import Any, Dict, List

try:
    from cryptography.fernet import Fernet
    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False


class BlackBoxLogger:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        fps = config.get("fps", 20)
        retention_seconds = config.get("retention_seconds", 60)
        self.maxlen = fps * retention_seconds
        
        self.buffer = collections.deque(maxlen=self.maxlen)
        self.events = []
        self._lock = threading.Lock()
        
        self.encryption_key = config.get("encryption_key", None)
        self.use_encryption = bool(self.encryption_key and HAS_CRYPTO)
        if self.use_encryption:
            self.cipher = Fernet(self.encryption_key)
        else:
            if config.get("encryption_key"):
                self.logger.warning("Encryption key provided but cryptography library not found.")
                
        self.log_dir = Path(config.get("log_dir", "/tmp/omnidrive_logs"))
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def log_frame(self, frame_id: int, timestamp: float, sensor_summary: dict, jepa_latent_summary: dict, rl_action, safety_flag, can_commands: list):
        data = {
            "type": "frame",
            "frame_id": frame_id,
            "timestamp": timestamp,
            "sensor": sensor_summary,
            "jepa": jepa_latent_summary,
            "action": {"steer": rl_action.steering, "throttle": rl_action.throttle, "brake": rl_action.brake} if rl_action else None,
            "safety_flag": safety_flag.name if hasattr(safety_flag, "name") else str(safety_flag),
            "can": can_commands
        }
        with self._lock:
            self.buffer.append(data)

    def log_event(self, event_type: str, data: dict):
        event = {
            "type": "event",
            "timestamp": time.time(),
            "event_type": event_type,
            "data": data
        }
        with self._lock:
            self.events.append(event)
            self.buffer.append(event)
        self.logger.info(f"Logged event: {event_type}")

    def dump_to_file(self, reason: str):
        # Fire and forget in a background thread
        threading.Thread(target=self._dump_async, args=(reason,), daemon=True).start()

    def _dump_async(self, reason: str):
        self.logger.info(f"Dumping black box due to: {reason}")
        with self._lock:
            dump_data = {
                "reason": reason,
                "dump_time": time.time(),
                "history": list(self.buffer),
                "events": list(self.events)
            }
        
        try:
            json_str = json.dumps(dump_data)
            out_path = self._get_output_path(reason)
            
            if self.use_encryption:
                data_to_write = self.cipher.encrypt(json_str.encode('utf-8'))
            else:
                data_to_write = json_str.encode('utf-8')
                
            with gzip.open(out_path, 'wb') as f:
                f.write(data_to_write)
                
            self.logger.info(f"Black box successfully dumped to {out_path}")
        except Exception as e:
            self.logger.error(f"Failed to dump black box: {e}")

    def _get_output_path(self, reason: str) -> Path:
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_reason = "".join([c if c.isalnum() else "_" for c in reason])
        ext = ".json.gz.enc" if self.use_encryption else ".json.gz"
        return self.log_dir / f"blackbox_{timestamp_str}_{safe_reason}{ext}"
