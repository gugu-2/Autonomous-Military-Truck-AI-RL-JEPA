import os

base_dir = r"c:\Users\majip\Downloads\rl-jepa-car ai\OMNIDRIVE_PROJECT\src"

def patch_safety_monitor():
    path = os.path.join(base_dir, "safety", "safety_monitor.py")
    with open(path) as f:
        content = f.read()

    content = content.replace("steering: float", "steering_angle: float")
    content = content.replace("action.steering", "action.steering_angle")
    content = content.replace("prev_action.steering", "prev_action.steering_angle")

    old_check_rl = """        if (
            action.steering_angle > 1.0
            or action.steering_angle < -1.0
            or action.throttle > 1.0
            or action.brake > 1.0
        ):"""
    new_check_rl = """        if (
            action.steering_angle > 1.0
            or action.steering_angle < -1.0
            or action.throttle > 1.0
            or action.throttle < 0.0
            or action.brake > 1.0
            or action.brake < 0.0
        ):"""
    content = content.replace(old_check_rl, new_check_rl)

    old_full_check = """    def full_check(
        self,
        jepa_latency: float,
        hazard_energy: float,
        action: DrivingAction,
        prev_action: DrivingAction,
        sensor_health: SensorHealthStatus,
    ) -> SafetyFlag:
        flags = [
            self.check_jepa_health(jepa_latency, hazard_energy),
            self.check_rl_output(action, prev_action, 0.05),  # assuming 20Hz dt
            self.check_sensor_health(sensor_health),
        ]"""
    new_full_check = """    def full_check(
        self,
        jepa_latency: float,
        hazard_energy: float,
        action: DrivingAction,
        prev_action: DrivingAction,
        sensor_health: SensorHealthStatus,
        last_can_ms: float = 0.0,
    ) -> SafetyFlag:
        flags = [
            self.check_jepa_health(jepa_latency, hazard_energy),
            self.check_rl_output(action, prev_action, 0.05),  # assuming 20Hz dt
            self.check_sensor_health(sensor_health),
            self.check_can_heartbeat(last_can_ms),
        ]"""
    content = content.replace(old_full_check, new_full_check)

    with open(path, "w") as f:
        f.write(content)


def patch_watchdog():
    path = os.path.join(base_dir, "safety", "watchdog.py")
    with open(path) as f:
        content = f.read()

    old_stop = """    def stop(self):
        with self._lock:
            self._running = False
        if self._thread:
            self._thread.join()
            self.logger.info("Watchdog stopped.")"""
    new_stop = """    def stop(self):
        with self._lock:
            self._running = False
        if self._thread:
            if threading.current_thread() is not self._thread:
                self._thread.join(timeout=2.0)
            self.logger.info("Watchdog stopped.")"""
    content = content.replace(old_stop, new_stop)

    old_monitor_meth = """    def _monitor_loop(self):
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
                                self.logger.error(f"Error in watchdog callback for {name}: {e}")"""
    new_monitor_meth = """    def _monitor_loop(self):
        while self._running:
            time.sleep(0.05)  # Check every 50ms
            now = time.time()
            callbacks_to_fire = []
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
                            callbacks_to_fire.append((callback, name))
            
            for cb, nm in callbacks_to_fire:
                try:
                    cb(nm)
                except Exception as e:
                    self.logger.error(f"Error in watchdog callback for {nm}: {e}")"""
    content = content.replace(old_monitor_meth, new_monitor_meth)

    with open(path, "w") as f:
        f.write(content)


def patch_failsafe():
    path = os.path.join(base_dir, "safety", "failsafe_controller.py")
    with open(path) as f:
        content = f.read()

    old_imports = """import logging
import threading
import time
from enum import Enum, auto"""
    new_imports = """import logging
import threading
import time
import struct
from enum import Enum, auto
from vehicle_interface.can_bus.can_encoder import CANCommandEncoder, DrivingAction"""
    content = content.replace(old_imports, new_imports)

    old_init = """    def __init__(self, vehicle_interface, config):
        self.vehicle_interface = vehicle_interface
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.state = FailsafeState.NOMINAL
        self._lock = threading.Lock()
        self._decel_thread = None"""
    new_init = """    def __init__(self, can_driver, can_decoder, config):
        self.can_driver = can_driver
        self.can_decoder = can_decoder
        self.can_encoder = CANCommandEncoder()
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.state = FailsafeState.NOMINAL
        self._lock = threading.Lock()
        self._decel_thread = None"""
    content = content.replace(old_init, new_init)

    old_em_stop = """    def trigger_emergency_stop(self, reason: str):
        self.logger.critical(f"EMERGENCY STOP Triggered: {reason}")
        try:
            self.vehicle_interface.apply_max_brake()
            self.vehicle_interface.disable_drive_power()
            self.state = FailsafeState.STOPPED
        except Exception as e:
            self.logger.error(f"Failed to apply emergency stop: {e}")"""
    new_em_stop = """    def trigger_emergency_stop(self, reason: str):
        self.logger.critical(f"EMERGENCY STOP Triggered: {reason}")
        try:
            action1 = DrivingAction(steering_angle=0.0, throttle=0.0, brake=1.0, gear=0)
            for msg in self.can_encoder.encode_action(action1):
                self.can_driver.send(msg.arbitration_id, msg.data, msg.is_extended_id)
            self.state = FailsafeState.STOPPED
        except Exception as e:
            self.logger.error(f"Failed to apply emergency stop: {e}")"""
    content = content.replace(old_em_stop, new_em_stop)

    old_decel_loop = """    def _decel_loop(self, target_decel: float):
        try:
            while True:
                current_speed = self.vehicle_interface.get_current_speed()
                if self.is_stopped(current_speed):
                    self.logger.info("Vehicle has fully stopped.")
                    with self._lock:
                        self.state = FailsafeState.STOPPED
                    self.vehicle_interface.apply_parking_brake()
                    break

                # Apply brake corresponding to target deceleration
                self.vehicle_interface.apply_brake_for_deceleration(target_decel)
                time.sleep(0.05)
        except Exception as e:
            self.logger.error(f"Error in controlled deceleration loop: {e}")
            self.trigger_emergency_stop("Deceleration loop failed")"""
    new_decel_loop = """    def _decel_loop(self, target_decel: float):
        try:
            while True:
                current_speed = self.can_decoder.get_vehicle_status().speed_ms
                if self.is_stopped(current_speed):
                    self.logger.info("Vehicle has fully stopped.")
                    with self._lock:
                        self.state = FailsafeState.STOPPED
                    action = DrivingAction(steering_angle=0.0, throttle=0.0, brake=1.0, gear=0)
                    for msg in self.can_encoder.encode_action(action):
                        self.can_driver.send(msg.arbitration_id, msg.data, msg.is_extended_id)
                    break

                # Apply brake corresponding to target deceleration
                brake_val = min(1.0, max(0.0, abs(target_decel) / 10.0))
                action = DrivingAction(steering_angle=0.0, throttle=0.0, brake=brake_val, gear=3)
                for msg in self.can_encoder.encode_action(action):
                    if msg.arbitration_id == 0x102:
                        self.can_driver.send(msg.arbitration_id, msg.data, msg.is_extended_id)
                time.sleep(0.05)
        except Exception as e:
            self.logger.error(f"Error in controlled deceleration loop: {e}")
            self.trigger_emergency_stop("Deceleration loop failed")"""
    content = content.replace(old_decel_loop, new_decel_loop)

    old_hazard = """    def activate_hazard_lights(self):
        try:
            self.vehicle_interface.send_can_message(0x350, [0x01])
            self.logger.info("Hazard lights activated.")
        except Exception as e:
            self.logger.error(f"Failed to activate hazard lights: {e}")"""
    new_hazard = """    def activate_hazard_lights(self):
        try:
            self.can_driver.send(0x350, bytes([0x01]), False)
            self.logger.info("Hazard lights activated.")
        except Exception as e:
            self.logger.error(f"Failed to activate hazard lights: {e}")"""
    content = content.replace(old_hazard, new_hazard)

    old_sos = """    def broadcast_sos(self, reason: str):
        self.logger.info(f"Broadcasting SOS to fleet API: {reason}")
        # In a real implementation, this would make an HTTP/gRPC request to fleet management
        pass"""
    new_sos = """    def broadcast_sos(self, reason: str):
        self.logger.info(f"Broadcasting SOS to fleet API: {reason}")
        try:
            current_speed = self.can_decoder.get_vehicle_status().speed_ms
            payload = b'SOS\\x00' + struct.pack('<f', current_speed)
            self.can_driver.send(0x7DF, payload, False)
        except Exception as e:
            self.logger.error(f"Failed to broadcast SOS: {e}")"""
    content = content.replace(old_sos, new_sos)

    with open(path, "w") as f:
        f.write(content)


def patch_black_box_logger():
    path = os.path.join(base_dir, "safety", "black_box_logger.py")
    with open(path) as f:
        content = f.read()

    new_encoder = """import json
import logging
import threading
import time
from pathlib import Path
import numpy as np

class _SafeJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        if hasattr(obj, 'item'):
            return obj.item()
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        return str(obj)"""
    content = content.replace("import json\nimport logging\nimport threading\nimport time\nfrom pathlib import Path", new_encoder)

    old_action_log = """            "action": (
                {
                    "steer": rl_action.steering,
                    "throttle": rl_action.throttle,
                    "brake": rl_action.brake,
                }
                if rl_action
                else None
            ),"""
    new_action_log = """            "action": (
                {
                    "steer": getattr(rl_action, 'steering_angle', getattr(rl_action, 'steering', 0.0)),
                    "throttle": rl_action.throttle,
                    "brake": rl_action.brake,
                }
                if rl_action
                else None
            ),"""
    content = content.replace(old_action_log, new_action_log)

    content = content.replace("json_str = json.dumps(dump_data)", "json_str = json.dumps(dump_data, cls=_SafeJSONEncoder)")

    with open(path, "w") as f:
        f.write(content)


def patch_can_decoder():
    path = os.path.join(base_dir, "vehicle_interface", "can_bus", "can_decoder.py")
    with open(path) as f:
        content = f.read()

    old_imports = """import struct
import time
from dataclasses import dataclass"""
    new_imports = """import struct
import time
import threading
import copy
import dataclasses
from dataclasses import dataclass"""
    content = content.replace(old_imports, new_imports)

    content = content.replace("        self._status = VehicleStatus()", "        self._status = VehicleStatus()\n        self._lock = threading.Lock()")

    old_speed = """            speed = ((fl + fr + rl + rr) / 4.0) * 0.01
            self._status.speed_ms = speed
            return speed
        return self._status.speed_ms"""
    new_speed = """            speed = ((fl + fr + rl + rr) / 4.0) * 0.01
            with self._lock:
                self._status.speed_ms = speed
            return speed
        with self._lock:
            return self._status.speed_ms"""
    content = content.replace(old_speed, new_speed)

    old_steer = """            angle = raw_angle * 0.1
            self._status.steering_angle_deg = angle
            return angle
        return self._status.steering_angle_deg"""
    new_steer = """            angle = raw_angle * 0.1
            with self._lock:
                self._status.steering_angle_deg = angle
            return angle
        with self._lock:
            return self._status.steering_angle_deg"""
    content = content.replace(old_steer, new_steer)

    old_engine = """            self._status.engine_rpm = float(rpm)
            self._status.gear = gear
            self._status.fault_code = faults
            return {"rpm": rpm, "gear": gear, "faults": faults}
        return {}"""
    new_engine = """            with self._lock:
                self._status.engine_rpm = float(rpm)
                self._status.gear = gear
                self._status.fault_code = faults
            return {"rpm": rpm, "gear": gear, "faults": faults}
        return {}"""
    content = content.replace(old_engine, new_engine)

    old_brake = """            pressure = pressure_raw * 0.1
            self._status.brake_pressure = pressure
            return {"brake_pressure": pressure}
        return {}"""
    new_brake = """            pressure = pressure_raw * 0.1
            with self._lock:
                self._status.brake_pressure = pressure
            return {"brake_pressure": pressure}
        return {}"""
    content = content.replace(old_brake, new_brake)

    old_msg = """        self._status.timestamp = time.time()"""
    new_msg = """        with self._lock:
            self._status.timestamp = time.time()"""
    content = content.replace(old_msg, new_msg)

    old_get_status = """    def get_vehicle_status(self) -> VehicleStatus:
        \"\"\"Returns the latest fused vehicle status.\"\"\"
        return self._status"""
    new_get_status = """    def get_vehicle_status(self) -> VehicleStatus:
        \"\"\"Returns the latest fused vehicle status.\"\"\"
        with self._lock:
            return dataclasses.replace(self._status)"""
    content = content.replace(old_get_status, new_get_status)

    with open(path, "w") as f:
        f.write(content)


def patch_can_driver():
    path = os.path.join(base_dir, "vehicle_interface", "can_bus", "can_driver.py")
    with open(path) as f:
        content = f.read()

    content = content.replace("self._running = False", "self._running = False\n        self._recovering = False")

    old_handle = """    def _handle_bus_error(self):
        \"\"\"Attempt bus-off recovery with exponential backoff.\"\"\"
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
                backoff = min(backoff * 2, max_backoff)"""

    new_handle = """    def _handle_bus_error(self):
        if getattr(self, '_recovering', False):
            return
        import threading
        threading.Thread(target=self._do_recovery, daemon=True).start()

    def _do_recovery(self):
        self._recovering = True
        try:
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
        finally:
            self._recovering = False"""
    content = content.replace(old_handle, new_handle)

    with open(path, "w") as f:
        f.write(content)


def patch_jaus():
    path = os.path.join(base_dir, "vehicle_interface", "military", "jaus_interface.py")
    with open(path) as f:
        content = f.read()

    old_init = """    def __init__(
        self, local_address: JausAddress, remote_address: JausAddress, udp_port: int = 3794
    ):
        self.local_address = local_address
        self.remote_address = remote_address
        self.udp_port = udp_port"""
    new_init = """    def __init__(
        self, local_address: JausAddress, remote_address: JausAddress, udp_port: int = 3794, remote_ip: str = '127.0.0.1'
    ):
        self.local_address = local_address
        self.remote_address = remote_address
        self.udp_port = udp_port
        self.remote_ip = remote_ip"""
    content = content.replace(old_init, new_init)

    old_send = """        # Assume broadcast or configured IP for remote. Using localhost for generic demonstration
        self.sock.sendto(data, ("127.0.0.1", self.udp_port))"""
    new_send = """        self.sock.sendto(data, (self.remote_ip, self.udp_port))"""
    content = content.replace(old_send, new_send)

    old_recv_1 = """            msg_type = struct.unpack("<H", data[14:16])[0]"""
    new_recv_1 = """            msg_type = struct.unpack("<H", data[15:17])[0]"""
    content = content.replace(old_recv_1, new_recv_1)

    old_recv_2 = """                    speed = struct.unpack("<d", data[16:24])[0]"""
    new_recv_2 = """                    speed = struct.unpack("<d", data[17:25])[0]"""
    content = content.replace(old_recv_2, new_recv_2)

    with open(path, "w") as f:
        f.write(content)


def patch_convoy():
    path = os.path.join(base_dir, "vehicle_interface", "military", "convoy_mode.py")
    with open(path) as f:
        content = f.read()

    old_halt = """    def emergency_convoy_halt(self):
        \"\"\"Propagates halt command immediately.\"\"\"
        logger.critical("EMERGENCY CONVOY HALT TRIGGERED.")
        # Network broadcast hook would go here"""
    new_halt = """    def emergency_convoy_halt(self):
        \"\"\"Propagates halt command immediately.\"\"\"
        logger.critical('EMERGENCY CONVOY HALT broadcast')
        import socket, struct, time
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            halt_payload = struct.pack('>4sI', b'HALT', int(time.time()))
            sock.sendto(halt_payload, ('<broadcast>', 5005))
            sock.close()
        except Exception as e:
            logger.error(f"Halt broadcast failed: {e}")"""
    content = content.replace(old_halt, new_halt)

    old_dist = """        # Compute distance to leader
        dx = self.leader_state.x - ego_state.x
        dy = self.leader_state.y - ego_state.y
        distance = (dx**2 + dy**2) ** 0.5

        # Error term
        error = distance - self.target_gap_m"""
    new_dist = """        import math
        # Compute distance to leader
        dx = self.leader_state.x - ego_state.x
        dy = self.leader_state.y - ego_state.y
        ego_heading = ego_state.heading if hasattr(ego_state, 'heading') else 0.0
        longitudinal_dist = dx * math.cos(ego_heading) + dy * math.sin(ego_heading)
        error = longitudinal_dist - self.target_gap_m"""
    content = content.replace(old_dist, new_dist)

    with open(path, "w") as f:
        f.write(content)


def patch_j1939():
    path = os.path.join(base_dir, "vehicle_interface", "truck", "j1939_interface.py")
    with open(path) as f:
        content = f.read()

    old_init = """    def __init__(self, driver, source_address: int = 0xF9):
        # driver is assumed to be an instance of CANDriver
        self.driver = driver
        self.source_address = source_address
        self.last_speed = 0.0"""
    new_init = """    def __init__(self, driver, source_address: int = 0xF9):
        # driver is assumed to be an instance of CANDriver
        self.driver = driver
        self.source_address = source_address
        self.last_speed = 0.0
        PGN_CCVS = 65265
        if hasattr(self.driver, 'register_callback'):
            self.driver.register_callback(PGN_CCVS, self._on_ccvs_message)
        elif hasattr(self.driver, 'subscribe'):
            self.driver.subscribe(PGN_CCVS, self._on_ccvs_message)

    def _on_ccvs_message(self, msg):
        msg_obj = self._decode_j1939_frame(msg)
        if msg_obj is not None:
            speed_raw = struct.unpack_from('<H', msg_obj.data, 1)[0]
            self.last_speed = speed_raw * 0.00390625  # 1/256 km/h per bit"""
    content = content.replace(old_init, new_init)

    with open(path, "w") as f:
        f.write(content)

def main():
    patch_safety_monitor()
    patch_watchdog()
    patch_failsafe()
    patch_black_box_logger()
    patch_can_decoder()
    patch_can_driver()
    patch_jaus()
    patch_convoy()
    patch_j1939()
    print("PATCHES APPLIED")

if __name__ == "__main__":
    main()
