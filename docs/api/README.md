# OMNIDRIVE System API Reference & Architecture Specifications
**OMNIDRIVE Autonomous Driving AI System**  
**Document Version:** 2.4.0  
**Target Platform:** Tactical Military Vehicles, Autonomous Heavy Freight Trucks, Urban Robot Taxis  
**Classification:** Technical Architecture & API Reference  

---

## 1. System APIs Overview

The OMNIDRIVE autonomous driving software stack is organized into five primary Application Programming Interfaces (APIs). These interfaces isolate perceptual processing, latent visual forecasting, reinforcement learning policy synthesis, natural language reasoning, and low-level drive-by-wire vehicle control.

```
+-----------------------------------------------------------------------------------+
|                            OMNIDRIVE SYSTEM API LAYER                             |
+-----------------------------------------------------------------------------------+
|  1. SensorFusionAPI     :: Ingests raw multi-modal sensors -> UnifiedWorldState   |
|  2. JEPA_API            :: Encodes state into latent space  -> LatentState        |
|  3. RLControllerAPI     :: Computes motion control actions  -> DrivingAction      |
|  4. ReasoningAPI        :: Performs natural language audit  -> Text Explanation   |
|  5. VehicleInterfaceAPI :: Dispatches drive-by-wire CAN     -> VehicleTelemetry   |
+-----------------------------------------------------------------------------------+
```

### 1.1 Summary of Core API Interfaces

| API Interface | Target Module | Frequency | Primary Input | Primary Output | Transport / Protocol |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`SensorFusionAPI`** | Layer 1 (Sensor Fusion) | 50 Hz | `SensorBatch` (Cameras, LiDAR, RADAR) | `UnifiedWorldState` | CUDA Zero-Copy Pointers / IPC |
| **`JEPA_API`** | Layer 2 (JEPA World Model) | 10 Hz | `UnifiedWorldState` | `LatentState` | PyTorch C++ Extension / TensorRT |
| **`RLControllerAPI`** | Layer 4 (RL Motion Planner) | 20 Hz | `LatentState`, Route Goal | `DrivingAction` | Shared Memory Lock-Free Queue |
| **`ReasoningAPI`** | Layer 3 (Reasoning Engine) | 2 Hz | `UnifiedWorldState`, `LatentState` | `ReasoningOutput` | Async gRPC / IPC |
| **`VehicleInterfaceAPI`**| Layer 5 (Vehicle Interface)| 100 Hz | `DrivingAction` | `VehicleTelemetry` | SocketCAN / Automotive Ethernet |

---

## 2. Common Data Types & Schemas (`common_types.py`)

All OMNIDRIVE modules share standard Python `@dataclass` definitions enforcing strict typing, unit standards (SI units: meters, seconds, radians), and tensor shape invariants.

### 2.1 Standard Tensor Dimensions Reference Table

| Tensor Variable Name | Description | Dimension Shape | Data Type | Value Range |
| :--- | :--- | :--- | :--- | :--- |
| `camera_tensor` | Multi-View RGB Image Batch | $(B, N_{\text{cam}}, C, H, W)$ | `torch.float32` | Normalized $[0.0, 1.0]$ |
| `lidar_bev_tensor` | Rasterized LiDAR Bird's-Eye-View | $(B, H_{\text{bev}}, W_{\text{bev}}, C_{\text{bev}})$ | `torch.float32` | $[0.0, 1.0]$ |
| `z_context` | JEPA Context Latent Embedding | $(B, D_{\text{latent}})$ | `torch.float32` | $D_{\text{latent}} = 1024$, Unbounded |
| `z_predicted` | JEPA Predicted Future Latent | $(B, T_{\text{pred}}, D_{\text{latent}})$ | `torch.float32` | $T_{\text{pred}} = 8$ steps |
| `trajectory_tensor` | Ego Trajectory Points | $(B, T, 3)$ | `torch.float32` | $(x, y, \psi)$ meters/radians |
| `action_tensor` | Drive-by-Wire Control Commands | $(B, 3)$ | `torch.float32` | $[\delta, \alpha, \beta]^T$ (Steer, Throttle, Brake) |

### 2.2 Python Type Definitions Implementation (`common_types.py`)

```python
"""
OMNIDRIVE Common Data Types & Dataclass Definitions
Shared across all 7 layers of the autonomous driving stack.
"""

import time
import torch
import numpy as np
from enum import Enum, IntEnum
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


class SystemStatus(IntEnum):
    UNINITIALIZED = 0
    INITIALIZING = 1
    NORMAL_OPERATION = 2
    DEGRADED_OPERATION = 3
    EMERGENCY_STOP = 4
    FATAL_ERROR = 5


class GearPosition(Enum):
    PARK = "P"
    REVERSE = "R"
    NEUTRAL = "N"
    DRIVE = "D"
    LOW = "L"


@dataclass
class EgoPose:
    """Rigid body transformation of ego vehicle in WGS84 / Local UTM Frame."""
    x: float                     # X position in meters (ISO-8855 local tangent plane)
    y: float                     # Y position in meters
    z: float                     # Z altitude in meters
    roll: float                  # Roll angle in radians [-pi, pi]
    pitch: float                 # Pitch angle in radians [-pi, pi]
    yaw: float                   # Yaw angle in radians [-pi, pi]
    vx: float                    # Longitudinal velocity in m/s
    vy: float                    # Lateral velocity in m/s
    vz: float                    # Vertical velocity in m/s
    ax: float                    # Longitudinal acceleration in m/s^2
    ay: float                    # Lateral acceleration in m/s^2
    yaw_rate: float              # Angular velocity around Z axis in rad/s
    timestamp_ns: int = field(default_factory=lambda: time.time_ns())


@dataclass
class UnifiedWorldState:
    """Output dataclass of Layer 1 Sensor Fusion."""
    timestamp_ns: int
    ego_pose: EgoPose
    camera_tensors: Dict[str, torch.Tensor]  # Maps camera position to (C, H, W) tensor
    lidar_bev_tensor: torch.Tensor          # (H_bev, W_bev, C_bev) Bird's-Eye-View tensor
    radar_tracks: List[Dict[str, float]]    # Tracked RADAR objects (id, x, y, vx, vy)
    sensors_healthy: bool
    fault_flags: List[str] = field(default_factory=list)


@dataclass
class LatentState:
    """Output dataclass of Layer 2 JEPA World Model Engine."""
    timestamp_ns: int
    z_context: torch.Tensor                 # (1, 1024) Current latent representation
    z_predicted: Optional[torch.Tensor]     # (1, T_pred, 1024) Predicted future latents
    confidence_score: float                  # Model uncertainty measure [0.0, 1.0]
    is_anomaly: bool                         # Latent out-of-distribution anomaly flag


@dataclass
class DrivingAction:
    """Control output consumed by Layer 5 Vehicle Interface."""
    steering_angle_rad: float                # Wheel steering angle in radians [-0.61, 0.61]
    throttle: float                          # Throttle pedal position [0.0, 1.0]
    brake: float                             # Brake pedal position [0.0, 1.0]
    target_gear: GearPosition = GearPosition.DRIVE
    emergency_brake: bool = False
    timestamp_ns: int = field(default_factory=lambda: time.time_ns())


@dataclass
class SafetyFlag:
    """System health snapshot generated by Layer 6 Safety System."""
    system_status: SystemStatus
    fallback_triggered: bool
    active_fault_codes: List[int]
    time_since_heartbeat_ms: float
    reason: str = "Nominal"
```

---

## 3. Inter-Module Communication Paradigms

OMNIDRIVE employs a hybrid communication strategy tailored to module latency and thread affinity constraints.

```
+-----------------------------------------------------------------------------------+
|                        COMMUNICATION PARADIGMS TOPOLOGY                           |
+-----------------------------------------------------------------------------------+
| 1. High-Bandwidth GPU Pipelines (Sensor Fusion -> JEPA World Model):               |
|    └── Direct In-Process CUDA Memory Pointers (Zero-Copy CUDA IPC)                |
|                                                                                   |
| 2. Real-Time Motion Control (JEPA -> RL Planner -> Vehicle Interface):            |
|    └── Shared-Memory Lock-Free Ring Buffers (POSIX SHM / ZeroMQ C++ IPC)           |
|                                                                                   |
| 3. Cross-Process & Tele-Operation (Vehicle Interface -> Safety & Remote Fleet):   |
|    └── ROS 2 Topics (FastDDS Middleware over eBPF Shared Memory)                 |
+-----------------------------------------------------------------------------------+
```

### 3.1 ROS 2 Topic Architecture & Message Definitions

For cross-process deployment (e.g., interfacing with Autoware navigation or ROS 2 node ecosystems), standard ROS 2 topic interfaces are provided:

| ROS 2 Topic Name | Message Type | Rate (Hz) | Publisher Module | Subscriber Modules |
| :--- | :--- | :--- | :--- | :--- |
| `/omnidrive/sensor_fusion/world_state` | `omnidrive_msgs/msg/UnifiedWorldState` | 50 Hz | Layer 1 Sensor Fusion | Layer 2 JEPA, Layer 3 Reasoning |
| `/omnidrive/jepa/latent_state` | `omnidrive_msgs/msg/LatentState` | 10 Hz | Layer 2 JEPA Brain | Layer 4 RL Controller |
| `/omnidrive/control/driving_action` | `omnidrive_msgs/msg/DrivingAction` | 20 Hz | Layer 4 RL Controller | Layer 5 Vehicle Interface |
| `/omnidrive/safety/system_status` | `omnidrive_msgs/msg/SafetyFlag` | 100 Hz | Layer 6 Safety System | All Modules |

### 3.2 Inter-Module Latency Comparison

| Communication Mechanism | Target Payload Size | Latency Overhead | CPU Usage Overhead | Ideal Use-Case |
| :--- | :--- | :--- | :--- | :--- |
| **CUDA Zero-Copy Pointers** | High-Res Tensors (50 MB) | $< 0.05 \text{ ms}$ | $0.1\%$ | In-Process Sensor -> JEPA |
| **POSIX Lock-Free SHM** | Dataclasses (1 KB) | $< 0.12 \text{ ms}$ | $0.4\%$ | RL Planner -> Actuation |
| **ROS 2 FastDDS Shared Mem**| Protocol Buffers / ROS Msgs | $< 0.85 \text{ ms}$ | $1.2\%$ | Multi-Process System Interop |
| **gRPC / Protocol Buffers** | JSON / Text Reasoning | $< 4.50 \text{ ms}$ | $3.5\%$ | Layer 3 Natural Language Audit |

---

## 4. Standardized Error Codes Registry

OMNIDRIVE uses a 16-bit hexadecimal error code structure divided into 7 functional domains:

```
+-----------------------------------------------------------------------------------+
|                            ERROR CODE SCHEME (16-BIT)                             |
+-----------------------------------------------------------------------------------+
|  [ 0x0000 - 0x0FFF ] :: Success & General Infrastructure Errors                  |
|  [ 0x1000 - 0x1FFF ] :: Layer 1 Sensor Fusion Subsystem Errors                    |
|  [ 0x2000 - 0x2FFF ] :: Layer 2 JEPA World Model Subsystem Errors                 |
|  [ 0x3000 - 0x3FFF ] :: Layer 4 RL Controller Subsystem Errors                     |
|  [ 0x4000 - 0x4FFF ] :: Layer 3 Reasoning Engine Subsystem Errors                  |
|  [ 0x5000 - 0x5FFF ] :: Layer 5 Vehicle Interface Subsystem Errors                  |
|  [ 0x6000 - 0x6FFF ] :: Layer 6 System Safety & Redundancy Errors                 |
+-----------------------------------------------------------------------------------+
```

### 4.1 Master Error Code Table

| Error Hex Code | Error Constant Symbol | Description | Corrective Recovery Action |
| :--- | :--- | :--- | :--- |
| `0x0000` | `ERR_SUCCESS` | Operation completed nominally | None |
| `0x1001` | `ERR_CAMERA_STREAM_TIMEOUT` | Frame missing from GStreamer pipeline > 30ms | Restart GStreamer pipeline thread |
| `0x1002` | `ERR_LIDAR_POINT_DROP` | Point cloud density fell below 10% expected | Switch to Camera-only fallback |
| `0x1003` | `ERR_CALIBRATION_DRIFT` | Extrinsic transform residual > 0.05m | Trigger online re-calibration |
| `0x2001` | `ERR_JEPA_CUDA_OOM` | GPU memory exhausted during visual prediction | Clear CUDA cache, downscale window |
| `0x2002` | `ERR_JEPA_LATENT_NAN` | NaN value detected in output embedding $z$ | Reset GRU memory state vector |
| `0x3001` | `ERR_RL_ACTION_OUT_OF_BOUNDS` | Action command violated physical envelope | Clamp action to safe bounds |
| `0x3002` | `ERR_RL_RECOVERY_TRIGGERED` | Model uncertainty exceeds critical threshold | Trigger classical APF planner |
| `0x5001` | `ERR_CAN_BUS_OFF` | Physical CAN bus entered Bus-Off fault state | Reset CAN controller interface |
| `0x5002` | `ERR_STEERING_ACTUATOR_FAULT` | Steering drive motor reported hardware fault | Engage emergency mechanical brake |
| `0x6001` | `ERR_SAFETY_HEARTBEAT_TIMEOUT` | Module heartbeat silent > 50ms | Trigger Layer 6 Minimum Risk Maneuver |

---

## 5. Configuration Loading Engine (OmegaConf / Hydra)

OMNIDRIVE utilizes **Hydra** and **OmegaConf** to parse YAML configuration trees, validate parameters, and enable runtime command-line overrides.

### 5.1 Directory Structure of Configuration Files

```
configs/
├── main.yaml                      # Core orchestration config
├── sensor_fusion/
│   ├── camera.yaml                # Camera resolutions, FOV, frame rates
│   └── lidar.yaml                 # LiDAR spatial bounds, voxel sizes
├── jepa/
│   └── vit_base.yaml              # JEPA transformer depth, heads, embedding dim
├── rl/
│   └── ppo_actor.yaml             # RL policy parameters, action limits
└── vehicle/
    └── comma3x_can.yaml           # CAN bus bitrates, DBC path
```

### 5.2 Python Configuration Loader Implementation (`config_loader.py`)

```python
"""
OMNIDRIVE OmegaConf / Hydra Configuration Manager
"""

import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path


class OMNIDRIVEConfigManager:
    @staticmethod
    def load_config(config_path: str = "configs", config_name: str = "main") -> DictConfig:
        """Loads and resolves YAML configuration tree."""
        with hydra.initialize(config_path=config_path, version_base="1.2"):
            cfg = hydra.compose(config_name=config_name)
            OmegaConf.resolve(cfg)
            return cfg

    @staticmethod
    def override_param(cfg: DictConfig, key_path: str, value: any) -> DictConfig:
        """Dynamically modifies configuration parameters at runtime."""
        OmegaConf.update(cfg, key_path, value)
        return cfg


if __name__ == "__main__":
    # Example usage
    cfg = OMNIDRIVEConfigManager.load_config()
    print("[INFO] Loaded Configuration Root:\n", OmegaConf.to_yaml(cfg))
```

---

## 6. Logging Convention & Telemetry Standard

OMNIDRIVE mandates structured **JSON logging** across all layers to ensure seamless ingestion into Grafana, ElasticSearch, and ROS bag analyzers.

### 6.1 Log Format Specification

```json
{
  "timestamp": "2026-08-10T01:51:16.123456Z",
  "timestamp_ns": 1786326676123456000,
  "module": "SensorFusion",
  "layer": 1,
  "level": "INFO",
  "error_code": "0x0000",
  "latency_ms": 12.45,
  "gpu_memory_mb": 4120.5,
  "message": "Processed 6 camera streams and 1 LiDAR cloud successfully.",
  "telemetry": {
    "fps": 50.1,
    "num_points": 128450,
    "hdop": 0.95
  }
}
```

### 6.2 Python Logger Setup (`logger.py`)

```python
"""
OMNIDRIVE Standardized JSON Logging Module
"""

import sys
import json
import logging
from datetime import datetime


class StructuredJSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_payload = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "module": getattr(record, "module_name", record.name),
            "level": record.levelname,
            "error_code": getattr(record, "error_code", "0x0000"),
            "latency_ms": getattr(record, "latency_ms", 0.0),
            "message": record.getMessage(),
        }
        return json.dumps(log_payload)


def get_omnidrive_logger(module_name: str) -> logging.Logger:
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(StructuredJSONFormatter())
    
    if not logger.handlers:
        logger.addHandler(handler)
        
    return logger
```

---

## 7. Threading Model, Concurrency & Safety

To prevent thread contention and satisfy real-time execution bounds, OMNIDRIVE enforces explicit thread pinning and CUDA stream isolation.

```
+-----------------------------------------------------------------------------------+
|                        THREAD & PROCESS MAP TOPOLOGY                              |
+-----------------------------------------------------------------------------------+
| Thread 1: Sensor Ingestion Thread (50 Hz, CPU Core 1, PREEMPT_RT, Lock-Free Ring) |
| Thread 2: JEPA Neural Worker     (10 Hz, CPU Core 2, Dedicated CUDA Stream 1)   |
| Thread 3: RL Planner Thread      (20 Hz, CPU Core 3, Dedicated CUDA Stream 2)   |
| Thread 4: CAN Actuation Thread   (100 Hz, CPU Core 4, Real-Time FIFO Priority)   |
| Thread 5: Safety Monitor Thread  (100 Hz, CPU Core 0, High Priority Heartbeat)    |
+-----------------------------------------------------------------------------------+
```

### 7.1 Thread Safety Guarantees
1. **Lock-Free Atomic Queues:** Module boundaries use lock-free single-producer single-consumer (SPSC) ring buffers to pass memory pointers without mutex locks.
2. **CUDA Stream Independence:** The JEPA World Model runs on `cudaStream_1`, while the RL Policy runs on `cudaStream_2`. Synchronization occurs via explicit `cudaStreamWaitEvent` calls without blocking CPU threads.
3. **Real-Time Memory Lock (`mlockall`):** Critical processes execute `mlockall(MCL_CURRENT | MCL_FUTURE)` on Linux target hardware to prevent page fault delays during runtime.

---

## 8. Quick Integration Guide

Below is a complete **10-line Python integration example** demonstrating how to initialize and run the OMNIDRIVE autonomous driving stack end-to-end:

```python
import torch
from src.sensor_fusion.fusion_engine import SensorFusionModule
from src.jepa_brain.jepa_engine import JEPABrainModule
from src.rl_controller.rl_policy import RLControllerModule
from src.vehicle_interface.can_bus import VehicleInterfaceModule

# 1. Initialize OMNIDRIVE System Modules
sensor_fusion = SensorFusionModule(config_path="configs/sensor_fusion.yaml")
jepa_brain = JEPABrainModule(model_path="weights/jepa_vit_base.pth")
rl_controller = RLControllerModule(policy_path="weights/rl_actor.pth")
vehicle_interface = VehicleInterfaceModule(can_channel="can0")

# 2. Main Autonomous Control Loop (Runs continuously)
while True:
    world_state = sensor_fusion.process_frame()                       # Ingest & Fuse Sensors
    latent_state = jepa_brain.encode_and_predict(world_state)         # Generate World Embeddings
    driving_action = rl_controller.compute_action(latent_state)      # Compute Motion Command
    vehicle_interface.dispatch_action(driving_action)                 # Send CAN Drive-by-Wire
```

---
**End of File 2: OMNIDRIVE System API Reference**
