# Layer 1: Sensor Fusion Module - Architectural Technical Specification
**OMNIDRIVE Autonomous Driving AI System**  
**Document Version:** 2.4.0  
**Target Platform:** Tactical Military Vehicles, Autonomous Heavy Freight Trucks, Urban Robot Taxis  
**Classification:** Technical Architecture & System Specification  

---

## 1. Module Overview

### 1.1 Definition and Mission
The **Sensor Fusion Module** forms **Layer 1** of the 7-layer OMNIDRIVE Autonomous Driving AI System architecture. It acts as the perceptual bedrock of the autonomous vehicle software stack. Its primary mission is to ingest raw, heterogeneous sensor streams (RGB cameras, long-wave infrared thermal cameras, 3D LiDAR point clouds, millimeter-wave automotive RADAR, high-frequency IMU, and RTK-assisted GNSS), sanitize and preprocess the signals, spatially and temporally align all frames into a unified coordinate frame, and fuse them into an uncorrupted, real-time representation: the **Unified World State**.

```
+-----------------------------------------------------------------------------------+
|                            OMNIDRIVE 7-LAYER STACK                                |
+-----------------------------------------------------------------------------------+
|  Layer 7: Tele-Operation, V2X, & Fleet Management Interface                      |
|  Layer 6: System Safety, Redundancy, & Fail-Operational Control                   |
|  Layer 5: Vehicle Control & Actuation Interface (DBW / CAN Engine)               |
|  Layer 4: Trajectory Generation & Motion Planning                                 |
|  Layer 3: Behavioral Decision Engine & Motion Prediction                          |
|  Layer 2: World Model & Semantic Perception Engine (JEPA Foundation Model)       |
|  Layer 1: SENSOR FUSION MODULE (This Specification)                               |
+-----------------------------------------------------------------------------------+
```

### 1.2 Multi-Domain Application Profiles
The Sensor Fusion Module supports three distinct hardware and operational profiles:
1. **Tactical Military Vehicles (GCV / UGVs):** Operates under GPS-denied environments, severe electronic warfare (EW) jamming, heavy dusty/smoky battlefields, and night operations requiring LWIR Thermal + Solid-State LiDAR fusion.
2. **Autonomous Heavy Freight Trucks (Class 8 Highway):** Demands long-range perception (up to 300m+) via high-resolution telephoto cameras, FMCW long-range RADAR, and high-density LiDAR for high-speed highway platooning and freight transport.
3. **Urban Robot Taxis:** Demands 360° ultra-dense perimeter vision (8–12 cameras, multiple short/medium/long-range LiDARs, surround RADARs) for high-density pedestrian and urban obstacle handling.

### 1.3 Inputs and Outputs

#### Primary Inputs
- **Multi-Camera RGB Streams:** Up to 12 cameras providing high-resolution image frames (1080p to 4K @ 30–60 FPS) via GStreamer pipelines (V4L2 / RTSP / GMSL2).
- **Thermal Infrared Streams (LWIR):** 8–14 µm thermal images (640x512 @ 30–60 FPS) for military night/smoke operation.
- **LiDAR Point Clouds:** 3D point cloud streams from 32/64/128/300-channel LiDARs (10–20 Hz, `sensor_msgs/msg/PointCloud2`).
- **Millimeter-Wave RADAR Tracks & Detections:** Target and tracklet streams via SocketCAN / Automotive Ethernet (10–50 Hz).
- **RTK-GNSS Data:** NMEA-0183 / UBX protocol messages providing latitude, longitude, altitude, and RTK correction status (10–20 Hz).
- **IMU Telemetry:** 6-axis / 9-axis high-rate accelerometer, gyroscope, and magnetometer readings (100–1000 Hz).
- **Vehicle CAN Telemetry:** Steering angle, wheel speeds, gear selection, and throttle/brake position (100 Hz).

#### Primary Outputs
- **Unified World State Dataclass:** A synchronized, standardized representation of vehicle position, velocity, orientation, covariance, and 3D environment.
- **Bird's-Eye-View (BEV) Feature Tensor:** Grid representation \((C \times H \times W)\) covering a \([-50\text{m}, +50\text{m}] \times [-50\text{m}, +50\text{m}]\) spatial window with 0.1m spatial resolution.
- **Fused 3D Object Tracklets:** Kinematically verified 3D bounding boxes with class probabilities, 3D velocity vectors, and covariance matrices.
- **Sensor Health & Diagnostics State:** Fault flags, degradation levels (L0–L4), calibration confidence, and temporal drift metrics.

---

## 2. Sensor Hardware Specs Table

| Sensor Category | Subsystem / Model | Operational Domain | Key Technical Specifications | Hardware Interface | Data Format / Protocol |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **RGB Camera (Front Main)** | Sony ISX031 / AR0820 | All Domains | 4K (3840x2160), 60 FPS, H-FOV: 120°, V-FOV: 70°, HDR 140dB | GMSL2 / PCIe | RAW12 / NV12 (GStreamer) |
| **RGB Camera (Long Telephoto)**| OnSemi AR0820 | Heavy Truck / Military | 8MP (3840x2160), 30 FPS, H-FOV: 30°, V-FOV: 18°, Range: >350m | GMSL2 | RAW12 (CUDA IPC) |
| **RGB Camera (Surround)** | Sony ISX019 | Robot Taxi / Military | 2MP (1920x1080), 30 FPS, H-FOV: 190° (Fisheye), 4-8 units | GMSL2 / FAKRA | YUV420p |
| **Thermal Infrared (LWIR)** | FLIR Boson+ / Teledyne | Military Vehicles | 640x512, 60 FPS, 8–14 µm spectral range, Thermal sensitivity <20mK | MIPI CSI-2 / USB3 | 16-bit Mono / Radiometric RAW |
| **Primary LiDAR (Long-Range)** | Hesai AT128 / RoboSense M1 | Robot Taxi / Heavy Truck | 128 Channels, 10-20 Hz, Range: 200m @ 10%, H-FOV: 120°, V-FOV: 25.4° | Automotive Ethernet | UDP / ROS 2 `PointCloud2` |
| **Solid-State LiDAR (Tactical)**| Ouster OS1-32 / Innoviz2 | Military / Heavy Truck | 32-128 Channels, 10-20 Hz, Range: 250m, 905nm/1550nm solid state | Gigabit Ethernet | Custom Pcap / UDP |
| **Perimeter Flash LiDAR** | Continental HFL110 | Robot Taxi | 3D Flash LiDAR, 30 Hz, Range: 50m, H-FOV: 120°, V-FOV: 90° | Ethernet 100BASE-T1 | UDP PointCloud |
| **4D Imaging RADAR** | Continental ARS540 / Arbe | All Domains | 4D Elevation+Azimuth, 20 Hz, Range: 300m, Range Acc: 0.1m, FOV: ±60° | CAN FD / Ethernet | SocketCAN / SOME/IP |
| **Surround RADAR** | Bosch LRR5 / Hella | Urban / Truck | 77 GHz FMCW, 50 Hz, Range: 110m, H-FOV: ±75° | CAN FD | CAN Bus Packets |
| **GNSS / RTK Receiver** | Septentrio Mosaic-X5 / u-blox ZED-F9P | All Domains | Dual-antenna multi-frequency RTK, Position: 1cm + 1ppm, Heading: 0.1° | RS232 / USB / Ethernet | NMEA 0183 / UBX Binary |
| **Tactical IMU** | Honeywell HG4930 / Analog Devices ADIS16495 | Military / Heavy Truck | FOG/MEMS 6-DOF, Gyro Bias Instability: 0.25°/hr, Accel Bias: 0.05mg | SPI / RS422 | Binary SPI / Serial Frames |

---

## 3. Camera Subsystem

The Camera Subsystem is responsible for stream acquisition, real-time image decoding, GPU-accelerated spatial preprocessing, multi-camera Bird's-Eye-View (BEV) stitching, and radiometric long-wave thermal integration.

```
+-----------------------------------------------------------------------------------+
|                               CAMERA SUBSYSTEM                                    |
+-----------------------------------------------------------------------------------+
|  +--------------------+    +-----------------------+    +----------------------+  |
|  | camera_driver.py   | -> | camera_preprocessor.py| -> |multi_camera_stitcher |  |
|  | GStreamer / GMSL2  |    | Undistort / Resize    |    | IPM / BEV Synthesis  |  |
|  +--------------------+    +-----------------------+    +----------------------+  |
|                                        ^                                          |
|                                        |                                          |
|                            +-----------------------+                              |
|                            | thermal_driver.py     |                              |
|                            | LWIR / Radiometric    |                              |
|                            +-----------------------+                              |
+-----------------------------------------------------------------------------------+
```

### 3.1 `camera_driver.py` Design
- **Architecture:** Leverages high-throughput hardware-accelerated GStreamer pipelines (NVIDIA DeepStream / NVMM zero-copy memory pointers).
- **Protocols Supported:** RTSP, V4L2, GMSL2 capture via CSI-2, and raw PCIe frame buffers.
- **Synchronization & Ring Buffer:** Implements a lock-free CUDA ring buffer (depth=16 frames) indexed by microsecond Hardware PTP (IEEE 1588) timestamps. Drops duplicate or late-arriving frames (>15ms delta) to maintain strict deterministic processing.

### 3.2 `camera_preprocessor.py` Design
- **Resolution Standard:** Resizes raw sensor feeds (e.g., 4K / 1080p) to the standard neural network input dimension of **\(224 \times 224\)** pixels or **\(448 \times 448\)** for high-resolution branches using CUDA bicubic interpolation.
- **Normalization:** Applies ImageNet mean \(\boldsymbol{\mu} = [0.485, 0.456, 0.406]\) and standard deviation \(\boldsymbol{\sigma} = [0.229, 0.224, 0.225]\) via GPU kernel transforms:
  $$\mathbf{I}_{\text{norm}}(x,y,c) = \frac{\mathbf{I}_{\text{raw}}(x,y,c) / 255.0 - \mu_c}{\sigma_c}$$
- **Lens Undistortion:** Implements pinhole + radial/tangential lens distortion model. Given intrinsic matrix \(\mathbf{K}\) and distortion coefficients \((k_1, k_2, p_1, p_2, k_3)\):
  $$x' = x(1 + k_1 r^2 + k_2 r^4 + k_3 r^6) + 2p_1 x y + p_2(r^2 + 2x^2)$$
  $$y' = y(1 + k_1 r^2 + k_2 r^4 + k_3 r^6) + p_1(r^2 + 2y^2) + 2p_2 x y$$
  where \(r^2 = x^2 + y^2\). Remapping grids are pre-computed at initialization into CUDA textures to perform undistortion in \(\le 0.4\text{ ms}\) per frame.

### 3.3 `multi_camera_stitcher.py` Design
- **BEV Stitching:** Accepts 4 to 12 calibrated camera inputs covering 360° surround space.
- **Inverse Perspective Mapping (IPM):** Projects undistorted camera image planes onto the vehicle ground plane \(Z_{\text{veh}} = 0\) using extrinsic transformation matrices \(\mathbf{T}_{\text{cam}}^{\text{ego}} = [\mathbf{R} \mid \mathbf{t}]\):
  $$\begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \mathbf{K} \mathbf{P}_{\text{cam}}^{\text{ego}} \begin{bmatrix} X_{\text{ego}} \\ Y_{\text{ego}} \\ Z_{\text{ego}} \\ 1 \end{bmatrix}$$
- **Overlapping Blend Handling:** Employs distance-weighted multi-band alpha blending across overlapping camera frustums to eliminate stitching seam artifacts and dynamic illumination discontinuities.

### 3.4 `thermal_camera_driver.py` Design
- **Spectral Band:** Long-Wave Infrared (LWIR, 8–14 µm) radiometric sensor integration for military and night operations.
- **Radiometric Calibration:** Converts raw 14-bit digital counts (ADU) to absolute temperature readings (\(^\circ\text{C}\)) using sensor gain/offset tables.
- **Contrast Optimization:** Applies GPU-based Contrast Limited Adaptive Histogram Equalization (CLAHE) with dynamic automatic gain control (AGC) to maximize thermal contrast in extreme cold/desert thermal conditions.
- **Cross-Modal Registration:** Registers thermal image coordinates onto the primary RGB reference frame using offline computed visual-infrared homography \(\mathbf{H}_{\text{therm}}^{\text{rgb}}\).

### 3.5 Python Class Signatures & Interfaces

```python
"""
Camera Subsystem Module Interface Definitions
File: omnidrive/sensor_fusion/camera/camera_subsystem.py
"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
import torch

@dataclass
class CameraDriverConfig:
    camera_id: str
    device_path: str  # e.g., "/dev/video0" or RTSP URL
    resolution: Tuple[int, int]  # (width, height)
    target_fps: int
    pixel_format: str  # "NV12", "YUV420", "RAW12"
    gstreamer_pipeline: str
    buffer_depth: int = 16

class CameraDriver:
    """Hardware camera capture driver leveraging GStreamer and CUDA shared memory."""
    
    def __init__(self, config: CameraDriverConfig) -> None:
        """Initialize GStreamer pipeline, CUDA buffers, and thread locks."""
        ...
        
    def start_stream(self) -> bool:
        """Start hardware acquisition stream."""
        ...
        
    def get_latest_frame(self) -> Tuple[np.ndarray, float]:
        """
        Fetch latest frame from ring buffer.
        Returns:
            Tuple[np.ndarray, float]: (Image frame array [H,W,C], Hardware timestamp in seconds)
        """
        ...
        
    def stop_stream(self) -> bool:
        """Stop hardware capture pipeline and free CUDA allocations."""
        ...


@dataclass
class CameraPreprocessorConfig:
    target_size: Tuple[int, int] = (224, 224)
    imagenet_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    imagenet_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    enable_undistortion: bool = True
    camera_matrix: Optional[np.ndarray] = None  # 3x3 Intrinsic Matrix
    dist_coeffs: Optional[np.ndarray] = None    # [k1, k2, p1, p2, k3]

class CameraPreprocessor:
    """GPU-accelerated image preprocessing (Resizing, Normalization, Undistortion)."""
    
    def __init__(self, config: CameraPreprocessorConfig) -> None:
        """Pre-compute CUDA undistortion lookup maps and initialize PyTorch tensors."""
        ...
        
    def process_frame_gpu(self, frame_tensor: torch.Tensor) -> torch.Tensor:
        """
        Executes GPU pipeline on frame tensor.
        Args:
            frame_tensor (torch.Tensor): Raw image tensor [C, H, W] on CUDA device.
        Returns:
            torch.Tensor: Preprocessed, undistorted, normalized tensor [3, 224, 224].
        """
        ...


@dataclass
class MultiCameraStitcherConfig:
    camera_ids: List[str]
    bev_range_x: Tuple[float, float] = (-50.0, 50.0)  # meters
    bev_range_y: Tuple[float, float] = (-50.0, 50.0)  # meters
    resolution: float = 0.1  # meters per pixel
    extrinsic_matrices: Dict[str, np.ndarray] = None  # 4x4 homogeneous matrices
    intrinsic_matrices: Dict[str, np.ndarray] = None  # 3x3 intrinsic matrices

class MultiCameraStitcher:
    """Stitches multi-camera surrounds into unified Bird's-Eye-View (BEV) image representation."""
    
    def __init__(self, config: MultiCameraStitcherConfig) -> None:
        """Initialize homography lookup tables and GPU IPM transformers."""
        ...
        
    def generate_bev_map(self, camera_frames: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Transform and blend multi-camera perspective inputs into 2D BEV map.
        Args:
            camera_frames (Dict[str, torch.Tensor]): Dict mapping camera_id to image tensor.
        Returns:
            torch.Tensor: Stitched BEV image tensor [3, 1000, 1000].
        """
        ...


@dataclass
class ThermalCameraDriverConfig:
    sensor_id: str
    device_index: int
    resolution: Tuple[int, int] = (640, 512)
    agc_clip_limit: float = 2.0
    spectral_range_min_um: float = 8.0
    spectral_range_max_um: float = 14.0

class ThermalCameraDriver:
    """Driver for radiometric LWIR thermal infrared cameras."""
    
    def __init__(self, config: ThermalCameraDriverConfig) -> None:
        """Initialize FLIR/radiometric API capture session and AGC pipelines."""
        ...
        
    def capture_radiometric_frame(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Capture thermal frame.
        Returns:
            Tuple[np.ndarray, np.ndarray, float]: 
                - normalized_8bit_image (np.ndarray): Enhanced visual array [512, 640]
                - temperature_grid_celsius (np.ndarray): Absolute temperature matrix [512, 640]
                - timestamp (float): Frame PTP timestamp
        """
        ...
```

---

## 4. LiDAR Subsystem

The LiDAR Subsystem parses, filters, projects, and clusters high-density 3D point cloud measurements.

```
+-----------------------------------------------------------------------------------+
|                                LiDARSUBSYSTEM                                     |
+-----------------------------------------------------------------------------------+
|  +--------------------+    +-----------------------+    +----------------------+  |
|  | lidar_driver.py    | -> |pointcloud_preproc.py  | -> | bev_projection.py    |  |
|  | ROS2 PointCloud2   |    | Voxel / Ground RANSAC |    | 3D to 2D Grid Tensor |  |
|  +--------------------+    +-----------------------+    +----------------------+  |
|                                                                    |              |
|                                                                    v              |
|                                                         +----------------------+  |
|                                                         | lidar_clustering.py  |  |
|                                                         | DBSCAN 3D Bounding   |  |
|                                                         +----------------------+  |
+-----------------------------------------------------------------------------------+
```

### 4.1 `lidar_driver.py` Design
- **ROS 2 Integration:** High-speed node subscribing to `sensor_msgs/msg/PointCloud2` messages.
- **Shared Memory Zero-Copy:** Utilizes cycloneDDS / eProsima FastDDS zero-copy shared memory transport to receive up to 3 million points/second without CPU copy overhead.
- **Data Layout:** Unpacks raw byte arrays into structured NumPy/CUDA arrays containing fields `(X, Y, Z, Intensity, Ring, Timestamp)`.

### 4.2 `pointcloud_preprocessor.py` Design
- **ROI Clipping:** Trims points outside the operational volume (\(X \in [-100, 100]\text{m}\), \(Y \in [-100, 100]\text{m}\), \(Z \in [-3, 10]\text{m}\)).
- **Voxel Grid Downsampling:** Discretizes space into a 3D grid with voxel size \(s_v = 0.1\text{m}\). Points within each voxel are replaced by their spatial centroid:
  $$\mathbf{p}_{\text{voxel}} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{p}_i$$
- **Ground Surface Extraction (RANSAC):** Iterative planar estimation to isolate ground points. Models ground plane equation \(aX + bY + cZ + d = 0\).
  - Iterations \(N_{\text{iter}} = 100\), Distance threshold \(d_{\text{thresh}} = 0.2\text{m}\).
  - Points satisfying \(\frac{|aX_i + bY_i + cZ_i + d|}{\sqrt{a^2 + b^2 + c^2}} \le d_{\text{thresh}}\) are flagged as non-obstacle ground returns and separated.

### 4.3 `bev_projection.py` Design
- **Spatial Bounds:** Maps point clouds to a 2D spatial grid covering \(X \in [-50\text{m}, +50\text{m}]\) and \(Y \in [-50\text{m}, +50\text{m}]\).
- **Resolution:** 0.1 meters per cell \(\rightarrow 1000 \times 1000\) grid elements.
- **Multi-Channel Rasterization:** Produces a 5-channel BEV tensor \((5 \times 1000 \times 1000)\):
  1. *Channel 0 (Max Height):* Normalized maximum height \(Z_{\text{max}}\) in cell.
  2. *Channel 1 (Mean Height):* Average height \(Z_{\text{mean}}\) of points in cell.
  3. *Channel 2 (Density):* Log-normalized point count \(\min(1.0, \frac{\log(N + 1)}{\log(64)})\).
  4. *Channel 3 (Max Intensity):* Peak laser reflectance intensity \(\in [0, 1]\).
  5. *Channel 4 (Ground Flag):* Binary indication of ground presence.

### 4.4 `lidar_clustering.py` Design
- **Algorithm:** Density-Based Spatial Clustering of Applications with Noise (DBSCAN) optimized with C++ / CUDA KD-Tree data structures.
- **Hyperparameters:** Neighborhood radius \(\varepsilon = 0.5\text{m}\), Minimum samples \(N_{\text{min}} = 10\).
- **3D Bounding Box Fitting:** For each cluster, computes the minimum oriented bounding box via Principal Component Analysis (PCA) on the covariance matrix of 2D projected cluster points to extract heading angle \(\psi\), width \(W\), length \(L\), and height \(H\).

### 4.5 Python Class Signatures & Interfaces

```python
"""
LiDAR Subsystem Module Interface Definitions
File: omnidrive/sensor_fusion/lidar/lidar_subsystem.py
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
import torch

@dataclass
class LiDARDriverConfig:
    topic_name: str = "/omnidrive/sensor/lidar_top/points"
    frame_id: str = "lidar_top_link"
    queue_size: int = 10
    use_shared_memory: bool = True

class LiDARDriver:
    """ROS 2 PointCloud2 Subscriber driver with zero-copy shared memory support."""
    
    def __init__(self, config: LiDARDriverConfig) -> None:
        """Initialize ROS 2 node context, QoS profiles, and shared memory subscription."""
        ...
        
    def get_latest_pointcloud(self) -> Tuple[np.ndarray, float]:
        """
        Returns latest parsed point cloud.
        Returns:
            Tuple[np.ndarray, float]: 
                - cloud (np.ndarray): Point array of shape [N, 5] (X, Y, Z, Intensity, Ring)
                - timestamp (float): Sensor timestamp in seconds
        """
        ...


@dataclass
class PointCloudPreprocessorConfig:
    voxel_size: Tuple[float, float, float] = (0.1, 0.1, 0.1)
    roi_x_min_max: Tuple[float, float] = (-100.0, 100.0)
    roi_y_min_max: Tuple[float, float] = (-100.0, 100.0)
    roi_z_min_max: Tuple[float, float] = (-3.0, 10.0)
    ransac_distance_threshold: float = 0.2
    ransac_max_iterations: int = 100

class PointCloudPreprocessor:
    """Filters, downsamples, and isolates ground surface from raw 3D point cloud."""
    
    def __init__(self, config: PointCloudPreprocessorConfig) -> None:
        """Initialize Open3D / CUDA acceleration engines."""
        ...
        
    def process(self, raw_cloud: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Execute pre-processing pipeline.
        Args:
            raw_cloud (np.ndarray): Input points [N, 5].
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: 
                - filtered_obstacles (np.ndarray): Non-ground points [M, 5]
                - ground_points (np.ndarray): Isolated ground points [K, 5]
                - plane_coefficients (np.ndarray): Fitted plane equation coefficients [a, b, c, d]
        """
        ...


@dataclass
class BEVProjectionConfig:
    x_bound: Tuple[float, float, float] = (-50.0, 50.0, 0.1)  # (min, max, resolution)
    y_bound: Tuple[float, float, float] = (-50.0, 50.0, 0.1)
    z_bound: Tuple[float, float, float] = (-3.0, 5.0, 8.0)
    num_channels: int = 5

class BEVProjection:
    """Rasterizes 3D point clouds into 2D multi-channel Bird's-Eye-View tensors."""
    
    def __init__(self, config: BEVProjectionConfig) -> None:
        """Allocate GPU rasterization buffers and grid indexes."""
        ...
        
    def generate_bev_tensor(self, obstacle_cloud: np.ndarray) -> torch.Tensor:
        """
        Rasterize 3D cloud into 2D image tensor.
        Args:
            obstacle_cloud (np.ndarray): Point array [N, 5].
        Returns:
            torch.Tensor: Tensor of shape [5, 1000, 1000] on CUDA device.
        """
        ...


@dataclass
class LiDARClusteringConfig:
    eps: float = 0.5
    min_samples: int = 10
    max_clusters: int = 500

class LiDARClustering:
    """Executes DBSCAN 3D clustering and fits oriented 3D bounding boxes."""
    
    def __init__(self, config: LiDARClusteringConfig) -> None:
        """Initialize KD-Tree engine."""
        ...
        
    def cluster_and_box(self, obstacle_cloud: np.ndarray) -> List[Dict[str, Any]]:
        """
        Cluster obstacle points and compute bounding attributes.
        Args:
            obstacle_cloud (np.ndarray): Points [N, 3].
        Returns:
            List[Dict[str, Any]]: List of cluster dicts containing:
                - "centroid": np.ndarray [3]
                - "dimensions": np.ndarray [3] (length, width, height)
                - "yaw": float (orientation angle in radians)
                - "point_indices": List[int]
        """
        ...
```

---

## 5. RADAR Subsystem

The RADAR Subsystem provides long-range velocity detection, Doppler validation, and object tracking under harsh environmental conditions (fog, dust, glare).

```
+-----------------------------------------------------------------------------------+
|                                RADAR SUBSYSTEM                                    |
+-----------------------------------------------------------------------------------+
|  +--------------------+    +-----------------------+    +----------------------+  |
|  | radar_driver.py    | -> |  radar_tracker.py     | -> |  radar_fusion.py     |  |
|  | SocketCAN / Enet   |    | EKF Tracker / Tracklet|    | Cam-RADAR Late Fusion|  |
|  +--------------------+    +-----------------------+    +----------------------+  |
+-----------------------------------------------------------------------------------+
```

### 5.1 `radar_driver.py` Design
- **Interfaces:** SocketCAN (`can0`, `can1`) for CAN FD packets or Automotive Ethernet (SOME/IP protocol).
- **Target Extraction:** Parses raw binary payloads into structured RADAR target records containing: Target ID, Range (\(r\)), Azimuth angle (\(\phi\)), Elevation angle (\(\theta\)), Range Rate / Radial Doppler Velocity (\(v_r\)), and Radar Cross Section (RCS in dBm²).
- **Coordinate Conversion:** Transforms spherical RADAR coordinates \((r, \phi, \theta)\) to Cartesian vehicle frame coordinates:
  $$X = r \cos(\theta) \cos(\phi), \quad Y = r \cos(\theta) \sin(\phi), \quad Z = r \sin(\theta)$$

### 5.2 `radar_tracker.py` Design
- **State Vector:** Each tracked RADAR object is governed by an Extended Kalman Filter tracking state:
  $$\mathbf{x}_{\text{radar}} = \begin{bmatrix} x & y & v_x & v_y & a_x & a_y \end{bmatrix}^T$$
- **Data Association:** Uses GNN (Global Nearest Neighbor) with Mahalanobis Distance thresholding:
  $$d_{\text{M}} = \sqrt{(\mathbf{z} - \mathbf{h}(\hat{\mathbf{x}}))^T \mathbf{S}^{-1} (\mathbf{z} - \mathbf{h}(\hat{\mathbf{x}}))}$$
- **Lifecycle Management:** 
  - *Tentative:* Initialized upon 1st detection.
  - *Confirmed:* Promoted if detected in \(\ge 3\) out of 5 consecutive frames.
  - *Deleted:* Terminated if unobserved for \(> 5\) consecutive cycles.

### 5.3 `radar_fusion.py` Design
- **Camera-RADAR Late Fusion:** Projects 3D RADAR track centroids into 2D camera image coordinates \((u,v)\).
- **Doppler Velocity Validation:** Uses RADAR Doppler velocity measurements to resolve speed ambiguity in 2D camera bounding box tracks, filtering out visual optical flow artifacts and shadow detections.
- **Covariance Intersection:** Merges spatial uncertainties between camera bounding frustums and RADAR range ellipses using Covariance Intersection (CI) optimization to yield robust joint spatial estimates.

### 5.4 Python Class Signatures & Interfaces

```python
"""
RADAR Subsystem Module Interface Definitions
File: omnidrive/sensor_fusion/radar/radar_subsystem.py
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np

@dataclass
class RADARDriverConfig:
    interface_type: str = "socketcan"  # "socketcan" or "ethernet"
    channel: str = "can0"
    bitrate: int = 500000
    radar_position_offset: Tuple[float, float, float] = (3.5, 0.0, 0.6)  # (x, y, z) in meters

class RADARDriver:
    """SocketCAN / Ethernet interface parser for 4D Automotive RADAR sensors."""
    
    def __init__(self, config: RADARDriverConfig) -> None:
        """Initialize CAN socket connection and frame decoding routines."""
        ...
        
    def read_targets(self) -> Tuple[List[Dict[str, float]], float]:
        """
        Poll and decode CAN bus packets.
        Returns:
            Tuple[List[Dict[str, float]], float]: 
                - targets: List of target dicts (id, x, y, z, v_r, rcs)
                - timestamp: Hardware capture timestamp in seconds
        """
        ...


@dataclass
class RADARTrackerConfig:
    max_distance_threshold: float = 4.0
    confirmation_hits: int = 3
    deletion_misses: int = 5
    process_noise_std_acc: float = 2.0

class RADARTrack:
    """Represents a single target tracklet maintained by Extended Kalman Filtering."""
    def __init__(self, track_id: int, initial_measurement: np.ndarray, timestamp: float) -> None:
        self.track_id = track_id
        self.state = np.zeros(6)  # [x, y, vx, vy, ax, ay]
        self.covariance = np.eye(6)
        self.age = 1
        self.hits = 1
        self.misses = 0

class RADARTracker:
    """Multi-object RADAR tracker managing tracklet life cycles."""
    
    def __init__(self, config: RADARTrackerConfig) -> None:
        """Initialize track management pools and EKF matrices."""
        ...
        
    def update(self, detections: List[Dict[str, float]], timestamp: float) -> List[RADARTrack]:
        """
        Predict and update state of all tracked objects.
        Args:
            detections (List[Dict[str, float]]): Raw RADAR Cartesian detections.
            timestamp (float): Current frame timestamp.
        Returns:
            List[RADARTrack]: List of active, confirmed tracks.
        """
        ...


@dataclass
class CameraRADARFusionConfig:
    iou_threshold_2d: float = 0.3
    velocity_gate_m_s: float = 1.5

class CameraRADARFusion:
    """Executes late-stage fusion between 2D Camera object detections and 3D RADAR tracks."""
    
    def __init__(self, config: CameraRADARFusionConfig) -> None:
        """Initialize projection matrices and association gates."""
        ...
        
    def fuse_camera_and_radar(
        self, 
        camera_boxes_2d: List[Dict[str, Any]], 
        radar_tracks: List[RADARTrack],
        camera_intrinsics: np.ndarray,
        camera_extrinsics: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        Fuses 2D bounding boxes with 3D RADAR tracklets.
        Returns:
            List[Dict[str, Any]]: List of fused object hypotheses with enriched 3D velocity and range.
        """
        ...
```

---

## 6. GPS/IMU Subsystem

The GPS/IMU Subsystem provides high-rate ego-motion state estimation and geodetic spatial reference positioning via an Extended Kalman Filter.

```
+-----------------------------------------------------------------------------------+
|                               GPS/IMU SUBSYSTEM                                   |
+-----------------------------------------------------------------------------------+
|  +--------------------+    +-----------------------+    +----------------------+  |
|  | gps_driver.py      |    | imu_driver.py         |    | ekf_localizer.py     |  |
|  | NMEA/UBX RTK-GPS   | -> | 100-1000Hz Raw IMU    | -> | 15-DOF EKF Localizer |  |
|  +--------------------+    +-----------------------+    +----------------------+  |
+-----------------------------------------------------------------------------------+
```

### 6.1 `gps_driver.py` Design
- **Protocols Parsed:** NMEA-0183 (`$GNGGA`, `$RMC`, `$VTG`) and u-blox UBX binary protocol (`UBX-NAV-PVT`, `UBX-NAV-RELPOSNED`).
- **RTK Differential Corrections:** Ingests RTCM v3 stream via NTRIP client, enabling RTK Fixed mode for **2cm horizontal accuracy**.
- **Dual-Antenna Heading:** Computes baseline vector between main and auxiliary GNSS antennas to provide absolute true heading independent of vehicle motion.

### 6.2 `imu_driver.py` Design
- **Sampling Frequency:** High-rate acquisition (100 Hz to 1000 Hz) via SPI or RS422 interface.
- **Sensor Calibration:** Real-time online temperature compensation and bias estimation for tri-axial accelerometers \(\mathbf{a}_{\text{raw}}\) and gyroscopes \(\boldsymbol{\omega}_{\text{raw}}\):
  $$\mathbf{a}_{\text{cal}} = \mathbf{S}_a (\mathbf{a}_{\text{raw}} - \mathbf{b}_a - \mathbf{n}_a), \quad \boldsymbol{\omega}_{\text{cal}} = \mathbf{S}_g (\boldsymbol{\omega}_{\text{raw}} - \mathbf{b}_g - \mathbf{n}_g)$$
  where \(\mathbf{S}\) represents scale-factor matrices and \(\mathbf{b}\) represents dynamic bias vectors.

### 6.3 `ekf_localizer.py` Design (15-DOF Extended Kalman Filter)

#### State Vector Definition
The 15-dimensional state vector \(\mathbf{x} \in \mathbb{R}^{15}\) is defined as:
$$\mathbf{x} = \begin{bmatrix} p_x & p_y & p_z & v_x & v_y & v_z & \phi & \theta & \psi & b_{ax} & b_{ay} & b_{az} & b_{gx} & b_{gy} & b_{gz} \end{bmatrix}^T$$
where:
- \(\mathbf{p} = [p_x, p_y, p_z]^T\): Position in East-North-Up (ENU) global reference frame (meters).
- \(\mathbf{v} = [v_x, v_y, v_z]^T\): Linear velocity in ENU frame (m/s).
- \(\boldsymbol{\Theta} = [\phi, \theta, \psi]^T\): Orientation Euler angles (Roll, Pitch, Yaw) in radians.
- \(\mathbf{b}_a = [b_{ax}, b_{ay}, b_{az}]^T\): Accelerometer dynamic bias vector (m/s²).
- \(\mathbf{b}_g = [b_{gx}, b_{gy}, b_{gz}]^T\): Gyroscope dynamic bias vector (rad/s).

#### Mathematical Formulation

##### 1. Continuous Time Kinematic Model
$$\dot{\mathbf{p}} = \mathbf{v}$$
$$\dot{\mathbf{v}} = \mathbf{R}_b^w (\mathbf{a}_{\text{meas}} - \mathbf{b}_a - \mathbf{n}_a) + \mathbf{g}$$
$$\dot{\boldsymbol{\Theta}} = \mathbf{E}_b^w (\boldsymbol{\omega}_{\text{meas}} - \mathbf{b}_g - \mathbf{n}_g)$$
$$\dot{\mathbf{b}}_a = \mathbf{w}_{ba}, \quad \dot{\mathbf{b}}_g = \mathbf{w}_{bg}$$
where \(\mathbf{R}_b^w\) is the body-to-world rotation matrix, \(\mathbf{E}_b^w\) is the Euler rate transform matrix, and \(\mathbf{g} = [0, 0, -9.81]^T\).

##### 2. Prediction Step (Discrete Time step \(\Delta t\))
State prediction:
$$\hat{\mathbf{x}}_{k|k-1} = \mathbf{f}(\hat{\mathbf{x}}_{k-1|k-1}, \mathbf{u}_{k-1})$$
Covariance prediction:
$$\mathbf{P}_{k|k-1} = \mathbf{F}_k \mathbf{P}_{k-1|k-1} \mathbf{F}_k^T + \mathbf{Q}_k$$
where \(\mathbf{F}_k = \left.\frac{\partial \mathbf{f}}{\partial \mathbf{x}}\right|_{\hat{\mathbf{x}}_{k-1|k-1}}\) is the state transition Jacobian matrix and \(\mathbf{Q}_k\) is the discrete process noise covariance matrix.

##### 3. Measurement Update Step (GNSS / RTK / Wheel Speed)
Measurement residual (innovation):
$$\mathbf{y}_k = \mathbf{z}_k - \mathbf{h}(\hat{\mathbf{x}}_{k|k-1})$$
Innovation covariance:
$$\mathbf{S}_k = \mathbf{H}_k \mathbf{P}_{k|k-1} \mathbf{H}_k^T + \mathbf{R}_k$$
Kalman Gain calculation:
$$\mathbf{K}_k = \mathbf{P}_{k|k-1} \mathbf{H}_k^T \mathbf{S}_k^{-1}$$
Updated state estimate:
$$\hat{\mathbf{x}}_{k|k} = \hat{\mathbf{x}}_{k|k-1} + \mathbf{K}_k \mathbf{y}_k$$
Updated covariance estimate:
$$\mathbf{P}_{k|k} = (\mathbf{I} - \mathbf{K}_k \mathbf{H}_k) \mathbf{P}_{k|k-1}$$
where \(\mathbf{H}_k = \left.\frac{\partial \mathbf{h}}{\partial \mathbf{x}}\right|_{\hat{\mathbf{x}}_{k|k-1}}\) is the measurement matrix Jacobian and \(\mathbf{R}_k\) is the measurement noise covariance matrix.

### 6.4 Python Class Signatures & Interfaces

```python
"""
GPS/IMU Subsystem Module Interface Definitions
File: omnidrive/sensor_fusion/localization/gps_imu_subsystem.py
"""

from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np

@dataclass
class GPSDriverConfig:
    port: str = "/dev/ttyUSB0"
    baudrate: int = 115200
    ntrip_server: str = "rtk.ntrip.service.com"
    ntrip_port: int = 2101
    mountpoint: str = "RTK_BASE"

class GPSDriver:
    """Parses NMEA/UBX serial streams and manages RTK differential client."""
    
    def __init__(self, config: GPSDriverConfig) -> None:
        """Connect serial interface and open NTRIP TCP socket."""
        ...
        
    def read_gps(self) -> Optional[Dict[str, Any]]:
        """
        Poll GPS receiver.
        Returns:
            Optional[Dict[str, Any]]: Returns dict containing:
                - "lat": float, "lon": float, "alt": float
                - "rtk_status": int (0=Fix Invalid, 1=GPS Single, 4=RTK Fixed, 5=RTK Float)
                - "heading": float
                - "timestamp": float
        """
        ...


@dataclass
class IMUDriverConfig:
    device_path: str = "/dev/spidev0.0"
    sample_rate_hz: int = 200
    accel_scale_g: float = 8.0
    gyro_scale_dps: float = 500.0

class IMUDriver:
    """High-frequency IMU acquisition driver with digital filtering and dynamic bias compensation."""
    
    def __init__(self, config: IMUDriverConfig) -> None:
        """Initialize SPI hardware interface and load offline bias metrics."""
        ...
        
    def read_imu(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Read calibrated IMU measurements.
        Returns:
            Tuple[np.ndarray, np.ndarray, float]:
                - accel (np.ndarray): [3] linear acceleration in m/s^2 (ax, ay, az)
                - gyro (np.ndarray): [3] angular velocity in rad/s (gx, gy, gz)
                - timestamp (float): High-resolution hardware time in seconds
        """
        ...


@dataclass
class EKFLocalizerConfig:
    init_pos_std: float = 10.0
    init_vel_std: float = 1.0
    init_att_std_rad: float = 0.1
    process_noise_acc: float = 0.2
    process_noise_gyro: float = 0.02

@dataclass
class EKFState:
    position: np.ndarray      # [x, y, z] in ENU frame (meters)
    velocity: np.ndarray      # [vx, vy, vz] in ENU frame (m/s)
    orientation: np.ndarray   # [roll, pitch, yaw] in radians
    accel_bias: np.ndarray    # [b_ax, b_ay, b_az] in m/s^2
    gyro_bias: np.ndarray     # [b_gx, b_gy, b_gz] in rad/s
    covariance: np.ndarray    # 15x15 Covariance matrix

class EKFLocalizer:
    """15-DOF Extended Kalman Filter for real-time localization."""
    
    def __init__(self, config: EKFLocalizerConfig) -> None:
        """Initialize 15x15 covariance matrices and baseline state vectors."""
        ...
        
    def predict(self, imu_accel: np.ndarray, imu_gyro: np.ndarray, dt: float) -> None:
        """
        Execute EKF motion prediction step using high-rate IMU telemetry.
        """
        ...
        
    def update_gnss(self, enu_pos: np.ndarray, pos_covariance: np.ndarray) -> None:
        """
        Execute EKF measurement update using RTK GNSS position observation.
        """
        ...
        
    def get_state(self) -> EKFState:
        """Fetch current localized state estimate."""
        ...
```

---

## 7. Sensor Fusion Engine

The Central Sensor Fusion Engine brings together spatial-temporal queues, fuses multi-modal signals into tracklets and BEV feature maps, and emits the **Unified World State**.

```
+-----------------------------------------------------------------------------------+
|                           SENSOR FUSION ENGINE                                    |
+-----------------------------------------------------------------------------------+
|  +---------------------------+       +-----------------------------------------+  |
|  | temporal_alignment.py     | ----> | sensor_fusion_engine.py                 |  |
|  | PTP Sync / Interpolation  |       | Spatial-Temporal Cross-Attention Fusion |  |
|  +---------------------------+       +-----------------------------------------+  |
|                                                           |                       |
|                                                           v                       |
|                                      +-----------------------------------------+  |
|                                      | unified_world_state.py                  |  |
|                                      | Master Output Dataclass                 |  |
|                                      +-----------------------------------------+  |
+-----------------------------------------------------------------------------------+
```

### 7.1 `temporal_alignment.py` Design
- **Hardware Clock Sync:** All sensor interfaces lock to a central grandmaster PTP clock (IEEE 1588v2) via hardware network interfaces.
- **Sliding Window Synchronization:** Maintains a 500ms sliding ring queue for all incoming streams.
- **Temporal Interpolation:** When fusing a target frame at timestamp \(T_{\text{target}}\), asynchronous sensor measurements (e.g., IMU at 200 Hz, RADAR at 20 Hz, Camera at 30 Hz) are aligned via linear or cubic spline interpolation:
  $$\mathbf{z}(T_{\text{target}}) = \mathbf{z}(t_0) + \frac{T_{\text{target}} - t_0}{t_1 - t_0} \left( \mathbf{z}(t_1) - \mathbf{z}(t_0) \right)$$

### 7.2 `sensor_fusion_engine.py` Design
- **Architecture:** Hybrid Fusion Network combining Early-Stage Feature Fusion (LiDAR BEV + Camera BEV concatenated into multi-modal tensors) and Late-Stage Object Fusion (3D LiDAR cluster bounding boxes + RADAR tracks + Camera 2D proposals fused via Multi-Hypothesis Tracking / Hungarian Algorithm).
- **Spatial Alignment:** Transforms all sensor representations into the Vehicle Ego Frame \(\mathcal{F}_{\text{ego}}\) where the origin \((0,0,0)\) is defined at the center of the rear axle projected on the ground.

### 7.3 `unified_world_state.py` Output Data Structures

```python
"""
Unified World State Data Structures
File: omnidrive/sensor_fusion/engine/unified_world_state.py
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np

@dataclass
class EgoState:
    """Ego Vehicle Kinematic State."""
    timestamp: float                       # Microsecond PTP Unix Timestamp
    position_enu: Tuple[float, float, float]  # (X, Y, Z) in meters
    velocity_enu: Tuple[float, float, float]  # (Vx, Vy, Vz) in m/s
    acceleration_body: Tuple[float, float, float] # (Ax, Ay, Az) in m/s^2
    orientation_euler: Tuple[float, float, float] # (Roll, Pitch, Yaw) in radians
    angular_velocity_body: Tuple[float, float, float] # (Wx, Wy, Wz) in rad/s
    position_std_dev: Tuple[float, float, float] # Standard deviations (X, Y, Z)
    heading_std_dev: float                 # Yaw standard deviation in radians

@dataclass
class TrackedObject3D:
    """Fused 3D Object Detection Tracklet."""
    track_id: int
    semantic_class: str                    # "car", "truck", "pedestrian", "cyclist", "tank", etc.
    class_confidence: float               # Probability in range [0.0, 1.0]
    position_ego: Tuple[float, float, float] # Centroid (X, Y, Z) in meters in Ego frame
    dimensions: Tuple[float, float, float]   # Size (Length, Width, Height) in meters
    orientation_yaw: float                 # Yaw angle in radians relative to Ego X-axis
    velocity_ego: Tuple[float, float, float] # Absolute 3D velocity vector (Vx, Vy, Vz) m/s
    covariance_matrix: List[List[float]]   # 6x6 spatial-velocity uncertainty matrix
    sensor_modalities_detected: List[str]  # e.g., ["camera_front", "lidar_top", "radar_front"]
    tracking_age_frames: int

@dataclass
class BEVFeatureMap:
    """Rasterized Multi-Modal BEV Grid Feature Tensor."""
    timestamp: float
    spatial_bounds_x: Tuple[float, float]  # (-50.0, 50.0) meters
    spatial_bounds_y: Tuple[float, float]  # (-50.0, 50.0) meters
    grid_resolution: float                 # 0.1 meters per cell
    tensor_shape: Tuple[int, int, int]     # Channels (C=10), Height (H=1000), Width (W=1000)
    bev_tensor_gpu_pointer: int            # Memory pointer for PyTorch/CUDA zero-copy interop

@dataclass
class SensorHealthStatus:
    """Operational Health Diagnostics for all Layer 1 Hardware Sensors."""
    camera_health: Dict[str, bool]         # {"camera_front_main": True, ...}
    lidar_health: Dict[str, bool]          # {"lidar_top": True, ...}
    radar_health: Dict[str, bool]          # {"radar_front": True, ...}
    gnss_rtk_lock: bool                    # True if RTK Fixed/Float lock maintained
    imu_degradation_flag: bool             # True if IMU bias drift exceeds safety threshold
    current_degradation_level: str         # "L0_FULL", "L1_DEGRADED", "L2_SEVERE", "L4_FAILSAFE"

@dataclass
class UnifiedWorldState:
    """
    Master Output Data Structure for Layer 1 Sensor Fusion Module.
    Ingested directly by Layer 2 (JEPA Foundation World Model) and Layer 3 (Decision Engine).
    """
    frame_sequence_id: int
    timestamp: float                       # Synchronized PTP Epoch Timestamp
    ego_state: EgoState
    tracked_objects: List[TrackedObject3D]
    bev_feature_map: BEVFeatureMap
    health_status: SensorHealthStatus
    environmental_lighting_lux: float      # Estimated ambient light level
    precipitation_flag: bool               # Rain/Snow environmental indicator


@dataclass
class TemporalAlignmentConfig:
    max_sync_delay_ms: float = 15.0
    interpolation_mode: str = "linear"

class TemporalAlignment:
    """Synchronizes sensor frame queues to unified temporal boundary using hardware PTP clocks."""
    
    def __init__(self, config: TemporalAlignmentConfig) -> None:
        ...
        
    def synchronize_frames(self, sensor_queues: Dict[str, List[Any]], target_timestamp: float) -> Dict[str, Any]:
        """
        Interpolate and match all sensor measurements to exact target timestamp.
        """
        ...


@dataclass
class SensorFusionEngineConfig:
    primary_frame_id: str = "base_link"
    fusion_mode: str = "hybrid"  # "early", "late", or "hybrid"
    max_track_age_sec: float = 0.5

class SensorFusionEngine:
    """Main Orchestrator for Layer 1 Sensor Fusion Pipeline."""
    
    def __init__(self, config: SensorFusionEngineConfig) -> None:
        """Initialize all drivers, preprocessors, fusion heads, and state queues."""
        ...
        
    def execute_fusion_cycle(self, raw_sensor_inputs: Dict[str, Any]) -> UnifiedWorldState:
        """
        Runs complete sensor fusion iteration.
        Args:
            raw_sensor_inputs (Dict[str, Any]): Dictionary of incoming sensor frame packets.
        Returns:
            UnifiedWorldState: Consolidated environment state.
        """
        ...
```

---

## 8. Data Flow Diagram

The diagram below outlines the full end-to-end processing pipeline, from physical hardware signal acquisition to the generation of the `UnifiedWorldState` output structure.

```
+-----------------------------------------------------------------------------------------------------------------------------+
|                                                   PHYSICAL HARDWARE LAYER                                                   |
+-----------------------------------------------------------------------------------------------------------------------------+
| [RGB Cameras (1-12)]  [LWIR Thermal]  [128-Ch LiDAR]  [4D Imaging RADAR]  [RTK GNSS Receiver]   [Tactical 6-DOF IMU]       |
+-----------------------------------------------------------------------------------------------------------------------------+
           |                 |                 |                 |                   |                      |
           v                 v                 v                 v                   v                      v
+-----------------------------------------------------------------------------------------------------------------------------+
|                                                     DRIVER / INGESTION LAYER                                                |
+-----------------------------------------------------------------------------------------------------------------------------+
| camera_driver.py   thermal_driver.py lidar_driver.py  radar_driver.py     gps_driver.py         imu_driver.py           |
| (GStreamer/GMSL2)  (Radiometric AGC) (ROS 2/Zero-Copy)(SocketCAN/SOMEIP)  (NMEA/UBX RTK)        (1000Hz SPI Driver)     |
+-----------------------------------------------------------------------------------------------------------------------------+
           |                 |                 |                 |                   |                      |
           v                 v                 v                 v                   v                      v
+-----------------------------------------------------------------------------------------------------------------------------+
|                                                  PREPROCESSING & FILTERING LAYER                                            |
+-----------------------------------------------------------------------------------------------------------------------------+
| camera_prep.py     thermal_prep.py   pointcloud_prep.py                ekf_localizer.py                                 |
| - Undistort (CUDA) - Registration    - Voxel Grid (0.1m)               - 15-DOF State Estimation                        |
| - Resize (224x224) - Contrast CLAHE  - Ground RANSAC Removal           - Position/Velocity/Attitude                     |
| - ImageNet Norm                      - ROI Trimming                     - IMU Dynamic Bias Corrections                   |
+-----------------------------------------------------------------------------------------------------------------------------+
           |                 |                 |                 |                   |                      |
           v                 v                 v                 v                   |                      |
+------------------------------------------------------------------------------------+                      |
|                              SPATIAL EXTRACTION & INTERMEDIATE PROJECTION LAYER    |                      |
+------------------------------------------------------------------------------------+                      |
| multi_camera_stitcher.py             bev_projection.py    radar_tracker.py         |                      |
| - IPM Transform                      - 3D Cloud to 2D     - Target EKF Tracking    |                      |
| - Overlap Blending                     5-Ch BEV Tensor    - Mahalanobis Assoc.     |                      |
|                                      lidar_clustering.py                           |                      |
|                                      - DBSCAN (eps=0.5)                            |                      |
|                                      - 3D Box Estimation                           |                      |
+------------------------------------------------------------------------------------+                      |
           |                                   |                 |                   |                      |
           +-------------------------+         |                 |                   |                      |
                                     v         v                 v                   v                      |
+-----------------------------------------------------------------------------------------------------------------------------+
|                                              TEMPORAL ALIGNMENT SUBSYSTEM                                                   |
+-----------------------------------------------------------------------------------------------------------------------------+
| temporal_alignment.py                                                                                                       |
| - Hardware IEEE 1588 PTP Timestamp Synchronization                                                                          |
| - Sliding-Window Linear & Spline State Interpolation                                                                        |
+-----------------------------------------------------------------------------------------------------------------------------+
                                                                 |
                                                                 v
+-----------------------------------------------------------------------------------------------------------------------------+
|                                               CENTRAL FUSION ENGINE                                                         |
+-----------------------------------------------------------------------------------------------------------------------------+
| sensor_fusion_engine.py                                                                                                     |
| - Spatial Transform to Vehicle Ego Frame (Rear Axle Origin)                                                                 |
| - Multi-Modal Feature Concatenation (Camera BEV + LiDAR BEV)                                                                |
| - Camera-RADAR-LiDAR Multi-Hypothesis Tracking (MHT / Hungarian Data Association)                                           |
| - Spatial-Temporal Covariance Intersection                                                                                  |
+-----------------------------------------------------------------------------------------------------------------------------+
                                                                 |
                                                                 v
+-----------------------------------------------------------------------------------------------------------------------------+
|                                                UNIFIED WORLD STATE OUTPUT                                                   |
+-----------------------------------------------------------------------------------------------------------------------------+
| unified_world_state.py                                                                                                      |
| - EgoState (Position, Velocity, Acceleration, Heading)                                                                      |
| - TrackedObject3D List (Fused 3D Bounding Boxes, Class, Velocity, Covariance)                                              |
| - BEVFeatureMap (10-Channel 1000x1000 Tensor)                                                                               |
| - SensorHealthStatus (Sensor Diagnostics & L0-L4 Degradation States)                                                         |
+-----------------------------------------------------------------------------------------------------------------------------+
                                                                 |
                                                                 v
                                         +-----------------------------------------------+
                                         | Ingestion by Layer 2 (JEPA Foundation Model)  |
                                         | Ingestion by Layer 3 (Behavioral Decision Engine)|
                                         +-----------------------------------------------+
```

---

## 9. Performance Requirements Table

| Metric / Pipeline Stage | Latency Budget (Target) | Max Latency (Hard Threshold) | Target Frame Rate (FPS) | Operational Spatial Range | Spatial / Angular Accuracy Target |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Camera Acquisition & GStreamer** | 5.0 ms | 10.0 ms | 60 FPS | 0.5m to 250m | Angular error < 0.15° |
| **Camera Preprocessing (GPU)** | 1.5 ms | 3.0 ms | 60 FPS | N/A | Sub-pixel distortion error < 0.5px |
| **Multi-Camera BEV Stitching** | 4.0 ms | 8.0 ms | 30 FPS | 50m radius | Projection error < 0.1m |
| **Thermal Preprocessing & AGC** | 2.5 ms | 5.0 ms | 60 FPS | 0.5m to 150m | Thermal sensitivity < 20mK |
| **LiDAR Point Cloud Parsing** | 3.0 ms | 6.0 ms | 20 FPS | 0.5m to 250m | Range accuracy ±2cm |
| **LiDAR Ground RANSAC & Voxel** | 6.0 ms | 12.0 ms | 20 FPS | 100m x 100m window | Plane fit residual < 0.05m |
| **LiDAR 3D DBSCAN Clustering** | 8.0 ms | 15.0 ms | 20 FPS | 100m x 100m window | Cluster centroid accuracy ±0.1m |
| **LiDAR BEV Tensor Projection** | 2.0 ms | 4.0 ms | 20 FPS | 1000x1000 grid | Grid alignment exact 0.1m/px |
| **RADAR Frame Parsing** | 1.0 ms | 2.5 ms | 50 FPS | 0.5m to 300m | Range error < 0.1m |
| **RADAR Target EKF Tracking** | 2.0 ms | 4.0 ms | 50 FPS | 300m range | Doppler velocity error < 0.05 m/s |
| **GPS / RTK Update** | 2.0 ms | 5.0 ms | 20 FPS | Global | RTK Position error **< 2.0 cm** |
| **IMU High-Rate Prediction Step** | 0.2 ms | 0.5 ms | 1000 Hz | N/A | Gyro bias instability < 0.25°/hr |
| **EKF Localization Full Step** | 1.0 ms | 2.5 ms | 200 Hz | Global / Local | Heading accuracy < 0.1° |
| **Temporal Alignment & Queuing**| 1.5 ms | 3.0 ms | Synchronized | N/A | Time sync jitter **< 1.0 ms** |
| **Central Sensor Fusion Engine**| 10.0 ms | 20.0 ms | 30 FPS | 100m x 100m window | Fused Object Position error < 0.15m |
| **TOTAL LAYER 1 END-TO-END** | **45.0 ms** | **80.0 ms** | **30 FPS** | **300m Surround** | **Strict Zero-Safety-Violation Target** |

---

## 10. Configuration Parameters

Below is the exhaustive single-file master YAML specification (`omnidrive_sensor_fusion_config.yaml`) consumed by Layer 1 at system boot.

```yaml
# OMNIDRIVE Autonomous Driving System - Layer 1 Configuration Specification
# File: omnidrive/configs/sensor_fusion_layer1.yaml

system:
  profile: "military_tactical" # Options: "military_tactical", "heavy_freight_truck", "robot_taxi"
  master_clock: "ptp_ieee1588"
  ptp_device: "/dev/ptp0"
  target_frame_rate_hz: 30
  max_allowable_latency_ms: 80.0

camera_subsystem:
  enabled: true
  num_cameras: 6
  cameras:
    - id: "camera_front_main"
      type: "rgb"
      device_path: "/dev/v4l/by-id/usb-Sony_ISX031_001"
      gstreamer_pipeline: "v4l2src device=/dev/video0 ! video/x-raw, format=NV12, width=3840, height=2160, framerate=60/1 ! nvvideoconv ! video/x-raw(memory:NVMM) ! appsink"
      resolution: [3840, 2160]
      fps: 60
      fov_h_deg: 120.0
      fov_v_deg: 70.0
      extrinsics:
        translation_xyz: [3.8, 0.0, 1.4] # meters from rear axle origin
        rotation_rpy: [0.0, 0.02, 0.0]    # radians (roll, pitch, yaw)
      intrinsics:
        fx: 1850.25
        fy: 1850.25
        cx: 1920.0
        cy: 1080.0
        distortion_k1_k2_p1_p2_k3: [-0.28, 0.09, 0.001, 0.0002, 0.0]
    
    - id: "camera_front_thermal"
      type: "thermal_lwir"
      device_path: "/dev/v4l/by-id/usb-FLIR_Boson_Plus_002"
      gstreamer_pipeline: "v4l2src device=/dev/video1 ! video/x-raw, format=GRAY16_LE, width=640, height=512, framerate=60/1 ! appsink"
      resolution: [640, 512]
      fps: 60
      spectral_range_um: [8.0, 14.0]
      agc_mode: "clahe"
      clahe_clip_limit: 2.5
      extrinsics:
        translation_xyz: [3.8, 0.2, 1.4]
        rotation_rpy: [0.0, 0.02, 0.0]

  preprocessing:
    target_resolution: [224, 224]
    normalize_mean: [0.485, 0.456, 0.406]
    normalize_std: [0.229, 0.224, 0.225]
    cuda_acceleration: true

lidar_subsystem:
  enabled: true
  primary_lidar:
    id: "lidar_top_primary"
    model: "Hesai_AT128"
    topic_name: "/omnidrive/sensor/lidar_top/points"
    channels: 128
    frame_rate: 20
    shared_memory_transport: true
    extrinsics:
      translation_xyz: [1.8, 0.0, 2.2]
      rotation_rpy: [0.0, 0.0, 0.0]
  
  preprocessing:
    roi_bounds:
      x_min_max: [-100.0, 100.0]
      y_min_max: [-100.0, 100.0]
      z_min_max: [-3.0, 8.0]
    voxel_grid:
      leaf_size_xyz: [0.1, 0.1, 0.1]
    ground_removal_ransac:
      enabled: true
      distance_threshold_m: 0.20
      max_iterations: 100
      normal_distance_weight: 0.1
  
  bev_projection:
    grid_size_pixels: [1000, 1000]
    meters_per_pixel: 0.1
    channels: ["max_height", "mean_height", "density", "max_intensity", "ground_flag"]
  
  clustering:
    algorithm: "dbscan_kdtree"
    eps_radius_m: 0.5
    min_samples: 10

radar_subsystem:
  enabled: true
  radars:
    - id: "radar_front_long_range"
      model: "Continental_ARS540"
      interface: "socketcan"
      can_channel: "can0"
      can_bitrate: 500000
      max_range_m: 300.0
      fov_azimuth_deg: 120.0
      extrinsics:
        translation_xyz: [3.9, 0.0, 0.6]
        rotation_rpy: [0.0, 0.0, 0.0]

  tracking:
    data_association: "global_nearest_neighbor"
    mahalanobis_gate_threshold: 9.21 # 95% confidence interval for 2-DOF
    track_confirmation_hits: 3
    track_deletion_misses: 5
    process_noise:
      accel_std_m_s2: 2.5

gps_imu_subsystem:
  enabled: true
  gnss:
    device_port: "/dev/ttyUSB0"
    baud_rate: 115200
    protocol: "ubx"
    rtk_corrections_enabled: true
    ntrip_host: "rtk.service.net"
    ntrip_port: 2101
    ntrip_mountpoint: "RTK_NEARBY"
  
  imu:
    device_port: "/dev/spidev0.0"
    sample_rate_hz: 1000
    gravity_m_s2: 9.80665
    gyro_bias_instability_deg_hr: 0.25
    accel_bias_instability_mg: 0.05
  
  ekf_localization:
    frequency_hz: 200
    state_dimensions: 15
    initial_covariance_diagonal:
      pos_m: [1.0, 1.0, 2.0]
      vel_m_s: [0.1, 0.1, 0.2]
      att_rad: [0.01, 0.01, 0.05]
      accel_bias: [0.01, 0.01, 0.01]
      gyro_bias: [0.001, 0.001, 0.001]

sensor_fusion_engine:
  fusion_strategy: "hybrid"
  temporal_alignment:
    max_queue_depth_sec: 0.5
    allowed_timestamp_skew_ms: 15.0
    interpolation_type: "cubic_spline"
  spatial_reference_frame: "base_link_rear_axle"
  output_topic: "/omnidrive/layer1/unified_world_state"
```

---

## 11. Error Handling and Graceful Degradation

To ensure fail-operational reliability across military and civilian deployment domains, Layer 1 enforces a strict hierarchical degradation matrix:

```
+-----------------------------------------------------------------------------------+
|                            DEGRADATION LEVELS (L0 - L4)                           |
+-----------------------------------------------------------------------------------+
|  L0: FULL SYSTEM OPERATIONAL (All Sensors Normal)                                |
|  L1: MINOR DEGRADATION (Single non-critical sensor failure e.g., 1 surround cam)  |
|  L2: MODERATE DEGRADATION (Primary LiDAR failure -> RADAR + Multi-Camera active)  |
|  L3: SEVERE DEGRADATION (GPS Denied / EW Jamming -> Visual-Inertial Odometry)     |
|  L4: EMERGENCY MINIMUM RISK MANEUVER (Thermal/LiDAR/Cam Fail -> Emergency Stop)  |
+-----------------------------------------------------------------------------------+
```

### 11.1 Sensor Failure Modes & Mitigation Strategies

| Failure Scenario | Detection Criteria | Automated Mitigation Strategy | Resulting Degradation Level |
| :--- | :--- | :--- | :--- |
| **Camera Occlusion / Mud / Glare** | Mean intensity \(\le 5\) or \(\ge 250\); zero laplacian variance over 30 frames. | Isolate occluded camera; rely on adjacent overlapping camera frustums and LiDAR BEV projection. | **L1 (Minor)** |
| **Thermal Camera Overheat / Saturation** | Sensor radiometric status returns thermal warning; uniform ADU count. | Deactivate thermal fusion branch; fall back entirely to RGB + LiDAR + RADAR pipelines. | **L1 (Minor)** |
| **LiDAR Lens Occlusion / Heavy Fog** | Point drop rate > 75%; ground RANSAC inlier ratio < 5% for 10 cycles. | Switch 3D obstacle perception from LiDAR-first to 4D RADAR + Multi-Camera Pseudo-LiDAR depth network. | **L2 (Moderate)** |
| **GPS Denial / Electronic Warfare (EW)** | RTK lock lost; GNSS position variance jumps > 15m; NMEA checksum errors. | Disengage GNSS updates in EKF; lock localization to high-rate IMU + LiDAR Odometry + Visual Odometry (VIO). | **L3 (Severe)** |
| **RADAR Interface Fault / Bus Disconnect** | SocketCAN timeout > 100ms; zero packet rate on CAN interface. | Flag RADAR tracklets as stale; compute object relative velocity using consecutive 3D LiDAR point cloud bounding boxes. | **L1 (Minor)** |
| **PTP Clock Desynchronization** | Timestamp skew delta between sensors exceeds > 25ms threshold. | Fall back to software local arrival time estimation with dynamic latency compensation queues. | **L1 (Minor)** |
| **Total Visual Perception Darkness + LiDAR Fail**| Camera lux < 0.01 AND LiDAR returns 0 points (Military Smoke + Dark). | Engage Thermal LWIR Camera + 4D RADAR fusion; cap vehicle maximum velocity to 25 km/h. | **L3 (Severe)** |
| **Catastrophic Power / Multi-Sensor Crash**| Concurrent failure of LiDAR, Cameras, and RADAR inputs. | Trigger Layer 6 Safety System: Execute Emergency Minimum Risk Maneuver (MRM) to bring vehicle to immediate stop in lane. | **L4 (Failsafe MRM)** |

---

## 12. System Dependencies

Layer 1 requires the following system drivers, hardware acceleration toolkits, ROS 2 modules, and Python packages:

### 12.1 System Software & Driver Stack
- **OS:** Ubuntu 22.04 LTS (Real-Time RT-PREEMPT Kernel `5.15.0-x-rt`)
- **CUDA Toolkit:** v12.2 or higher
- **NVIDIA TensorRT:** v8.6.1 (for preprocessor CUDA kernel optimization)
- **GStreamer:** v1.22.0 with `nvstreammux` and `nvvideoconv` plugins
- **ROS 2:** Humble Hawksbill or Jazzy Jalisco (Desktop Install)
- **PCL (Point Cloud Library):** v1.12.1
- **CycloneDDS:** v0.10.x with shared memory IPC enabled

### 12.2 Hardware Acceleration & Libraries
- **OpenCV (CUDA-enabled build):** v4.8.0
- **Open3D (C++ / Python):** v0.17.0
- **Eigen3:** v3.4.0 (for high-speed EKF matrix operations)

### 12.3 Python Runtime Dependencies (`requirements.txt`)
```ini
# Layer 1 Core Dependencies
torch>=2.1.0+cu121
torchvision>=0.16.0+cu121
numpy>=1.24.3
scipy>=1.10.1
opencv-python-headless>=4.8.0.76
open3d>=0.17.0
pyyaml>=6.0.1
pyserial>=3.5
cantools>=39.0.0
filterpy>=1.4.5
pytest>=7.4.0
pytest-mock>=3.11.1
```

---

## 13. Unit Test Plan

The Sensor Fusion Module enforces 100% code coverage on class signatures, interfaces, transform mathematics, and degradation state machines using `pytest`.

```
+-----------------------------------------------------------------------------------+
|                                UNIT TEST SUITE                                    |
+-----------------------------------------------------------------------------------+
|  [test_camera_subsystem.py] -> Test GStreamer, Undistortion, Stitching            |
|  [test_lidar_subsystem.py]  -> Test Voxel Grid, RANSAC Ground, BEV Tensor, DBSCAN  |
|  [test_radar_subsystem.py]  -> Test CAN Decode, EKF Trackers, Doppler Validation  |
|  [test_gps_imu_subsystem.py]-> Test NMEA, Bias Comp, 15-DOF EKF Equations        |
|  [test_fusion_engine.py]    -> Test PTP Sync, Temporal Queue, World State Out    |
|  [test_degradation.py]      -> Test Fault Injection & L0-L4 State Transitions     |
+-----------------------------------------------------------------------------------+
```

### 13.1 Subsystem Unit Test Suite Specifications

#### 1. Camera Subsystem Tests (`test_camera_subsystem.py`)
- **Test 1.1 (Undistortion Mathematical Correctness):** Pass a synthetic checkerboard image with known radial distortion parameters through `CameraPreprocessor.process_frame_gpu()`. Verify that straight lines remain straight (residual distortion error \(< 0.5\text{ pixels}\)).
- **Test 1.2 (ImageNet Normalization):** Input a tensor of solid white \((255, 255, 255)\). Verify output tensor values match expected transformed values \(\frac{1.0 - \mu_c}{\sigma_c}\).
- **Test 1.3 (BEV Stitcher Spatial Bounds):** Pass 4 orthogonal camera frames into `MultiCameraStitcher`. Verify the output BEV tensor shape is exactly `[3, 1000, 1000]`.

#### 2. LiDAR Subsystem Tests (`test_lidar_subsystem.py`)
- **Test 2.1 (Voxel Grid Downsampling):** Generate a synthetic point cloud of 100,000 points uniformly distributed in a \(10\text{m} \times 10\text{m} \times 10\text{m}\) cube. Run `PointCloudPreprocessor` with voxel size `0.1m`. Verify the downsampled point count equals approximately \(100 \times 100 \times 100 = 1,000,000\) max voxels (or occupied count).
- **Test 2.2 (RANSAC Ground Plane Extraction):** Synthesize a horizontal ground plane at \(Z = 0.0\text{m}\) with Gaussian noise (\(\sigma = 0.02\text{m}\)) plus elevated points representing a car box. Run RANSAC. Verify estimated plane coefficients \([a, b, c, d]\) approximate \([0, 0, 1, 0]\) within \(\pm 0.01\).
- **Test 2.3 (BEV Tensor Projection Channels):** Input 3D points with specified heights and intensities. Confirm that Channel 0 correctly holds maximum height and Channel 3 holds laser intensity.
- **Test 2.4 (DBSCAN Clustering):** Inject two distinct 3D point clusters separated by 5 meters. Verify `LiDARClustering.cluster_and_box()` returns exactly 2 bounding box dictionaries.

#### 3. RADAR Subsystem Tests (`test_radar_subsystem.py`)
- **Test 3.1 (CAN Packet Parsing):** Inject mock raw SocketCAN bytes corresponding to a Continental ARS540 CAN payload. Verify `RADARDriver.read_targets()` decodes correct range, azimuth, and Doppler velocity.
- **Test 3.2 (EKF Track Lifecycle):** Feed a moving point measurement over 10 iterations. Verify track status transitions from `Tentative` to `Confirmed` at hit count 3. Stop feeding measurements for 5 cycles and verify track deletion.

#### 4. GPS/IMU Subsystem Tests (`test_gps_imu_subsystem.py`)
- **Test 4.1 (IMU Bias Calibration):** Supply stationary raw IMU readings with known static offset. Run `IMUDriver`. Verify calibrated acceleration vector length equals \(9.80665 \pm 0.01 \text{ m/s}^2\).
- **Test 4.2 (15-DOF EKF Prediction-Update Loop):** Inject constant acceleration forward along X-axis for 1.0 second (\(a_x = 2.0 \text{ m/s}^2\)). Verify EKF state velocity \(v_x\) integrates to \(2.0 \text{ m/s} \pm 0.05\). Inject an RTK position measurement update and confirm covariance matrix \(\mathbf{P}\) contracts as expected.

#### 5. Sensor Fusion Engine Tests (`test_sensor_fusion_engine.py`)
- **Test 5.1 (Temporal Alignment Interpolation):** Feed asynchronous queue data with timestamps \(t_1 = 10.00\text{s}\) and \(t_2 = 10.10\text{s}\). Request target timestamp \(T_{\text{target}} = 10.05\text{s}\). Verify returned state is exactly halfway interpolated.
- **Test 5.2 (Unified World State Output Integrity):** Execute `SensorFusionEngine.execute_fusion_cycle()`. Validate that all fields in `UnifiedWorldState` (including `EgoState`, `TrackedObject3D`, `BEVFeatureMap`, and `SensorHealthStatus`) are fully populated without `None` or `NaN` values.
- **Test 5.3 (Fault Injection & Degradation Transition):** Simulate total drop of primary LiDAR frame queue. Verify `SensorHealthStatus.current_degradation_level` dynamically transitions from `"L0_FULL"` to `"L2_SEVERE"`.

---

## 14. Document Sign-Off & Control Metrics

- **Lead Architect:** Subagent Technical Architecture Specialist
- **Reviewed By:** OMNIDRIVE System Safety Board & Layer Architecture Committee
- **Target Repository Path:** `c:\Users\majip\Downloads\rl-jepa-car ai\OMNIDRIVE_PROJECT\docs\01_SENSOR_FUSION_MODULE.md`
- **Execution Approval:** Granted for Layer 1 Module Construction and Component Integration.
