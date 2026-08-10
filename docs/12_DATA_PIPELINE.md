# OMNIDRIVE Data Pipeline & Ingestion Architecture Specification
**OMNIDRIVE Autonomous Driving AI System**  
**Document Version:** 2.4.0  
**Target Platform:** Tactical Military Vehicles, Autonomous Heavy Freight Trucks, Urban Robot Taxis  
**Classification:** Technical Architecture & Data Specification  

---

## 1. Data Overview & Ingestion Requirements

The **OMNIDRIVE Data Pipeline** is designed to ingest, validate, sanitize, preprocess, and format petabyte-scale multi-modal sensory data to power the 7-layer OMNIDRIVE AI architecture. Data flows through three primary paradigms corresponding to the training objectives of OMNIDRIVE's core neural engines:

```
+-----------------------------------------------------------------------------------+
|                            OMNIDRIVE DATA PIPELINE                                |
+-----------------------------------------------------------------------------------+
| 1. Unlabeled Video Streams   --> [JEPA Foundation Model Pretraining]               |
|    (Cameras, LWIR, LiDAR)        Joint-Embedding Predictive Architecture          |
|                                                                                   |
| 2. Labeled Scenario Datasets --> [Alpamayo Fine-Tuning & Multi-Task Heads]        |
|    (3D BBoxes, Trajectories)     Supervised Latent Alignment & Trajectory Tuning    |
|                                                                                   |
| 3. Closed-Loop Sim Rollouts --> [RL Controller Policy Optimization]               |
|    (CARLA, Isaac Sim, Ground)    Model-Based Reinforcement Learning               |
+-----------------------------------------------------------------------------------+
```

### 1.1 Unlabeled Driving Video for JEPA Pretraining
- **Purpose:** Self-supervised spatio-temporal representation learning using the Joint-Embedding Predictive Architecture (JEPA). The model learns world physics, object permanence, and motion dynamics without human labels.
- **Data Modalities:** Multi-camera RGB (8–12 streams), Long-Wave Infrared (LWIR Thermal), 3D LiDAR point clouds (rasterized into BEV), and vehicle CAN ego-kinematics.
- **Volume Target:** >10,000 hours of continuous driving video across diverse geographies, weather, and illumination conditions.

### 1.2 Labeled Scenario Datasets for Alpamayo Fine-Tuning
- **Purpose:** Supervised alignment of latent predictor representations, fine-tuning task-specific decoders (3D object detection, semantic occupancy grids, trajectory forecasting, turn-signal prediction), and behavioral cloning.
- **Data Modalities:** Calibrated camera-LiDAR streams paired with 3D bounding box cuboids, HD map vector layers, pedestrian pose keypoints, lane boundaries, and future ego-trajectory ground truth ($T_{\text{future}} = 8.0\text{ seconds}$ at 10 Hz).
- **Volume Target:** >2,000 hours of meticulously annotated safety-critical scenarios (intersections, highway cut-ins, off-road obstacles, tactical convoy formations).

### 1.3 Synthetic Simulation Data for RL Controller Training
- **Purpose:** Training the model-based Reinforcement Learning (RL) actor-critic policy within simulator environments (CARLA, NVIDIA Isaac Sim) and offline re-simulation logs.
- **Data Modalities:** Closed-loop rollout buffers containing state vectors $\mathbf{s}_t \in \mathbb{R}^{D_{\text{latent}}}$, control actions $\mathbf{a}_t = [\delta, \alpha, \beta]^T$ (steering, throttle, brake), scalar rewards $r_t$, done flags $d_t$, and privilege ground truth (actor velocities, friction coefficients).
- **Volume Target:** >50,000,000 synthetic interaction steps with dynamic domain randomization.

---

## 2. Public Datasets Table

OMNIDRIVE integrates multiple open-source autonomous driving benchmarks alongside proprietary sensor feeds. The complete dataset inventory is detailed below:

| Dataset Name | Total Size | Duration / Volume | Modalities Included | Key Features & Annotations | Official Download URL |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **nuScenes** | ~500 GB | 1,000 scenes (23 hours), 700 train / 150 val | 6 Cameras, 1 LiDAR (32-ch), 5 RADARs, GNSS/IMU | 1.4M 3D bounding boxes, 23 object classes, map layers, weather tags | [https://www.nuscenes.org](https://www.nuscenes.org) |
| **Waymo Open Dataset** | ~2.0 TB | 1,950 segments (20 Hz, 20s each), 1,150 train | 5 High-Res Cameras, 5 LiDARs (top + 4 sides), GNSS/IMU | 12M 3D LiDAR boxes, 12M 2D camera boxes, tracking IDs, camera-LiDAR sync | [https://waymo.com/open](https://waymo.com/open) |
| **nuPlan** | ~4.0 TB | 1,500 hours, 1600+ urban cities (Boston, SF, Vegas) | Sensor trajectories, HD Vector Maps, raw camera/LiDAR | World's largest trajectory dataset for ML motion planning, discrete graph paths | [https://www.nuplan.org](https://www.nuplan.org) |
| **BDD100K** | ~1.0 TB | 100,000 HD video sequences (40s each, 720p @ 30 FPS) | Front Camera RGB, GPS/IMU trajectories | Diverse weather, day/night, 100K 2D boxes, instance segmentation, drivable area | [https://bdd-data.berkeley.edu](https://bdd-data.berkeley.edu) |
| **KITTI Vision Benchmark** | ~50 GB | 7,481 training / 7,518 testing stereo frames | Stereo RGB Cameras, Velodyne HDL-64E LiDAR, GPS/IMU | Benchmark classic for 3D detection, optical flow, stereo vision, visual odometry | [https://www.cvlibs.net/datasets/kitti](https://www.cvlibs.net/datasets/kitti) |
| **CARLA Synthetic** | Unlimited | Procedurally generated (Custom generation script) | Multi-Camera RGB, Depth, Semantic, LiDAR, RADAR | Perfect ground-truth annotations, customizable extreme weather & failure modes | Local Generation Script (`dataset_downloader.py --carla`) |

---

## 3. Dataset Download Scripts (`dataset_downloader.py`)

The `dataset_downloader.py` utility automates downloading, checksum verification, and decompression of public and cloud-hosted datasets.

### 3.1 Script Execution Interface

```bash
# Download nuScenes trainval schema and mini set
python scripts/dataset_downloader.py --dataset nuscenes --version v1.0-trainval --target-dir /mnt/storage/raw_data/nuscenes

# Download Waymo Open Dataset perception v1.4.2
python scripts/dataset_downloader.py --dataset waymo --version v1.4.2 --target-dir /mnt/storage/raw_data/waymo --num-workers 16

# Generate synthetic CARLA dataset locally
python scripts/dataset_downloader.py --dataset carla --town Town04 --num-episodes 500 --target-dir /mnt/storage/raw_data/carla
```

### 3.2 Expected Download Times Matrix

| Dataset | Uncompressed Size | Download Time (10 Gbps Fiber) | Download Time (1 Gbps Commercial) | Download Time (100 Mbps) |
| :--- | :--- | :--- | :--- | :--- |
| **nuScenes (Full)** | 480 GB | ~7 minutes | ~68 minutes | ~11.3 hours |
| **Waymo Open (Perception)**| 2,100 GB | ~30 minutes | ~4.8 hours | ~48.0 hours |
| **nuPlan (Full)** | 4,200 GB | ~60 minutes | ~9.6 hours | ~96.0 hours |
| **BDD100K (Full Video)** | 1,050 GB | ~15 minutes | ~2.4 hours | ~24.0 hours |
| **KITTI (3D Detection)** | 45 GB | ~40 seconds | ~6 minutes | ~60 minutes |

### 3.3 Python Downloader Implementation (`dataset_downloader.py`)

```python
"""
OMNIDRIVE Dataset Downloader Utility
Automates secure downloading, authentication, verification, and extraction of public benchmarks.
"""

import os
import sys
import argparse
import hashlib
import requests
import subprocess
from typing import Dict, Any
from pathlib import Path
from tqdm import tqdm


DATASET_REGISTRY: Dict[str, Dict[str, Any]] = {
    "nuscenes": {
        "url_base": "https://www.nuscenes.org/data/",
        "files": ["v1.0-trainval01_blobs.tgz", "v1.0-trainval02_blobs.tgz", "v1.0-meta.tgz"],
        "auth_required": True,
        "env_token": "NUSCENES_AUTH_TOKEN",
    },
    "waymo": {
        "gs_bucket": "gs://waymo_open_dataset_v_1_4_2/individual_files/training/",
        "auth_required": True,
        "env_token": "GCP_SERVICE_ACCOUNT_KEY",
    },
    "kitti": {
        "url_base": "https://s3.eu-central-1.amazonaws.com/avg-kitti/",
        "files": ["data_object_image_2.zip", "data_object_velodyne.zip", "data_object_calib.zip"],
        "auth_required": False,
    },
    "bdd100k": {
        "url_base": "https://dl.drivable.ai/bdd100k/",
        "files": ["bdd100k_videos_train_00.zip", "bdd100k_labels_release.zip"],
        "auth_required": True,
        "env_token": "BDD100K_AUTH_TOKEN",
    }
}


class DatasetDownloader:
    def __init__(self, dataset_name: str, target_dir: str, num_workers: int = 8):
        self.dataset_name = dataset_name.lower()
        if self.dataset_name not in DATASET_REGISTRY and self.dataset_name != "carla":
            raise ValueError(f"Unknown dataset: {dataset_name}. Valid options: {list(DATASET_REGISTRY.keys()) + ['carla']}")
        
        self.target_dir = Path(target_dir)
        self.target_dir.mkdir(parents=True, exist_ok=True)
        self.num_workers = num_workers

    def _verify_auth(self, meta: Dict[str, Any]) -> None:
        if meta.get("auth_required", False):
            token_env = meta.get("env_token")
            if not os.getenv(token_env):
                raise PermissionError(
                    f"Dataset {self.dataset_name} requires authentication. "
                    f"Please set environment variable '{token_env}' before running."
                )

    def download_file(self, url: str, destination: Path) -> None:
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get("content-length", 0))
        block_size = 1024 * 1024  # 1 MB

        with open(destination, "wb") as f, tqdm(
            desc=destination.name,
            total=total_size,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(block_size):
                size = f.write(data)
                bar.update(size)

    def download_kitti(self) -> None:
        meta = DATASET_REGISTRY["kitti"]
        for filename in meta["files"]:
            file_url = meta["url_base"] + filename
            dest_path = self.target_dir / filename
            if not dest_path.exists():
                print(f"[INFO] Downloading KITTI archive: {filename}")
                self.download_file(file_url, dest_path)
                print(f"[INFO] Extracting {filename}...")
                subprocess.run(["unzip", "-q", str(dest_path), "-d", str(self.target_dir)], check=True)

    def download_waymo(self) -> None:
        meta = DATASET_REGISTRY["waymo"]
        self._verify_auth(meta)
        print(f"[INFO] Syncing Waymo Open Dataset from {meta['gs_bucket']} via gsutil...")
        cmd = ["gsutil", "-m", "cp", "-r", meta["gs_bucket"], str(self.target_dir)]
        subprocess.run(cmd, check=True)

    def run(self) -> None:
        if self.dataset_name == "kitti":
            self.download_kitti()
        elif self.dataset_name == "waymo":
            self.download_waymo()
        else:
            print(f"[INFO] Executing downloader for {self.dataset_name}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OMNIDRIVE Dataset Downloader")
    parser.add_argument("--dataset", required=True, choices=["nuscenes", "waymo", "nuplan", "bdd100k", "kitti", "carla"])
    parser.add_argument("--target-dir", required=True, help="Destination directory on SSD/NAS")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of parallel download threads")
    args = parser.parse_args()

    downloader = DatasetDownloader(args.dataset, args.target_dir, args.num_workers)
    downloader.run()
```

---

## 4. Unified HDF5 Data Format & Schemas

To achieve ultra-high throughput I/O during GPU multi-node distributed training, all heterogeneous sensor data is converted into standardized **HDF5 (`.h5`) file archives**.

```
+-----------------------------------------------------------------------------------+
|                            UNIFIED HDF5 ARCHIVE SCHEMA                            |
+-----------------------------------------------------------------------------------+
|  /cameras                                                                         |
|     ├── rgb_front      (B, T, C, H, W) [uint8]                                   |
|     ├── rgb_left       (B, T, C, H, W) [uint8]                                   |
|     ├── rgb_right      (B, T, C, H, W) [uint8]                                   |
|     └── thermal_front  (B, T, 1, H, W) [uint16]                                  |
|  /lidar                                                                           |
|     └── bev_grid       (B, H_bev, W_bev, C_bev) [float32]                         |
|  /kinematics                                                                      |
|     ├── ego_trajectory (B, T, 3) [float32] (x, y, yaw)                            |
|     └── ego_velocity   (B, T, 3) [float32] (vx, vy, yaw_rate)                     |
|  /labels                                                                          |
|     ├── boxes_3d       (B, T, N_max, 10) [float32] (x,y,z,dx,dy,dz,r,p,y,class)   |
|     └── occupancy_grid (B, H_occ, W_occ, Z_occ) [uint8]                           |
+-----------------------------------------------------------------------------------+
```

### 4.1 Detailed Dataset Tensor Schemas

#### 1. Camera Frame Tensor (`/cameras/rgb_*`)
- **Shape:** $(B, T, C, H, W)$
  - $B$: Batch size / episode sequence count
  - $T$: Temporal window length ($T = 16$ frames at 10 Hz = 1.6s history/future)
  - $C$: Channels ($C=3$ for RGB, $C=1$ for Thermal LWIR)
  - $H, W$: Spatial dimensions ($H = 224, W = 224$ for standard JEPA, or $448 \times 448$ for high-res)
- **Data Type:** `uint8` (0–255 range, unnormalized on disk to minimize storage volume by $4\times$ relative to `float32`).

#### 2. LiDAR BEV Feature Tensor (`/lidar/bev_grid`)
- **Shape:** $(B, H_{\text{bev}}, W_{\text{bev}}, C_{\text{bev}})$
  - $H_{\text{bev}} = 256$, $W_{\text{bev}} = 256$ (covering $[-50\text{m}, +50\text{m}]$ spatial range at $0.39\text{m/pixel}$)
  - $C_{\text{bev}} = 8$ channels:
    1. Channel 0: Height density / intensity map
    2. Channels 1–5: Max height slices ($Z \in [-2\text{m}, +4\text{m}]$ split into 5 height bands)
    3. Channel 6: Point reflectivity mean
    4. Channel 7: LiDAR pulse ring density
- **Data Type:** `float32` compressed via GZIP level 4 in HDF5.

#### 3. Ego-Trajectory Tensor (`/kinematics/ego_trajectory`)
- **Shape:** $(B, T, 3)$ where columns represent $(x, y, \psi)$ in ISO-8855 ego-vehicle centered frame at $t=0$.
- **Units:** Meters for $(x, y)$, Radians for heading $\psi$.
- **Data Type:** `float32`.

#### 4. 3D Object Label Matrix (`/labels/boxes_3d`)
- **Shape:** $(B, T, N_{\text{max}}, 10)$ padded to $N_{\text{max}} = 128$ objects.
- **Attributes per object:** $(x, y, z, \text{dx}, \text{dy}, \text{dz}, \text{roll}, \text{pitch}, \text{yaw}, \text{class\_id})$.
- **Data Type:** `float32`.

### 4.2 HDF5 Dataset Handler Code (`hdf5_handler.py`)

```python
"""
OMNIDRIVE HDF5 Read/Write Handler with Chunking and Compression Optimization
"""

import h5py
import numpy as np
from typing import Dict, Any
from pathlib import Path


class HDF5DatasetHandler:
    def __init__(self, filepath: Path, mode: str = "r"):
        self.filepath = Path(filepath)
        self.mode = mode
        self.file_handle = h5py.File(self.filepath, self.mode)

    def create_schema(self, batch_size: int, seq_len: int = 16, height: int = 224, width: int = 224):
        """Initializes empty datasets with optimal chunking and GZIP compression."""
        cameras_grp = self.file_handle.create_group("cameras")
        cameras_grp.create_dataset(
            "rgb_front",
            shape=(batch_size, seq_len, 3, height, width),
            maxshape=(None, seq_len, 3, height, width),
            dtype=np.uint8,
            chunks=(1, seq_len, 3, height, width),
            compression="gzip",
            compression_opts=4
        )

        lidar_grp = self.file_handle.create_group("lidar")
        lidar_grp.create_dataset(
            "bev_grid",
            shape=(batch_size, 256, 256, 8),
            maxshape=(None, 256, 256, 8),
            dtype=np.float32,
            chunks=(1, 256, 256, 8),
            compression="gzip",
            compression_opts=4
        )

        kin_grp = self.file_handle.create_group("kinematics")
        kin_grp.create_dataset(
            "ego_trajectory",
            shape=(batch_size, seq_len, 3),
            maxshape=(None, seq_len, 3),
            dtype=np.float32
        )

    def write_sample(self, index: int, sample_data: Dict[str, np.ndarray]):
        """Writes a single data sample into the HDF5 archive."""
        self.file_handle["cameras/rgb_front"][index] = sample_data["rgb_front"]
        self.file_handle["lidar/bev_grid"][index] = sample_data["bev_grid"]
        self.file_handle["kinematics/ego_trajectory"][index] = sample_data["ego_trajectory"]

    def close(self):
        if self.file_handle:
            self.file_handle.close()
```

---

## 5. Data Preprocessing Pipeline (`dataset_preprocessor.py`)

The preprocessing pipeline converts raw video (MP4/AVI), point cloud binaries (`.bin`/`.pcap`), and telemetry JSON logs into unified HDF5 archives.

```
+-----------------------------------------------------------------------------------+
|                        PREPROCESSING PIPELINE FLOW                                |
+-----------------------------------------------------------------------------------+
| Raw Video (MP4)   --> Decode Frames --> CUDA Resize (224x224) --> Normalize        |
|                                                                    |              |
| Raw LiDAR (.bin)  --> Ground Filter --> Voxelize --> BEV (256x256) | --> HDF5 Write|
|                                                                    |              |
| CAN/GPS Log (CSV) --> Coordinate Align --> Temporal Interpolate ----+              |
+-----------------------------------------------------------------------------------+
```

### 5.1 Pipeline Step Breakdown

1. **Video Stream Decoding:** Decodes raw video streams via hardware-accelerated NVIDIA NVDEC / PyAV.
2. **Frame Extraction & Resizing:** Extracts synchronized frames at 10 Hz, resizes to $224 \times 224$ via CUDA bicubic interpolation.
3. **Photometric Normalization:** Normalizes RGB frames to zero mean and unit variance using ImageNet coefficients:
   $$\mathbf{I}_{\text{norm}}(c, y, x) = \frac{\mathbf{I}(c, y, x) / 255.0 - \mu_c}{\sigma_c}$$
   where $\boldsymbol{\mu} = [0.485, 0.456, 0.406]$ and $\boldsymbol{\sigma} = [0.229, 0.224, 0.225]$.
4. **Temporal Windowing:** Constructs overlapping sliding windows of $T=16$ frames ($1.6\text{ seconds}$ context) with step size $\Delta t = 2$ frames.
5. **LiDAR Voxelization & BEV Rasterization:**
   - Filters out ground plane points ($z < -2.0\text{m}$) using RANSAC or height thresholding.
   - Discretizes spatial bounds $x \in [-50\text{m}, +50\text{m}]$, $y \in [-50\text{m}, +50\text{m}]$, $z \in [-2\text{m}, +4\text{m}]$ into voxel grid cell size $\Delta x = \Delta y = 0.39\text{m}$.
   - Aggregates point density, max height, and mean intensity into an 8-channel BEV image $(256 \times 256 \times 8)$.

### 5.2 Python Preprocessor Implementation (`dataset_preprocessor.py`)

```python
"""
OMNIDRIVE Dataset Preprocessor Engine
Executes parallel multi-threaded video decoding, LiDAR BEV rasterization, and HDF5 serialization.
"""

import os
import cv2
import h5py
import torch
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any


class DatasetPreprocessor:
    def __init__(self, target_h5_path: Path, img_size: Tuple[int, int] = (224, 224)):
        self.target_h5_path = Path(target_h5_path)
        self.img_size = img_size
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

    def preprocess_video_frames(self, video_path: Path, seq_len: int = 16) -> np.ndarray:
        """Decodes video file, resizes frames, normalizes, and packages into temporal windows."""
        cap = cv2.VideoCapture(str(video_path))
        frames = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize
            frame_resized = cv2.resize(frame, self.img_size, interpolation=cv2.INTER_CUBIC)
            frames.append(frame_resized)

        cap.release()

        frames_arr = np.array(frames, dtype=np.uint8)  # (N_total, H, W, C)
        
        # Ensure we have at least seq_len frames
        if len(frames_arr) < seq_len:
            pad_width = ((0, seq_len - len(frames_arr)), (0, 0), (0, 0), (0, 0))
            frames_arr = np.pad(frames_arr, pad_width, mode="edge")

        # Extract sequence (T, C, H, W)
        seq_frames = frames_arr[:seq_len]
        seq_frames = np.transpose(seq_frames, (0, 3, 1, 2))  # (T, C, H, W)
        return seq_frames

    def lidar_to_bev(self, point_cloud: np.ndarray, grid_size: Tuple[int, int] = (256, 256)) -> np.ndarray:
        """Converts raw (N, 4) point cloud (x, y, z, intensity) into (256, 256, 8) BEV representation."""
        bev = np.zeros((grid_size[0], grid_size[1], 8), dtype=np.float32)
        
        # Spatial filtering
        mask = (point_cloud[:, 0] >= -50.0) & (point_cloud[:, 0] <= 50.0) & \
               (point_cloud[:, 1] >= -50.0) & (point_cloud[:, 1] <= 50.0) & \
               (point_cloud[:, 2] >= -2.0) & (point_cloud[:, 2] <= 4.0)
        pts = point_cloud[mask]

        if len(pts) == 0:
            return bev

        # Map meters to pixel coordinates
        x_img = ((pts[:, 0] + 50.0) / 100.0 * (grid_size[1] - 1)).astype(np.int32)
        y_img = ((pts[:, 1] + 50.0) / 100.0 * (grid_size[0] - 1)).astype(np.int32)

        # Height slice channels (5 slices from -2.0 to 4.0m)
        z_bins = np.linspace(-2.0, 4.0, 6)
        for i in range(5):
            z_mask = (pts[:, 2] >= z_bins[i]) & (pts[:, 2] < z_bins[i+1])
            bev[y_img[z_mask], x_img[z_mask], i] = 1.0

        # Density map (Channel 5)
        np.add.at(bev[:, :, 5], (y_img, x_img), 0.1)
        bev[:, :, 5] = np.clip(bev[:, :, 5], 0.0, 1.0)

        # Intensity mean map (Channel 6)
        bev[y_img, x_img, 6] = pts[:, 3]

        return bev
```

---

## 6. Custom Dashcam Collection Infrastructure (`dashcam_collector.py`)

OMNIDRIVE integrates custom real-world video capture pipelines from comma 3X hardware or custom GoPro Hero 12 rigs equipped with RTK-GNSS receivers and OBD-II CAN bus decoders.

```
+-----------------------------------------------------------------------------------+
|                        CUSTOM DASHCAM COLLECTION PIPELINE                         |
+-----------------------------------------------------------------------------------+
| Hardware Rig (comma 3X / GoPro + RTK) --> Raw MP4 + CAN CSV                        |
|                                                     |                             |
| Central Upload via Wi-Fi 6 / 5G -------------> Central Storage                    |
|                                                     |                             |
| Automatic Privacy Scrubbing (DeepFace / LPRNet) --> Anonymized Dataset            |
+-----------------------------------------------------------------------------------+
```

### 6.1 Hardware Specifications

- **Option A (Comma 3X Hardware):**
  - Triple Sony ISX021 HDR image sensors ($120^\circ, 150^\circ, 20^\circ$ FOV).
  - Qualcomm Snapdragon 845 SoC running openpilot logger daemon.
  - Built-in u-blox M8 GNSS + 9-axis LSM6DS3 IMU.
  - Direct OBD-II CAN bus tapping (Panda interface).

- **Option B (GoPro + RTK Rig):**
  - Dual GoPro Hero 12 Black ($4\text{K} @ 60\text{ FPS}$, Wide FOV).
  - Septentrio Mosaic-X5 Dual Antenna RTK-GNSS (1 cm positioning accuracy).
  - Custom Raspberry Pi 4 OBD-II CAN bus reader logging at 100 Hz.

### 6.2 Automatic Privacy Scrubbing & Anonymization

Before ingested dashcam video is moved to active training pools, it passes through an automated pipeline:
- **Face Anonymization:** Uses **DeepFace** with a RetinaFace backend to detect human faces and applies a $25 \times 25$ Gaussian kernel blur.
- **License Plate Anonymization:** Uses **LPRNet** / YOLO-v8-Plate to locate automotive license plates and applies dense pixelation.

### 6.3 Dashcam Collector & Anonymizer Implementation (`dashcam_collector.py`)

```python
"""
OMNIDRIVE Custom Dashcam Collector & Privacy Scrubbing Pipeline
Scrub faces and license plates from raw MP4 video before uploading to central NAS/S3 storage.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import List


class PrivacyAnonymizer:
    def __init__(self, blur_kernel_size: int = 25):
        self.blur_kernel = (blur_kernel_size, blur_kernel_size)
        # Load cascade classifier fallbacks (or DeepFace/YOLO modules)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')

    def anonymize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Detects faces and license plates and applies Gaussian blur."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            roi = frame[y:y+h, x:x+w]
            frame[y:y+h, x:x+w] = cv2.GaussianBlur(roi, self.blur_kernel, 30)

        # Detect license plates
        plates = self.plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(25, 15))
        for (x, y, w, h) in plates:
            roi = frame[y:y+h, x:x+w]
            frame[y:y+h, x:x+w] = cv2.GaussianBlur(roi, self.blur_kernel, 30)

        return frame


class DashcamCollector:
    def __init__(self, input_video: Path, output_video: Path):
        self.input_video = Path(input_video)
        self.output_video = Path(output_video)
        self.anonymizer = PrivacyAnonymizer()

    def process_and_upload(self, s3_target_bucket: str = None):
        """Processes raw video frame by frame, scrubs PII, and outputs clean MP4."""
        cap = cv2.VideoCapture(str(self.input_video))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(self.output_video), fourcc, fps, (width, height))

        print(f"[INFO] Anonymizing video {self.input_video.name}...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            clean_frame = self.anonymizer.anonymize_frame(frame)
            out.write(clean_frame)

        cap.release()
        out.release()
        print(f"[SUCCESS] Saved anonymized video to {self.output_video}")

        if s3_target_bucket:
            print(f"[INFO] Uploading {self.output_video.name} to s3://{s3_target_bucket}...")
            # AWS S3 sync command execution
            os.system(f"aws s3 cp {self.output_video} s3://{s3_target_bucket}/dashcam_raw/")


if __name__ == "__main__":
    collector = DashcamCollector(Path("raw_trip_001.mp4"), Path("clean_trip_001.mp4"))
    collector.process_and_upload(s3_target_bucket="omnidrive-fleet-data")
```

---

## 7. Data Augmentation Strategies

Data augmentation is critical to prevent overfitting and ensure model robustness under out-of-distribution driving conditions.

### 7.1 Augmentation Strategies by Training Phase

```
+-----------------------------------------------------------------------------------+
|                         DATA AUGMENTATION TAXONOMY                                |
+-----------------------------------------------------------------------------------+
| 1. JEPA Pretraining Augmentations:                                                |
|    ├── 3D Spatio-Temporal Token Masking (75% random block removal)                 |
|    ├── Photometric Color Jittering (Brightness +/-20%, Contrast +/-15%)            |
|    ├── Horizontal Spatial Mirroring (Flip image + invert trajectory sign)          |
|    └── Temporal Rate Jittering (Sub-sample frames at 5Hz, 10Hz, 20Hz)              |
|                                                                                   |
| 2. RL Controller (CARLA) Augmentations:                                           |
|    ├── Dynamic Weather Randomization (Sun Angle, Heavy Rain, Fog, Puddle Wetness) |
|    ├── Sensor Noise Injection (LiDAR Dropout 5-15%, Camera Flare, RADAR Clutter)  |
|    └── Dynamic Mass & Tire Friction Perturbations (Friction [0.4, 1.2])            |
+-----------------------------------------------------------------------------------+
```

### 7.2 PyTorch Data Augmentation Pipeline Implementation

```python
"""
OMNIDRIVE PyTorch Data Augmentation Engine for JEPA Pretraining & RL Fine-Tuning
"""

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random
from typing import Tuple, Dict


class JEPADataAugmentor:
    def __init__(self, mask_ratio: float = 0.75, patch_size: int = 16):
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.color_jitter = T.ColorJitter(brightness=0.2, contrast=0.15, saturation=0.1, hue=0.05)

    def apply_spatiotemporal_masking(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies 3D block masking to frame tensor (T, C, H, W).
        Returns masked tensor and binary boolean mask tensor.
        """
        T_dim, C, H, W = tensor.shape
        num_patches_h = H // self.patch_size
        num_patches_w = W // self.patch_size
        total_patches = num_patches_h * num_patches_w

        # Generate random mask for each patch
        num_masked = int(total_patches * self.mask_ratio)
        mask_idx = torch.randperm(total_patches)[:num_masked]

        mask = torch.zeros((total_patches,), dtype=torch.bool)
        mask[mask_idx] = True
        mask = mask.view(num_patches_h, num_patches_w)

        # Upsample mask to image size
        mask_expanded = mask.repeat_interleave(self.patch_size, dim=0).repeat_interleave(self.patch_size, dim=1)
        mask_4d = mask_expanded.unsqueeze(0).unsqueeze(0).repeat(T_dim, C, 1, 1)

        masked_tensor = tensor.clone()
        masked_tensor[mask_4d] = 0.0  # Zero out masked patches

        return masked_tensor, mask

    def augment_sequence(self, frames: torch.Tensor, trajectory: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies synchronized photometric jitter and spatial mirroring across temporal frames.
        frames: (T, C, H, W), trajectory: (T, 3) -> (x, y, yaw)
        """
        # Photometric jitter applied consistently to all frames
        frames = self.color_jitter(frames)

        # 50% chance horizontal flip
        if random.random() > 0.5:
            frames = TF.hflip(frames)
            # Invert lateral y-coordinate and yaw angle in trajectory
            trajectory[:, 1] = -trajectory[:, 1]  # y = -y
            trajectory[:, 2] = -trajectory[:, 2]  # yaw = -yaw

        return frames, trajectory
```

---

## 8. Data Versioning with DVC (Data Version Control)

OMNIDRIVE uses **Data Version Control (DVC)** coupled with Git to track dataset revisions, pipeline configurations, and preprocessed HDF5 archives across cloud and local NVMe storage.

```
+-----------------------------------------------------------------------------------+
|                            DVC DATA VERSIONING ARCHITECTURE                       |
+-----------------------------------------------------------------------------------+
|  Git Repository (Code & Metadata)               Remote Storage (S3 / NAS)          |
|  ├── src/                                       ├── datasets/                     |
|  ├── configs/                                   │   ├── nuscenes_v1.0.h5 (500GB)  |
|  ├── dataset_preprocessor.py                    │   └── waymo_v1.4.2.h5 (2TB)     |
|  └── datasets.dvc  ---------------------------> └── carla_sim_v2.h5 (1TB)       |
+-----------------------------------------------------------------------------------+
```

### 8.1 Basic DVC Commands Workflow

```bash
# Initialize DVC in the OMNIDRIVE repository
dvc init

# Configure S3 remote storage target
dvc remote add -d s3remote s3://omnidrive-data-bucket/dvc_store
dvc remote modify s3remote profile omnidrive-aws-profile

# Add large HDF5 dataset to tracking
dvc add /mnt/storage/preprocessed/nuscenes_v1.0_train.h5

# Git commit the lightweight pointer file generated by DVC
git add /mnt/storage/preprocessed/nuscenes_v1.0_train.h5.dvc .gitignore
git commit -m "feat(data): add preprocessed nuScenes v1.0 HDF5 dataset version 2.4"

# Push heavy binaries to S3 cold/hot storage
dvc push

# On a new GPU training node, pull exact matching data version
git checkout main
dvc pull
```

### 8.2 Sample `dvc.yaml` Pipeline Definition

```yaml
stages:
  preprocess_nuscenes:
    cmd: python scripts/dataset_preprocessor.py --input /mnt/storage/raw_data/nuscenes --output /mnt/storage/preprocessed/nuscenes_v1.0_train.h5
    deps:
      - scripts/dataset_preprocessor.py
      - /mnt/storage/raw_data/nuscenes
    outs:
      - /mnt/storage/preprocessed/nuscenes_v1.0_train.h5:
          cache: true
```

---

## 9. Data Storage & Infrastructure Architecture

The storage topology follows a three-tiered infrastructure balancing low-latency GPU memory feeding with cost-effective multi-petabyte archiving.

```
+-----------------------------------------------------------------------------------+
|                         THREE-TIER STORAGE TOPOLOGY                               |
+-----------------------------------------------------------------------------------+
| TIER 1: Local NVMe RAID-0 Arrays  (PCIe Gen5, >14 GB/s read, 30 TB per GPU Node)  |
|          └── Active Training Hot Cache (Cached HDF5 Chunks)                       |
|                                                                                   |
| TIER 2: On-Premises Enterprise NAS (TrueNAS ZFS, 100GbE QSFP28, 500 TB Capacity)  |
|          └── Preprocessed HDF5 Master Registry                                    |
|                                                                                   |
| TIER 3: Cloud Cold Backup (AWS S3 Glacier / Google Cloud Storage Deep Archive)    |
|          └── Raw Uncompressed Datasets & Disaster Recovery                        |
+-----------------------------------------------------------------------------------+
```

### 9.1 Storage Cost Breakdown & Capacity Estimation (1 Petabyte Dataset Lifecycle)

| Storage Tier | Technology | Bandwidth / Latency | Installed Capacity | Monthly Cost / TB | Total Monthly Cost (1 PB) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Tier 1 (Hot NVMe Cache)** | Kioxia CM6 PCIe Gen4 NVMe RAID-0 | 14.5 GB/s / < 20 $\mu\text{s}$ | 60 TB (Across 4 nodes) | $65.00 / TB | $3,900 |
| **Tier 2 (On-Prem NAS)** | TrueNAS ZFS Pool (100GbE NICs) | 4.8 GB/s / < 1 $\text{ms}$ | 600 TB | $18.00 / TB | $10,800 |
| **Tier 3 (Cloud Cold S3)** | AWS S3 Glacier Flexible Retrieval | Variable / Hours | 1,000 TB (1 PB) | $3.60 / TB | $3,600 |
| **Total Combined** | Hybrid Multi-Tier Arch | — | **1.66 PB Total** | — | **$18,300 / month** |

---

## 10. Privacy, Security & Regulatory Compliance

### 10.1 GDPR Compliance for Public Road Dashcam Data
- **Lawful Basis & Consent:** Fleet dashcam collection operates under Legitimate Interest for AI safety research (GDPR Article 6(1)(f)).
- **Automated PII Scrubbing:** Facial identities and vehicle license plates are irreversibly blurred at the edge or immediately during server ingestion before dataset commitment.
- **Right to be Forgotten (Article 17):** OMNIDRIVE maintains a hash-indexed registry of collection timestamps and GPS coordinates. If a data subject requests erasure, the corresponding HDF5 temporal slice is expunged using `h5repack`.

### 10.2 Military & Defense Data Classification Handling
- **Security Partitioning:** Military tactical convoy datasets operate under strict **CUI (Controlled Unclassified Information)** or **SECRET** classification protocols.
- **Air-Gapped Processing:** Tactical military data processing pipelines run entirely on physically isolated, air-gapped server clusters with disabled wireless radios.
- **Encryption Standards:** All military data stored at rest is encrypted using **AES-256-GCM** hardware encryption compliant with FIPS 140-3 standards.
- **ITAR / EAR Export Control:** Synthetic defense scenario models and sensor logs are flagged with ITAR compliance metadata, restricting network replication to cleared U.S. defense cloud endpoints.

---

## 11. Automated Data Quality & Validation Engine

Before raw logs are converted to HDF5 archives, an automated quality control suite validates sensory inputs.

```
+-----------------------------------------------------------------------------------+
|                         DATA QUALITY ASSURANCE CHECKS                             |
+-----------------------------------------------------------------------------------+
| 1. Camera Integrity Checks:                                                        |
|    ├── Brightness Range Check: Reject if mean intensity < 15 or > 240 (Under/Over) |
|    └── Blur Metric (Laplacian Variance): Reject if Var(Laplacian) < 100.0          |
|                                                                                   |
| 2. Temporal & Rate Consistency:                                                   |
|    └── Inter-frame delta check: Alert if Δt > 15ms (Nominal 10ms for 100Hz IMU)    |
|                                                                                   |
| 3. GPS / RTK Accuracy Validation:                                                 |
|    ├── HDOP Check: Reject frame if Horizontal Dilution of Precision > 2.0          |
|    └── Satellite Count: Require Satellite Count >= 8 for valid position           |
+-----------------------------------------------------------------------------------+
```

### 11.1 Python Quality Assurance Script (`data_quality_validator.py`)

```python
"""
OMNIDRIVE Automated Data Quality & Validation Engine
"""

import cv2
import numpy as np
from typing import Dict, Any, List


class DataQualityValidator:
    def __init__(self, min_laplacian_var: float = 100.0, max_hdop: float = 2.0):
        self.min_laplacian_var = min_laplacian_var
        self.max_hdop = max_hdop

    def validate_camera_frame(self, frame_bgr: np.ndarray) -> Dict[str, Any]:
        """Checks camera image brightness, contrast, and focus sharpness."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        mean_brightness = float(np.mean(gray))
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        is_valid = True
        failures = []

        if mean_brightness < 15.0:
            is_valid = False
            failures.append(f"Image underexposed (brightness={mean_brightness:.1f} < 15.0)")
        elif mean_brightness > 240.0:
            is_valid = False
            failures.append(f"Image overexposed (brightness={mean_brightness:.1f} > 240.0)")

        if laplacian_var < self.min_laplacian_var:
            is_valid = False
            failures.append(f"Image blurred (Laplacian variance={laplacian_var:.1f} < {self.min_laplacian_var})")

        return {
            "valid": is_valid,
            "brightness": mean_brightness,
            "sharpness": laplacian_var,
            "failures": failures
        }

    def validate_gps_telemetry(self, hdop: float, num_satellites: int) -> Dict[str, Any]:
        """Validates GNSS RTK positioning quality."""
        is_valid = True
        failures = []

        if hdop > self.max_hdop:
            is_valid = False
            failures.append(f"Poor GPS precision (HDOP={hdop:.2f} > {self.max_hdop})")

        if num_satellites < 8:
            is_valid = False
            failures.append(f"Insufficient satellites (count={num_satellites} < 8)")

        return {
            "valid": is_valid,
            "hdop": hdop,
            "satellites": num_satellites,
            "failures": failures
        }


if __name__ == "__main__":
    validator = DataQualityValidator()
    
    # Test frame check
    dummy_frame = np.zeros((224, 224, 3), dtype=np.uint8)
    cam_result = validator.validate_camera_frame(dummy_frame)
    print("[TEST] Camera Validation Result:", cam_result)

    # Test GPS check
    gps_result = validator.validate_gps_telemetry(hdop=1.2, num_satellites=12)
    print("[TEST] GPS Validation Result:", gps_result)
```

---
**End of File 1: OMNIDRIVE Data Pipeline Technical Specification**
