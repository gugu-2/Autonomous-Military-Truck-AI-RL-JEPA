# Layer 10: OMNIDRIVE System Deployment & Operations Manual
**OMNIDRIVE Autonomous Driving AI System**  
**Document Version:** 2.4.0  
**Target Platforms:** Tactical Military Vehicles (GCV/UGV), Heavy Freight Trucks (Class 8), Urban RobotTaxis (L4 Mobility)  
**Classification:** Technical Operations & Deployment Specification  

---

## 1. Deployment Overview

The **OMNIDRIVE Autonomous Driving AI System** is engineered as a modular, 7-layer autonomous software stack capable of operating across three fundamentally distinct operational domains: **Tactical Military Operations**, **Class 8 Heavy Freight Trucking**, and **Urban RobotTaxi Passenger Mobility**.

```
+-----------------------------------------------------------------------------------+
|                        OMNIDRIVE SYSTEM DEPLOYMENT ARCHITECTURE                    |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  +--------------------+     +---------------------+     +----------------------+  |
|  | MILITARY PROFILE   |     | HEAVY TRUCK PROFILE |     | ROBOTAXI PROFILE     |  |
|  | - Air-Gapped / EW  |     | - SAE J1939 CAN-FD  |     | - DBW Kit Integrator |  |
|  | - AES-256 / TPM2.0 |     | - Heavy Vehicle DBW |     | - Fleet HMI & Cloud  |  |
|  | - JAUS Protocol    |     | - Weigh-In-Motion   |     | - Dynamic Geofencing |  |
|  +--------------------+     +---------------------+     +----------------------+  |
|            |                           |                           |              |
|            +---------------------------+---------------------------+              |
|                                        |                                          |
|                                        v                                          |
|  +-----------------------------------------------------------------------------+  |
|  |                     OMNIDRIVE CORE RUNTIME CONTAINER                        |  |
|  |  [Layer 1: Sensor Fusion] -----> [Layer 2: JEPA World Model & Brain]       |  |
|  |  [Layer 3: Behavioral Engine] --> [Layer 4: Trajectory & Motion Planner]    |  |
|  |  [Layer 5: Vehicle Interface] --> [Layer 6: Safety Redundancy / Guardian]   |  |
|  |  [Layer 7: Telematics / Tele-Op / V2X Gateway]                             |  |
|  +-----------------------------------------------------------------------------+  |
|                                        |                                          |
|                                        v                                          |
|  +-----------------------------------------------------------------------------+  |
|  |                   TARGET HARDWARE & ACTUATION ENGINE                        |  |
|  |  - NVIDIA DRIVE AGX Orin (Dual SoC) / Jetson AGX Orin 64GB                   |  |
|  |  - Real-Time POSIX OS (QNX / Ubuntu 22.04 LTS RT Kernel PREEMPT_RT)           |  |
|  +-----------------------------------------------------------------------------+  |
+-----------------------------------------------------------------------------------+
```

### 1.1 Multi-Domain Comparison Matrix

The table below outlines the core operational, architectural, and security differences between the three supported deployment modes:

| Operational Parameter | Tactical Military Deployment (`military`) | Heavy Freight Truck Deployment (`truck`) | Urban RobotTaxi Deployment (`robotaxi`) |
| :--- | :--- | :--- | :--- |
| **Primary Network Connectivity** | Completely Air-Gapped (Zero Cellular/Internet) | Dual-SIM 5G/LTE + Satellite (Starlink) | Multi-Carrier 5G/LTE + V2X C-V2X |
| **Actuation & Bus Protocol** | JAUS (AS4869/AS5669) over Encrypted Ethernet | SAE J1939 CAN-FD / Commercial DBW | Automotive CAN-FD / Dataspeed DBW Kit |
| **Security & Key Management** | TPM 2.0 / Hardware HSM, AES-256 Weights | PKI / TLS 1.3 mTLS Fleet Certificates | OAuth 2.0 / JWT Cloud Gateway Token |
| **Map & Routing Source** | Classified DTED/CADRG Offline Maps | Commercial HD Vector Maps (HERE/TomTom) | Centrally Updated Dynamic HD Maps |
| **Safety Integrity Standard** | MIL-STD-882E / MIL-STD-810H Ruggedization | ISO 26262 ASIL-D / FMVSS Heavy Vehicle | ISO 26262 ASIL-D / SAE J3016 Level 4 |
| **Passenger / User HMI** | Tactical Rugged Display / Operator Handset | Driver Telematics Tablet / Fleet Screen | In-Cabin Passenger Touch Screen + App |
| **Regulatory Compliance** | DoD Directive 3000.09 Autonomous Weapons | FMCSA / DOT / Weigh Station Bypass | NHTSA AV-TEST / State DOT Taxi Permit |

---

## 2. Pre-Deployment Verification Checklist

Prior to initiating software deployment on any physical vehicle, the field deployment engineer must execute and sign off on the following four-stage verification protocol:

```
+-----------------------------------------------------------------------------------+
|                        PRE-DEPLOYMENT VERIFICATION WORKFLOW                       |
+-----------------------------------------------------------------------------------+
|  Stage 1: Hardware Integrity & Power Check                                        |
|  --> Inspect 12V/24V Power Rails, E-Stop Physical Loop, CAN Termination (120 Ohm) |
|                                        |                                          |
|  Stage 2: Software Image & Hash Audit                                             |
|  --> SHA-256 Digest Validation, Container Cosign Signature Audit, TRT Engine Check|
|                                        |                                          |
|  Stage 3: Sensor Alignment & Spatial Extrinsics Audit                             |
|  --> Camera-LiDAR Calibration Residual Check (<1.5px), IMU Baseline Vector Check   |
|                                        |                                          |
|  Stage 4: Cold-Start Drive-by-Wire Handshake                                      |
|  --> Actuator Sweep Test, Mechanical Brake Override Test, Watchdog Timeout Check |
+-----------------------------------------------------------------------------------+
```

### 2.1 Pre-Flight Check Command Sequence

Run the automated system diagnostic tool prior to container boot:

```bash
# Step 1: Validate Hardware Interfaces & CAN Bus Communication
sudo omnidrive-diag --check-hardware --can-interface can0,can1 --baud 500000

# Step 2: Validate Cryptographic Hash Signatures of Deployment Artifacts
sha256sum -c /opt/omnidrive/checksums.sha256

# Step 3: Verify GPU TensorRT Engine Integrity
python3 -m omnidrive.tools.engine_verifier --engine-dir /opt/omnidrive/models/trt_engines/

# Step 4: Run Cold-Start DBW Actuator Test (Vehicle must be on jack stands or clear area)
sudo omnidrive-dbw-test --mode cold-start --max-steering-angle 5.0 --brake-pressure 100.0
```

---

## 3. Military Deployment (`deployment/military/`)

### 3.1 Air-Gapped Network Architecture

The military operational profile requires complete isolation from external RF signatures and public telecommunications infrastructure to prevent electronic warfare (EW) exploitation, RF tracking, and remote interception.

- **Physical Disconnection:** All cellular (LTE/5G), Wi-Fi (IEEE 802.11), and Bluetooth hardware modules are physically removed or permanently disabled at the kernel level via devicetree bindings.
- **Ethernet Unidirectional Diode:** Data export to onboard flight recorders uses hardware optical data diodes permitting transmit-only (`TX-only`) communication.
- **Kernel Hardening:**
  ```bash
  # Disable network wireless subsystems in Linux Kernel
  sudo modprobe -r iwlwifi btusb rfcomm
  echo "blacklist iwlwifi" | sudo tee -a /etc/modprobe.d/blacklist-military.conf
  echo "blacklist btusb" | sudo tee -a /etc/modprobe.d/blacklist-military.conf
  ```

### 3.2 Encrypted Model Weights & Key Management

All neural network weights (JEPA World Model, Vision Transformers, RL Controller policy networks) are stored encrypted at rest using **AES-256-GCM**.

```
+-----------------------------------------------------------------------------------+
|                     TPM 2.0 HARDWARE MODEL UNWRAP PROCESS                         |
+-----------------------------------------------------------------------------------+
|  Encrypted Weight Artifact (.bin.enc)                                             |
|        |                                                                          |
|        v                                                                          |
|  TPM 2.0 Security Chip (PCR 0-7 Validation) ---> [Key Release Approved?]          |
|                                                              | Yes                |
|                                                              v                    |
|  AES-256 Symmetric Key Unwrapped in Secure Memory Ring                            |
|        |                                                                          |
|        v                                                                          |
|  In-Memory Decryption directly into CUDA Unified Locked Memory (mlock)           |
+-----------------------------------------------------------------------------------+
```

#### Decryption & Initialization Script Fragment:
```python
import os
import tpm2_pytss
from Cryptodome.Cipher import AES

def load_encrypted_jepa_weights(encrypted_path: str, tpm_key_handle: int) -> bytes:
    """Unwraps model decryption key via TPM 2.0 PCR policies and decrypts weight buffer."""
    tpm_context = tpm2_pytss.TPMContext()
    # Ensure PCR states match expected secure boot hashes
    pcr_digest = tpm_context.pcr_read(pcr_selection="sha256:0,1,2,3,7")
    
    # Decrypt AES-256 Key using TPM
    raw_aes_key = tpm_context.unseal(tpm_key_handle)
    
    with open(encrypted_path, "rb") as f:
        nonce = f.read(16)
        tag = f.read(16)
        ciphertext = f.read()
        
    cipher = AES.new(raw_aes_key, AES.MODE_GCM, nonce=nonce)
    decrypted_weights = cipher.decrypt_and_verify(ciphertext, tag)
    
    # Memory-lock payload to prevent swapping to disk
    tpm2_pytss.mlock(decrypted_weights)
    return decrypted_weights
```

### 3.3 Secure Boot & Measured Execution

1. **UEFI Secure Boot:** Signed kernel (`vmlinuz-omnidrive-rt`), initramfs, and Devicetree Blob (DTB) using custom military PKCS#12 keys stored in PK-DB.
2. **DM-Verity RootFS:** Read-only root file system verified block-by-block against a signed cryptographic tree root.
3. **TPM PCR Measurements:**
   - `PCR 0`: Core UEFI firmware code
   - `PCR 4`: Boot Loader & Kernel command line
   - `PCR 7`: Secure Boot state and DB key certificates

### 3.4 JAUS (Joint Architecture for Unmanned Systems) Configuration

The military deployment communicates with higher-tier C4ISR systems via the SAE AS4869 / AS5669 **JAUS** standard over encrypted Ethernet.

```
+-----------------------------------------------------------------------------------+
|                         JAUS NETWORK MESSAGING TOPOLOGY                           |
+-----------------------------------------------------------------------------------+
|  Tactical Command Unit (Subsystem ID: 100)                                        |
|        |                                                                          |
|        | (JAUS AS5669 Transport over Encrypted UDP 3794)                         |
|        v                                                                          |
|  OMNIDRIVE JAUS Bridge Node (Subsystem ID: 200, Node ID: 1, Component ID: 10)     |
|        |                                                                          |
|        +---> Query Status (Msg ID: 0x0002) -----> Report Status (Msg ID: 0x4002)  |
|        +---> Set Element  (Msg ID: 0x0404) -----> Execute Path  (Msg ID: 0x040A)  |
|        +---> Set Emergency(Msg ID: 0x000E) -----> Safe Mechanical Stop Instant    |
+-----------------------------------------------------------------------------------+
```

#### JAUS Node Configuration File (`deployment/military/jaus_config.yaml`):
```yaml
jaus:
  subsystem_id: 200
  node_id: 1
  component_id: 10
  transport: "UDP"
  port: 3794
  multicast_address: "239.255.0.1"
  heartbeat_frequency_hz: 10.0
  supported_services:
    - "urn:jaus:jts:Core:Transport"
    - "urn:jaus:jts:Core:Events"
    - "urn:jaus:jts:Mobility:WayPointDriver"
    - "urn:jaus:jts:Mobility:GlobalVectorDriver"
  access_control:
    authority_level: 255
    timeout_seconds: 2.0
```

### 3.5 `launch_military.sh` Walkthrough

```bash
#!/usr/bin/env bash
# OMNIDRIVE Tactical Military Launch Script
set -eo pipefail

echo "[+] Initializing OMNIDRIVE Tactical Military Environment..."

# 1. Verify Root Privileges & Air-Gap Network Status
if [ "$EUID" -ne 0 ]; then
  echo "[-] ERROR: Must run as root for real-time thread pinning and hardware access."
  exit 1
fi

# Disable non-military network interfaces
ip link set dev wlan0 down 2>/dev/null || true
ip link set dev wwan0 down 2>/dev/null || true

# 2. Unseal Model Keys from TPM 2.0
echo "[+] Accessing TPM 2.0 Security Subsystem..."
tpm2_startup -c || true
tpm2_pcrread sha256:0,1,2,3,7 > /tmp/pcr_state.txt
python3 -m omnidrive.security.tpm_unseal --pcr-file /tmp/pcr_state.txt --output /dev/shm/key.bin

# 3. Mount Encrypted Tactical Maps (DTED Level 4)
echo "[+] Decrypting & Mounting Tactical Map Storage..."
cryptsetup open /dev/nvme0n1p3 tactical_maps_decrypted --key-file /dev/shm/key.bin
mount -t ext4 -o ro /dev/mapper/tactical_maps_decrypted /mnt/tactical_maps

# Clean up memory key immediately
rm -f /dev/shm/key.bin

# 4. Set Kernel Real-Time Scheduling & CPU Affinity
# Pin ROS 2 Perception to Cores 0-3, JEPA Brain to Cores 4-7, Safety Loop to Core 8
echo "[+] Configuring CPU Core Isolation and PREEMPT_RT Priorities..."
sysctl -w kernel.sched_rt_runtime_us=-1

# 5. Launch Air-Gapped Docker Container Stack
echo "[+] Launching Military Container Stack..."
docker compose -f /opt/omnidrive/deployment/military/docker-compose.military.yml up -d

# 6. Verify System Startup & JAUS Handshake
echo "[+] Awaiting JAUS Controller Handshake..."
python3 -m omnidrive.tools.jaus_health_check --timeout 15

echo "[+] OMNIDRIVE Tactical Military Mission Execution Ready."
```

### 3.6 MIL-STD Hardening Requirements

- **MIL-STD-810H Compliance:** Operating temperature range \(-40^\circ\text{C}\) to \(+85^\circ\text{C}\), ballistic vibration profiles (Method 514.8), functional shock (50G 11ms), and 95% non-condensing humidity tolerance.
- **MIL-STD-461G EMI/EMC:** Shielded IP67 aluminum chassis with MIL-DTL-38999 connectors protecting internal CAN/Ethernet lines against high-altitude electromagnetic pulses (HEMP) and radar interference.

### 3.7 Classified Tactical Map Loading Procedure

Tactical military maps are ingested via offline encrypted media drives:

1. Insert USB/SATA encrypted storage cartridge containing DTED-4 / CADRG format vector tiles.
2. Provide operator multi-factor authentication card (CAC/PIV) + PIN.
3. Validate digital signature of map package:
   ```bash
   openssl dgst -sha256 -verify /etc/omnidrive/keys/tactical_map_pubkey.pem \
     -signature /media/tactical_cartridge/map_manifest.sig \
     /media/tactical_cartridge/map_manifest.json
   ```
4. Copy raster elevation and obstacle maps into volatile ramdisk (`/dev/shm/tactical_map_cache`).

---

## 4. Heavy Freight Truck Deployment (`deployment/truck/`)

### 4.1 Heavy Vehicle Profiles

OMNIDRIVE supports three primary commercial Class 8 truck chassis profiles. Differences in vehicle kinematics and dynamics are parametrized in profile YAMLs:

```
+-----------------------------------------------------------------------------------+
|                        CLASS 8 TRUCK VEHICLE PROFILES                             |
+-----------------------------------------------------------------------------------+
|  Freightliner Cascadia        Volvo VNL 860                  Kenworth T680        |
|  - Wheelbase: 6.2m            - Wheelbase: 5.8m              - Wheelbase: 6.0m    |
|  - Pneumatic Brake Delay: 0.35s- Pneumatic Brake Delay: 0.28s- Pneumatic Delay: 0.32s|
|  - Engine Compression: 450HP  - Engine Brake: Volvo VEB+     - Engine Brake: PACCAR|
|  - Hitch Angle Max: 82 deg    - Hitch Angle Max: 85 deg      - Hitch Angle: 80 deg|
+-----------------------------------------------------------------------------------+
```

#### Comparison of Heavy Vehicle Parameters:

| Specification Parameter | Freightliner Cascadia | Volvo VNL 860 | Kenworth T680 |
| :--- | :--- | :--- | :--- |
| **Gross Vehicle Weight (GVWR)** | 80,000 lbs (36,287 kg) | 80,000 lbs (36,287 kg) | 80,000 lbs (36,287 kg) |
| **Pneumatic Brake Lag (\(\tau_{brake}\))** | 350 ms | 280 ms | 320 ms |
| **Compression Engine Brake Power** | Detroit DT12 Jake Brake (450 HP)| Volvo VEB+ Engine Brake (500 HP)| PACCAR MX-13 Engine Brake (470 HP)|
| **Fifth-Wheel Kingpin Offset** | +0.45 m forward of tandem | +0.40 m forward of tandem | +0.42 m forward of tandem |
| **Minimum Turning Radius** | 12.8 m | 12.2 m | 12.5 m |
| **J1939 Primary Channel** | CAN0 (500 kbps) | CAN0 (500 kbps) / J1939 FD | CAN0 (500 kbps) |

### 4.2 SAE J1939 Bus Connection Procedure

Class 8 trucks use the **SAE J1939** protocol over CAN-FD for engine, transmission, brake, and retarder actuation.

```
+-----------------------------------------------------------------------------------+
|                         SAE J1939 CAN-FD INTERFACE DIAGRAM                        |
+-----------------------------------------------------------------------------------+
|  Vehicle Deutsch HD10 9-Pin Connector                                             |
|  [Pin A: GND | Pin B: V+ | Pin C: CAN-High | Pin D: CAN-Low | Pin J: J1939-FD High]  |
|                                        |                                          |
|                                        v                                          |
|  Vector VN1640A / Kvaser Leaf Light Transceiver                                  |
|                                        |                                          |
|                                        v                                          |
|  Linux SocketCAN Driver Interface (`can0`, `can1`)                               |
|                                        |                                          |
|                                        v                                          |
|  OMNIDRIVE J1939 Interface Daemon (`src/vehicle_interface/j1939_bridge.cpp`)      |
|  - Engine Torque Control: PGN 61444 (EEC1)                                        |
|  - Electronic Brake Control: PGN 61441 (EBC1)                                     |
|  - Engine Retarder Control: PGN 61440 (ERC1)                                      |
+-----------------------------------------------------------------------------------+
```

#### SocketCAN J1939 Setup Command Sequence:
```bash
# Configure socketcan for J1939 with 500kbps nominal baud rate
sudo ip link set can0 type can bitrate 500000 dbitrate 2000000 fd on restart-ms 100
sudo ip link set can0 up
sudo ip link set j1939-can0 type j1939
sudo ip link set j1939-can0 up
```

### 4.3 `launch_truck.sh` Walkthrough

```bash
#!/usr/bin/env bash
# OMNIDRIVE Autonomous Heavy Truck Launch Script
set -eo pipefail

TRUCK_PROFILE=${1:-"freightliner_cascadia"}
echo "[+] Starting OMNIDRIVE Fleet Stack for Vehicle Profile: ${TRUCK_PROFILE}"

# 1. Load CAN / J1939 Kernel Modules
sudo modprobe can
sudo modprobe can_raw
sudo modprobe can_j1939

# 2. Initialize Dual CAN Interfaces
sudo ip link set can0 type can bitrate 500000 restart-ms 100
sudo ip link set can0 up
sudo ip link set can1 type can bitrate 500000 restart-ms 100
sudo ip link set can1 up

# 3. Export Environment Variables
export OMNIDRIVE_VEHICLE_TYPE="truck"
export OMNIDRIVE_PROFILE="${TRUCK_PROFILE}"
export VEHICLE_CONFIG_PATH="/opt/omnidrive/configs/truck/${TRUCK_PROFILE}.yaml"

# 4. Launch Heavy Truck Docker Compose Stack
docker compose -f /opt/omnidrive/deployment/truck/docker-compose.yml up -d

# 5. Initialize Commercial Fleet Telemetry Daemon
python3 -m omnidrive.telemetry.fleet_gateway --profile ${TRUCK_PROFILE} --server mqtts://fleet.omnidrive.ai:8883 &

echo "[+] OMNIDRIVE Freight Truck Deployment Active."
```

### 4.4 Weigh Station Compliance Settings

Commercial autonomous freight trucks must comply with DOT Weigh-In-Motion (WIM) and weigh station bypass systems (Drivewyze / PrePass).

```yaml
weigh_station:
  bypass_system_enabled: true
  provider: "DRIVEWYZE"
  transponder_id: "WIM-998234-TX"
  wim_approach_distance_m: 1500.0
  auto_lane_change_to_weigh_lane: true
  max_axle_load_lbs:
    steer_axle: 12000
    drive_tandem: 34000
    trailer_tandem: 34000
  speed_limit_scale_kmh: 48.0
```

### 4.5 Commercial Fleet Management API

Remote monitoring and command interface uses secured MQTT/gRPC endpoints.

#### Sample Telemetry Payload (`POST /api/v2/fleet/telemetry`):
```json
{
  "vin": "1FUJGLDR8NLBP1092",
  "timestamp_utc": 1786326700.123,
  "vehicle_mode": "AUTONOMOUS_ENGAGED",
  "pose": {
    "latitude": 32.7767,
    "longitude": -96.7970,
    "heading_deg": 184.2,
    "speed_mps": 28.5
  },
  "diagnostics": {
    "engine_rpm": 1250,
    "trailer_hitch_angle_deg": 0.4,
    "pneumatic_brake_pressure_psi": 115.0,
    "j1939_bus_load_percent": 34.2
  },
  "jepa_brain_status": {
    "world_model_loss": 0.0142,
    "prediction_horizon_sec": 5.0,
    "confidence_score": 0.992
  }
}
```

---

## 5. RobotTaxi Deployment (`deployment/robotaxi/`)

### 5.1 Urban Vehicle Profiles

OMNIDRIVE supports high-efficiency electric and hybrid urban passenger vehicles:

| Parameter | Toyota Camry Hybrid | Hyundai IONIQ 5 EV | Nissan Leaf EV |
| :--- | :--- | :--- | :--- |
| **Powertrain** | 2.5L 4-Cyl Hybrid | Dual-Motor AWD Electric | Single-Motor FWD Electric |
| **DBW Provider** | Dataspeed DBW Kit | AutonomouStuff DBW | Dataspeed DBW Kit |
| **Auxiliary 12V Power Capacity** | 100A DC-DC Converter | 180A High-Voltage DC-DC | 90A Auxiliary Battery |
| **Turn Circle Diameter** | 11.4 m | 10.8 m | 10.6 m |
| **Steering Torque Limit** | 8.5 Nm | 10.0 Nm | 7.5 Nm |

### 5.2 Drive-By-Wire (DBW) Installation & Calibration

1. **Hardware Splice / CAN Tap:** Connect Dataspeed DBW controller directly to the OEM steering column CAN, brake pedal position sensor, and electronic throttle controller.
2. **Steering Angle Calibration Polynomial:**
   Map target wheel angle \(\delta_{\text{wheel}}\) to OEM steering wheel angle \(\delta_{\text{column}}\):
   $$\delta_{\text{column}} = 14.82 \cdot \delta_{\text{wheel}} + 0.003 \cdot \delta_{\text{wheel}}^2$$
3. **Override Detection:** Mechanical driver override triggers if driver steering torque exceeds \(3.5\text{ Nm}\) or brake pedal pressure exceeds \(15\text{ PSI}\).

### 5.3 Passenger App Integration & Cabin HMI

```
+-----------------------------------------------------------------------------------+
|                        ROBOTAXI PASSENGER APP WORKFLOW                            |
+-----------------------------------------------------------------------------------+
| Rider App (iOS/Android) ---> REST/gRPC Cloud Gateway                              |
|                                   |                                               |
|                                   v                                               |
| Rider Dispatch Request (Pickup / Dropoff Lat-Lon)                                |
|                                   |                                               |
|                                   v                                               |
| RobotTaxi Onboard Telematics Bridge (`hmi_bridge_node`)                           |
|                                   |                                               |
|                                   v                                               |
| In-Cabin Display Screen (ROS 2 HMI Gateway)                                       |
| - Shows live BEV trajectory visualizer                                            |
| - Buttons: "Start Ride", "Pull Over Now", "Contact Support"                       |
+-----------------------------------------------------------------------------------+
```

### 5.4 Geofencing Configuration (`geofence_config.json`)

```json
{
  "geofence_id": "SAN_FRANCISCO_DOWNTOWN_V4",
  "bounding_polygon": [
    {"lat": 37.7892, "lon": -122.4014},
    {"lat": 37.7954, "lon": -122.3932},
    {"lat": 37.7781, "lon": -122.3881},
    {"lat": 37.7712, "lon": -122.3995}
  ],
  "speed_limits_default_kmh": 40.0,
  "forbidden_zones": [
    {
      "name": "Pedestrian_Mall_Market_St",
      "polygon": [
        {"lat": 37.7851, "lon": -122.4062},
        {"lat": 37.7865, "lon": -122.4041}
      ]
    }
  ],
  "odd_violation_action": "SAFE_PULLOVER_IMMEDIATE"
}
```

### 5.5 `launch_robotaxi.sh` Walkthrough

```bash
#!/usr/bin/env bash
# OMNIDRIVE Urban RobotTaxi Launch Script
set -eo pipefail

PROFILE=${1:-"hyundai_ioniq5"}
echo "[+] Initializing OMNIDRIVE Urban RobotTaxi Stack [Profile: ${PROFILE}]..."

# 1. Bring up Dataspeed DBW CAN bus
sudo ip link set can0 type can bitrate 500000 dbitrate 2000000 fd on
sudo ip link set can0 up

# 2. Launch Passenger HMI Node and Cabin Occupancy Sensor
python3 -m omnidrive.hmi.cabin_monitor --camera-id /dev/video4 &
python3 -m omnidrive.hmi.passenger_screen_server --port 8080 &

# 3. Boot Containerized Stack
docker compose -f /opt/omnidrive/deployment/robotaxi/docker-compose.yml up -d

# 4. Perform DBW System Handshake Check
python3 -m omnidrive.tools.dbw_handshake --profile ${PROFILE}

echo "[+] RobotTaxi Station Ready for Dispatch."
```

---

## 6. Docker Deployment

### 6.1 Production `docker-compose.yml`

```yaml
version: '3.8'

services:
  sensor_fusion:
    image: omnidrive/sensor-fusion:2.4.0
    container_name: omnidrive_sensor_fusion
    restart: always
    network_mode: host
    ipc: host
    privileged: true
    devices:
      - "/dev/bus/usb:/dev/bus/usb"
    volumes:
      - /opt/omnidrive/configs:/etc/omnidrive
      - /tmp/nvmm_shm:/tmp/nvmm_shm
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu, video]
    ulimits:
      rtprio: 99
      memlock: -1

  jepa_brain:
    image: omnidrive/jepa-brain:2.4.0
    container_name: omnidrive_jepa_brain
    restart: always
    network_mode: host
    ipc: host
    volumes:
      - /opt/omnidrive/models:/opt/omnidrive/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    ulimits:
      rtprio: 95
      memlock: -1

  vehicle_interface:
    image: omnidrive/vehicle-interface:2.4.0
    container_name: omnidrive_vehicle_interface
    restart: always
    network_mode: host
    privileged: true
    volumes:
      - /dev:/dev
    ulimits:
      rtprio: 99
```

### 6.2 Air-Gapped Tactical `docker-compose.military.yml`

```yaml
version: '3.8'

services:
  military_core:
    image: omnidrive/military-core:2.4.0-airgapped
    container_name: omnidrive_military_core
    read_only: true
    network_mode: none
    ipc: host
    cap_add:
      - SYS_NICE
      - IPC_LOCK
      - SYS_RAWIO
    volumes:
      - type: tmpfs
        target: /tmp
        tmpfs:
          size: 4096m
      - /mnt/tactical_maps:/mnt/tactical_maps:ro
      - /dev/shm:/dev/shm
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    ulimits:
      rtprio: 99
      memlock: -1
```

---

## 7. Over-the-Air (OTA) Updates

### 7.1 A/B Dual Partition Update Framework

The OTA system utilizes an **A/B Dual-Partition architecture** managed by `RAUC` (Robust Auto-Update Controller) to ensure fail-safe updates across commercial truck and taxi fleets.

```
+-----------------------------------------------------------------------------------+
|                        OTA A/B PARTITION ROLLOVER ARCHITECTURE                    |
+-----------------------------------------------------------------------------------+
|  Slot A (Active System v2.4.0)           Slot B (Inactive / Target Partition)     |
|  [RootFS A: Active Boot]                 [RootFS B: Flashing v2.5.0 Image]        |
|                                                        |                          |
|                                                        v                          |
|  1. Ed25519 Cryptographic Verification of Bundle Metadata                         |
|  2. Flash Signed Squashed Image to Slot B                                         |
|  3. Set Bootloader (U-Boot/GRUB) default flag to Slot B                           |
|  4. Cold Warm Reboot -> Execute Automated Diagnostic Health Audit                |
|  5. If Health Audit Passes  --> Commit Slot B as Active                           |
|  6. If Health Audit Fails   --> Fallback to Slot A instantly                      |
+-----------------------------------------------------------------------------------+
```

### 7.2 OTA Client Update Script Execution
```bash
# Trigger OTA Bundle Installation
rauc install /var/downloads/omnidrive-update-v2.5.0.raucb

# Query Status
rauc status
```

---

## 8. Monitoring & Telemetry

### 8.1 Prometheus Metrics Endpoint

The telemetry gateway exposes system runtime metrics at `http://localhost:9090/metrics`:

```ini
# HELP omnidrive_jepa_inference_latency_ms Latency of JEPA world model prediction
# TYPE omnidrive_jepa_inference_latency_ms gauge
omnidrive_jepa_inference_latency_ms{gpu="0"} 12.4

# HELP omnidrive_sensor_frame_drop_count Cumulative dropped frames per sensor
# TYPE omnidrive_sensor_frame_drop_count counter
omnidrive_sensor_frame_drop_count{sensor_id="cam_front_center"} 0
omnidrive_sensor_frame_drop_count{sensor_id="lidar_top"} 0

# HELP omnidrive_safety_guardian_status Current status of safety guardian state machine
# TYPE omnidrive_safety_guardian_status gauge
omnidrive_safety_guardian_status 1.0
```

### 8.2 Live Fleet Management Dashboard Architecture

```
[Onboard GPU Workstation] 
       | (Protobuf over TLS 1.3 / gRPC)
       v
[Cloud Fleet Server / InfluxDB] ----> [Grafana Operations Dashboard]
                                      - Real-time vehicle positions
                                      - Battery / Fuel State
                                      - Disengagement Alerts
```

---

## 9. Rollback Procedure

In the event of a critical software defect, sensor synchronization regression, or disengagement anomaly post-deployment, run the automated zero-downtime rollback procedure:

```bash
#!/usr/bin/env bash
# OMNIDRIVE System Rollback Command
set -eo pipefail

echo "[!] CRITICAL: Initiating OMNIDRIVE System Rollback to Last Known Good Version..."

# 1. Trigger Vehicle Safety Guardian Emergency Stop Hold
python3 -m omnidrive.safety.trigger_safe_stop --reason "OPERATOR_INITIATED_ROLLBACK"

# 2. Revert Docker Container Tag to Previous Release
docker compose -f /opt/omnidrive/deployment/docker-compose.yml down
ln -sfn /opt/omnidrive/releases/v2.3.9 /opt/omnidrive/current

# 3. Revert A/B Bootloader Partition if OTA was recently flashed
if command -v rauc &> /dev/null; then
  rauc status mark-bad
fi

# 4. Restart Services
docker compose -f /opt/omnidrive/deployment/docker-compose.yml up -d

echo "[+] Rollback complete. System reverted to v2.3.9."
```

---

## 10. Deployment Configuration YAML Specifications (`deployment_config.yaml`)

Below is the complete reference schema and field description for `deployment_config.yaml`:

```yaml
# OMNIDRIVE Master Deployment Configuration Specification
version: "2.4.0"

deployment:
  mode: "truck"                       # Options: military | truck | robotaxi
  target_hardware: "DRIVE_AGX_ORIN"   # Options: DRIVE_AGX_ORIN | JETSON_ORIN | X86_WORKSTATION
  realtime_priority: 99               # POSIX SCHED_FIFO priority (1-99)

security:
  enable_airgap: false
  tpm_unseal_pcr_index: 7
  weight_encryption:
    enabled: true
    algorithm: "AES-256-GCM"
    key_source: "TPM2.0"

network:
  jaus:
    enabled: false
    subsystem_id: 200
    node_id: 1
    component_id: 10
  telemetry:
    enabled: true
    broker_url: "mqtts://fleet.omnidrive.ai:8883"
    publish_interval_ms: 100

vehicle:
  profile: "freightliner_cascadia"
  can_interface: "can0"
  can_protocol: "SAE_J1939"
  baud_rate: 500000

safety:
  watchdog_timeout_ms: 100
  override_torque_threshold_nm: 3.5
  emergency_brake_decel_mps2: 8.5
```

### Detailed Field Reference Table:

| YAML Key Path | Type | Allowed Values | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `deployment.mode` | String | `military`, `truck`, `robotaxi` | `robotaxi` | Determines sensor pipeline setup, CAN bus driver, and network protocols. |
| `deployment.target_hardware` | String | `DRIVE_AGX_ORIN`, `JETSON_ORIN`, `X86_WORKSTATION` | `DRIVE_AGX_ORIN` | Selects TensorRT acceleration profiles and CUDA device allocations. |
| `deployment.realtime_priority` | Integer | `1` to `99` | `99` | Sets PREEMPT_RT kernel thread scheduling priority for safety execution. |
| `security.enable_airgap` | Boolean | `true`, `false` | `false` | When true, disables all external socket connections and wireless modules. |
| `security.tpm_unseal_pcr_index` | Integer | `0` to `23` | `7` | Selects TPM 2.0 PCR register for secure unsealing of model encryption keys. |
| `network.jaus.enabled` | Boolean | `true`, `false` | `false` | Enables JAUS AS4869/AS5669 military protocol bridge. |
| `vehicle.can_protocol` | String | `SAE_J1939`, `DATASPEED_CAN`, `AUTONOMOUSTUFF_CAN` | `SAE_J1939` | Protocol parser used by Layer 5 (Vehicle Interface Module). |
| `safety.watchdog_timeout_ms` | Integer | `10` to `500` | `100` | Hardware watchdog timeout. Triggers emergency mechanical brake if heartbeat missed. |
