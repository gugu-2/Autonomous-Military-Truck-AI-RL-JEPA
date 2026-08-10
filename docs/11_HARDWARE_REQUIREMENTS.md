# Layer 11: OMNIDRIVE Hardware Requirements & Physical Specifications
**OMNIDRIVE Autonomous Driving AI System**  
**Document Version:** 2.4.0  
**Target Platforms:** Tactical Military Vehicles (GCV/UGV), Heavy Freight Trucks (Class 8), Urban RobotTaxis (L4 Mobility)  
**Classification:** Technical Hardware Architecture & Physical Specification  

---

## 1. Compute Hardware Platforms

The OMNIDRIVE Autonomous Driving AI System requires high-throughput, low-latency heterogeneous compute platforms capable of running dense 3D perception transformers, JEPA world models, dynamic trajectory planners, and ASIL-D safety watchdog loops concurrently under strict real-time constraints.

```
+-----------------------------------------------------------------------------------+
|                        COMPUTE HARDWARE HIERARCHY & ROLES                         |
+-----------------------------------------------------------------------------------+
|  1. NVIDIA DRIVE AGX Orin (Production Fleet Deployment - Standard)               |
|     - Dual Orin System-on-Chip (SoC) + Dual ConnectX-7 NIC                        |
|     - Primary Platform for Autonomous Operation & In-Vehicle Safety Integrity     |
|                                                                                   |
|  2. NVIDIA Jetson AGX Orin 64GB (Development, Testbeds & Light Vehicles)          |
|     - Compact 275 TOPS Industrial Board                                          |
|     - Ideal for Rapid Prototyping & Single-SoC Field Validation                   |
|                                                                                   |
|  3. High-Performance x86 Industrial Workstation (Bench Testing & HIL Simulation)  |
|     - Dual Intel Xeon / AMD EPYC + Dual NVIDIA RTX 6000 Ada / RTX 4090 GPUs        |
|     - Hardware-in-the-Loop (HIL) Simulator & Sensor Replay System                |
+-----------------------------------------------------------------------------------+
```

### 1.1 Compute Platform Comparison Table

| Technical Specification | NVIDIA DRIVE AGX Orin (Recommended Production) | NVIDIA Jetson AGX Orin 64GB (Dev / Prototype) | x86 High-Performance Industrial Workstation (HIL / Bench) |
| :--- | :--- | :--- | :--- |
| **Part Number / SKU** | `940-63710-0000-000` | `900-13701-0000-000` | Custom Rugged Industrial Rack |
| **CPU Architecture** | 12-core Arm Cortex-A78AE v8.2 64-Bit @ 2.2 GHz (Dual SoC = 24 Cores) | 12-core Arm Cortex-A78AE v8.2 64-Bit @ 2.2 GHz | Intel Xeon W-1390P (8-Core / 16-Thread @ 5.3 GHz) |
| **System RAM** | 64GB LPDDR5 per SoC (128GB Total @ 409.6 GB/s) | 64GB LPDDR5 (204.8 GB/s) | 128GB DDR4-3200 ECC Unbuffered RAM |
| **GPU Architecture** | 2x NVIDIA Ampere GPUs (2048 CUDA Cores + 64 Tensor Cores each) | 1x NVIDIA Ampere GPU (2048 CUDA Cores + 64 Tensor Cores) | 2x NVIDIA RTX 4090 24GB (16,384 CUDA Cores each) |
| **Deep Learning Accelerators**| 4x DLA 2.0 Engines (Dual SoC) | 2x DLA 2.0 Engines | N/A (GPU CUDA Parallel Execution) |
| **Peak AI Performance** | **500 TOPS (INT8)** / 275 TFLOPS (FP16) | **275 TOPS (INT8)** / 138 TFLOPS (FP16) | **1,650 TFLOPS (FP16)** / 3,300 TOPS (INT8) |
| **GPU VRAM** | Unified 64GB LPDDR5 per SoC | Unified 64GB LPDDR5 | 2x 24GB GDDR6X Dedicated VRAM (48GB Total) |
| **Storage Subsystem** | 2TB NVMe M.2 PCIe Gen4 x4 Solid State Drive | 64GB eMMC 5.1 + 2TB NVMe M.2 SSD | 4TB NVMe Enterprise PCIe Gen4 SSD (RAID-1) |
| **I/O Connectivity** | 16x GMSL2 Camera inputs, 4x 10GbE, 2x CAN-FD | 16x MIPI CSI-2, 2x 10GbE, 2x CAN-FD | 8x PCIe Gen4 slots, 4x 10GbE, USB 3.2 Gen2 |
| **Power Consumption** | **130W** (Max Peak) / 90W (Nominal) | **60W** (Max Peak) / 40W (Nominal) | **850W** (Max Peak) / 650W (Nominal) |
| **Operating Temperature** | \(-40^\circ\text{C}\) to \(+85^\circ\text{C}\) (ASIL-D Automotive) | \(-25^\circ\text{C}\) to \(+80^\circ\text{C}\) Industrial | \(0^\circ\text{C}\) to \(+50^\circ\text{C}\) Commercial |
| **Target Vehicle Domain** | Tactical Military & Commercial Heavy Truck | Prototype RobotTaxi & Edge Testbeds | Hardware-in-the-Loop Simulation Rig |

---

## 2. Sensor Suite Architecture Per Vehicle Type

OMNIDRIVE implements specialized sensor topologies customized for the unique operational design domains (ODDs) of military vehicles, freight trucks, and robotaxis.

```
+-----------------------------------------------------------------------------------+
|                        SENSOR SUITE OVERVIEW PER DOMAIN                           |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  TACTICAL MILITARY VEHICLE         CLASS 8 FREIGHT TRUCK          URBAN ROBOTAXI   |
|  - 12x RGB Cameras                 - 8x RGB Cameras               - 8x RGB Cameras |
|  - 2x LWIR Thermal Cameras         - 1x 128-Beam 360 LiDAR       - 1x 360 LiDAR   |
|  - 2x Solid-State LiDARs           - 2x FMCW Long-Range RADARs    - 1x 4D RADAR    |
|  - 4x 4D Imaging RADARs            - 1x RTK-GNSS + Tactical IMU   - 1x RTK-GNSS    |
|  - Dual-Antenna RTK-GNSS + IMU     - 1x Trailer Hitch Angle Sensor                 |
+-----------------------------------------------------------------------------------+
```

### 2.1 Domain Sensor Topology Specifications

| Sensor Category | Tactical Military Vehicle (`military`) | Heavy Freight Truck (`truck`) | Urban RobotTaxi (`robotaxi`) |
| :--- | :--- | :--- | :--- |
| **Visible Spectrum Cameras** | 12x Basler ace2 GMSL2 (Front 120°, Side 90°, Rear 120°, Telephoto 30°) | 8x Basler ace2 GMSL2 (Front Long-Tele 15°, Front Mid 60°, Side Wide, Rear) | 8x Basler ace2 GMSL2 (Surround 360° Ring + Interior Cabin Camera) |
| **Long-Wave Thermal IR (LWIR)**| 2x FLIR Boson+ (640x512, 60 FPS, Night/Smoke Vision) | None (Optional Upgrade for Night Highway) | None |
| **3D LiDAR Subsystem** | 2x Ouster OS1-128 (Roof Dual Pod, Armored Mounts) | 1x Ouster OS1-128 (Cab Roof Top Center Mount) | 1x Velodyne VLP-32C / Ouster OS1-64 (Roof Center) |
| **Millimeter-Wave RADAR** | 4x Continental ARS548 4D RADAR (Front, Rear, Left, Right) | 2x Continental ARS548 4D RADAR (Front Grille + Trailer Rear) | 1x Bosch MRR3 / Continental ARS548 (Front Bumper) |
| **GNSS & Navigation** | Septentrio AsteRx-i D (Dual Antenna RTK + Tactical IMU) | Septentrio AsteRx-i D (RTK-GNSS + IMU Combo) | Septentrio AsteRx-i D / u-blox ZED-F9P |
| **Primary Physical Interface** | MIL-DTL-38999 Shielded Circular Connectors | M12 Industrial Ethernet + Deutsch HD Connector | FAKRA Z-Coded Cables + OBD-II Harness |

---

## 3. Camera Subsystem Specifications

### 3.1 Visible Spectrum Camera: Basler ace2 (`a2A1920-51gcBAS` / `a2A3840-45gucBAS`)

- **Sensor Type:** Sony Pregius S Global Shutter CMOS (`IMX541` / `IMX545`)
- **Native Resolution:** 1920 x 1200 (2.3 MP) for Surround / 3840 x 2160 (8.3 MP 4K) for Long-Range Telephoto
- **Pixel Size:** 2.74 µm x 2.74 µm
- **Frame Rate:** 60 FPS continuous streaming @ uncompressed NV12 / RAW12
- **Dynamic Range (HDR):** 140 dB multi-exposure HDR (essential for tunnel exit & glare mitigation)
- **Optical Interface:** GMSL2 via FAKRA Z-Code coaxial connection / GigE Vision with IEEE 1588 PTP
- **Hardware Trigger Sync:** Opto-isolated hardware input trigger (<1 µs synchronization jitter)
- **Power Consumption:** 4.2 W per camera unit @ 12V DC input

### 3.2 Thermal Infrared Camera: FLIR Boson+ (`Boson+ 640`)

```
+-----------------------------------------------------------------------------------+
|                        FLIR BOSON+ SPECIFICATION MATRIX                           |
+-----------------------------------------------------------------------------------+
|  Detector Array: Uncooled VOx Microbolometer                                      |
|  Spectral Response: 8 µm to 14 µm (Long-Wave Infrared - LWIR)                     |
|  Thermal Sensitivity (NETD): <20 mK (Ultra-sensitive thermal distinction)         |
|  Pixel Pitch: 12 µm                                                               |
|  Array Format: 640 x 512 pixels @ 60 Hz                                           |
|  Output Interface: CMOS / MIPI CSI-2 via Military Sealed Enclosure                |
|  Power Consumption: 1.6 W @ 5V DC                                                 |
|  Environmental Protection: IP67 Sealed Germanium Lens Window                      |
+-----------------------------------------------------------------------------------+
```

---

## 4. LiDAR Models & Comparative Analysis

OMNIDRIVE supports high-density spinning and solid-state 3D LiDAR sensors. The primary choices for high-performance deployment are the **Ouster OS1-128 (Rev 7)** and **Velodyne VLP-32C (Ultra Puck)**.

```
+-----------------------------------------------------------------------------------+
|                       LIDAR ARCHITECTURAL BEAM CONFIGURATION                      |
+-----------------------------------------------------------------------------------+
|  Ouster OS1-128 (128 Uniform / Gradient Vertical Channels)                        |
|  - Beam Spacing: 0.35° Vertical Resolution                                        |
|  - Range: 90m @ 10% reflectivity / 200m @ 80%                                     |
|                                                                                   |
|  Velodyne VLP-32C (32 Non-Linear Concentrated Channels)                           |
|  - Beam Spacing: 0.33° Center Dense Zone / 1.0° Outer Zone                         |
|  - Range: 200m @ 20% reflectivity                                                 |
+-----------------------------------------------------------------------------------+
```

### 4.1 Detailed LiDAR Comparison Matrix

| Specification | Ouster OS1-128 (Rev 7) (Recommended) | Velodyne VLP-32C (Ultra Puck) |
| :--- | :--- | :--- |
| **Number of Vertical Channels** | **128 Channels** | **32 Channels** |
| **Range (10% Reflectivity)** | 90 meters | 100 meters |
| **Maximum Range (80% Reflectivity)**| 200 meters | 200 meters |
| **Precision / Accuracy** | \(\pm 1.5\text{ cm}\) | \(\pm 3.0\text{ cm}\) |
| **Vertical Field of View (V-FOV)** | \(45^\circ\) (\(-22.5^\circ\) to \(+22.5^\circ\)) | \(40^\circ\) (\(-25^\circ\) to \(+15^\circ\)) |
| **Vertical Angular Resolution** | **0.35°** (Uniform spatial distribution) | 0.33° (Center) to 1.0° (Periphery) |
| **Horizontal Angular Resolution** | 0.18° (2048 points/scan @ 10 Hz) | 0.1° to 0.4° |
| **Points Generated Per Second** | **5,242,880 pts/sec** | 1,200,000 pts/sec |
| **Data Protocol / Output** | UDP Packets / PTP IEEE 1588 v2 | UDP Packets / NMEA 0183 |
| **Data Interface** | 1000BASE-T Gigabit Ethernet | 100BASE-TX Ethernet |
| **Ingress Protection Rating** | **IP68 & IP69K** (High-pressure washdown) | IP67 |
| **Operating Power Draw** | 14 W (Nominal) / 20 W (Peak Heater) | 10 W (Nominal) / 18 W (Peak) |
| **Weight / Dimensions** | 447 grams / \(\varnothing 85\text{mm} \times 73\text{mm}\) | 925 grams / \(\varnothing 103\text{mm} \times 86\text{mm}\) |

---

## 5. RADAR Subsystem Specifications

### 5.1 Continental ARS548 4D Premium Imaging RADAR (Military & Freight Truck)

- **Operating Frequency:** 76 GHz to 77 GHz FMCW (Fast Chirp Modulation)
- **Measurement Dimensions:** 4D (Range, Azimuth Angle, Elevation Angle, Doppler Radial Velocity)
- **Maximum Detection Range:** 300 meters (Long-Range Mode) / 100 meters (Short-Range Mode)
- **Azimuth Field of View:** \(\pm 60^\circ\) (Near Field) / \(\pm 9^\circ\) (Far Field)
- **Elevation Field of View:** \(\pm 14^\circ\)
- **Range Resolution:** 0.10 meters
- **Track Capacity:** Up to 800 independent object tracks simultaneously
- **Host Interface:** Ethernet 100BASE-T1 / CAN-FD (Autosar PDU format)
- **Operating Voltage & Power:** 12V DC / 12 W nominal

### 5.2 Bosch MRR3 / MRR5 Mid-Range RADAR (Urban RobotTaxi)

- **Operating Frequency:** 76 GHz to 77 GHz
- **Maximum Detection Range:** 160 meters
- **Horizontal Field of View:** \(\pm 45^\circ\)
- **Host Interface:** Automotive CAN-FD (500 kbps / 2 Mbps)
- **Power Draw:** 4.5 W

---

## 6. High-Precision GNSS / IMU Subsystem

### 6.1 Septentrio AsteRx-i D (RTK-GNSS + Tactical IMU Combo)

```
+-----------------------------------------------------------------------------------+
|                     SEPTENTRIO ASTERX-I D ARCHITECTURE                            |
+-----------------------------------------------------------------------------------+
|  Dual GNSS Antennas (Polant-MC Antenna Pair)                                      |
|  - Tracks GPS (L1/L2/L5), GLONASS, Galileo, BeiDou, QZSS, NavIC                   |
|                                        |                                          |
|                                        v                                          |
|  Multi-Frequency RTK Core (0.6 cm Horizontal Accuracy + 1 ppm)                    |
|                                        |                                          |
|                                        v                                          |
|  Tightly-Coupled Tactical MEMS IMU                                                |
|  - Accelerometer Bias Instability: 0.02 mg                                        |
|  - Gyroscope Bias Instability: 0.25 °/hr                                          |
|  - Dual Antenna Heading Precision: 0.1° at 1.0m baseline                          |
|                                        |                                          |
|                                        v                                          |
|  Hardware Output Interface: RS-232 / USB / 100BASE-T Ethernet (NMEA + UBX)        |
+-----------------------------------------------------------------------------------+
```

---

## 7. CAN Interface Hardware

Reliable vehicle bus interfacing requires galvanic isolation and low-latency CAN-FD frame transfer to the Linux Kernel SocketCAN interface.

| Interface Model | Manufacturer | Channel Count | Protocol Support | Isolation Rating | Operating Temp Range | Linux Driver |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **PEAK PCAN-USB Pro FD** | PEAK-System | 2x CAN-FD + 2x LIN | CAN 2.0A/B & CAN-FD (8 Mbps) | 500 V Galvanic | \(-40^\circ\text{C}\) to \(+85^\circ\text{C}\) | `peak_usb` (Native Kernel) |
| **Kvaser Leaf Light v2** | Kvaser AB | 1x CAN Channel | CAN 2.0A/B & CAN-FD | 1000 V Galvanic | \(-40^\circ\text{C}\) to \(+85^\circ\text{C}\) | `kvaser_usb` |
| **Vector VN1640A** | Vector Informatik | 4x CAN-FD / LIN | High-Speed CAN-FD | 1500 V Galvanic | \(-40^\circ\text{C}\) to \(+80^\circ\text{C}\) | Vector XL Driver Library |

---

## 8. Wiring, Electrical System & Vehicle Power Budget

### 8.1 System Electrical Power Draw Distribution

```
+-----------------------------------------------------------------------------------+
|                        SYSTEM ELECTRICAL POWER BUDGET                             |
+-----------------------------------------------------------------------------------+
| Component Subsystem        | Voltage Input | Nominal Power | Peak Power (Heater/Boost)
+----------------------------+---------------+---------------+----------------------+
| Compute Box (AGX Orin)     | 12V DC        | 90 W          | 130 W                |
| Cameras (8x-12x Visible)   | 12V DC (PoC)  | 35 W          | 50 W                 |
| Thermal IR Cameras (2x)    | 5V DC         | 3.2 W         | 5.0 W                |
| LiDAR Units (1x-2x)        | 12V / 24V DC  | 28 W          | 40 W                 |
| 4D RADAR Array (1x-4x)     | 12V DC        | 24 W          | 48 W                 |
| RTK-GNSS / IMU             | 12V DC        | 6 W           | 10 W                 |
| DBW Gateway & Actuators    | 12V / 24V DC  | 60 W          | 180 W (Full Lock Brake)
| Roof Pod Thermal Cooling   | 12V DC        | 45 W          | 80 W                 |
+----------------------------+---------------+---------------+----------------------+
| TOTAL SYSTEM POWER BUDGET  | 12V / 24V DC  | **291.2 W**   | **543.0 W**          |
+-----------------------------------------------------------------------------------+
```

### 8.2 Redundant Power Distribution Unit (PDU) & UPS Architecture

- **Primary Source:** Vehicle Heavy-Duty Alternator (24V 270A for Military/Truck, 12V 180A for RobotTaxi).
- **Secondary Power Sinks:** Dual AGM Aux Batteries connected via a Smart Isolator Diode.
- **Uninterruptible Power Supply (UPS):** Solid-state Supercapacitor Bank providing **60 seconds of full compute power (500W)** in the event of total engine stalls or main battery cable disconnects, allowing safe mechanical stop maneuvers.

---

## 9. Mounting & Mechanical Calibration Guidelines

### 9.1 Military Tactical Vehicle Sensor Layout (ASCII Schematic)

```
                       [ FRONT OF VEHICLE ]
  +-----------------------------------------------------------+
  |  [RADAR FL]              [CAM FRONT MAIN]      [RADAR FR] |
  |  Continental             Basler 120° FOV       Continental|
  |                                                           |
  |  [THERMAL L]                                 [THERMAL R]  |
  |  FLIR Boson+                                 FLIR Boson+  |
  |                                                           |
  |    +-------------------------------------------------+    |
  |    |        ROOF MOUNTED ARMORED SENSOR POD          |    |
  |    |  [LiDAR LEFT]    [SEPTENTRIO GNSS]  [LiDAR RIGHT]|    |
  |    |  Ouster OS1-128  Dual RTK Antenna   Ouster OS1-128 |    |
  |    +-------------------------------------------------+    |
  |                                                           |
  |  [CAM SIDE L]                               [CAM SIDE R]  |
  |  Basler 90° FOV                             Basler 90° FOV|
  |                                                           |
  |  [RADAR RL]              [CAM REAR MAIN]       [RADAR RR] |
  |  Continental             Basler 120° FOV       Continental|
  +-----------------------------------------------------------+
                        [ REAR OF VEHICLE ]
```

### 9.2 Class 8 Heavy Freight Truck Sensor Placement (ASCII Schematic)

```
                        [ TRUCK CAB FRONT ]
         +-----------------------------------------------+
         | [CAM TELE 15°]  [4D RADAR]  [CAM MID 60°]    |
         | Grille Mount    Front Bumper Grille Mount     |
         +-----------------------------------------------+
                                 |
         +-----------------------------------------------+
         |      CAB ROOF TOP POD (HEIGHT: 3.8m)          |
         | [Ouster OS1-128] [GNSS ANT 1] [GNSS ANT 2]    |
         +-----------------------------------------------+
            /                                         \
           /                                           \
   [MIRROR ARM POD L]                          [MIRROR ARM POD R]
   - Cam Side Forward 60°                      - Cam Side Forward 60°
   - Cam Rearward Blindspot                    - Cam Rearward Blindspot
                                 |
                        [ FIFTH WHEEL KINGPIN ]
                        - Trailer Hitch Angle Sensor
                                 |
                        [ TRAILER END BUMPER ]
                        - Rear 4D RADAR + Rear Backup Cam
```

### 9.3 Mechanical Installation Standards

- **LiDAR Roof Pod:** Machined billet T6-6061 aluminum frame hard-bolted to structural roof roll-cage pillars. Vibration dampening isolators must maintain a resonant frequency above \(120\text{ Hz}\).
- **Camera Calibration Stability:** Rigidity requirement of \(<0.05^\circ\) flex under 5G acceleration to preserve extrinsic camera-to-LiDAR calibration matrices.

---

## 10. Environmental & Ruggedization Requirements

### 10.1 MIL-STD-810H Compliance Matrix

| Environmental Test | MIL-STD-810H Method | Test Parameters / Conditions | Pass Criteria |
| :--- | :--- | :--- | :--- |
| **High Temperature** | Method 501.7 | \(+85^\circ\text{C}\) Operational (Procedure II), \(+105^\circ\text{C}\) Storage | Zero thermal shutdown or performance degradation |
| **Low Temperature** | Method 502.7 | \(-40^\circ\text{C}\) Operational (Procedure II) | Internal lens heaters engage; boot time < 45s |
| **Vibration** | Method 514.8 | Category 4 (Composite Wheeled Vehicle Profile), 5–500 Hz | Structural integrity maintained; no calibration drift |
| **Functional Shock** | Method 516.8 | Procedure I: 50G peak acceleration, 11 ms saw-tooth pulse | Compute and sensors maintain full operation |
| **Humidity** | Method 507.6 | 10 24-hour cycles @ 95% RH non-condensing | Zero internal corrosion or moisture ingress |
| **Salt Fog** | Method 509.7 | 5% NaCl saline atomizer spray for 48 hours | Chassis finish intact; zero pin corrosion |

### 10.2 Ingress Protection (IP) Standards

- **Roof Pod Sensors (LiDAR/Cameras):** **IP68 & IP69K** (Resistant to high-pressure hot water steam jets @ 100 bar, \(80^\circ\text{C}\)).
- **Compute Enclosure:** **IP67** dust-tight and water immersion up to 1 meter depth for 30 minutes.

---

## 11. Bill of Materials (BOM) & System Cost Breakdown

### 11.1 Tactical Military Vehicle Bill of Materials

| Item # | Component Description | Manufacturer & Part Number | Qty | Unit Cost (USD) | Total Cost (USD) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | NVIDIA DRIVE AGX Orin Compute Unit | NVIDIA `940-63710-0000-000` | 1 | $18,500 | $18,500 |
| 2 | Basler ace2 4K GMSL2 Visible Cameras | Basler `a2A3840-45gucBAS` | 12 | $1,200 | $14,400 |
| 3 | FLIR Boson+ Thermal IR Cameras | Teledyne FLIR `Boson+ 640` | 2 | $4,800 | $9,600 |
| 4 | Ouster OS1-128 3D LiDAR (Rev 7) | Ouster `OS1-128-REV7` | 2 | $14,000 | $28,000 |
| 5 | Continental ARS548 4D Imaging RADAR | Continental `ARS548` | 4 | $2,500 | $10,000 |
| 6 | Septentrio RTK-GNSS + Tactical IMU | Septentrio `AsteRx-i D` | 1 | $7,500 | $7,500 |
| 7 | PEAK PCAN-USB Pro FD CAN Interface | PEAK-System `IPEH-004062` | 2 | $850 | $1,700 |
| 8 | Rugged Armored Wiring Harness & Enclosure| Custom MIL-DTL-38999 Assembly | 1 | $5,500 | $5,500 |
| **TOTAL** | **Tactical Military System Cost** | | | | **$95,200** |

### 11.2 Autonomous Heavy Freight Truck Bill of Materials

| Item # | Component Description | Manufacturer & Part Number | Qty | Unit Cost (USD) | Total Cost (USD) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | NVIDIA DRIVE AGX Orin Compute Unit | NVIDIA `940-63710-0000-000` | 1 | $18,500 | $18,500 |
| 2 | Basler ace2 GMSL2 Visible Cameras | Basler `a2A1920-51gcBAS` | 8 | $950 | $7,600 |
| 3 | Ouster OS1-128 3D LiDAR (Rev 7) | Ouster `OS1-128-REV7` | 1 | $14,000 | $14,000 |
| 4 | Continental ARS548 4D Imaging RADAR | Continental `ARS548` | 2 | $2,500 | $5,000 |
| 5 | Septentrio RTK-GNSS + Tactical IMU | Septentrio `AsteRx-i D` | 1 | $7,500 | $7,500 |
| 6 | Kvaser Leaf Light v2 CAN Interface | Kvaser `73-30130-00241-6` | 2 | $450 | $900 |
| 7 | Heavy Duty Mounting Pod & Cables | Custom Class 8 Bracket Kit | 1 | $2,800 | $2,800 |
| **TOTAL** | **Class 8 Heavy Freight Truck System Cost** | | | | **$56,300** |

### 11.3 Urban RobotTaxi Bill of Materials

| Item # | Component Description | Manufacturer & Part Number | Qty | Unit Cost (USD) | Total Cost (USD) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | NVIDIA Jetson AGX Orin 64GB / DRIVE AGX | NVIDIA `900-13701-0000-000` | 1 | $2,499 | $2,499 |
| 2 | Basler ace2 Surround Cameras | Basler `a2A1920-51gcBAS` | 8 | $750 | $6,000 |
| 3 | Velodyne VLP-32C / Ouster OS1-64 LiDAR | Ouster / Velodyne `VLP-32C` | 1 | $9,500 | $9,500 |
| 4 | Bosch MRR3 Mid-Range RADAR | Bosch `MRR3` | 1 | $650 | $650 |
| 5 | u-blox ZED-F9P RTK-GNSS + IMU | u-blox `ZED-F9P` | 1 | $450 | $450 |
| 6 | Dataspeed Drive-By-Wire Kit | Dataspeed `DBW-CAMRY-V2` | 1 | $8,500 | $8,500 |
| 7 | Urban Roof Pod & Wiring Harness | Custom Composite Roof Pod | 1 | $1,800 | $1,800 |
| **TOTAL** | **Urban RobotTaxi System Cost** | | | | **$29,399** |
