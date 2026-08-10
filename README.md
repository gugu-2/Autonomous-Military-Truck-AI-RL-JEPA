# 🧠 OMNIDRIVE
## Omni-Purpose Autonomous Driving AI Brain

> **The world's most advanced open-source autonomous driving AI — combining JEPA World Models, Model-Based RL, Vision-Language Reasoning, and Autoware Navigation into a single unified brain.**

**Targets:** Military Vehicles · Autonomous Trucks · Robot Taxis  
**Autonomy Level:** L4 (Full Urban) / L4+ (Military Off-Road)  
**Core Tech:** Drive-JEPA + DreamerV3 + Alpamayo + Autoware (ROS 2)

---

## Architecture Overview

```
SENSORS → SENSOR FUSION → JEPA WORLD MODEL → RL CONTROLLER → AUTOWARE NAV → VEHICLE
                              ↑
                         ALPAMAYO VLA (Reasoning)
                              ↑
                         SAFETY SYSTEM (Cross-cutting)
```

## Documentation Index

All implementation documentation is in the `docs/` folder:

| Doc | Module | Description |
|---|---|---|
| [00_MASTER_DIRECTORY_STRUCTURE.md](docs/00_MASTER_DIRECTORY_STRUCTURE.md) | Project | Complete file & folder structure |
| [01_SENSOR_FUSION_MODULE.md](docs/01_SENSOR_FUSION_MODULE.md) | Layer 1 | Camera, LiDAR, Radar, GPS/IMU fusion |
| [02_JEPA_BRAIN_MODULE.md](docs/02_JEPA_BRAIN_MODULE.md) | Layers 2-3 | JEPA World Model (Imagination Engine) |
| [03_RL_CONTROLLER_MODULE.md](docs/03_RL_CONTROLLER_MODULE.md) | Layer 5 | DreamerV3 Model-Based RL Controller |
| [04_REASONING_MODULE.md](docs/04_REASONING_MODULE.md) | Layer 4 | Alpamayo VLA Reasoning |
| [05_AUTOWARE_NAVIGATION_MODULE.md](docs/05_AUTOWARE_NAVIGATION_MODULE.md) | Layer 6 | Autoware ROS 2 Navigation |
| [06_VEHICLE_INTERFACE_MODULE.md](docs/06_VEHICLE_INTERFACE_MODULE.md) | Layer 7 | CAN / JAUS / J1939 Vehicle Control |
| [07_SAFETY_SYSTEM_MODULE.md](docs/07_SAFETY_SYSTEM_MODULE.md) | Safety | ISO 26262 / ASIL-D Safety Systems |
| [08_TRAINING_PIPELINE.md](docs/08_TRAINING_PIPELINE.md) | Training | Full 3-phase training workflow |
| [09_TEST_STRATEGY.md](docs/09_TEST_STRATEGY.md) | Testing | Complete test suite strategy |
| [10_DEPLOYMENT_GUIDE.md](docs/10_DEPLOYMENT_GUIDE.md) | Deploy | Per-vehicle deployment guide |
| [11_HARDWARE_REQUIREMENTS.md](docs/11_HARDWARE_REQUIREMENTS.md) | Hardware | BOM + sensor specs |
| [12_DATA_PIPELINE.md](docs/12_DATA_PIPELINE.md) | Data | Dataset management + collection |

## Open Source Stack

| Component | Source | License |
|---|---|---|
| Autoware (Navigation) | autowarefoundation/autoware | Apache 2.0 |
| Drive-JEPA (AI Brain) | linhanwang/Drive-JEPA | MIT |
| CarDreamer (RL Trainer) | ucd-dare/CarDreamer | MIT |
| V-JEPA (Foundation) | facebookresearch/jepa | CC-NC |
| Alpamayo (Reasoning) | nvidia/alpamayo-recipes | OpenMDW-1.1 |
| CARLA (Simulator) | carla-simulator/carla | MIT |
