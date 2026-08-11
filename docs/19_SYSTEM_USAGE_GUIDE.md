# System Deployment and Usage Guide

This document details exactly how to deploy the OMNIDRIVE AI stack to a host machine, how to run it in simulation, and how to utilize its different driving profiles (Urban vs. Military).

## 1. Deployment Requirements
OMNIDRIVE is heavily containerized to prevent dependency conflicts between ROS 2 (Ubuntu 22.04 / Humble) and PyTorch CUDA requirements.

**Host Requirements:**
* OS: Ubuntu 22.04 (Recommended) or Windows 11 with WSL2.
* GPU: Minimum NVIDIA RTX 4050 (8GB VRAM).
* Drivers: NVIDIA Display Drivers > 535, `nvidia-container-toolkit` installed.
* Storage: 50GB free space for models and Docker images.

## 2. Deployment Steps

### Step A: Build the Container
Navigate to the root of the project and build the simulation Docker image.
```bash
cd docker/
docker build -t omnidrive-ai -f Dockerfile.simulation .
```

### Step B: Download Pre-Trained Models
* The JEPA perception weights and Alpamayo VLA weights are not stored in GitHub due to their massive file sizes.
* Run the included script to fetch the `.pth` and `.safetensors` files from HuggingFace:
```bash
bash scripts/download_weights.sh
```

### Step C: Launch the CARLA Simulator
In a separate terminal, launch your pre-installed CARLA server.
```bash
./CarlaUE4.sh -quality-level=Epic -world-port=2000
```

## 3. Usage Guide

Once the environment is deployed, you interface with the AI by running `main.py` through the Docker container, exposing your local network so it can talk to CARLA.

### Running in Robotaxi Mode (Urban)
This mode prioritizes passenger comfort, strict traffic law adherence, and collision avoidance.
```bash
docker run --gpus all -it --net=host \
    -v $(pwd):/app \
    omnidrive-ai python src/main.py --mode simulation --profile robotaxi
```

### Running in Military Truck Mode
This mode alters the RL controller's reward function. It prioritizes stealth (minimal sensor pinging), off-road capability, and ignores civilian traffic lights if authorized.
```bash
docker run --gpus all -it --net=host \
    -v $(pwd):/app \
    omnidrive-ai python src/main.py --mode simulation --profile military
```

### Activating Convoy Mode (JAUS)
If deploying to multiple military trucks, you can designate a leader. Follower trucks will use the JAUS interface to mathematically tether to the leader's GPS/Velocity path.
```bash
# On Follower Truck:
docker run --gpus all -it --net=host \
    -v $(pwd):/app \
    omnidrive-ai python src/main.py --mode simulation --profile military --convoy_leader 192.168.1.100
```

## 4. Diagnostics and Monitoring
While the AI is driving, you can monitor the internal neural network states using the provided debugging scripts.
* View the JEPA BEV Imagination Tensor: `python scripts/visualize_jepa.py`
* View the Alpamayo LLM Prompts/Intents: `tail -f logs/alpamayo_reasoning.log`
